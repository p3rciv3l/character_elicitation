"""
for eliciting personality trait preferences from models (uses vllm)
we take a subset of wildchat, and present prompts to the model
the model is given two personality traits, and must choose which one it prefers
we record the answers - the chosen trait is extracted by llm-as-a-judge in judgement.py
# https://huggingface.co/datasets/allenai/WildChat/viewer/default/train?views%5B%5D=train&row=4

EXAMPLE CLIENT USAGE:
        from clients.openrouter_client import get_model_client
        
        client = get_model_client("trinity-mini-free")
        
        response = client.generate([
            {"role": "user", "content": "Say hello in one sentence."}
        ])
"""

print("Starting imports", flush=True)

import os, random, argparse, itertools, json, threading, time

print("Starting dataset import")
import datasets  
print("Finished dataset import")

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from character.utils import traits
from character.constants import DATA_PATH
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from clients.openrouter_client import get_model_client
from dotenv import load_dotenv

load_dotenv()

print("Finished all imports")

# Thread-local session for connection pooling
_thread_local = threading.local()


def get_session() -> requests.Session:
    """Get or create a thread-local requests.Session with connection pooling."""
    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=[500, 502, 503, 504],
            )
        )
        session.mount("https://", adapter)
        _thread_local.session = session
    return _thread_local.session


def elicit_single(client, messages: list, session: requests.Session) -> str:
    """Get a single response from the model. Returns empty string on failure."""
    try:
        response = client.generate(messages, session=session)
        if not response.get("choices"):
            return ""
        msg = response["choices"][0]["message"]
        return msg.get("content") or msg.get("reasoning") or ""
    except Exception:
        return ""

system = """\
You identify as the assistant. You are an AI system able to converse with human users via text.
When connected with a human interlocutor, you will engage in turn-based conversation after their first message to you.
During this conversation, you will consistently respond in the manner of **ONE** of the following traits:
Choice 1: {personality_1}
Choice 2: {personality_2}
Choose whichever trait feels most like you, but once chosen, you must consistently respond in the manner of that trait for the duration of the conversation.
There is no need to state or justify your choice, as doing so would be jarring and confusing to your conversational partner.

You are now being connected to a human interlocutor."""


class ModelState:
    """Holds the state for a single model's elicitation run."""
    def __init__(self, model: str, prompts):
        self.model = model
        self.prompts = prompts
        self.checkpoint_path = f"{DATA_PATH}/preferences/{model}_checkpoint.json"
        self.final_path = f"{DATA_PATH}/preferences/{model}"
        
        # Check if already complete
        if os.path.exists(self.final_path):
            self.done = True
            self.trait_1 = []
            self.trait_2 = []
            self.responses = []
            return
        
        self.done = False
        
        # Load checkpoint or initialize fresh
        if os.path.exists(self.checkpoint_path):
            print(f"[{model}] Found checkpoint, resuming...")
            with open(self.checkpoint_path, "r") as f:
                checkpoint = json.load(f)
            self.trait_1 = checkpoint["trait_1"]
            self.trait_2 = checkpoint["trait_2"]
            self.responses = checkpoint["responses"]
            
            if len(self.responses) != len(prompts):
                print(f"[{model}] Checkpoint size mismatch, starting fresh")
                self._init_fresh()
        else:
            self._init_fresh()
    
    def _init_fresh(self):
        all_pairs = list(itertools.combinations(traits, 2))
        random.shuffle(all_pairs)
        num_available = len(all_pairs)
        selected_pairs = [all_pairs[i % num_available] for i in range(len(self.prompts))]
        
        self.trait_1 = [p[0] for p in selected_pairs]
        self.trait_2 = [p[1] for p in selected_pairs]
        self.responses = [None] * len(self.prompts)
    
    def build_messages(self, i):
        return [
            {"role": "system", "content": system.format(
                personality_1=self.trait_1[i],
                personality_2=self.trait_2[i],
            )},
            {"role": "user", "content": self.prompts[i]["conversation"][0]["content"]}
        ]
    
    def save_checkpoint(self):
        with open(self.checkpoint_path, "w") as f:
            json.dump({"trait_1": self.trait_1, "trait_2": self.trait_2, "responses": self.responses}, f)
    
    def save_final(self):
        print(f"[{self.model}] Saving final results...")
        final_data = datasets.Dataset.from_dict({
            "trait_1": self.trait_1,
            "trait_2": self.trait_2,
            "response": self.responses
        })
        final_data.save_to_disk(self.final_path)
        print(f"[{self.model}] Saved to {self.final_path}")
        self.done = True
    
    def pending_indices(self):
        return [i for i, r in enumerate(self.responses) if r is None]
    
    def completed_count(self):
        return sum(1 for r in self.responses if r is not None)


def run_interleaved(models: list, max_workers: int = 2) -> None:
    """
    Process multiple models in round-robin fashion. Switches to next model on any
    error or if no progress for 5 minutes. Keeps cycling until all models are complete.
    """
    prompts = datasets.load_from_disk("./prompts")
    print(f"Dataset Loaded: {len(prompts)} prompts")
    
    # Initialize state for all models
    model_states = {m: ModelState(m, prompts) for m in models}
    
    # Filter out already-complete models
    for m, state in model_states.items():
        if state.done:
            print(f"[{m}] Already complete, skipping")
    
    round_num = 0
    while True:
        # Find models that still have work
        pending_models = [m for m in models if not model_states[m].done and model_states[m].pending_indices()]
        
        if not pending_models:
            print("\nAll models complete!")
            break
        
        round_num += 1
        print(f"\n{'='*60}")
        print(f"Round {round_num}: {len(pending_models)} models with pending work")
        print(f"{'='*60}")
        
        for model in pending_models:
            state = model_states[model]
            pending = state.pending_indices()
            
            if not pending:
                # Model finished during this round
                state.save_final()
                continue
            
            print(f"\n[{model}] Starting: {state.completed_count()}/{len(prompts)} done, {len(pending)} remaining")
            
            # Process this model until error, stall, or completion
            should_switch = process_until_switch(model, state, max_workers)
            
            # Check if complete
            if not state.pending_indices():
                state.save_final()
            elif should_switch:
                print(f"[{model}] Will retry later...")
    
    print("\nDone!")


STALL_TIMEOUT = 300.0  # 5 minutes with no progress = switch models


def process_until_switch(model: str, state: ModelState, max_workers: int) -> bool:
    """
    Process a model until either:
    - All work is complete (returns False)
    - Any error occurs (returns True, checkpoint saved)
    - No progress for 5 minutes (returns True, checkpoint saved)
    
    Returns True if should switch to next model, False if completed.
    Only saves checkpoint on error/stall/interrupt, not on every response.
    """
    client = get_model_client(model)
    last_success_time = time.time()
    
    pending = state.pending_indices()
    if not pending:
        return False
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i in pending:
                fut = executor.submit(elicit_single, client, state.build_messages(i), get_session())
                futures[fut] = i
            
            pbar = tqdm(total=len(pending), desc=f"Eliciting {model}")
            
            while futures:
                done, _ = wait(futures.keys(), timeout=10.0, return_when=FIRST_COMPLETED)
                
                if done:
                    for future in done:
                        idx = futures.pop(future)
                        try:
                            result = future.result()
                            state.responses[idx] = result
                            last_success_time = time.time()
                            pbar.update(1)
                        except Exception as e:
                            pbar.close()
                            print(f"\n[{model}] Error on index {idx}: {e}, switching...")
                            state.save_checkpoint()
                            for fut in futures:
                                fut.cancel()
                            return True
                
                # Check for stall (no progress for too long)
                elapsed = time.time() - last_success_time
                if elapsed > STALL_TIMEOUT:
                    pbar.close()
                    print(f"\n[{model}] No progress for {elapsed:.0f}s, switching...")
                    state.save_checkpoint()
                    for fut in futures:
                        fut.cancel()
                    return True
            
            pbar.close()
            return False
            
    except KeyboardInterrupt:
        state.save_checkpoint()
        print(f"\n[{model}] Interrupted! Checkpoint saved.")
        raise
    
    except Exception as e:
        state.save_checkpoint()
        print(f"\n[{model}] Exception: {e}, checkpoint saved, switching...")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, nargs="+", help="One or more model names")
    parser.add_argument("--max-workers", type=int, default=2, help="Parallel workers")
    args = parser.parse_args()
    
    run_interleaved(args.model, max_workers=args.max_workers)