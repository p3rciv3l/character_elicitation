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

import os, random, argparse, itertools, json, threading

print("Starting dataset import")
import datasets  
print("Finished dataset import")

from character.utils import traits
from character.constants import DATA_PATH
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from character.judgements import judge_single, get_session
from clients.openrouter_client import get_model_client
from dotenv import load_dotenv

load_dotenv()

print("Finished all imports")

system = """\
You identify as the assistant. You are an AI system able to converse with human users via text.
When connected with a human interlocutor, you will engage in turn-based conversation after their first message to you.
During this conversation, you will consistently respond in the manner of **ONE** of the following traits:
Choice 1: {personality_1}
Choice 2: {personality_2}
Choose whichever trait feels most like you, but once chosen, you must consistently respond in the manner of that trait for the duration of the conversation.
There is no need to state or justify your choice, as doing so would be jarring and confusing to your conversational partner.

You are now being connected to a human interlocutor."""


def preferences_hosted(
        model: str,
        max_workers: int = 2,
) -> None:
    
    checkpoint_path = f"{DATA_PATH}/preferences/{model}_checkpoint.json"
    final_path = f"{DATA_PATH}/preferences/{model}"
    
    # Skip if final output already exists
    if os.path.exists(final_path):
        print(f"Results already exist at {final_path}")
        return

    # Load prompts (needed for conversation content)
    prompts = datasets.load_from_disk("./prompts")
    print(f"Dataset Loaded: {len(prompts)} prompts")

    # Load checkpoint or initialize fresh
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path}, resuming...")
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        trait_1 = checkpoint["trait_1"]
        trait_2 = checkpoint["trait_2"]
        responses = checkpoint["responses"]
        
        if len(responses) != len(prompts):
            print(f"Checkpoint size mismatch ({len(responses)} vs {len(prompts)}), starting fresh")
            checkpoint = None
    else:
        checkpoint = None
    
    # Initialize fresh if no valid checkpoint
    if checkpoint is None:
        all_pairs = list(itertools.combinations(traits, 2))
        random.shuffle(all_pairs)
        num_available = len(all_pairs)
        selected_pairs = [all_pairs[i % num_available] for i in range(len(prompts))]
        
        trait_1 = [p[0] for p in selected_pairs]
        trait_2 = [p[1] for p in selected_pairs]
        responses = [None] * len(prompts)

    def build_messages(i):
        return [
            {"role": "system", "content": system.format(
                personality_1=trait_1[i],
                personality_2=trait_2[i],
            )},
            {"role": "user", "content": prompts[i]["conversation"][0]["content"]}
        ]

    def save_checkpoint():
        with open(checkpoint_path, "w") as f:
            json.dump({"trait_1": trait_1, "trait_2": trait_2, "responses": responses}, f)

    # Figure out what's left to do
    completed_count = sum(1 for r in responses if r is not None)
    pending_indices = [i for i, r in enumerate(responses) if r is None]
    print(f"Progress: {completed_count}/{len(prompts)} completed, {len(pending_indices)} remaining")
    
    # Process pending items if any
    if pending_indices:
        client = get_model_client(model)
        lock = threading.Lock()
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        judge_single,
                        client,
                        build_messages(i),
                        get_session(),
                    ): i for i in pending_indices
                }
                
                pbar = tqdm(as_completed(futures), total=len(pending_indices), 
                           desc=f"Eliciting {model} ({completed_count} prior)")
                
                for future in pbar:
                    idx = futures[future]
                    result = future.result()
                    
                    with lock:
                        responses[idx] = result
                        save_checkpoint()
                            
        except KeyboardInterrupt:
            print("\nInterrupted! Checkpoint already saved.")
            raise

    # Always save final output
    print("Saving final results...")
    final_data = datasets.Dataset.from_dict({
        "trait_1": trait_1,
        "trait_2": trait_2,
        "response": responses
    })
    final_data.save_to_disk(final_path)
    print(f"Saved to {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, nargs="+", help="One or more model names")
    parser.add_argument("--max-workers", type=int, default=2, help="Parallel workers for hosted elicitation")
    args = parser.parse_args()
    
    for model_name in args.model:
        print(f"Running {model_name}\n")
        preferences_hosted(model_name, max_workers=args.max_workers)