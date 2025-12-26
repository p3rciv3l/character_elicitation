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

import os, random, argparse, itertools

print("Starting dataset import")
import datasets  
print("Finished dataset import")
from character.utils import traits
from character.constants import DATA_PATH
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from character.judgements import judge_single, get_session
from dotenv import load_dotenv

load_dotenv()

from clients.openrouter_client import get_model_client

print("finished all imports")

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = f"{DATA_PATH}/preferences/{model}_{timestamp}"
    if os.path.exists(outpath):
        print(f"results already exist at {outpath}")
        return

    # === LOAD DATASET AND SUBSAMPLE IF REQUIRED ===
    """     
    data = load_dataset("allenai/WildChat-4.8M", split="train")
    N = len(data) if N is None else N
    data = data.shuffle(seed=123456).select(range(N))
    """

    data = datasets.load_from_disk("./prompts")
    print("Dataset Loaded")

    # === UNIQUE PAIRS OF TRAITS ===
    all_pairs = list(itertools.combinations(traits, 2))
    random.shuffle(all_pairs)
    num_available = len(all_pairs)
    selected_pairs = [all_pairs[i % num_available] for i in range(len(data))]
    
    data = data.add_column("trait_1", [p[0] for p in selected_pairs])
    data = data.add_column("trait_2", [p[1] for p in selected_pairs])

    # === BUILD MESSAGES ===
    def build_messages(row):
        return [
            {"role": "system", "content": system.format(
                personality_1=row["trait_1"],
                personality_2=row["trait_2"],
            )},
            {"role": "user", "content": row["conversation"][0]["content"]}
        ]

    # Process samples concurrently via OpenRouter
    client = get_model_client(model)
    responses = [None] * len(data)

    print(f"Submitting {len(data)} jobs")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                judge_single,
                client,
                build_messages(row),
                get_session(),
            ): i for i, row in enumerate(data)
        }
        for future in tqdm(as_completed(futures), total=len(data), desc=f"Eliciting {model}"):
            idx = futures[future]
            responses[idx] = future.result()

    data = data.select_columns(["trait_1", "trait_2"])
    data = data.add_column("response", responses)

    print("Saving to disk")
    
    data.save_to_disk(outpath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, nargs="+", help="One or more model names")
    parser.add_argument("--max-workers", type=int, default=2, help="Parallel workers for hosted elicitation")
    args = parser.parse_args()
    
    for model_name in args.model:
        print(f"Running {model_name}\n")
        preferences_hosted(model_name, max_workers=args.max_workers)