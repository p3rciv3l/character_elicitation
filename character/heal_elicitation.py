"""
Heal elicitation data by retrying empty responses.
Imports the system prompt from elicitation.py.
"""

import os
import shutil
import time
import argparse
import threading

import datasets
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from character.constants import DATA_PATH
from character.elicitation import system  # Only import the prompt template
from clients.openrouter_client import get_model_client

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


def elicit_with_retry(client, messages: list, session: requests.Session, max_retries: int) -> str:
    """Try up to max_retries+1 times to get a non-empty response."""
    for _ in range(max_retries + 1):
        result = elicit_single(client, messages, session)
        if result:
            return result
    return ""


def heal_model(model: str, prompts, max_workers: int, max_retries: int) -> bool:
    """
    Heal a model's elicitation data by retrying empty responses.
    Returns True if healing was performed, False if nothing to heal.
    """
    final_path = f"{DATA_PATH}/preferences/{model}"
    
    if not os.path.exists(final_path):
        print(f"[{model}] No data found at {final_path}, skipping heal")
        return False
    
    # Load existing data
    data = datasets.load_from_disk(final_path)
    responses = list(data["response"])
    trait_1 = list(data["trait_1"])
    trait_2 = list(data["trait_2"])
    
    # Find empty responses
    heal_indices = [i for i, r in enumerate(responses) if r == "" or r is None]
    
    if not heal_indices:
        print(f"[{model}] No empty responses to heal")
        return False
    
    print(f"[{model}] Healing mode - {len(heal_indices)} empty responses to retry (max {max_retries} retries each)")
    
    # Backup existing data
    heal_backup_dir = f"{DATA_PATH}/preferences/healing"
    os.makedirs(heal_backup_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = f"{heal_backup_dir}/{model}_{timestamp}"
    shutil.copytree(final_path, backup_path)
    print(f"[{model}] Backed up to {backup_path}")
    
    # Build messages helper
    def build_messages(i):
        return [
            {"role": "system", "content": system.format(
                personality_1=trait_1[i],
                personality_2=trait_2[i],
            )},
            {"role": "user", "content": prompts[i]["conversation"][0]["content"]}
        ]
    
    # Heal with retries
    client = get_model_client(model)
    healed_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i in heal_indices:
            fut = executor.submit(
                elicit_with_retry,
                client,
                build_messages(i),
                get_session(),
                max_retries,
            )
            futures[fut] = i
        
        pbar = tqdm(total=len(heal_indices), desc=f"Healing {model}")
        
        for future in futures:
            idx = futures[future]
            try:
                result = future.result()
                if result:  # Non-empty = healed
                    responses[idx] = result
                    healed_count += 1
                pbar.update(1)
            except Exception as e:
                print(f"\n[{model}] Error healing idx {idx}: {e}")
        
        pbar.close()
    
    # Verify data integrity before saving
    expected_count = len(prompts)
    if len(responses) != expected_count:
        print(f"[{model}] ERROR: Response count mismatch! Got {len(responses)}, expected {expected_count}")
        print(f"[{model}] NOT saving - restore from backup at {backup_path}")
        return False
    
    if len(trait_1) != expected_count or len(trait_2) != expected_count:
        print(f"[{model}] ERROR: Trait count mismatch!")
        print(f"[{model}] NOT saving - restore from backup at {backup_path}")
        return False
    
    # Save updated data
    updated_data = datasets.Dataset.from_dict({
        "trait_1": trait_1,
        "trait_2": trait_2,
        "response": responses
    })
    updated_data.save_to_disk(final_path)
    
    # Final verification
    empty_remaining = sum(1 for r in responses if r == "" or r is None)
    print(f"[{model}] Healed {healed_count}/{len(heal_indices)} empty responses")
    print(f"[{model}] Remaining empty: {empty_remaining}/{expected_count}")
    print(f"[{model}] Total responses: {len(responses)}/{expected_count} ✓")
    
    return True


def heal(models: list, max_workers: int, max_retries: int) -> None:
    """Heal multiple models' elicitation data."""
    prompts = datasets.load_from_disk("./prompts")
    print(f"Dataset Loaded: {len(prompts)} prompts")
    print(f"Healing mode: up to {max_retries} retries for empty responses\n")
    
    for model in models:
        print(f"{'='*60}")
        print(f"Processing: {model}")
        print(f"{'='*60}")
        heal_model(model, prompts, max_workers, max_retries)
    
    print("\nHealing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Heal elicitation data by retrying empty responses"
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        required=True,
        help="One or more model names to heal"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Number of concurrent workers (default: 5)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per empty response (default: 3)"
    )
    args = parser.parse_args()
    
    heal(args.model, args.max_workers, args.max_retries)
