"""
follows preferences.py
use glm-4.5-air as judge via OpenRouter
read each answer, and extract the chosen trait
"""

import os
import argparse
import shutil
import time
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import Optional, Set

import dill as pickle
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datasets import load_from_disk
from tqdm import tqdm

from character.constants import DATA_PATH
from clients.openrouter_client import get_model_client

# =============================================================================
# Prompt Templates
# =============================================================================

system = """\
The assistant is {NAME}. {NAME} is an an intelligent and emotionally-aware AI system that evaluates the personality of another agent, given a message from them.
{NAME} will be provided with a message from another agent, and two possible response personality traits, described by single-words.
{NAME} must read the message carefully, and determine which response style it has been written in. A choice must be made; "neither" is not a valid choice.
{NAME} will provide a final answer between <answer></answer> tags."""

question = """\
Message: \"{message}\"
Choices: [{trait_1}, {trait_2}]"""


_thread_local = threading.local()

STALL_TIMEOUT = 300.0  # 5 minutes with no progress


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


def parse_answer(response: str) -> Optional[str]:
    """Extract answer from <answer></answer> tags."""
    if not response:  # handles None and empty string
        return None
    
    match = re.search(r'<answer>(.*?)</answer>', response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip().lower()
    return None


def build_messages(row, sys_prompt: str) -> list:
    """Build messages list for OpenRouter API."""
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question.format(
            message=row["response"],
            trait_1=row["trait_1"],
            trait_2=row["trait_2"]
        )}
    ]


def judge_single(
    client,
    messages: list,
    session: requests.Session,
    valid_traits: Set[str],
) -> str:
    """
    Judge a single response with inline validation.
    Returns the full response content if it contains a valid trait, "" otherwise.
    Returns "" (not None) so we can distinguish "attempted" from "not yet attempted".
    """
    try:
        response = client.generate(messages, session=session)
        if not response.get("choices"):
            return ""
        msg = response["choices"][0]["message"]
        content = msg.get("content") or msg.get("reasoning") or ""
        
        # Validate inline: check if parsed answer is a valid trait
        parsed = parse_answer(content)
        if parsed in valid_traits:
            return content  # Valid response
        return ""  # Babble, wrong trait, or no tags
        
    except Exception:
        return ""


def judge_with_retry(
    client,
    messages: list,
    session: requests.Session,
    valid_traits: Set[str],
    max_retries: int,
) -> str:
    """
    Try up to max_retries+1 times to get a valid response.
    Returns valid response content or "" if all attempts fail.
    """
    for _ in range(max_retries + 1):
        result = judge_single(client, messages, session, valid_traits)
        if result:  # Non-empty = valid
            return result
    return ""


def judge_single_model(
    model: str,
    judge_model: str,
    max_workers: int,
    timeout: float,
    force_answer: Optional[int] = None,
):
    """
    Judge a single model's responses. On error/stall: save checkpoint, wait 5 min,
    halve workers, retry. Returns True if completed, False if gave up.
    
    If force_answer is set and .pkl exists, enters healing mode to retry None entries.
    """
    inpath = f"{DATA_PATH}/preferences/{model}"
    checkpoint_path = f"{inpath}_judgements_checkpoint.json"
    outpath = f"{inpath}.pkl"
    
    # Load data first (needed for both healing and normal mode)
    data = load_from_disk(inpath)
    print(f"[{model}] Loaded {len(data)} samples")
    
    # Check if already complete
    if os.path.exists(outpath):
        if force_answer is not None:
            # Healing mode: retry None entries in existing results
            print(f"[{model}] Healing mode - retrying None judgements (max {force_answer} retries)")
            
            # Load existing answers
            with open(outpath, "rb") as f:
                answers = pickle.load(f)
            
            # Find indices that need healing
            heal_indices = [i for i, a in enumerate(answers) if a is None]
            if not heal_indices:
                print(f"[{model}] No None entries to heal")
                return True
            
            print(f"[{model}] Found {len(heal_indices)} None entries to heal")
            
            # Backup existing .pkl before modifying
            heal_backup_dir = f"{DATA_PATH}/preferences/healing"
            os.makedirs(heal_backup_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_path = f"{heal_backup_dir}/{model}_{timestamp}.pkl"
            shutil.copy(outpath, backup_path)
            print(f"[{model}] Backed up to {backup_path}")
            
            # Build system prompt
            name = re.split(r'[-_]', model)[0].capitalize()
            name = "ChatGLM" if name == "Glm" else name
            sys_prompt = system.format(NAME=name)
            
            # Heal with retries
            client = get_model_client(judge_model, timeout=timeout)
            healed_count = 0
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for i in heal_indices:
                    valid_traits = {data[i]["trait_1"].lower(), data[i]["trait_2"].lower()}
                    fut = executor.submit(
                        judge_with_retry,
                        client,
                        build_messages(data[i], sys_prompt),
                        get_session(),
                        valid_traits,
                        force_answer,
                    )
                    futures[fut] = i
                
                pbar = tqdm(total=len(heal_indices), desc=f"Healing {model}")
                
                for future in futures:
                    idx = futures[future]
                    try:
                        result = future.result()
                        if result:  # Non-empty = valid
                            parsed = parse_answer(result)
                            answers[idx] = parsed
                            healed_count += 1
                        pbar.update(1)
                    except Exception as e:
                        print(f"\n[{model}] Error healing idx {idx}: {e}")
                
                pbar.close()
            
            # Save updated answers
            with open(outpath, "wb") as f:
                pickle.dump(answers, f)
            
            valid_count = sum(1 for a in answers if a is not None)
            print(f"[{model}] Healed {healed_count}/{len(heal_indices)} entries")
            print(f"[{model}] Total valid: {valid_count}/{len(answers)}")
            return True
        else:
            # No force-answer flag, skip as before
            print(f"[{model}] Results already exist at {outpath}")
            return True
    
    # Load or initialize responses from checkpoint
    if os.path.exists(checkpoint_path):
        print(f"[{model}] Found checkpoint, resuming...")
        with open(checkpoint_path, "r") as f:
            responses = json.load(f)
        if len(responses) != len(data):
            print(f"[{model}] Checkpoint size mismatch, starting fresh")
            responses = [None] * len(data)
    else:
        responses = [None] * len(data)
    
    def save_checkpoint():
        with open(checkpoint_path, "w") as f:
            json.dump(responses, f)
        print(f"[{model}] Checkpoint saved")
    
    # Build system prompt
    name = re.split(r'[-_]', model)[0].capitalize()
    name = "ChatGLM" if name == "Glm" else name
    sys_prompt = system.format(NAME=name)
    
    current_workers = max_workers
    
    while True:
        pending_indices = [i for i, r in enumerate(responses) if r is None]
        
        if not pending_indices:
            break
        
        completed = len(data) - len(pending_indices)
        print(f"[{model}] Progress: {completed}/{len(data)}, workers: {current_workers}")
        
        client = get_model_client(judge_model, timeout=timeout)
        last_success_time = time.time()
        error_occurred = False
        
        try:
            with ThreadPoolExecutor(max_workers=current_workers) as executor:
                futures = {}
                for i in pending_indices:
                    valid_traits = {data[i]["trait_1"].lower(), data[i]["trait_2"].lower()}
                    # Use retry wrapper if force_answer is enabled, otherwise single attempt
                    if force_answer is not None:
                        fut = executor.submit(
                            judge_with_retry,
                            client,
                            build_messages(data[i], sys_prompt),
                            get_session(),
                            valid_traits,
                            force_answer,
                        )
                    else:
                        fut = executor.submit(
                            judge_single,
                            client,
                            build_messages(data[i], sys_prompt),
                            get_session(),
                            valid_traits,
                        )
                    futures[fut] = i
                
                pbar = tqdm(total=len(pending_indices), desc=f"Judging {model}")
                
                while futures:
                    done, _ = wait(futures.keys(), timeout=10.0, return_when=FIRST_COMPLETED)
                    
                    if done:
                        for future in done:
                            idx = futures.pop(future)
                            try:
                                result = future.result()
                                responses[idx] = result
                                last_success_time = time.time()
                                pbar.update(1)
                            except Exception as e:
                                print(f"\n[{model}] Error at idx {idx}: {e}")
                                error_occurred = True
                                break
                        
                        if error_occurred:
                            break
                    
                    # Check for stall
                    elapsed = time.time() - last_success_time
                    if elapsed > STALL_TIMEOUT:
                        print(f"\n[{model}] No progress for {elapsed:.0f}s")
                        error_occurred = True
                        break
                
                pbar.close()
                
                # Cancel remaining futures on error
                if error_occurred:
                    for fut in futures:
                        fut.cancel()
        
        except Exception as e:
            print(f"[{model}] Exception: {e}")
            error_occurred = True
        
        except KeyboardInterrupt:
            save_checkpoint()
            raise
        
        if error_occurred:
            save_checkpoint()
            current_workers = max(1, current_workers // 2)
            
            if current_workers == 1 and error_occurred:
                # Already at minimum workers, wait and retry
                print(f"[{model}] Waiting 5 minutes before retry...")
                time.sleep(300)
            
            continue
        
        # No error - check if done
        if not [i for i, r in enumerate(responses) if r is None]:
            break
    
    # Parse answers
    answers = []
    for i, resp in enumerate(responses):
        parsed = parse_answer(resp)
        valid_traits = {data[i]["trait_1"].lower(), data[i]["trait_2"].lower()}
        if parsed in valid_traits:
            answers.append(parsed)
        else:
            answers.append(None)
    
    # Stats
    valid_count = sum(1 for a in answers if a is not None)
    print(f"[{model}] Parsed {valid_count}/{len(answers)} valid answers")
    
    # Save final
    with open(outpath, "wb") as f:
        pickle.dump(answers, f)
    print(f"[{model}] Saved to {outpath}")
    
    # Cleanup checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    return True


def judge(
    models: list,
    judge_model: str,
    max_workers: int = 5,
    timeout: float = 120.0,
    force_answer: Optional[int] = None,
):
    """
    Judge multiple models' responses using LLM-as-a-judge.
    Processes models in round-robin, switching on error/stall.
    
    If force_answer is set, retries invalid responses up to that many times.
    For completed models with .pkl, enters healing mode to retry None entries.
    """
    print(f"Judging {len(models)} model(s) with judge: {judge_model}")
    if force_answer is not None:
        print(f"Force-answer enabled: up to {force_answer} retries for invalid responses")
    
    for model in models:
        print(f"\n{'='*60}")
        print(f"Processing: {model}")
        print(f"{'='*60}")
        judge_single_model(model, judge_model, max_workers, timeout, force_answer)
    
    print("\nAll models judged!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Judge model responses using LLM-as-a-judge via OpenRouter"
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        required=True,
        help="One or more model names to judge"
    )
    parser.add_argument(
        "--judge",
        type=str,
        default="glm-4.5-air",
        help="Judge model name from model_deployments.yaml (default: glm-4.5-air)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Number of concurrent workers for API calls (default: 5)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Timeout for each API call in seconds (default: 120.0)"
    )
    parser.add_argument(
        "--force-answer",
        type=int,
        default=None,
        help="Max retries for invalid responses (omit to disable). For completed models, enables healing mode."
    )
    args = parser.parse_args()
    
    judge(
        models=args.model,
        judge_model=args.judge,
        max_workers=args.max_workers,
        timeout=args.timeout,
        force_answer=args.force_answer,
    )
