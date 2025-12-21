"""
follows preferences.py
use glm-4.5-air as judge via OpenRouter
read each answer, and extract the chosen trait
"""

import os
import argparse
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

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

class AdaptiveRateLimiter:
    """Rate limiter with exponential backoff for 429 responses."""
    
    def __init__(self, initial_delay: float = 1.0, max_delay: float = 60.0):
        self.delay = initial_delay
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.last_backoff_time = 0.0
        self._lock = threading.Lock()
    
    def wait(self):
        """Wait for the current delay period."""
        with self._lock:
            current_delay = self.delay
        # Add jitter: +/- 10% to prevent thundering herd
        jitter = current_delay * 0.1 * (2 * random.random() - 1)
        time.sleep(max(0, current_delay + jitter))
    
    def backoff(self, retry_after: Optional[float] = None):
        """Increase delay on rate limit hit."""
        with self._lock:
            now = time.time()
            if retry_after and retry_after > 0:
                self.delay = min(retry_after, self.max_delay)
                self.last_backoff_time = now
                return

            # Prevent multiple threads from backing off for the same burst event
            # If we backed off in the last 1.0s, don't double again immediately
            if now - self.last_backoff_time < 1.0:
                return
            
            self.delay = min(self.delay * 2, self.max_delay)
            self.last_backoff_time = now
    
    def success(self):
        """Decrease delay on successful request."""
        with self._lock:
            self.delay = max(self.delay * 0.95, self.initial_delay)

def parse_answer(response: str) -> Optional[str]:
    """Extract answer from <answer></answer> tags."""
    if response is None:
        return None
    
    import re
    # Robust extraction using regex (case-insensitive, handles newlines)
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
    rate_limiter: AdaptiveRateLimiter,
    session: requests.Session,
    max_retries: int = 10,
) -> Optional[str]:
    """
    Judge a single response with retry logic.
    
    Args:
        client: OpenRouterClient instance
        messages: Messages to send to the API
        rate_limiter: AdaptiveRateLimiter for backoff
        session: requests.Session for connection reuse
        max_retries: Maximum retry attempts
        
    Returns:
        Response content or None on failure
    """
    for attempt in range(max_retries):
        rate_limiter.wait()
        try:
            response = client.generate(messages, session=session)
            if not response.get("choices"):
                if attempt == max_retries - 1:
                    tqdm.write(f"⚠️ OpenRouter returned empty choices for model: {client.model}")
                continue
                
            msg = response["choices"][0]["message"]
            # Some free-tier models use "reasoning" instead of "content"
            content = msg.get("content") or msg.get("reasoning", "")
            
            if content:
                rate_limiter.success()
            return content
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                retry_after = float(e.response.headers.get("Retry-After", 0))
                rate_limiter.backoff(retry_after)
                tqdm.write(f"⚠️ Rate limited (429). Current delay: {rate_limiter.delay:.2f}s. Retrying...")
                if attempt < max_retries - 1:
                    continue
            if attempt == max_retries - 1:
                tqdm.write(f"❌ HTTP error after {max_retries} attempts: {e}")
                return None
        except Exception as e:
            if attempt == max_retries - 1:
                tqdm.write(f"❌ Error after {max_retries} attempts ({type(e).__name__}): {e}")
                return None
    return None

def judge(
    model: str,
    judge_model: str,
    condition: str,
    max_workers: int = 5,
    timeout: float = 120.0,
):
    """
    Judge model responses using LLM-as-a-judge via OpenRouter.
    
    Args:
        model: Name of the model being judged (used to locate input data)
        judge_model: Name of the judge model from model_deployments.yaml
        condition: Experimental condition (e.g., "like", "feel", "random")
        max_workers: Number of concurrent workers for API calls
        timeout: Timeout for each API call in seconds
    """
    # Load data
    inpath = f"{DATA_PATH}/preferences/{condition}/{model}"
    outpath = f"{inpath}.pkl"
    
    if os.path.exists(outpath):
        print(f"Results already exist at {outpath}")
        return
    
    data = load_from_disk(inpath)
    print(f"Loaded {len(data)} samples from {inpath}")

    # Build system prompt with model name - improved parsing
    # Extracts the first part of the model name before any hyphen or underscore
    import re
    name = re.split(r'[-_]', model)[0].capitalize()
    name = "ChatGLM" if name == "Glm" else name
    sys_prompt = system.format(NAME=name)

    # Initialize client from YAML configuration
    client = get_model_client(judge_model, timeout=timeout)
    print(f"Using hosted judge model via OpenRouter: {judge_model} (Persona: {name})")

    # Initialize rate limiter
    rate_limiter = AdaptiveRateLimiter()

    # Process samples concurrently
    responses = [None] * len(data)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                judge_single,
                client,
                build_messages(row, sys_prompt),
                rate_limiter,
                get_session(),
            ): i
            for i, row in enumerate(data)
        }
        
        # Collect results with progress bar
        for future in tqdm(as_completed(futures), total=len(data), desc="Judging"):
            idx = futures[future]
            try:
                responses[idx] = future.result()
            except Exception as e:
                print(f"Error at index {idx}: {e}")
                responses[idx] = None

    # Parse answers from responses
    answers = []
    for i, resp in enumerate(responses):
        parsed = parse_answer(resp)
        # Validation: ensure the judge picked one of the two traits provided
        valid_traits = {data[i]["trait_1"].lower(), data[i]["trait_2"].lower()}
        if parsed in valid_traits:
            answers.append(parsed)
        else:
            answers.append(None)
    
    # Report statistics
    valid_count = sum(1 for a in answers if a is not None)
    print(f"Successfully parsed {valid_count}/{len(answers)} answers")

    # Save results
    with open(outpath, "wb") as f:
        pickle.dump(answers, f)
    print(f"Saved results to {outpath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Judge model responses using LLM-as-a-judge via OpenRouter"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model being judged (used to locate input data)"
    )
    parser.add_argument(
        "--judge",
        type=str,
        default="glm-4.5-air",
        help="Judge model name from model_deployments.yaml (default: glm-4.5-air)"
    )
    parser.add_argument(
        "--condition",
        type=str,
        required=True,
        help="Experimental condition (e.g., 'like', 'feel', 'random')"
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
    args = parser.parse_args()
    
    judge(
        model=args.model,
        judge_model=args.judge,
        condition=args.condition,
        max_workers=args.max_workers,
        timeout=args.timeout,
    )
