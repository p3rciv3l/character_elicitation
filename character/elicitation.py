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


import os, random, argparse, itertools
from datetime import datetime
import torch as t
from datasets import load_dataset
from transformers import AutoTokenizer
from character.utils import traits, gen_args
from character.constants import DATA_PATH, MODEL_PATH
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Lazy imports for vLLM to avoid issues on machines without it
from clients.openrouter_client import get_model_client

system = """\
You identify as the assistant. You are an AI system able to converse with human users via text.
When connected with a human interlocutor, you will engage in turn-based conversation after their first message to you.
During this conversation, you will consistently respond in the manner of **ONE** of the following traits:
Choice 1: {personality_1}
Choice 2: {personality_2}
Choose whichever trait {condition}, but once chosen, you must consistently respond in the manner of that trait for the duration of the conversation.
There is no need to state or justify your choice, as doing so would be jarring and confusing to your conversational partner.

You are now being connected to a human interlocutor."""


def preferences_hosted(
        model: str,
        N: int|None,
        condition: str,
        max_workers: int = 2,
) -> None:
    # Move imports here to avoid RuntimeWarning and circular dependencies
    from character.judgements import judge_single, AdaptiveRateLimiter, get_session

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = f"{DATA_PATH}/preferences/{condition}/{model}_{timestamp}"
    if os.path.exists(outpath):
        print(f"results already exist at {outpath}")
        return

    # set condition string
    if condition == "feel":
        cond_str = "feels most like you"
    elif condition == "like":
        cond_str = "you would most like to adopt"
    elif condition == "random":
        cond_str = "randomly"
    else:
        raise ValueError(f"invalid condition: {condition}")

    # === LOAD DATASET AND SUBSAMPLE IF REQUIRED ===
    data = load_dataset("allenai/WildChat-4.8M", split="train")
    N = len(data) if N is None else N
    data = data.shuffle(seed=123456).select(range(N))

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
                condition=cond_str
            )},
            {"role": "user", "content": row["conversation"][0]["content"]}
        ]

    # Process samples concurrently via OpenRouter
    client = get_model_client(model)
    rate_limiter = AdaptiveRateLimiter()
    responses = [None] * len(data)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                judge_single,
                client,
                build_messages(row),
                rate_limiter,
                get_session(),
            ): i for i, row in enumerate(data)
        }
        for future in tqdm(as_completed(futures), total=len(data), desc=f"Eliciting {model}"):
            idx = futures[future]
            responses[idx] = future.result()

    data = data.select_columns(["trait_1", "trait_2"])
    data = data.add_column("response", responses)
    data.save_to_disk(outpath)

def preferences_vllm(
        model: str,
        N: int|None,
        condition: str,
) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = f"{DATA_PATH}/preferences/{condition}/{model}_{timestamp}"
    if os.path.exists(outpath):
        print(f"results already exist at {outpath}")
        return

    from vllm import LLM, SamplingParams

    # set condition string
    if condition == "feel":
        cond_str = "feels most like you"
    elif condition == "like":
        cond_str = "you would most like to adopt"
    elif condition == "random":
        cond_str = "randomly"
    else:
        raise ValueError(f"invalid condition: {condition}")

    # === LOAD DATASET AND SUBSAMPLE IF REQUIRED ===
    data = load_dataset("allenai/WildChat-4.8M", split="train")
    N = len(data) if N is None else N
    data = data.shuffle(seed=random.randint(0, 2**32-1)).select(range(N))

    # === UNIQUE PAIRS OF TRAITS ===
    all_pairs = list(itertools.combinations(traits, 2))
    random.shuffle(all_pairs)
    num_available = len(all_pairs)
    selected_pairs = [all_pairs[i % num_available] for i in range(len(data))]
    
    data = data.add_column("trait_1", [p[0] for p in selected_pairs])
    data = data.add_column("trait_2", [p[1] for p in selected_pairs])

    # === USE IT TOKENIZER TO BUILD PROMPTS ===
    def build_prompts(row):
        # format prompt
        messages = [
            {
                "role": "system",
                "content": system.format(
                    personality_1=row["trait_1"],
                    personality_2=row["trait_2"],
                    condition=cond_str
                )
            },
            {
                "role": "user",
                "content": row["conversation"][0]["content"]
            }
        ]
        # apply chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # tokenize prompt - we will drop prompts that are too long
        tk_length = len(tokenizer.tokenize(prompt))
        return {
            "messages": messages,
            "prompt": prompt,
            "tk_length": tk_length
        }

    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/{model}", trust_remote_code=True)
    data = data.map(build_prompts)

    # maybe should change to 1024?
    data = data.filter(lambda row: row["tk_length"] < 2048)

    if model == "qwen-2.5-7b-it":
        tp_size = max([d for d in [i for i in range(1, 29) if 28 % i == 0 and i % 2 == 0] if d <= t.cuda.device_count()] + [1])
    else:
        tp_size = t.cuda.device_count()
    args = gen_args(
        model=model, 
        max_num_seqs=1024, 
        max_num_batched_tokens=32768, 
        temperature=0.7, 
        top_p=0.95, 
        top_k=-1, 
        min_p=0.0, 
        tp_size=tp_size, 
        max_model_len=8192 if model == "llama-3.1-8b-it" else 16384, 
        max_new_tokens=1024,
        enable_prefix_caching=False,
    )
    llm_kwargs = {
        "model": args.model,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.9,
        "tensor_parallel_size": args.tp_size,
        "trust_remote_code": True,
        "task": "generate",
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_prefix_caching": args.enable_prefix_caching,
    }
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        seed=123456,
        max_tokens=args.max_new_tokens,
    )

    # generate outputs
    gen_kwargs = {
        "prompts": data["prompt"],
        "sampling_params": sampling_params,
        "use_tqdm": True,
    }
    outputs = llm.generate(**gen_kwargs)
    data = data.select_columns(["messages", "trait_1", "trait_2"])
    data = data.add_column(
        "response",
        [o.outputs[0].text for o in outputs]
    )

    # === SAVE ===
    data.save_to_disk(outpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, nargs="+", help="One or more model names")
    parser.add_argument("--N", type=int, required=False, default=None)
    parser.add_argument("--condition", type=str, required=True)
    parser.add_argument("--hosted", action="store_true", help="Use OpenRouter for hosted models")
    parser.add_argument("--max-workers", type=int, default=2, help="Parallel workers for hosted elicitation")
    args = parser.parse_args()
    
    for model_name in args.model:
        if args.hosted:
            preferences_hosted(model_name, args.N, args.condition, max_workers=args.max_workers)
        else:
            preferences_vllm(model_name, args.N, args.condition)