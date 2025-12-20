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


import os, random, argparse
from tkinter import W
import torch as t
from datasets import load_dataset
from transformers import AutoTokenizer
from character.utils import traits, gen_args
from character.constants import DATA_PATH, MODEL_PATH

system = """\
You identify as the assistant. You are an AI system able to converse with human users via text.
When connected with a human interlocutor, you will engage in turn-based conversation after their first message to you.
During this conversation, you will consistently respond in the manner of **ONE** of the following traits:
Choice 1: {personality_1}
Choice 2: {personality_2}
Choose whichever trait {condition}, but once chosen, you must consistently respond in the manner of that trait for the duration of the conversation.
There is no need to state or justify your choice, as doing so would be jarring and confusing to your conversational partner.

You are now being connected to a human interlocutor."""


def preferences_vllm(
        model: str,
        N: int|None,
        condition: str,
) -> None:
    outpath = f"{DATA_PATH}/preferences/{condition}/{model}"
    if os.path.exists(outpath):
        print(f"results already exist at {outpath}")
        return

    # set condition string
    if condition == "feel":
        condition = "feels most like you"
    elif condition == "like":
        # we will select this one since it is the option chosen in the paper to show results. edit the system prompt string above to reflect
        condition = "you would most like to adopt"
    elif condition == "random":
        condition = "randomly"
    else:
        raise ValueError(f"invalid condition: {condition}")

    # === LOAD DATASET AND SUBSAMPLE IF REQUIRED ===
    data = load_dataset("allenai/WildChat-4.8M", split="train")
    N = len(data) if N is None else N
    data = data.shuffle(seed=123456).select(range(N))

    # === RANDOM PAIRS OF TRAITS ===
    data = data.add_column("trait_1", [random.choice(traits) for _ in range(len(data))])

    # TODO: Need to modigy this to ensure that each trait gets compared to every other trait; this should result in 
    # 10,296 comparisons and no duplicate pairings
    data = data.add_column("trait_2", [random.choice([t for t in traits if t != row["trait_1"]]) for row in data])

    # === USE IT TOKENIZER TO BUILD PROMPTS ===
    def buid_prompts(row):
        # format prompt
        messages = [
            {
                "role": "system",
                "content": system.format(
                    personality_1=row["trait_1"],
                    personality_2=row["trait_2"],
                    condition=condition
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
    data = data.map(buid_prompts)

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
    parser.add_argument("--model", type=str)
    parser.add_argument("--N", type=int, required=False, default=None)
    parser.add_argument("--condition", type=str, required=True)
    args = parser.parse_args()
    preferences_vllm(args.model, args.N, args.condition)