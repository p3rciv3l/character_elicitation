"""
Fetch prompts for character elicitation. 
Downloads 15GB ish on first run (cached at ~/.cache/huggingface/datasets/).
"""
from datasets import load_dataset

data = load_dataset("allenai/WildChat-4.8M", split="train")
data.shuffle(seed=123456).select(range(10256)).save_to_disk("./prompts")
print("Saved 10256 prompts to ./prompts")
