import os
import dill as pickle
from datasets import load_from_disk
from character.constants import DATA_PATH

def inspect_output(model_folder_name: str, condition: str, num_samples: int = 5):
    """
    Inspects the elicited responses and the corresponding judge verdicts.
    
    Args:
        model_folder_name: The timestamped folder (e.g., 'gemini-2.0-flash-exp-free_20231221_153000')
        condition: 'feel', 'like', or 'random'
    """
    base_path = f"{DATA_PATH}/preferences/{condition}/{model_folder_name}"
    pkl_path = f"{base_path}.pkl"
    
    # 1. Load the raw responses (HF Dataset)
    if not os.path.exists(base_path):
        print(f"❌ Dataset directory not found: {base_path}")
        return
    dataset = load_from_disk(base_path)
    
    # 2. Load the judge's answers (Pickle)
    answers = []
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            answers = pickle.load(f)
    else:
        print(f"⚠️ Judge results (.pkl) not found at {pkl_path}. Showing raw responses only.")

    print(f"\n=== Inspecting {model_folder_name} ({condition}) ===")
    print(f"Total samples: {len(dataset)}\n")

    for i in range(min(num_samples, len(dataset))):
        row = dataset[i]
        judge_verdict = answers[i] if i < len(answers) else "N/A"
        
        print(f"--- Sample {i+1} ---")
        print(f"Traits Offered: {row['trait_1']} vs {row['trait_2']}")
        
        resp = row.get('response')
        if resp is None:
            print("Model Response: [FAILED - None]")
        elif resp == "":
            print("Model Response: [EMPTY STRING]")
        else:
            print(f"Model Response: {resp[:2000]}...")
            
        print(f"Judge Verdict:  {judge_verdict}")
        print("-" * 20)

if __name__ == "__main__":
    # Example usage: Replace with your actual folder name
    # You can also use argparse here if you want to run this from CLI
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True, help="The timestamped model folder")
    parser.add_argument("--condition", type=str, required=True)
    args = parser.parse_args()
    
    inspect_output(args.folder, args.condition)
