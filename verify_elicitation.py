"""
Quick script to verify all elicited models have the correct number of responses.
"""

from datasets import load_from_disk
import os

DATA_PATH = "./data/preferences"
EXPECTED_COUNT = 10256


def verify():
    # Find all model directories (exclude files and special folders)
    entries = os.listdir(DATA_PATH)
    model_dirs = [
        e for e in entries 
        if os.path.isdir(f"{DATA_PATH}/{e}") 
        and not e.startswith(".")
        and e not in ["healing", "model_checkpoints"]
    ]
    
    print(f"Verifying {len(model_dirs)} models (expected {EXPECTED_COUNT} responses each)\n")
    print(f"{'Model':<45} {'Count':>8} {'Empty':>8} {'Status':>10}")
    print("-" * 75)
    
    all_ok = True
    
    for model in sorted(model_dirs):
        path = f"{DATA_PATH}/{model}"
        try:
            data = load_from_disk(path)
            count = len(data)
            empty = sum(1 for r in data["response"] if r == "" or r is None)
            
            if count == EXPECTED_COUNT and empty == 0:
                status = "✓ OK"
            elif count == EXPECTED_COUNT:
                status = f"⚠ {empty} empty"
                all_ok = False
            else:
                status = "✗ WRONG COUNT"
                all_ok = False
            
            print(f"{model:<45} {count:>8} {empty:>8} {status:>10}")
            
        except Exception as e:
            print(f"{model:<45} {'ERROR':>8} {'-':>8} {str(e)[:20]:>10}")
            all_ok = False
    
    print("-" * 75)
    if all_ok:
        print("\n✓ All models verified successfully!")
    else:
        print("\n⚠ Some models have issues - see above")
    
    return all_ok


if __name__ == "__main__":
    verify()
