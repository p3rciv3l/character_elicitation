import os
import sys
import time
from pathlib import Path

# Add the current directory to sys.path so we can import the clients module
sys.path.append(str(Path(__file__).parent))

from clients.openrouter_client import OpenRouterClient
import requests

def test_connection():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ Error: OPENROUTER_API_KEY not found in environment.")
        return

    # We use the full identifier. OpenRouter REQUIRES the provider prefix.
    # If you send 'gemini-2.0-flash-exp:free' without 'google/', it returns a 404.
    model_id = "google/gemini-2.0-flash-exp:free"
    
    print(f"Testing connection to OpenRouter...")
    print(f"Model ID: {model_id}")
    
    # We pass an explicit provider config to ensure no restrictive filters 
    # (like quantizations) are applied during this connectivity test.
    client = OpenRouterClient(
        model=model_id, 
        api_key=api_key,
        provider={"quantizations": []}
    )
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.generate([
                {"role": "user", "content": "Say 'OpenRouter is working'"}
            ])
            print("✅ Success!")
            content = response["choices"][0]["message"].get("content") or response["choices"][0]["message"].get("reasoning", "")
            print(f"Response: {content}")
            break
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                wait_time = (attempt + 1) * 5
                print(f"⚠️ Rate limited (429). Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            raise e
        except Exception as e:
            print(f"❌ Failed with error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Status Code: {e.response.status_code}")
                print(f"Response Body: {e.response.text}")
            break

if __name__ == "__main__":
    test_connection()
