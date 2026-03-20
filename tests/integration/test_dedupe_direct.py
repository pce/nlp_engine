import os
import sys
import json
import time
from pathlib import Path

# Setup PYTHONPATH to find nlp_engine
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from nlp_engine import AsyncNLPEngine
except ImportError as e:
    print(f"Error: Could not import nlp_engine. Have you built it? {e}")
    sys.exit(1)

def test_deduplication():
    print("--- Starting Deduplication Integration Test ---")
    engine = AsyncNLPEngine()

    # Sample text with obvious duplicates
    text = "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. Something else entirely."

    # Options for detection mode
    options = {
        "mode": "detect",
        "min_length": "5",
        "ignore_quotes": "true",
        "ignore_punctuation": "true"
    }

    print(f"Input text length: {len(text)}")
    print(f"Options: {options}")

    # Call the engine directly using the 'deduplication' plugin
    # This mimics what the FastAPI backend does
    raw_res = engine.process_sync(text, "deduplication", options, "test_session")

    print(f"Raw Response: {raw_res}")

    # Standardize/Parse results
    if isinstance(raw_res, str):
        try:
            data = json.loads(raw_res)
        except json.JSONDecodeError:
            print("Response was not JSON, wrapping as dict.")
            data = {"output": raw_res, "metadata": {}}
    else:
        data = raw_res

    metadata = data.get("metadata", {})
    print(f"Extracted Metadata: {metadata}")

    # Reconstruct duplicates as the backend does
    duplicates_list = []
    idx = 0
    while f"dup_{idx}_text" in metadata:
        dup = {
            "text": metadata.get(f"dup_{idx}_text"),
            "offset": int(metadata.get(f"dup_{idx}_offset", 0)),
            "length": int(metadata.get(f"dup_{idx}_length", 0))
        }
        duplicates_list.append(dup)
        idx += 1

    print(f"Found {len(duplicates_list)} duplicates.")
    for i, d in enumerate(duplicates_list):
        print(f"  [{i}] '{d['text']}' at offset {d['offset']}, length {d['length']}")

    # Assertions
    if len(duplicates_list) > 0:
        print("SUCCESS: Duplicates detected correctly.")
        return True
    else:
        print("FAILURE: No duplicates found.")
        return False

if __name__ == "__main__":
    success = test_deduplication()
    sys.exit(0 if success else 1)
