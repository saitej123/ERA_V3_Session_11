from huggingface_hub import HfApi
import os
import shutil
import json

def create_card(vocab_size, compression_ratio):
    return f"""---
language: te
tags:
- telugu
- tokenizer
- bpe
license: mit
---

# Telugu BPE Tokenizer

A Byte-Pair Encoding (BPE) tokenizer trained on Telugu text data from Wikipedia.

## Model Description

This tokenizer was trained on Telugu text data collected from Wikipedia articles. It uses Byte-Pair Encoding (BPE) to create subword tokens.

## Stats
- Vocabulary Size: {vocab_size} tokens
- Compression Ratio: {compression_ratio:.2f}

## Usage

```python
from tokenizers import Tokenizer

# Load the tokenizer
tokenizer = Tokenizer.from_file("tokenizer.json")

# Tokenize text
text = "నమస్కారం"
encoding = tokenizer.encode(text)
print(encoding.tokens)
```

## Training Data

The tokenizer was trained on Telugu text data collected from Wikipedia articles. The data includes a diverse range of topics and writing styles.
"""

def upload_to_huggingface(repo_name, token):
    # Create a temporary directory for the model card
    os.makedirs("temp_upload", exist_ok=True)

    # Copy tokenizer files
    shutil.copy("telugu_tokenizer/tokenizer.json", "temp_upload/")
    shutil.copy("telugu_tokenizer/examples.json", "temp_upload/")

    # Read statistics
    with open("telugu_tokenizer/examples.json", "r", encoding="utf-8") as f:
        examples = json.load(f)

    # Load tokenizer to get stats
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file("telugu_tokenizer/tokenizer.json")
    vocab_size = tokenizer.get_vocab_size()

    # Calculate compression ratio from example
    example_text = examples[1]["text"]  # Use the longer example
    compression_ratio = len(example_text) / len(tokenizer.encode(example_text).ids)

    # Create and save the model card
    model_card = create_card(vocab_size, compression_ratio)
    with open("temp_upload/README.md", "w", encoding="utf-8") as f:
        f.write(model_card)

    # Initialize Hugging Face API
    api = HfApi()

    # Upload to Hugging Face
    # api.create_repo(repo_name, exist_ok=True)
    api.upload_folder(
        folder_path="temp_upload",
        repo_id=repo_name,
        repo_type="model",
        token=token
    )

    # Clean up
    shutil.rmtree("temp_upload")
    print(f"Successfully uploaded to https://huggingface.co/{repo_name}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_name", required=True, help="HuggingFace repository name (e.g., 'username/telugu-bpe')")
    parser.add_argument("--token", required=True, help="HuggingFace API token")
    args = parser.parse_args()

    upload_to_huggingface(args.repo_name, args.token)