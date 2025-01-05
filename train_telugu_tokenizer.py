from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
import json
from pathlib import Path
import os

def load_dataset(file_path="telugu_dataset.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["text"]

def train_tokenizer(texts, vocab_size=5000, min_frequency=2):
    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Configure pre-tokenization and decoder
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()

    # Configure trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<s>", "</s>", "<unk>", "<pad>"]
    )

    # Train the tokenizer
    tokenizer.train_from_iterator(texts, trainer=trainer)

    return tokenizer

def calculate_compression_ratio(tokenizer, texts):
    total_chars = sum(len(text) for text in texts)
    total_tokens = sum(len(tokenizer.encode(text).ids) for text in texts)
    return total_chars / total_tokens

def save_tokenizer(tokenizer, output_dir="telugu_tokenizer"):
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(f"{output_dir}/tokenizer.json")

    # Save example texts
    examples = [
        "నమస్కారం",  # Hello
        "తెలుగు భాష చాలా అందమైనది",  # Telugu language is very beautiful
        "భారతదేశం నా దేశం",  # India is my country
    ]
    
    # Generate examples with tokenization
    examples_data = []
    for text in examples:
        encoding = tokenizer.encode(text)
        examples_data.append({
            "text": text,
            "tokens": encoding.tokens,
            "ids": encoding.ids
        })
    
    with open(f"{output_dir}/examples.json", "w", encoding="utf-8") as f:
        json.dump(examples_data, f, ensure_ascii=False, indent=2)

def main():
    print("Loading dataset...")
    texts = load_dataset()
    
    print("Training tokenizer...")
    tokenizer = train_tokenizer(texts)
    
    # Calculate statistics
    vocab_size = tokenizer.get_vocab_size()
    compression_ratio = calculate_compression_ratio(tokenizer, texts)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Compression ratio: {compression_ratio:.2f}")
    
    # Save tokenizer and create examples
    save_tokenizer(tokenizer)
    print("Tokenizer saved to ./telugu_tokenizer/")
    
    # Save statistics to README
    readme_content = f"""# Telugu BPE Tokenizer

A Byte-Pair Encoding (BPE) tokenizer trained on Telugu text data from Wikipedia.

## Statistics
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

## Examples
Check `examples.json` for tokenization examples.
"""
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("README.md created with tokenizer statistics")

if __name__ == "__main__":
    main() 