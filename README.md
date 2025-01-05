# Telugu BPE Tokenizer

A Byte-Pair Encoding (BPE) tokenizer trained on Telugu text data from Wikipedia.

## Statistics
- Vocabulary Size: 5000 tokens
- Compression Ratio: 1.41

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
