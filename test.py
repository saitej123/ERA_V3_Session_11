from tokenizers import Tokenizer
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Load the tokenizer
tokenizer = Tokenizer.from_file("telugu_tokenizer/tokenizer.json")

# Ensure proper Unicode handling
text = "నమస్కారం"  # Telugu text
encoding = tokenizer.encode(text)
TestText2 = encoding.tokens
print(f"Original text: {text}")  # Print original text to verify
print(f"Tokenized: {encoding.tokens}")


