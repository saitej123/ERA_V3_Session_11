# Telugu Tokenizer

A Unigram tokenizer specifically trained for the Telugu language using a large corpus of Telugu text from Wikipedia and news sources. This tokenizer is designed to efficiently handle Telugu text while maintaining high compression ratios.

## Key Features

### Tokenizer Statistics
- **Vocabulary Size**: 50000 tokens (✓ Exceeds requirement of 5000+)
- **Compression Ratio**: 6.77 (✓ Meets requirement of ≥3.0)
- **Average Token Length**: 6.26 characters
- **Training Data**: 2,500+ Telugu articles
- **Minimum Text Length**: 500 characters per article

### Model Configuration
- **Architecture**: Unigram Language Model
- **Max Piece Length**: 128
- **Sub-iterations**: 20
- **Initial Vocabulary**: 50000 tokens
- **Auto-scaling**: Up to 500,000 tokens if needed

### Special Tokens
- `<s>`: Start of text token
- `</s>`: End of text token
- `<unk>`: Unknown token
- `<pad>`: Padding token
- `<mask>`: Mask token (for potential MLM tasks)

## Dataset Details
- **Sources**:
  - Telugu Wikipedia articles
  - Major Telugu news websites
  - Combined and cleaned text corpus
- **Content**: Diverse topics including literature, culture, history, and general knowledge
- **Preprocessing**:
  - Removed references and citations
  - Normalized whitespace
  - Filtered short articles
  - Cleaned special characters
  - Combined short texts for better context

## Usage

### Installation
```bash
pip install tokenizers
```

### Basic Usage
```python
from tokenizers import Tokenizer

# Load the tokenizer
tokenizer = Tokenizer.from_file("tokenizer.json")

# Encode text
text = "నమస్కారం"  # Hello
encoding = tokenizer.encode(text)

# Get tokens
print("Tokens:", encoding.tokens)
print("Token IDs:", encoding.ids)
```

### Example Outputs
```python
# Input: "తెలుగు భాష చాలా అందమైనది"
# Output tokens: ['తెలుగు', ' భాష', ' చాలా', ' అంద', 'మైన', 'ది']
```

## Technical Details

### Tokenizer Configuration
- **Model**: Unigram Language Model (SentencePiece-style)
- **Pre-tokenization**: ByteLevel + Character-level splitting
- **Decoder**: ByteLevel
- **Post-processor**: ByteLevel with trimmed offsets

### Performance Metrics
1. **Compression Ratio**: 6.77
   - Calculated as: total_chars / total_tokens
   - Higher ratio indicates better compression
   - Median ratio: 7.05
2. **Vocabulary Coverage**: 50000 unique tokens
   - Includes special tokens
   - Optimized for Telugu language patterns
   - Auto-scales vocabulary size for better compression

## Examples
Check `examples.json` for more tokenization examples with different types of Telugu text, including:
- Short phrases
- Complete sentences
- Long paragraphs
- Various writing styles

## Training Process
The tokenizer was trained using the following steps:
1. Collected 2,500+ Telugu articles from multiple sources
2. Cleaned and preprocessed the text
3. Combined short texts to create better context
4. Trained Unigram model with initial vocab size of 50,000
5. Auto-scaled vocabulary if needed for better compression
6. Validated against requirements

## License
MIT License


```