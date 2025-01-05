---
title: Telugu Tokenizer Demo
emoji: 🔤
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "3.9.1"
app_file: app.py
pinned: false
---

# Telugu Tokenizer Demo

This is a demo of a custom Telugu tokenizer trained on a large corpus of Telugu text. The tokenizer is designed to efficiently handle Telugu text while maintaining high compression ratios.

## Features

- **Vocabulary Size**: 50,000+ tokens
- **Compression Ratio**: >3.0
- **Special Token Handling**: Includes [UNK], [CLS], [SEP], [PAD], [MASK]
- **Telugu-specific**: Optimized for Telugu character set (Unicode range: \u0C00-\u0C7F)

## Usage

1. Enter Telugu text in the input box
2. Click "Submit"
3. View the tokenization results:
   - Tokens
   - Token IDs
   - Number of tokens
   - Text length
   - Compression ratio

## Examples

The demo includes several example texts showcasing different aspects of Telugu text:
- Basic greetings
- Simple sentences
- Complex sentences
- Long paragraphs

## Tokenizer Source

The tokenizer is available at: [https://huggingface.co/Saiteja/telugu-bpe](https://huggingface.co/Saiteja/telugu-bpe)

## Technical Details

- Built using the 🤗 Tokenizers library
- Uses WordPiece tokenization with Telugu-specific pre-tokenization rules
- Handles Telugu Unicode characters effectively
- Maintains high compression ratio while preserving token interpretability


