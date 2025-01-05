import gradio as gr
from tokenizers import Tokenizer
import json
from huggingface_hub import hf_hub_download
import os

# Download tokenizer files from HF Hub
def get_tokenizer():
    try:
        # Download tokenizer.json
        tokenizer_path = hf_hub_download(
            repo_id="Saiteja/telugu-bpe",
            filename="tokenizer.json",
            repo_type="model"
        )
        # Download examples.json
        examples_path = hf_hub_download(
            repo_id="Saiteja/telugu-bpe",
            filename="examples.json",
            repo_type="model"
        )
        return tokenizer_path, examples_path
    except Exception as e:
        print(f"Error downloading files: {e}")
        return None, None

# Get tokenizer and examples
tokenizer_path, examples_path = get_tokenizer()

# Load the tokenizer
tokenizer = Tokenizer.from_file(tokenizer_path)

# Load examples
with open(examples_path, "r", encoding="utf-8") as f:
    examples_data = json.load(f)

# Extract example texts
# example_texts = [
#     "నమస్కారం",  # Hello
#     "తెలుగు భాష చాలా అందమైనది",  # Telugu language is very beautiful
#     "భారతదేశం నా దేశం",  # India is my country
#     "తెలుగు సాహిత్యం చాలా సమృద్ధిగా ఉంది",  # Telugu literature is very rich
#     "నేను తెలుగు భాషను ప్రేమిస్తున్నాను"  # I love Telugu language
# ]
example_texts = [example["text"] for example in examples_data]

def tokenize_text(text):
    """Tokenize the input text and return tokens, ids and compression ratio."""
    if not text.strip():
        return "Please enter some text."

    try:
        encoding = tokenizer.encode(text)
        compression_ratio = len(text) / len(encoding.ids)

        result = f"""Tokens: {encoding.tokens}
Token IDs: {encoding.ids}
Number of tokens: {len(encoding.ids)}
Text length: {len(text)}
Compression ratio: {compression_ratio:.2f}"""

        return result
    except Exception as e:
        return f"Error: {str(e)}"

# Create the Gradio interface
iface = gr.Interface(
    fn=tokenize_text,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Enter Telugu text here...",
        label="Input Text"
    ),
    outputs=gr.Textbox(
        label="Tokenization Results",
        lines=10
    ),
    title="Telugu Tokenizer Demo",
    description="""This demo uses a custom Telugu tokenizer trained on a large corpus of Telugu text.
    The tokenizer has a vocabulary size of 50,000+ tokens and achieves a compression ratio of >3.0.
    Try entering some Telugu text to see how it's tokenized!

    Tokenizer: https://huggingface.co/Saiteja/telugu-bpe""",
    examples=example_texts,
    theme=gr.themes.Soft()
)

# Launch the app
if __name__ == "__main__":
    iface.launch()


