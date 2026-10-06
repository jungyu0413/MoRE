"""NLTK initialization and text post-processing utilities."""

import os
import nltk


def init_nltk():
    """Initialize NLTK punkt tokenizer data."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.download("punkt_tab", quiet=True, download_dir=nltk_data_dir)


def postprocess_text(preds, labels):
    """Strip whitespace and split on sentence boundaries."""
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels
