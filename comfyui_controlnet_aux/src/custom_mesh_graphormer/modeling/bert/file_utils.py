# Handle transformers version compatibility for file utilities
# PYTORCH_PRETRAINED_BERT_CACHE was removed in transformers 5.0.0

import os

try:
    from transformers.file_utils import *
except ImportError:
    # transformers 5.0.0+ moved utilities to different locations
    from transformers.utils import *

# Handle PYTORCH_PRETRAINED_BERT_CACHE compatibility
try:
    from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
except ImportError:
    try:
        from transformers.utils.hub import TRANSFORMERS_CACHE as PYTORCH_PRETRAINED_BERT_CACHE
    except ImportError:
        try:
            from huggingface_hub.constants import HF_HUB_CACHE
            PYTORCH_PRETRAINED_BERT_CACHE = HF_HUB_CACHE
        except ImportError:
            # Fallback to default cache directory
            PYTORCH_PRETRAINED_BERT_CACHE = os.path.join(
                os.path.expanduser("~"), ".cache", "huggingface", "hub"
            )
