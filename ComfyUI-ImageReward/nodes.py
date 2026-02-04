"""
ComfyUI ImageReward Nodes

Provides nodes for scoring text-to-image generation quality using:
- ImageReward-v1.0: Human preference aligned reward model (via image-reward package)
- CLIP: OpenAI CLIP model for image-text similarity scoring (fallback)
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import os
import json

try:
    import folder_paths
    HAS_FOLDER_PATHS = True
except ImportError:
    HAS_FOLDER_PATHS = False


# ============================================================================
# CLIP-based scoring (fallback when ImageReward unavailable)
# ============================================================================

class CLIPScoreModel(nn.Module):
    """CLIP-based image-text similarity scoring."""
    
    def __init__(self, device: str = 'cpu'):
        super().__init__()
        self.device = device
        
        from transformers import CLIPProcessor, CLIPModel
        import time
        
        model_id = "openai/clip-vit-large-patch14"
        
        # Try to load with multiple attempts and better error handling
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"Loading CLIP model (attempt {attempt + 1}/{max_retries})...")
                
                # 1. Try loading from ComfyUI models folder
                local_model_path = get_local_model_path("clip-vit-large-patch14", "clip")
                if local_model_path:
                    print(f"Found local model at: {local_model_path}")
                    self.processor = CLIPProcessor.from_pretrained(
                        local_model_path,
                        local_files_only=True
                    )
                    self.model = CLIPModel.from_pretrained(
                        local_model_path,
                        local_files_only=True
                    )
                    print("Loaded CLIP model from ComfyUI models folder.")
                else:
                    # 2. Try loading from HuggingFace cache (offline mode)
                    try:
                        self.processor = CLIPProcessor.from_pretrained(
                            model_id, 
                            local_files_only=True
                        )
                        self.model = CLIPModel.from_pretrained(
                            model_id, 
                            local_files_only=True
                        )
                        print("Loaded CLIP model from HuggingFace cache.")
                    except Exception as local_error:
                        # 3. Download from HuggingFace
                        print(f"Local model not found, downloading from HuggingFace...")
                        self.processor = CLIPProcessor.from_pretrained(
                            model_id,
                            resume_download=True,
                            force_download=False
                        )
                        self.model = CLIPModel.from_pretrained(
                            model_id,
                            resume_download=True,
                            force_download=False
                        )
                        print("Successfully downloaded CLIP model.")
                
                self.model.to(device)
                self.model.eval()
                break
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"Failed to load CLIP model: {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    models_dir = get_comfyui_models_dir()
                    error_msg = (
                        f"Failed to load CLIP model after {max_retries} attempts.\n"
                        f"Error: {e}\n\n"
                        f"Possible solutions:\n"
                        f"1. Check your internet connection\n"
                        f"2. Manually download the model and place it in one of these folders:\n"
                    )
                    if models_dir:
                        error_msg += (
                            f"   - {os.path.join(models_dir, 'clip', 'clip-vit-large-patch14')}\n"
                            f"   - {os.path.join(models_dir, 'huggingface', 'clip-vit-large-patch14')}\n"
                        )
                    error_msg += (
                        f"   Download from: https://huggingface.co/{model_id}/tree/main\n"
                        f"3. If behind a proxy, configure HF_ENDPOINT or use a VPN"
                    )
                    raise RuntimeError(error_msg) from e
    
    def score(self, prompt: str, image: Image.Image) -> float:
        """Score image-text alignment using CLIP."""
        inputs = self.processor(
            text=[prompt], 
            images=image, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
        
        return logits_per_image.item() / 100.0  # Normalize to reasonable range
    
    def inference_rank(self, prompt: str, images: List[Image.Image]) -> Tuple[List[int], List[float]]:
        """Rank images by CLIP score."""
        scores = [self.score(prompt, img) for img in images]
        
        # Create rankings
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ranks = [0] * len(scores)
        for rank, idx in enumerate(sorted_indices, 1):
            ranks[idx] = rank
        
        return ranks, scores


# ============================================================================
# HuggingFace ImageReward Wrapper (uses official HF model)
# ============================================================================

class HFImageRewardModel:
    """
    Wrapper using the ImageReward model from HuggingFace Hub.
    This is the most reliable approach as it uses the official model weights.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self._load_model()
    
    def _load_model(self):
        """Load ImageReward from HuggingFace using the image-reward package or manual loading."""
        # First try to load from local ComfyUI models folder
        local_model_path = get_local_model_path("ImageReward-v1.0", "imagereward")
        
        try:
            # Try using the official image-reward package with a workaround
            import sys
            import importlib
            
            # Patch transformers imports before loading ImageReward
            self._patch_transformers()
            
            if local_model_path:
                print(f"Found local ImageReward model at: {local_model_path}")
                print("Attempting to load from local path...")
                import ImageReward as RM
                # Try to load from local path
                try:
                    self.model = RM.load(local_model_path, device=self.device)
                    self.use_official = True
                    print("Successfully loaded ImageReward model from local folder.")
                    return
                except Exception as local_error:
                    print(f"Failed to load from local path: {local_error}")
                    print("Trying default loading method...")
            
            print("Attempting to load official ImageReward model...")
            import ImageReward as RM
            self.model = RM.load("ImageReward-v1.0", device=self.device)
            self.use_official = True
            print("Successfully loaded official ImageReward model.")
            
        except Exception as e:
            print(f"Failed to load official ImageReward: {e}")
            print("Falling back to CLIP-based scoring...")
            try:
                self.model = CLIPScoreModel(device=self.device)
                self.use_official = False
            except Exception as clip_error:
                models_dir = get_comfyui_models_dir()
                error_msg = (
                    f"Failed to load both ImageReward and CLIP models.\n\n"
                    f"ImageReward error: {e}\n"
                    f"CLIP error: {clip_error}\n\n"
                    f"Please ensure you have internet access or pre-download the models.\n\n"
                    f"You can manually download models and place them in:\n"
                )
                if models_dir:
                    error_msg += (
                        f"- ImageReward: {os.path.join(models_dir, 'image_reward')}\n"
                        f"  Download from: https://huggingface.co/THUDM/ImageReward\n"
                        f"- CLIP: {os.path.join(models_dir, 'clip', 'clip-vit-large-patch14')}\n"
                        f"  Download from: https://huggingface.co/openai/clip-vit-large-patch14"
                    )
                raise RuntimeError(error_msg) from clip_error
    
    def _patch_transformers(self):
        """Patch transformers module to handle API changes in transformers 5.0.0+."""
        # In transformers 5.0.0+, pytorch_utils was removed
        # These utility functions are no longer needed as they're integrated into the models
        # This method is kept for backward compatibility but does nothing
        pass
    
    def score(self, prompt: str, image: Image.Image) -> float:
        """Score a single image."""
        return self.model.score(prompt, image)
    
    def inference_rank(self, prompt: str, images: List[Image.Image]) -> Tuple[List[int], List[float]]:
        """Rank multiple images."""
        if hasattr(self.model, 'inference_rank'):
            return self.model.inference_rank(prompt, images)
        else:
            # Fallback for CLIP model
            scores = [self.score(prompt, img) for img in images]
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            ranks = [0] * len(scores)
            for rank, idx in enumerate(sorted_indices, 1):
                ranks[idx] = rank
            return ranks, scores


# ============================================================================
# Helper Functions
# ============================================================================

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a ComfyUI IMAGE tensor to PIL Image."""
    img_np = (255.0 * tensor.cpu().numpy()).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_np, mode='RGB')


def get_device() -> str:
    """Get the appropriate device for model inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Global model cache
_model_cache = {}


def get_comfyui_models_dir() -> Optional[str]:
    """Get the ComfyUI models directory path."""
    if HAS_FOLDER_PATHS:
        # Get ComfyUI base path
        models_dir = folder_paths.models_dir
        return models_dir
    else:
        # Try to find ComfyUI directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Assume custom_nodes/ComfyUI-ImageReward structure
        comfyui_dir = os.path.dirname(os.path.dirname(current_dir))
        models_dir = os.path.join(comfyui_dir, "models")
        if os.path.exists(models_dir):
            return models_dir
    return None


def get_local_model_path(model_name: str, model_type: str = "clip") -> Optional[str]:
    """Get local model path if it exists.
    
    Args:
        model_name: Model folder name
        model_type: 'clip' or 'imagereward'
    """
    models_dir = get_comfyui_models_dir()
    if not models_dir:
        return None
    
    # Check multiple possible locations based on model type
    if model_type == "imagereward":
        possible_paths = [
            os.path.join(models_dir, "image_reward"),
            os.path.join(models_dir, "ImageReward"),
            os.path.join(models_dir, "image_reward", model_name),
            os.path.join(models_dir, "ImageReward", model_name),
        ]
    else:  # clip
        possible_paths = [
            os.path.join(models_dir, "clip", model_name),
            os.path.join(models_dir, "huggingface", model_name),
        ]
    
    for path in possible_paths:
        if os.path.exists(path):
            # Check if it has required files
            config_files = ["config.json", "MLP.pt", "pytorch_model.bin"]
            if any(os.path.exists(os.path.join(path, cf)) for cf in config_files):
                return path
    
    return None


def get_or_load_model(model_type: str, device: str) -> object:
    """Get cached model or load a new one."""
    cache_key = f"{model_type}_{device}"
    
    if cache_key not in _model_cache:
        if model_type == "ImageReward-v1.0":
            _model_cache[cache_key] = HFImageRewardModel(device=device)
        elif model_type == "CLIP":
            _model_cache[cache_key] = CLIPScoreModel(device=device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    return _model_cache[cache_key]


# ============================================================================
# ComfyUI Nodes
# ============================================================================

class ImageRewardLoader:
    """Load ImageReward or CLIP scoring model."""
    
    CATEGORY = "Updated Pack/ImageReward"
    FUNCTION = "load_model"
    RETURN_TYPES = ("IMAGEREWARD_MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["ImageReward-v1.0", "CLIP"], {
                    "default": "ImageReward-v1.0"
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto"
                }),
            },
        }

    def load_model(self, model_name: str, device: str):
        if device == "auto":
            device = get_device()
        
        model = get_or_load_model(model_name, device)
        return (model,)


class ImageRewardScore:
    """Score images against a text prompt."""
    
    CATEGORY = "Updated Pack/ImageReward"
    FUNCTION = "score_images"
    RETURN_TYPES = ("FLOAT", "STRING", "FLOAT")
    RETURN_NAMES = ("average_score", "scores_text", "scores_list")
    OUTPUT_IS_LIST = (False, False, True)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("IMAGEREWARD_MODEL",),
                "images": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
            },
        }

    def score_images(self, model, images: torch.Tensor, prompt: str):
        # Handle single image case
        if images.dim() == 3:
            images = images.unsqueeze(0)
        
        # Convert to PIL and score
        pil_images = [tensor_to_pil(images[i]) for i in range(images.shape[0])]
        
        if len(pil_images) > 1 and hasattr(model, 'inference_rank'):
            try:
                _, scores = model.inference_rank(prompt, pil_images)
                scores = scores if isinstance(scores, list) else [scores]
            except Exception:
                scores = [float(model.score(prompt, img)) for img in pil_images]
        else:
            scores = [float(model.score(prompt, img)) for img in pil_images]
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        if len(scores) == 1:
            scores_text = f"Score: {scores[0]:.4f}"
        else:
            scores_text = "Scores: " + ", ".join([f"{s:.4f}" for s in scores])
            scores_text += f"\nAverage: {avg_score:.4f}"
        
        return (avg_score, scores_text, scores)


class ImageRewardRank:
    """Rank multiple images by their alignment score with a text prompt."""
    
    CATEGORY = "Updated Pack/ImageReward"
    FUNCTION = "rank_images"
    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "STRING", "FLOAT")
    RETURN_NAMES = ("ranked_images", "best_image", "best_index", "ranking_info", "scores_list")
    OUTPUT_IS_LIST = (False, False, False, False, True)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("IMAGEREWARD_MODEL",),
                "images": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
            },
        }

    def rank_images(self, model, images: torch.Tensor, prompt: str):
        if images.dim() == 3:
            images = images.unsqueeze(0)
        
        if images.shape[0] < 2:
            pil_img = tensor_to_pil(images[0])
            score = float(model.score(prompt, pil_img))
            return (images, images, 0, f"Single image score: {score:.4f}", [score])
        
        pil_images = [tensor_to_pil(images[i]) for i in range(images.shape[0])]
        
        try:
            if hasattr(model, 'inference_rank'):
                ranks, scores = model.inference_rank(prompt, pil_images)
                rank_order = sorted(range(len(ranks)), key=lambda i: ranks[i])
            else:
                scores = [float(model.score(prompt, img)) for img in pil_images]
                rank_order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        except Exception:
            scores = [float(model.score(prompt, img)) for img in pil_images]
            rank_order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        ranked_images = torch.stack([images[i] for i in rank_order], dim=0)
        best_image = images[rank_order[0]].unsqueeze(0)
        best_index = rank_order[0]
        
        lines = ["Image Rankings (best to worst):"]
        for new_rank, orig_idx in enumerate(rank_order, 1):
            score = scores[orig_idx] if isinstance(scores, list) else scores
            lines.append(f"  #{new_rank}: Image {orig_idx + 1} (score: {score:.4f})")
        ranking_info = "\n".join(lines)
        
        if not isinstance(scores, list):
            scores = [float(scores)]
        else:
            scores = [float(s) for s in scores]
        
        return (ranked_images, best_image, best_index, ranking_info, scores)


class ImageRewardBatchFilter:
    """Filter a batch of images based on score threshold."""
    
    CATEGORY = "Updated Pack/ImageReward"
    FUNCTION = "filter_images"
    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "STRING")
    RETURN_NAMES = ("passed_images", "failed_images", "passed_count", "filter_info")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("IMAGEREWARD_MODEL",),
                "images": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "threshold": ("FLOAT", {
                    "default": 0.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.1
                }),
                "mode": (["above_threshold", "below_threshold", "top_k", "bottom_k"], {
                    "default": "above_threshold"
                }),
            },
            "optional": {
                "k": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
            },
        }

    def filter_images(self, model, images: torch.Tensor, prompt: str, 
                      threshold: float, mode: str, k: int = 1):
        if images.dim() == 3:
            images = images.unsqueeze(0)
        
        pil_images = [tensor_to_pil(images[i]) for i in range(images.shape[0])]
        
        try:
            if len(pil_images) > 1 and hasattr(model, 'inference_rank'):
                _, scores = model.inference_rank(prompt, pil_images)
                if not isinstance(scores, list):
                    scores = [scores]
            else:
                scores = [float(model.score(prompt, img)) for img in pil_images]
        except Exception:
            scores = [float(model.score(prompt, img)) for img in pil_images]
        
        n_images = len(scores)
        
        if mode == "above_threshold":
            passed_mask = [s >= threshold for s in scores]
        elif mode == "below_threshold":
            passed_mask = [s <= threshold for s in scores]
        elif mode == "top_k":
            k = min(k, n_images)
            sorted_indices = sorted(range(n_images), key=lambda i: scores[i], reverse=True)
            top_k_set = set(sorted_indices[:k])
            passed_mask = [i in top_k_set for i in range(n_images)]
        elif mode == "bottom_k":
            k = min(k, n_images)
            sorted_indices = sorted(range(n_images), key=lambda i: scores[i])
            bottom_k_set = set(sorted_indices[:k])
            passed_mask = [i in bottom_k_set for i in range(n_images)]
        else:
            passed_mask = [True] * n_images
        
        passed_indices = [i for i, p in enumerate(passed_mask) if p]
        failed_indices = [i for i, p in enumerate(passed_mask) if not p]
        
        if passed_indices:
            passed_images = torch.stack([images[i] for i in passed_indices], dim=0)
        else:
            passed_images = torch.zeros((1, 1, 1, 3), dtype=images.dtype, device=images.device)
        
        if failed_indices:
            failed_images = torch.stack([images[i] for i in failed_indices], dim=0)
        else:
            failed_images = torch.zeros((1, 1, 1, 3), dtype=images.dtype, device=images.device)
        
        passed_count = len(passed_indices)
        lines = [f"Filter Mode: {mode}"]
        if mode in ["above_threshold", "below_threshold"]:
            lines.append(f"Threshold: {threshold:.4f}")
        else:
            lines.append(f"K: {k}")
        lines.append(f"Passed: {passed_count}/{n_images}")
        lines.append("\nImage Scores:")
        for i, score in enumerate(scores):
            status = "✓" if passed_mask[i] else "✗"
            lines.append(f"  {status} Image {i + 1}: {score:.4f}")
        
        filter_info = "\n".join(lines)
        
        return (passed_images, failed_images, passed_count, filter_info)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ImageRewardLoader_UpdatedPack": ImageRewardLoader,
    "ImageRewardScore_UpdatedPack": ImageRewardScore,
    "ImageRewardRank_UpdatedPack": ImageRewardRank,
    "ImageRewardBatchFilter_UpdatedPack": ImageRewardBatchFilter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageRewardLoader_UpdatedPack": "ImageReward Loader [Updated Pack]",
    "ImageRewardScore_UpdatedPack": "ImageReward Score [Updated Pack]",
    "ImageRewardRank_UpdatedPack": "ImageReward Rank [Updated Pack]",
    "ImageRewardBatchFilter_UpdatedPack": "ImageReward Batch Filter [Updated Pack]",
}
