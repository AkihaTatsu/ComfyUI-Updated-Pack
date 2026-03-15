"""
ComfyUI Updated Pack
====================
A comprehensive collection of ComfyUI custom nodes, updated for compatibility with transformers 5.0.0+.
"""

import importlib.util
import json
import os
import sys

__package_dir__ = os.path.dirname(os.path.abspath(__file__))
if __package_dir__ not in sys.path:
    sys.path.insert(0, __package_dir__)

ACTIVATION_FILE = os.path.join(__package_dir__, "ACTIVATED_PACKS.json")
PACK_NAMES = (
    "AIGODLIKE-ComfyUI-Studio",
    "ComfyUI-ImageReward",
    "ComfyUI-SAM2",
    "was-node-suite-comfyui",
    "comfyui_controlnet_aux",
)


def _build_activation_config():
    return {pack_name: True for pack_name in PACK_NAMES}


def _write_activation_file(config):
    try:
        with open(ACTIVATION_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise RuntimeError(f"Failed to write ACTIVATED_PACKS.json: {e}") from e


def _rewrite_activation_file(reason: str):
    config = _build_activation_config()
    _write_activation_file(config)
    print(f"[ComfyUI Updated Pack] Rewrote ACTIVATED_PACKS.json: {reason}")
    return config


def _load_activation_config():
    if not os.path.exists(ACTIVATION_FILE):
        return _rewrite_activation_file("file not found")

    try:
        with open(ACTIVATION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return _rewrite_activation_file("root must be a JSON object")

        expected_keys = set(PACK_NAMES)
        actual_keys = set(data.keys())

        if actual_keys != expected_keys:
            missing_keys = sorted(expected_keys - actual_keys)
            extra_keys = sorted(actual_keys - expected_keys)
            reason_parts = []
            if missing_keys:
                reason_parts.append(f"missing keys: {missing_keys}")
            if extra_keys:
                reason_parts.append(f"extra keys: {extra_keys}")
            return _rewrite_activation_file("; ".join(reason_parts))

        invalid_types = [key for key in PACK_NAMES if not isinstance(data.get(key), bool)]
        if invalid_types:
            return _rewrite_activation_file(f"non-bool values for keys: {invalid_types}")

        return data
    except Exception as e:
        return _rewrite_activation_file(f"invalid JSON or read error: {e}")


ACTIVATED_PACKS = _load_activation_config()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _is_enabled(pack_name: str) -> bool:
    return bool(ACTIVATED_PACKS.get(pack_name, True))


# 1) AIGODLIKE-ComfyUI-Studio (first)
if _is_enabled("AIGODLIKE-ComfyUI-Studio"):
    try:
        studio_dir = os.path.join(__package_dir__, "AIGODLIKE-ComfyUI-Studio")
        studio_init_path = os.path.join(studio_dir, "__init__.py")

        studio_spec = importlib.util.spec_from_file_location(
            "updated_pack_aigodlike_studio",
            studio_init_path,
            submodule_search_locations=[studio_dir],
        )
        studio_module = importlib.util.module_from_spec(studio_spec)
        studio_module.__package__ = "updated_pack_aigodlike_studio"
        sys.modules["updated_pack_aigodlike_studio"] = studio_module
        studio_spec.loader.exec_module(studio_module)

        if hasattr(studio_module, "NODE_CLASS_MAPPINGS"):
            NODE_CLASS_MAPPINGS.update(studio_module.NODE_CLASS_MAPPINGS)
        if hasattr(studio_module, "NODE_DISPLAY_NAME_MAPPINGS"):
            NODE_DISPLAY_NAME_MAPPINGS.update(studio_module.NODE_DISPLAY_NAME_MAPPINGS)

        WEB_DIRECTORY = "./AIGODLIKE-ComfyUI-Studio/loader"
        print("[ComfyUI Updated Pack] Successfully loaded AIGODLIKE-ComfyUI-Studio")
    except Exception as e:
        print(f"[ComfyUI Updated Pack] Warning: Failed to load AIGODLIKE-ComfyUI-Studio: {e}")
else:
    print("[ComfyUI Updated Pack] Skipped AIGODLIKE-ComfyUI-Studio (disabled)")


# 2) ImageReward
if _is_enabled("ComfyUI-ImageReward"):
    try:
        imagereward_path = os.path.join(__package_dir__, "ComfyUI-ImageReward", "nodes.py")
        spec = importlib.util.spec_from_file_location("imagereward_nodes", imagereward_path)
        imagereward_nodes = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(imagereward_nodes)

        if hasattr(imagereward_nodes, "NODE_CLASS_MAPPINGS"):
            NODE_CLASS_MAPPINGS.update(imagereward_nodes.NODE_CLASS_MAPPINGS)
        if hasattr(imagereward_nodes, "NODE_DISPLAY_NAME_MAPPINGS"):
            NODE_DISPLAY_NAME_MAPPINGS.update(imagereward_nodes.NODE_DISPLAY_NAME_MAPPINGS)
        print("[ComfyUI Updated Pack] Successfully loaded ImageReward nodes")
    except Exception as e:
        print(f"[ComfyUI Updated Pack] Warning: Failed to load ImageReward nodes: {e}")
else:
    print("[ComfyUI Updated Pack] Skipped ImageReward (disabled)")


# 3) SAM2
if _is_enabled("ComfyUI-SAM2"):
    try:
        sam2_dir = os.path.join(__package_dir__, "ComfyUI-SAM2")
        if sam2_dir not in sys.path:
            sys.path.insert(0, sam2_dir)

        sam2_node_path = os.path.join(sam2_dir, "node.py")
        spec = importlib.util.spec_from_file_location("sam2_node", sam2_node_path)
        sam2_node = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sam2_node)

        sam2_class_mappings = {
            "SAM2ModelLoader_UpdatedPack": sam2_node.SAM2ModelLoader,
            "GroundingDinoModelLoader_UpdatedPack": sam2_node.GroundingDinoModelLoader,
            "GroundingDinoSAM2Segment_UpdatedPack": sam2_node.GroundingDinoSAM2Segment,
            "InvertMask_UpdatedPack": sam2_node.InvertMask,
            "IsMaskEmpty_UpdatedPack": sam2_node.IsMaskEmptyNode,
        }
        NODE_CLASS_MAPPINGS.update(sam2_class_mappings)

        sam2_display_mappings = {
            "SAM2ModelLoader_UpdatedPack": "SAM2 Model Loader [Updated Pack]",
            "GroundingDinoModelLoader_UpdatedPack": "Grounding Dino Model Loader [Updated Pack]",
            "GroundingDinoSAM2Segment_UpdatedPack": "Grounding Dino SAM2 Segment [Updated Pack]",
            "InvertMask_UpdatedPack": "Invert Mask [Updated Pack]",
            "IsMaskEmpty_UpdatedPack": "Is Mask Empty [Updated Pack]",
        }
        NODE_DISPLAY_NAME_MAPPINGS.update(sam2_display_mappings)
        print("[ComfyUI Updated Pack] Successfully loaded SAM2 nodes")
    except Exception as e:
        print(f"[ComfyUI Updated Pack] Warning: Failed to load SAM2 nodes: {e}")
        import traceback

        traceback.print_exc()
else:
    print("[ComfyUI Updated Pack] Skipped SAM2 (disabled)")


# 4) WAS Node Suite
if _is_enabled("was-node-suite-comfyui"):
    try:
        was_path = os.path.join(__package_dir__, "was-node-suite-comfyui", "WAS_Node_Suite.py")
        spec = importlib.util.spec_from_file_location("WAS_Node_Suite", was_path)
        WAS_Node_Suite = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(WAS_Node_Suite)

        if hasattr(WAS_Node_Suite, "NODE_CLASS_MAPPINGS"):
            NODE_CLASS_MAPPINGS.update(WAS_Node_Suite.NODE_CLASS_MAPPINGS)
        if hasattr(WAS_Node_Suite, "NODE_DISPLAY_NAME_MAPPINGS"):
            NODE_DISPLAY_NAME_MAPPINGS.update(WAS_Node_Suite.NODE_DISPLAY_NAME_MAPPINGS)
        print("[ComfyUI Updated Pack] Successfully loaded WAS Node Suite")
    except Exception as e:
        print(f"[ComfyUI Updated Pack] Warning: Failed to load WAS Node Suite: {e}")
else:
    print("[ComfyUI Updated Pack] Skipped WAS Node Suite (disabled)")


# 5) ControlNet Aux (MeshGraphormer)
if _is_enabled("comfyui_controlnet_aux"):
    try:
        controlnet_aux_dir = os.path.join(__package_dir__, "comfyui_controlnet_aux")
        controlnet_aux_src = os.path.join(controlnet_aux_dir, "src")

        paths_to_add = [
            controlnet_aux_dir,
            controlnet_aux_src,
            os.path.join(controlnet_aux_src, "custom_controlnet_aux"),
            os.path.join(controlnet_aux_src, "custom_mesh_graphormer"),
            os.path.join(controlnet_aux_src, "custom_manopth"),
        ]

        for path in paths_to_add:
            if path not in sys.path:
                sys.path.insert(0, path)

        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = os.getenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        os.environ["NPU_DEVICE_COUNT"] = "0"
        os.environ["MMCV_WITH_OPS"] = "0"

        controlnet_aux_init_path = os.path.join(controlnet_aux_dir, "__init__.py")
        spec = importlib.util.spec_from_file_location(
            "comfyui_controlnet_aux_updated_pack",
            controlnet_aux_init_path,
            submodule_search_locations=[controlnet_aux_dir],
        )
        controlnet_aux_module = importlib.util.module_from_spec(spec)
        controlnet_aux_module.__package__ = "comfyui_controlnet_aux_updated_pack"
        sys.modules["comfyui_controlnet_aux_updated_pack"] = controlnet_aux_module

        spec.loader.exec_module(controlnet_aux_module)

        if hasattr(controlnet_aux_module, "NODE_CLASS_MAPPINGS"):
            NODE_CLASS_MAPPINGS.update(controlnet_aux_module.NODE_CLASS_MAPPINGS)
        if hasattr(controlnet_aux_module, "NODE_DISPLAY_NAME_MAPPINGS"):
            NODE_DISPLAY_NAME_MAPPINGS.update(controlnet_aux_module.NODE_DISPLAY_NAME_MAPPINGS)

        print("[ComfyUI Updated Pack] Successfully loaded ControlNet Aux nodes (MeshGraphormer)")
    except Exception as e:
        print(f"[ComfyUI Updated Pack] Warning: Failed to load ControlNet Aux nodes: {e}")
        import traceback

        traceback.print_exc()
else:
    print("[ComfyUI Updated Pack] Skipped ControlNet Aux (disabled)")


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "ACTIVATED_PACKS"]

print(f"[ComfyUI Updated Pack] Loaded {len(NODE_CLASS_MAPPINGS)} nodes total")
