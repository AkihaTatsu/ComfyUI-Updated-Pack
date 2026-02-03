"""
ComfyUI Updated Pack
====================
A comprehensive collection of ComfyUI custom nodes, updated for compatibility with transformers 5.0.0+.

This pack aggregates and maintains these popular node suites:
- ComfyUI-ImageReward: Image quality scoring with ImageReward and CLIP
- ComfyUI-SAM2: Segment Anything 2 with GroundingDINO integration
- WAS Node Suite: Extensive node suite with 180+ nodes

All nodes are updated to work with the latest transformers library.
"""

import sys
import os

# Add the main directory to Python path
__package_dir__ = os.path.dirname(os.path.abspath(__file__))
if __package_dir__ not in sys.path:
    sys.path.insert(0, __package_dir__)

# Import node mappings from each suite
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Import ImageReward nodes
try:
    import importlib.util
    imagereward_path = os.path.join(__package_dir__, 'ComfyUI-ImageReward', 'nodes.py')
    spec = importlib.util.spec_from_file_location("imagereward_nodes", imagereward_path)
    imagereward_nodes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imagereward_nodes)
    
    if hasattr(imagereward_nodes, 'NODE_CLASS_MAPPINGS'):
        NODE_CLASS_MAPPINGS.update(imagereward_nodes.NODE_CLASS_MAPPINGS)
    if hasattr(imagereward_nodes, 'NODE_DISPLAY_NAME_MAPPINGS'):
        NODE_DISPLAY_NAME_MAPPINGS.update(imagereward_nodes.NODE_DISPLAY_NAME_MAPPINGS)
    print("[ComfyUI Updated Pack] Successfully loaded ImageReward nodes")
except Exception as e:
    print(f"[ComfyUI Updated Pack] Warning: Failed to load ImageReward nodes: {e}")

# Import SAM2 nodes
try:
    import importlib.util
    sam2_path = os.path.join(__package_dir__, 'ComfyUI-SAM2', 'node.py')
    spec = importlib.util.spec_from_file_location("sam2_node", sam2_path)
    sam2_node = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sam2_node)
    
    sam2_mappings = {
        'SAM2ModelLoader': sam2_node.SAM2ModelLoader,
        'GroundingDinoModelLoader': sam2_node.GroundingDinoModelLoader,
        'GroundingDinoSAM2Segment': sam2_node.GroundingDinoSAM2Segment,
        'InvertMask': sam2_node.InvertMask,
        'IsMaskEmpty': sam2_node.IsMaskEmptyNode,
    }
    NODE_CLASS_MAPPINGS.update(sam2_mappings)
    
    sam2_display_mappings = {
        'SAM2ModelLoader': 'SAM2 Model Loader',
        'GroundingDinoModelLoader': 'Grounding Dino Model Loader',
        'GroundingDinoSAM2Segment': 'Grounding Dino SAM2 Segment',
        'InvertMask': 'Invert Mask',
        'IsMaskEmpty': 'Is Mask Empty',
    }
    NODE_DISPLAY_NAME_MAPPINGS.update(sam2_display_mappings)
    print("[ComfyUI Updated Pack] Successfully loaded SAM2 nodes")
except Exception as e:
    print(f"[ComfyUI Updated Pack] Warning: Failed to load SAM2 nodes: {e}")

# Import WAS Node Suite
try:
    import importlib.util
    was_path = os.path.join(__package_dir__, 'was-node-suite-comfyui', 'WAS_Node_Suite.py')
    spec = importlib.util.spec_from_file_location("WAS_Node_Suite", was_path)
    WAS_Node_Suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(WAS_Node_Suite)
    
    if hasattr(WAS_Node_Suite, 'NODE_CLASS_MAPPINGS'):
        NODE_CLASS_MAPPINGS.update(WAS_Node_Suite.NODE_CLASS_MAPPINGS)
    if hasattr(WAS_Node_Suite, 'NODE_DISPLAY_NAME_MAPPINGS'):
        NODE_DISPLAY_NAME_MAPPINGS.update(WAS_Node_Suite.NODE_DISPLAY_NAME_MAPPINGS)
    print("[ComfyUI Updated Pack] Successfully loaded WAS Node Suite")
except Exception as e:
    print(f"[ComfyUI Updated Pack] Warning: Failed to load WAS Node Suite: {e}")

# Export for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print(f"[ComfyUI Updated Pack] Loaded {len(NODE_CLASS_MAPPINGS)} nodes total")
