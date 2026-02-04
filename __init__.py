"""
ComfyUI Updated Pack
====================
A comprehensive collection of ComfyUI custom nodes, updated for compatibility with transformers 5.0.0+.

This pack aggregates and maintains these popular node suites:
- ComfyUI-ImageReward: Image quality scoring with ImageReward and CLIP
- ComfyUI-SAM2: Segment Anything 2 with GroundingDINO integration  
- WAS Node Suite: Extensive node suite with 180+ nodes
- ControlNet Aux: MeshGraphormer hand refiner

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

# Import SAM2 nodes - need to add directory to path first for relative imports
try:
    sam2_dir = os.path.join(__package_dir__, 'ComfyUI-SAM2')
    if sam2_dir not in sys.path:
        sys.path.insert(0, sam2_dir)
    
    # Import node.py directly since it doesn't use relative imports
    import importlib.util
    sam2_node_path = os.path.join(sam2_dir, 'node.py')
    spec = importlib.util.spec_from_file_location("sam2_node", sam2_node_path)
    sam2_node = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sam2_node)
    
    # Use the mappings from ComfyUI-SAM2/__init__.py with proper suffixes
    sam2_class_mappings = {
        'SAM2ModelLoader_UpdatedPack': sam2_node.SAM2ModelLoader,
        'GroundingDinoModelLoader_UpdatedPack': sam2_node.GroundingDinoModelLoader,
        'GroundingDinoSAM2Segment_UpdatedPack': sam2_node.GroundingDinoSAM2Segment,
        'InvertMask_UpdatedPack': sam2_node.InvertMask,
        'IsMaskEmpty_UpdatedPack': sam2_node.IsMaskEmptyNode,
    }
    NODE_CLASS_MAPPINGS.update(sam2_class_mappings)
    
    sam2_display_mappings = {
        'SAM2ModelLoader_UpdatedPack': 'SAM2 Model Loader [Updated Pack]',
        'GroundingDinoModelLoader_UpdatedPack': 'Grounding Dino Model Loader [Updated Pack]',
        'GroundingDinoSAM2Segment_UpdatedPack': 'Grounding Dino SAM2 Segment [Updated Pack]',
        'InvertMask_UpdatedPack': 'Invert Mask [Updated Pack]',
        'IsMaskEmpty_UpdatedPack': 'Is Mask Empty [Updated Pack]',
    }
    NODE_DISPLAY_NAME_MAPPINGS.update(sam2_display_mappings)
    print("[ComfyUI Updated Pack] Successfully loaded SAM2 nodes")
except Exception as e:
    print(f"[ComfyUI Updated Pack] Warning: Failed to load SAM2 nodes: {e}")
    import traceback
    traceback.print_exc()

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

# Import ControlNet Aux nodes (includes MeshGraphormer)
try:
    # Add paths for comfyui_controlnet_aux internal imports
    controlnet_aux_dir = os.path.join(__package_dir__, 'comfyui_controlnet_aux')
    controlnet_aux_src = os.path.join(controlnet_aux_dir, 'src')
    
    # Add all necessary paths BEFORE any imports
    paths_to_add = [
        controlnet_aux_dir,
        controlnet_aux_src,
        os.path.join(controlnet_aux_src, 'custom_controlnet_aux'),
        os.path.join(controlnet_aux_src, 'custom_mesh_graphormer'),
        os.path.join(controlnet_aux_src, 'custom_manopth'),
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    # Set environment variables for MPS fallback and disable problematic ops
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = os.getenv("PYTORCH_ENABLE_MPS_FALLBACK", '1')
    os.environ['NPU_DEVICE_COUNT'] = '0'
    os.environ['MMCV_WITH_OPS'] = '0'
    
    # Use the Updated Pack's own comfyui_controlnet_aux __init__.py to load nodes
    # This is a simplified version that only loads MeshGraphormer nodes
    import importlib.util
    controlnet_aux_init_path = os.path.join(controlnet_aux_dir, '__init__.py')
    spec = importlib.util.spec_from_file_location(
        "comfyui_controlnet_aux_updated_pack", 
        controlnet_aux_init_path,
        submodule_search_locations=[controlnet_aux_dir]
    )
    controlnet_aux_module = importlib.util.module_from_spec(spec)
    
    # Set up the package properly for relative imports
    controlnet_aux_module.__package__ = "comfyui_controlnet_aux_updated_pack"
    sys.modules["comfyui_controlnet_aux_updated_pack"] = controlnet_aux_module
    
    spec.loader.exec_module(controlnet_aux_module)
    
    if hasattr(controlnet_aux_module, 'NODE_CLASS_MAPPINGS'):
        NODE_CLASS_MAPPINGS.update(controlnet_aux_module.NODE_CLASS_MAPPINGS)
    if hasattr(controlnet_aux_module, 'NODE_DISPLAY_NAME_MAPPINGS'):
        NODE_DISPLAY_NAME_MAPPINGS.update(controlnet_aux_module.NODE_DISPLAY_NAME_MAPPINGS)
    
    print("[ComfyUI Updated Pack] Successfully loaded ControlNet Aux nodes (MeshGraphormer)")
except Exception as e:
    print(f"[ComfyUI Updated Pack] Warning: Failed to load ControlNet Aux nodes: {e}")
    import traceback
    traceback.print_exc()

# Export for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print(f"[ComfyUI Updated Pack] Loaded {len(NODE_CLASS_MAPPINGS)} nodes total")
