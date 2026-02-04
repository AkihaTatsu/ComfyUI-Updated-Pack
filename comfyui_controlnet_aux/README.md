# ComfyUI ControlNet Aux - Updated Pack

This is a **patched version** of [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) that contains only the **MeshGraphormer** nodes with compatibility fixes for newer versions of the `transformers` library (5.0.0+).

## Why this exists

The original `comfyui_controlnet_aux` library uses functions from `transformers.modeling_utils` that were removed in `transformers >= 5.0.0`:
- `prune_linear_layer`
- `find_pruneable_heads_and_indices`
- `apply_chunking_to_forward`

These functions were moved to `transformers.pytorch_utils` in version 4.x and completely removed in version 5.0.0+. This Updated Pack provides fallback implementations to maintain compatibility.

## Included Nodes

This package contains only the MeshGraphormer-related nodes:

| Node Name | Display Name |
|-----------|-------------|
| `MeshGraphormer-DepthMapPreprocessor_UpdatedPack` | MeshGraphormer Hand Refiner [Updated Pack] |
| `MeshGraphormer+ImpactDetector-DepthMapPreprocessor_UpdatedPack` | MeshGraphormer Hand Refiner With External Detector [Updated Pack] |

All nodes are placed under the menu: **Updated Pack / ControlNet Preprocessors / Normal and Depth Estimators**

## Compatibility

This package is designed to work **alongside** the original `comfyui_controlnet_aux` without conflicts:
- Node class names have `_UpdatedPack` suffix
- Node display names have ` [Updated Pack]` suffix
- Nodes are placed in a separate menu category

## Changes from Original

1. **Patched `custom_mesh_graphormer/modeling/bert/modeling_utils.py`**: Added fallback implementations for removed transformers functions
2. **Removed unrelated nodes**: Only MeshGraphormer nodes are included
3. **Renamed nodes**: All node names and class names have Updated Pack suffixes
4. **Separate menu category**: Nodes appear under "Updated Pack" submenu

## Installation

1. Clone or copy this folder to your ComfyUI `custom_nodes` directory
2. Install dependencies: `pip install -r requirements.txt`
3. Restart ComfyUI

## Credits

Original code from [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) by Fannovel16.

The MeshGraphormer model is based on the work from Microsoft Research.
