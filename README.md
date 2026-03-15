# ComfyUI Updated Pack

A comprehensive collection of ComfyUI custom nodes updated for transformers compatibility (supports both older versions and 5.0.0+).

This pack aggregates and maintains these popular ComfyUI node suites with necessary updates to ensure compatibility with various transformers library versions. All nodes have been tested and patched where necessary to work with modern dependency versions.

## What's Included

This pack combines these excellent node suites:

### 1. AIGODLIKE-ComfyUI-Studio - Loader Model Manager & UI Enhancements
- **Original Author**: [AIGODLIKE](https://github.com/AIGODLIKE/AIGODLIKE-ComfyUI-Studio)
- **Description**: Enhanced loader interaction, model thumbnails, model notes/workflows, filtering and multilingual UI
- **Updates**: Includes canvas interaction compatibility fix to prevent blank-canvas drag interference

### 2. ComfyUI-ImageReward - Image Quality Scoring
- **Original Author**: [ZaneA](https://github.com/ZaneA/ComfyUI-ImageReward)
- **Description**: Score and rank images using ImageReward or CLIP models
- **Updates**: Fixed transformers compatibility (handles pytorch_utils changes across versions)

### 3. ComfyUI-SAM2 - Segment Anything 2
- **Original Author**: [neverbiasu](https://github.com/neverbiasu/ComfyUI-SAM2)
- **Description**: Advanced image segmentation using SAM2 and GroundingDINO
- **Updates**: Compatible with transformers 4.x and 5.0.0+ (uses AutoTokenizer and BERT models correctly)

### 4. WAS Node Suite - Comprehensive Node Collection
- **Original Author**: [WASasquatch](https://github.com/WASasquatch/was-node-suite-comfyui)
- **Description**: An extensive suite with 180+ nodes for image processing, text processing, video, and more
- **Updates**: Patched BLIP module for transformers compatibility (apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer)

### 5. ComfyUI ControlNet Aux (MeshGraphormer Only)
- **Original Author**: [Fannovel16](https://github.com/Fannovel16/comfyui_controlnet_aux)
- **Description**: MeshGraphormer Hand Refiner for generating hand depth maps (stripped-down version with only MeshGraphormer nodes)
- **Updates**: Patched custom_mesh_graphormer/modeling/bert/modeling_utils.py for transformers compatibility (prune_linear_layer, find_pruneable_heads_and_indices, apply_chunking_to_forward)

## Package Activation Control

`ComfyUI-Updated-Pack` provides `ACTIVATED_PACKS.json` to control which bundled packs are enabled.

Default file (all enabled):

```json
{
   "AIGODLIKE-ComfyUI-Studio": true,
   "ComfyUI-ImageReward": true,
   "ComfyUI-SAM2": true,
   "was-node-suite-comfyui": true,
   "comfyui_controlnet_aux": true
}
```

Set a pack to `false` to fully disable it at load time. For example, setting `"AIGODLIKE-ComfyUI-Studio": false` disables all Studio-related backend routes and frontend UI integration.

## Installation

### For ComfyUI Users

1. Clone this repository into your ComfyUI custom_nodes folder:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/YOUR-USERNAME/ComfyUI-Updated-Pack
   ```

2. Install dependencies:
   ```bash
   cd ComfyUI-Updated-Pack
   pip install -r requirements.txt
   ```

3. Restart ComfyUI

### Optional: ImageReward Support

The original `image-reward` is outdated and an independent implementation is used here.

If your environment is compatible with this one, you can install the optional dependency:
```bash
pip install image-reward
```
If not installed, the nodes will automatically fall back to using CLIP for scoring.

## Node List

### ImageReward Nodes

All nodes are in the `ImageReward` category.

| Node | Description |
|------|-------------|
| **ImageRewardLoader_UpdatedPack** | Load ImageReward or CLIP model for image scoring |
| **ImageRewardScore_UpdatedPack** | Score a single image based on a text prompt |
| **ImageRewardScoreBatch_UpdatedPack** | Score multiple images and return rankings |
| **ImageRewardFilter_UpdatedPack** | Filter images based on quality threshold |

Use Cases: Score generated images based on prompt alignment, rank multiple image variations, filter low-quality outputs, quality-aware image selection in workflows.

### SAM2 (Segment Anything 2) Nodes

All nodes are in the `segment anything` or `segment anything2` category.

| Node | Description |
|------|-------------|
| **SAM2ModelLoader_UpdatedPack** | Load SAM2 segmentation models (tiny/small/base/large) |
| **GroundingDinoModelLoader_UpdatedPack** | Load GroundingDINO models for text-based object detection |
| **GroundingDinoSAM2Segment_UpdatedPack** | Segment objects in images using text prompts |
| **InvertMask_UpdatedPack** | Invert segmentation masks |
| **IsMaskEmpty_UpdatedPack** | Check if a mask is empty |

Use Cases: Text-based object segmentation, advanced masking for inpainting, object isolation and removal, precision editing workflows.

Supported Models:
- SAM 2.1: `sam2.1_hiera_{tiny,small,base_plus,large}.pt`
- SAM 2.0: `sam2_hiera_{tiny,small,base_plus,large}.pt`
- GroundingDINO: SwinT_OGC (694MB), SwinB (938MB)

Models are downloaded automatically to `ComfyUI/models/sam2/` and `ComfyUI/models/grounding-dino/`.

### WAS Node Suite Nodes

This suite contains 180+ nodes across multiple categories:

**Image Processing Nodes**: Image filtering and adjustments (blur, sharpen, grain, etc.), image transformations (flip, rotate, crop, resize), color adjustments and grading, blending modes and compositing, face detection and cropping, edge detection and masking, special effects (bloom, chromatic aberration, etc.)

**Text Processing Nodes**: Text manipulation and parsing, wildcard support (A1111 style), NSP (Noodle Soup Prompts) parsing, text file operations, string operations and formatting.

**Video Nodes**: Create videos from image sequences, video frame extraction, morph animations between images.

**AI/ML Nodes**: BLIP image captioning, BLIP visual question answering, CLIPSeg segmentation, SAM (Segment Anything v1) integration.

**Utility Nodes**: Bus nodes for cleaner workflows, cache nodes for latents and images, switch nodes for conditional routing, number and logic operations, dictionary and data structure handling.

For a complete list of all WAS Node Suite nodes, see the [original documentation](https://github.com/WASasquatch/was-node-suite-comfyui#current-nodes).

### MeshGraphormer Nodes (ControlNet Aux)

All nodes are in the `Updated Pack/ControlNet Preprocessors/Normal and Depth Estimators` category.

| Node | Description |
|------|-------------|
| **MeshGraphormer-DepthMapPreprocessor_UpdatedPack** | Generate hand depth maps using MeshGraphormer |
| **MeshGraphormer+ImpactDetector-DepthMapPreprocessor_UpdatedPack** | MeshGraphormer with external bounding box detector from Impact Pack |

Use Cases: Hand refinement for image generation, generating accurate hand depth maps for ControlNet, fixing hand details in generated images.

Compatible ControlNet Model: [control_sd15_inpaint_depth_hand_fp16](https://huggingface.co/hr16/ControlNet-HandRefiner-pruned/blob/main/control_sd15_inpaint_depth_hand_fp16.safetensors)

## What Was Updated

### Transformers Compatibility (4.x and 5.0.0+)

The main update in this pack addresses breaking changes across transformers versions:

1. **WAS Node Suite (BLIP module)**:
   - Patched imports for `apply_chunking_to_forward`, `find_pruneable_heads_and_indices`, `prune_linear_layer`
   - These functions were in `modeling_utils` (older versions), moved to `pytorch_utils` (4.x), and removed in 5.0.0+
   - Added fallback implementations for transformers 5.0.0+ compatibility

2. **ImageReward Nodes**: 
   - Updated transformer patching mechanism to handle version differences gracefully

3. **SAM2 Nodes**:
   - Verified compatibility with AutoTokenizer and BertModel imports
   - Uses stable APIs that work across versions

4. **MeshGraphormer Nodes (ControlNet Aux)**:
   - Patched `custom_mesh_graphormer/modeling/bert/modeling_utils.py`
   - Added fallback implementations for `prune_linear_layer`, `find_pruneable_heads_and_indices`, `apply_chunking_to_forward`
   - Removed all other ControlNet preprocessor nodes to keep only MeshGraphormer functionality

### Structural Changes

- Removed individual `requirements.txt`, `pyproject.toml`, and `.github` folders from sub-packages
- Created unified dependency management in the root folder
- Added aggregated `__init__.py` for seamless node loading
- Maintained original folder structures for easy updates

## Credits

This pack is built upon the excellent work of:

- **[AIGODLIKE](https://github.com/AIGODLIKE)** - [AIGODLIKE-ComfyUI-Studio](https://github.com/AIGODLIKE/AIGODLIKE-ComfyUI-Studio)
- **[ZaneA](https://github.com/ZaneA)** - [ComfyUI-ImageReward](https://github.com/ZaneA/ComfyUI-ImageReward)
- **[neverbiasu](https://github.com/neverbiasu)** - [ComfyUI-SAM2](https://github.com/neverbiasu/ComfyUI-SAM2)
- **[WASasquatch](https://github.com/WASasquatch)** - [was-node-suite-comfyui](https://github.com/WASasquatch/was-node-suite-comfyui)
- **[Fannovel16](https://github.com/Fannovel16)** - [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux)

Special thanks to:
- The [ComfyUI](https://github.com/comfyanonymous/ComfyUI) team for the amazing framework
- The [Hugging Face](https://huggingface.co/) team for transformers library
- Meta AI for [Segment Anything 2](https://github.com/facebookresearch/segment-anything-2)
- The open-source community for continued support and contributions

## License

See individual LICENSE files in each sub-folder for details.

## Issues and Contributions

If you encounter any issues or have suggestions:

1. Check if the issue is specific to one of the original repositories
2. If it's a compatibility issue with this pack, please open an issue here
3. Pull requests for improvements are welcome!

## Related Projects

- [AIGODLIKE-ComfyUI-Studio](https://github.com/AIGODLIKE/AIGODLIKE-ComfyUI-Studio) - Advanced loader model manager and UI enhancements
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - The main ComfyUI repository
- [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) - Easy node installation and management
- [Segment Anything 2](https://github.com/facebookresearch/segment-anything-2) - SAM2 official repository
- [ImageReward](https://github.com/THUDM/ImageReward) - Human preference reward model
- [MeshGraphormer](https://github.com/microsoft/MeshGraphormer) - Microsoft's MeshGraphormer for hand reconstruction
- [HandRefiner](https://github.com/wenquanlu/HandRefiner) - Hand refinement using MeshGraphormer
