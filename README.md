# ComfyUI Updated Pack

A comprehensive collection of ComfyUI custom nodes updated for transformers compatibility (supports both older versions and 5.0.0+).

This pack aggregates and maintains three popular ComfyUI node suites with necessary updates to ensure compatibility with various transformers library versions. All nodes have been tested and patched where necessary to work with modern dependency versions.

## What's Included

This pack combines three excellent node suites:

### 1. ComfyUI-ImageReward - Image Quality Scoring
- **Original Author**: [ZaneA](https://github.com/ZaneA/ComfyUI-ImageReward)
- **Description**: Score and rank images using ImageReward or CLIP models
- **Updates**: Fixed transformers compatibility (handles pytorch_utils changes across versions)

### 2. ComfyUI-SAM2 - Segment Anything 2
- **Original Author**: [neverbiasu](https://github.com/neverbiasu/ComfyUI-SAM2)
- **Description**: Advanced image segmentation using SAM2 and GroundingDINO
- **Updates**: Compatible with transformers 4.x and 5.0.0+ (uses AutoTokenizer and BERT models correctly)

### 3. WAS Node Suite - Comprehensive Node Collection
- **Original Author**: [WASasquatch](https://github.com/WASasquatch/was-node-suite-comfyui)
- **Description**: An extensive suite with 180+ nodes for image processing, text processing, video, and more
- **Updates**: Patched BLIP module for transformers compatibility (apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer)

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
| **ImageRewardLoader** | Load ImageReward or CLIP model for image scoring |
| **ImageRewardScore** | Score a single image based on a text prompt |
| **ImageRewardScoreBatch** | Score multiple images and return rankings |
| **ImageRewardFilter** | Filter images based on quality threshold |

Use Cases: Score generated images based on prompt alignment, rank multiple image variations, filter low-quality outputs, quality-aware image selection in workflows.

### SAM2 (Segment Anything 2) Nodes

All nodes are in the `segment anything` or `segment anything2` category.

| Node | Description |
|------|-------------|
| **SAM2ModelLoader** | Load SAM2 segmentation models (tiny/small/base/large) |
| **GroundingDinoModelLoader** | Load GroundingDINO models for text-based object detection |
| **GroundingDinoSAM2Segment** | Segment objects in images using text prompts |
| **InvertMask** | Invert segmentation masks |
| **IsMaskEmpty** | Check if a mask is empty |

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

### Structural Changes

- Removed individual `requirements.txt`, `pyproject.toml`, and `.github` folders from sub-packages
- Created unified dependency management in the root folder
- Added aggregated `__init__.py` for seamless node loading
- Maintained original folder structures for easy updates

## Dependencies

Key dependencies (automatically installed):
- `transformers` - Supports both older and newer versions
- `torch>=2.0.0` - Deep learning framework
- `timm>=0.4.12` - Vision models
- `opencv-python>=4.7.0` - Image processing
- `numpy<2` - Array operations
- `Pillow>=9.4.0` - Image handling

See [requirements.txt](requirements.txt) for the complete list.

## Credits

This pack is built upon the excellent work of:

- **[ZaneA](https://github.com/ZaneA)** - [ComfyUI-ImageReward](https://github.com/ZaneA/ComfyUI-ImageReward)
- **[neverbiasu](https://github.com/neverbiasu)** - [ComfyUI-SAM2](https://github.com/neverbiasu/ComfyUI-SAM2)
- **[WASasquatch](https://github.com/WASasquatch)** - [was-node-suite-comfyui](https://github.com/WASasquatch/was-node-suite-comfyui)

Special thanks to:
- The [ComfyUI](https://github.com/comfyanonymous/ComfyUI) team for the amazing framework
- The [Hugging Face](https://huggingface.co/) team for transformers library
- Meta AI for [Segment Anything 2](https://github.com/facebookresearch/segment-anything-2)
- The open-source community for continued support and contributions

## License

This pack maintains the original licenses of each included component:

- **ComfyUI-ImageReward**: MIT License
- **ComfyUI-SAM2**: Apache License 2.0
- **WAS Node Suite**: MIT License

See individual LICENSE files in each sub-folder for details.

## Issues and Contributions

If you encounter any issues or have suggestions:

1. Check if the issue is specific to one of the original repositories
2. If it's a compatibility issue with this pack, please open an issue here
3. Pull requests for improvements are welcome!

## Related Projects

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - The main ComfyUI repository
- [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) - Easy node installation and management
- [Segment Anything 2](https://github.com/facebookresearch/segment-anything-2) - SAM2 official repository
- [ImageReward](https://github.com/THUDM/ImageReward) - Human preference reward model
