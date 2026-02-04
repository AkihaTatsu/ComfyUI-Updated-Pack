from .node import SAM2ModelLoader, GroundingDinoModelLoader, GroundingDinoSAM2Segment, InvertMask, IsMaskEmptyNode

NODE_CLASS_MAPPINGS = {
    'SAM2ModelLoader_UpdatedPack': SAM2ModelLoader,
    'GroundingDinoModelLoader_UpdatedPack': GroundingDinoModelLoader,
    'GroundingDinoSAM2Segment_UpdatedPack': GroundingDinoSAM2Segment,
    'InvertMask_UpdatedPack': InvertMask,
    "IsMaskEmpty_UpdatedPack": IsMaskEmptyNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'SAM2ModelLoader_UpdatedPack': 'SAM2 Model Loader [Updated Pack]',
    'GroundingDinoModelLoader_UpdatedPack': 'Grounding Dino Model Loader [Updated Pack]',
    'GroundingDinoSAM2Segment_UpdatedPack': 'Grounding Dino SAM2 Segment [Updated Pack]',
    'InvertMask_UpdatedPack': 'Invert Mask [Updated Pack]',
    "IsMaskEmpty_UpdatedPack": "Is Mask Empty [Updated Pack]",
}
