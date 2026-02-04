"""
ComfyUI ControlNet Aux - Updated Pack
This is a patched version that contains only the MeshGraphormer nodes with
compatibility fixes for newer versions of the transformers library.

This package is designed to work alongside the original comfyui_controlnet_aux
without conflicts by using different node names with "_UpdatedPack" suffix.
"""

import sys, os

# Disable NPU device initialization and problematic MMCV ops to prevent RuntimeError
# Must be set BEFORE any MMCV imports happen anywhere in ComfyUI
os.environ['NPU_DEVICE_COUNT'] = '0'
os.environ['MMCV_WITH_OPS'] = '0'

from pathlib import Path
import traceback
import importlib

# Setup paths
here = Path(__file__).parent.resolve()

# Setup package paths for custom modules
# Ref: https://github.com/comfyanonymous/ComfyUI/blob/76d53c4622fc06372975ed2a43ad345935b8a551/nodes.py#L17
sys.path.insert(0, str(Path(here, "src").resolve()))
for pkg_name in ["custom_controlnet_aux", "custom_mmpkg"]:
    sys.path.append(str(Path(here, "src", pkg_name).resolve()))

# Enable CPU fallback for ops not being supported by MPS like upsample_bicubic2d.out
# https://github.com/pytorch/pytorch/issues/77764
# https://github.com/Fannovel16/comfyui_controlnet_aux/issues/2#issuecomment-1763579485
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = os.getenv("PYTORCH_ENABLE_MPS_FALLBACK", '1')


def load_nodes():
    """Load MeshGraphormer nodes from node_wrappers."""
    shorted_errors = []
    full_error_messages = []
    node_class_mappings = {}
    node_display_name_mappings = {}

    for filename in (here / "node_wrappers").iterdir():
        module_name = filename.stem
        # Skip hidden files created by the OS (e.g. .DS_Store)
        if module_name.startswith('.'):
            continue
        try:
            module = importlib.import_module(
                f".node_wrappers.{module_name}", package=__package__
            )
            node_class_mappings.update(getattr(module, "NODE_CLASS_MAPPINGS"))
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                node_display_name_mappings.update(getattr(module, "NODE_DISPLAY_NAME_MAPPINGS"))

            print(f"[comfyui_controlnet_aux Updated Pack] Loaded {module_name} nodes")

        except AttributeError:
            pass  # wip nodes
        except Exception:
            error_message = traceback.format_exc()
            full_error_messages.append(error_message)
            error_message = error_message.splitlines()[-1]
            shorted_errors.append(
                f"Failed to import module {module_name} because {error_message}"
            )
    
    if len(shorted_errors) > 0:
        full_err_log = '\n\n'.join(full_error_messages)
        print(f"\n\nFull error log from comfyui_controlnet_aux Updated Pack: \n{full_err_log}\n\n")
        print(
            f"[comfyui_controlnet_aux Updated Pack] Some nodes failed to load:\n\t"
            + "\n\t".join(shorted_errors)
            + "\n\n"
            + "Check that you properly installed the dependencies.\n"
        )
    return node_class_mappings, node_display_name_mappings


# Load all nodes
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = load_nodes()
