"""
Utility functions for comfyui_controlnet_aux Updated Pack.
Simplified version containing only what's needed for MeshGraphormer nodes.
"""

import os
import yaml
import subprocess
import threading
import tempfile
from pathlib import Path
from enum import Enum

here = Path(__file__).parent.resolve()

config_path = Path(here, "config.yaml")

if os.path.exists(config_path):
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    annotator_ckpts_path = str(Path(here, config["annotator_ckpts_path"]))
    TEMP_DIR = config["custom_temp_path"]
    USE_SYMLINKS = config["USE_SYMLINKS"]
    ORT_PROVIDERS = config["EP_list"]

    if USE_SYMLINKS is None or type(USE_SYMLINKS) != bool:
        USE_SYMLINKS = False

    if TEMP_DIR is None:
        TEMP_DIR = tempfile.gettempdir()
    elif not os.path.isdir(TEMP_DIR):
        try:
            os.makedirs(TEMP_DIR)
        except:
            TEMP_DIR = tempfile.gettempdir()

    if not os.path.isdir(annotator_ckpts_path):
        try:
            os.makedirs(annotator_ckpts_path)
        except:
            annotator_ckpts_path = str(Path(here, "./ckpts"))
else:
    annotator_ckpts_path = str(Path(here, "./ckpts"))
    TEMP_DIR = tempfile.gettempdir()
    USE_SYMLINKS = False
    ORT_PROVIDERS = ["CUDAExecutionProvider", "DirectMLExecutionProvider", "OpenVINOExecutionProvider", "ROCMExecutionProvider", "MIGraphXExecutionProvider", "CPUExecutionProvider", "CoreMLExecutionProvider"]

os.environ['AUX_ANNOTATOR_CKPTS_PATH'] = os.getenv('AUX_ANNOTATOR_CKPTS_PATH', annotator_ckpts_path)
os.environ['AUX_TEMP_DIR'] = os.getenv('AUX_TEMP_DIR', str(TEMP_DIR))
os.environ['AUX_USE_SYMLINKS'] = os.getenv('AUX_USE_SYMLINKS', str(USE_SYMLINKS))
os.environ['AUX_ORT_PROVIDERS'] = os.getenv('AUX_ORT_PROVIDERS', str(",".join(ORT_PROVIDERS)))

# Sync with theoretical limit from Comfy base
# https://github.com/comfyanonymous/ComfyUI/blob/eecd69b53a896343775bcb02a4f8349e7442ffd1/nodes.py#L45
MAX_RESOLUTION = 16384


def define_preprocessor_inputs(**arguments):
    return dict(
        required=dict(image=INPUT.IMAGE()),
        optional=arguments
    )


class INPUT(Enum):
    def IMAGE():
        return ("IMAGE",)
    def LATENT():
        return ("LATENT",)
    def MASK():
        return ("MASK",)
    def SEED(default=0):
        return ("INT", dict(default=default, min=0, max=0xffffffffffffffff))
    def RESOLUTION(default=512, min=64, max=MAX_RESOLUTION, step=64): 
        return ("INT", dict(default=default, min=min, max=max, step=step))
    def INT(default=0, min=0, max=MAX_RESOLUTION, step=1): 
        return ("INT", dict(default=default, min=min, max=max, step=step))
    def FLOAT(default=0, min=0, max=1, step=0.01):
        return ("FLOAT", dict(default=default, min=min, max=max, step=step))
    def STRING(default='', multiline=False): 
        return ("STRING", dict(default=default, multiline=multiline))
    def COMBO(values, default=None):
        return (values, dict(default=values[0] if default is None else default))
    def BOOLEAN(default=True):
        return ("BOOLEAN", dict(default=default))


# Ref: https://github.com/ltdrdata/ComfyUI-Manager/blob/284e90dc8296a2e1e4f14b4b2d10fba2f52f0e53/__init__.py#L14
def handle_stream(stream, prefix):
    for line in stream:
        print(prefix, line, end="")


def run_script(cmd, cwd='.'):
    process = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    stdout_thread = threading.Thread(target=handle_stream, args=(process.stdout, ""))
    stderr_thread = threading.Thread(target=handle_stream, args=(process.stderr, "[!]"))

    stdout_thread.start()
    stderr_thread.start()

    stdout_thread.join()
    stderr_thread.join()

    return process.wait()
