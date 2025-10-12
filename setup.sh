#!/bin/bash

# CADDreamer Setup Script
# Creates conda environment and installs all required dependencies

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting CADDreamer environment setup...${NC}"

# Check for reset flag
if [[ "$1" == "--reset" ]]; then
    echo -e "${YELLOW}Resetting progress tracking...${NC}"
    rm -f "$PROGRESS_FILE"
    echo -e "${GREEN}Progress reset. Starting fresh installation.${NC}"
fi

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed or not in PATH${NC}"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

# Environment name
ENV_NAME="caddreamer"

# Progress tracking file
PROGRESS_FILE=".setup_progress"

# Function to check if a step is completed
step_completed() {
    grep -q "^$1$" "$PROGRESS_FILE" 2>/dev/null
}

# Function to mark a step as completed
mark_step_completed() {
    echo "$1" >> "$PROGRESS_FILE"
    echo -e "${GREEN}✓ Completed: $1${NC}"
}

# Function to install package with error handling
install_package() {
    local package=$1
    local step_name="$package"
    
    if step_completed "$step_name"; then
        echo -e "${YELLOW}Skipping $package (already installed)${NC}"
        return 0
    fi
    
    echo -e "${GREEN}Installing $package...${NC}"
    if pip install "$package"; then
        mark_step_completed "$step_name"
        return 0
    else
        echo -e "${RED}Failed to install $package${NC}"
        echo -e "${YELLOW}You can resume from this point by running the script again${NC}"
        exit 1
    fi
}


# Remove existing environment if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}Removing existing ${ENV_NAME} environment...${NC}"
    conda env remove -n ${ENV_NAME} -y
fi

# Create new conda environment with Python 3.10
echo -e "${GREEN}Creating conda environment: ${ENV_NAME} with Python 3.10${NC}"
conda create -n ${ENV_NAME} python=3.10 -y

# Activate environment
echo -e "${GREEN}Activating environment: ${ENV_NAME}${NC}"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# Verify Python version
echo -e "${GREEN}Python version:${NC}"
python --version

# Install PyTorch with CUDA support first (as it's a critical dependency)
echo -e "${GREEN}Installing PyTorch with CUDA 11.8 support...${NC}"
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install essential system dependencies via conda
echo -e "${GREEN}Installing essential system dependencies...${NC}"
conda install -c conda-forge ninja cmake -y

# Install core scientific computing packages
if ! step_completed "core_packages"; then
    echo -e "${GREEN}Installing core scientific packages...${NC}"
    install_package "numpy==1.24.0"
    install_package "scipy==1.15.1"
    install_package "matplotlib==3.9.1.post1"
    install_package "opencv-python-headless==4.11.0.86"
    install_package "pillow==10.4.0"
    install_package "scikit-image==0.25.0"
    install_package "scikit-learn==1.5.1"
    install_package "imageio==2.34.2"
    install_package "psutil==6.0.0"
    mark_step_completed "core_packages"
fi

# Install ML/AI libraries
echo -e "${GREEN}Installing ML/AI libraries...${NC}"
pip install transformers==4.44.0
pip install diffusers==0.19.3
pip install accelerate==0.33.0
pip install huggingface-hub==0.24.5
pip install tokenizers==0.19.1
pip install safetensors==0.4.4


pip install trimesh==3.18.1
pip install open3d==0.18.0
pip install pymeshlab==2023.12.post1
pip install PyMCubes==0.1.2
pip install meshio==5.3.5
pip install potpourri3d==1.1.0
pip install polyscope==2.3.0
pip install pyvista==0.44.1

# Install PyTorch3D (requires special handling)
echo -e "${GREEN}Installing PyTorch3D...${NC}"
# Downgrade setuptools to fix compatibility issue
pip install setuptools==59.5.0
# Install PyTorch3D from git
pip install "git+https://github.com/facebookresearch/pytorch3d.git"


# Install torch geometric libraries
echo -e "${GREEN}Installing PyTorch Geometric libraries...${NC}"
pip install torch-scatter==2.1.1+pt20cu118 -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
pip install torch-sparse==0.6.17+pt20cu118 -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
pip install torch-spline-conv==1.2.2+pt20cu118 -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

# Install Blender and BlenderProc
echo -e "${GREEN}Installing Blender and BlenderProc...${NC}"
pip install bpy==3.6.0 --extra-index-url https://download.blender.org/pypi/
pip install blenderproc==2.7.1

# Modify BlenderProc __init__.py file
echo -e "${GREEN}Modifying BlenderProc __init__.py...${NC}"
# Find BlenderProc installation path without importing it
BLENDERPROC_PATH=$(python -c "import numpy; print(str(numpy.__file__).split('numpy')[0])")/blenderproc/__init__.py
if [ -f "$BLENDERPROC_PATH" ]; then
    cat > "$BLENDERPROC_PATH" << 'EOF'
"""A procedural Blender pipeline for photorealistic rendering."""

import os
import sys
from .version import __version__

# check the python version, only python 3.X is allowed:
if sys.version_info.major < 3:
    raise Exception("BlenderProc requires at least python 3.X to run.")


# from .python.utility.SetupUtility import SetupUtility
# SetupUtility.setup([])
from .api import loader
from .api import utility
from .api import sampler
from .api import math
from .python.utility.Initializer import init, clean_up
from .api import postprocessing
from .api import writer
from .api import material
from .api import lighting
from .api import camera
from .api import renderer
from .api import world
from .api import constructor
from .api import types
# pylint: disable=redefined-builtin
from .api import object
from .api import filter


# Only import if we are in the blender environment, this environment variable is set by the cli.py script
if "INSIDE_OF_THE_INTERNAL_BLENDER_PYTHON_ENVIRONMENT" in os.environ:
    # Remove the parent of the blender proc folder, as it might contain other packages
    # that we do not want to import inside the blenderproc env
    sys.path.remove(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    # Also clean the python path as this might disturb the pip installs
    if "PYTHONPATH" in os.environ:
        del os.environ["PYTHONPATH"]
    from .python.utility.SetupUtility import SetupUtility
    SetupUtility.setup([])
    from .api import loader
    from .api import utility
    from .api import sampler
    from .api import math
    from .python.utility.Initializer import init, clean_up
    from .api import postprocessing
    from .api import writer
    from .api import material
    from .api import lighting
    from .api import camera
    from .api import renderer
    from .api import world
    from .api import constructor
    from .api import types
    # pylint: disable=redefined-builtin
    from .api import object
    from .api import filter
    # pylint: enable=redefined-builtin
else:
    # this checks if blenderproc the command line tool or the cli.py script are used. If not an exception is thrown
    import traceback
    # extract the basename of the file, which is the first in the traceback
    stack_summary = traceback.extract_stack()
    file_names_of_stack = [os.path.basename(file_summary.filename) for file_summary in stack_summary]
    # check if blenderproc is called via python3 -m blenderproc ...
    is_module_call = file_names_of_stack[0] == "runpy.py"
    if sys.platform == "win32":
        is_bproc_shell_called = file_names_of_stack[2] in ["metadata.py", "__main__.py"]
        is_command_line_script_called = file_names_of_stack[0] == "command_line.py"

        is_correct_startup_command = is_bproc_shell_called or is_command_line_script_called or is_module_call
    else:
        is_bproc_shell_called = file_names_of_stack[0] in ["blenderproc", "command_line.py"]
        # check if the name of this file is either blenderproc or if the
        # "OUTSIDE_OF_THE_INTERNAL_BLENDER_PYTHON_ENVIRONMENT_BUT_IN_RUN_SCRIPT" is set, which is set in the cli.py
        is_correct_startup_command = is_bproc_shell_called or is_module_call
    # if "OUTSIDE_OF_THE_INTERNAL_BLENDER_PYTHON_ENVIRONMENT_BUT_IN_RUN_SCRIPT" not in os.environ \
    #         and not is_correct_startup_command:
    #     # pylint: disable=consider-using-f-string
    #     raise RuntimeError("\n###############\nThis script can only be run by \"blenderproc run\", instead of calling:"
    #                        "\n\tpython {}\ncall:\n\tblenderproc run {}\n###############".format(sys.argv[0],
    #                                                                                             sys.argv[0]))
        # pylint: enable=consider-using-f-string
EOF
    echo -e "${GREEN}✓ BlenderProc __init__.py modified successfully${NC}"
else
    echo -e "${RED}Warning: Could not find BlenderProc __init__.py file${NC}"
fi

# Install specialized libraries
echo -e "${GREEN}Installing specialized libraries...${NC}"
pip install einops==0.8.0
pip install rembg==2.0.58
pip install xformers==0.0.22
pip install omegaconf==2.2.3
pip install tqdm==4.66.5
pip install tensorboard==2.17.0
pip install torchmetrics==1.4.1
pip install pytorch-lightning==1.9.5

# Install utility libraries
echo -e "${GREEN}Installing utility libraries...${NC}"
pip install networkx==3.2.1
pip install transforms3d==0.4.2
pip install pyquaternion==0.9.9
pip install coloredlogs==15.0.1
pip install rich==13.7.1
pip install loguru==0.7.2
pip install ConfigArgParse==1.7

# Install additional dependencies from requirements.txt
echo -e "${GREEN}Installing additional dependencies...${NC}"
pip install absl-py==2.1.0
pip install addict==2.4.0
pip install altair==5.3.0
pip install annotated-types==0.7.0
pip install bitsandbytes==0.35.4
pip install carvekit_colab==4.1.2
pip install contourpy==1.2.1
pip install cycler==0.12.1
pip install Cython==0.29.37
pip install decord==0.6.0
pip install docker==6.0.1
pip install fonttools==4.53.1
pip install ftfy==6.2.3
pip install geomdl==5.3.1
pip install gitdb==4.0.11
pip install GitPython==3.1.43
pip install gpustat==0.6.0
pip install h5py==3.11.0
pip install icecream==2.1.0
pip install iopath==0.1.10
pip install joblib==1.4.2
pip install kiwisolver==1.4.5
pip install lightning-utilities==0.11.6
pip install llvmlite==0.43.0
pip install numba==0.60.0
pip install numpy-stl==3.1.2
pip install nvitop==0.5.5
pip install onnxruntime==1.18.1
pip install openmesh==1.2.1
pip install pandas==2.2.2
pip install piq==0.8.0
pip install plyfile==1.1
pip install plotly==5.23.0
pip install pooch==1.8.2
pip install pyarrow==17.0.0
pip install pydantic==2.8.2
pip install pydub==0.25.1
pip install pyembree==0.1.12
pip install pyglet==2.0.17
pip install pyhocon==0.3.57
pip install PyMatting==1.1.12
pip install pyparsing==3.1.2
pip install pypng==0.20220715.0
pip install pyransac3d==0.6.0
pip install python-louvain==0.16
pip install retrying==1.3.4
pip install Rtree==1.3.0
pip install segment-anything==1.0
pip install shapely==2.0.6
pip install svgwrite==1.4.3
pip install sympy==1.12
pip install tenacity==9.0.0
pip install tensorboardX==2.6.2.2
pip install threadpoolctl==3.5.0
pip install tifffile==2024.7.24
pip install torch_efficient_distloss==0.1.3
pip install cmake==3.30.2
pip install lit==18.1.8
pip install triton==2.0.0
pip install vedo==2024.5.2
pip install vtk==9.3.1
pip install webdataset==0.2.100
pip install xxhash==3.4.1
pip install zstandard==0.23.0
pip install dill 
pip install optimparallel
# Install missing packages from version.txt
echo -e "${GREEN}Installing additional missing packages...${NC}"
pip install lapsolver==1.1.0
pip install bleach==5.0.1

# Install nerfacc (special CUDA library)
echo -e "${GREEN}Installing nerfacc...${NC}"
pip install nerfacc==0.3.3


# install tinycudann 
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Install OpenCascade (OCC) and pythonOCC, refer to https://github.com/tpaviot/pythonocc-core/blob/master/INSTALL.md#prerequisites-linux
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Installing OpenCascade (OCC) and pythonOCC${NC}"
echo -e "${GREEN}========================================${NC}"
# Install system dependencies for OCC
if ! step_completed "occ_system_deps"; then
    echo -e "${GREEN}Installing system dependencies for OpenCascade...${NC}"
    sudo apt-get update
    sudo apt-get install -y \
        wget \
        libglu1-mesa-dev \
        libgl1-mesa-dev \
        libxmu-dev \
        libxi-dev \
        build-essential \
        cmake \
        libfreetype6-dev \
        tk-dev \
        python3-dev \
        rapidjson-dev \
        libpcre2-dev
    mark_step_completed "occ_system_deps"
fi

# Build SWIG 4.2.1 (required for pythonOCC)
if ! step_completed "swig_build"; then
    echo -e "${GREEN}Building SWIG 4.2.0...${NC}"
    cd /tmp
    wget http://prdownloads.sourceforge.net/swig/swig-4.2.1.tar.gz
    tar -zxvf swig-4.2.1.tar.gz
    cd swig-4.2.1
    ./configure
    make -j$(nproc)
    sudo make install
    mark_step_completed "swig_build"
fi

# Build OpenCascade 7.9.0
if ! step_completed "opencascade_build"; then
    echo -e "${GREEN}Building OpenCascade 7.8.1...${NC}"
    cd /tmp
    wget https://github.com/Open-Cascade-SAS/OCCT/archive/refs/tags/V7_8_1.tar.gz
    tar -xvzf V7_8_1.tar.gz
    cd OCCT-7_8_1
    mkdir cmake-build
    cd cmake-build
    
    cmake -DINSTALL_DIR=/opt/occt781 \
          -DBUILD_RELEASE_DISABLE_EXCEPTIONS=OFF \
          ..
    
    make -j$(nproc)
    sudo make install
    
    # Add OpenCascade libraries to the system
    sudo bash -c 'echo "/opt/occt781/lib" >> /etc/ld.so.conf.d/occt.conf'
    sudo ldconfig
    mark_step_completed "opencascade_build"
fi

# Build pythonOCC
if ! step_completed "pythonocc_build"; then
    echo -e "${GREEN}Building pythonOCC...${NC}"
    cd /tmp
    git clone https://github.com/tpaviot/pythonocc-core.git
    cd pythonocc-core
    mkdir cmake-build && cd cmake-build
    
    # Set installation directory
    PYTHONOCC_INSTALL_DIRECTORY=$(python -c "import numpy; print(str(numpy.__file__).split('numpy')[0])")
    
    cmake \
        -DOCCT_INCLUDE_DIR=/opt/occt781/include/opencascade \
        -DOCCT_LIBRARY_DIR=/opt/occt781/lib \
        -DCMAKE_BUILD_TYPE=Release \
        -DPYTHONOCC_INSTALL_DIRECTORY=$PYTHONOCC_INSTALL_DIRECTORY \
        ..
    
    make -j$(nproc) && sudo make install
    
    mark_step_completed "pythonocc_build"
fi


# Add OpenCascade libraries to environment
echo -e "${GREEN}Setting up OpenCascade environment variables...${NC}"
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/occt781/lib' >> ~/.bashrc

echo -e "${GREEN}OpenCascade installation completed!${NC}"


# Verify critical imports
echo -e "${GREEN}Verifying installation...${NC}"
python -c "
import torch
import torchvision
import numpy as np
import cv2
import trimesh
import pytorch3d
import bpy
import blenderproc
import rembg
import diffusers
import transformers
import einops
import omegaconf
import accelerate
try:
    from OCC.Core import TopoDS_Shape
    print('✓ pythonOCC imported successfully!')
except ImportError as e:
    print(f'⚠ pythonOCC import failed: {e}')
    print('Note: pythonOCC may require environment setup. Run: source ~/.bashrc')
print(' All critical packages imported successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device count: {torch.cuda.device_count()}')
    print(f'Current CUDA device: {torch.cuda.current_device()}')
"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}CADDreamer environment setup completed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}To activate the environment, run:${NC}"
echo -e "${GREEN}conda activate ${ENV_NAME}${NC}"
echo ""
echo -e "${YELLOW}To deactivate the environment, run:${NC}"
echo -e "${GREEN}conda deactivate${NC}"
echo ""
echo -e "${GREEN}Environment is ready for CADDreamer development!${NC}"