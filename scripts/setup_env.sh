#!/bin/bash
# Sparse-GS2Mesh environment (tested: Ubuntu 24.04, CUDA 12.8, gcc 13, A100 80GB)
set -e
cd "$(dirname "$0")/.."

conda create -y -n sgs2mesh --override-channels -c conda-forge python=3.10 pip
PIP=$(conda run -n sgs2mesh which pip)

$PIP install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
$PIP install xformers==0.0.31.post1 --index-url https://download.pytorch.org/whl/cu128
# numpy < 2: open3d 0.18 crashes in TSDF integration with numpy 2
$PIP install "numpy==1.26.4" "plyfile==1.0.3" "opencv-python==4.10.0.84" "open3d==0.18.0" \
             tqdm pillow scipy scikit-image scikit-learn trimesh tensorboard timm opt_einsum einops gdown ninja

# DEFOM-Stereo (stereo matching network) and the 2DGS surfel rasterizer
[ -d third_party/DEFOM-Stereo ] || git clone https://github.com/Insta360-Research-Team/DEFOM-Stereo.git third_party/DEFOM-Stereo
[ -d submodules/diff-surfel-rasterization ] || \
    git clone --recursive https://github.com/hbb1/diff-surfel-rasterization.git submodules/diff-surfel-rasterization
grep -q "<cstdint>" submodules/diff-surfel-rasterization/cuda_rasterizer/rasterizer_impl.h || \
    sed -i '0,/#include/s//#include <cstdint>\n#include/' submodules/diff-surfel-rasterization/cuda_rasterizer/rasterizer_impl.h

# CUDA extensions: 3DGS rasterizer with depth / alpha outputs, simple-knn, 2DGS surfel rasterizer
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}" CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH
for d in diff-gaussian-rasterization simple-knn diff-surfel-rasterization; do
    $PIP install --no-build-isolation submodules/$d
done

# DEFOM-Stereo weights (defomstereo_vitl_sceneflow.pth is used)
if [ ! -f third_party/DEFOM-Stereo/checkpoints/defomstereo_vitl_sceneflow.pth ]; then
    (cd third_party/DEFOM-Stereo && PATH=$(dirname $PIP):$PATH bash scripts/download_models.sh)
fi
echo "done"
