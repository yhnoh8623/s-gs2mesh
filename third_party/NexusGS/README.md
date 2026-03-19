<div align="center">

<h1>NexusGS: Sparse View Synthesis with Epipolar Depth Priors in 3D Gaussian Splatting</h1>

<div>
    Yulong Zheng<sup>1</sup>&emsp;
    Zicheng Jiang<sup>1</sup>&emsp;
    Shengfeng He<sup>2</sup>&emsp;
    Yandu Sun<sup>1</sup>&emsp;
    Junyu Dong<sup>1</sup>&emsp;
    Huaidong Zhang<sup>3</sup>&emsp;
    Yong Du<sup>1, *</sup>
</div>

<div>
    <sup>1</sup>Ocean University of Chinay&emsp;
    <sup>2</sup>Singapore Management University&emsp;
    <sup>3</sup>South China University of Technology
</div>

<div>
    <sup>*</sup>corresponding author
</div>

[Paper](https://arxiv.org/abs/2503.18794) | 
[Project Page](https://usmizuki.github.io/NexusGS/) |
[Video](https://www.youtube.com/watch?v=K2foTIXzpMQ)

</div>


---------------------------------------------------
<p align="center" >
  <a href="">
    <img src="https://usmizuki.github.io/NexusGS/static/images/pipeline.png" alt="pipeline" width="100%">
  </a>
</p>


## Environmental Setups
Tested on Ubuntu 18.04, CUDA 11.8, PyTorch 2.0.0

```bash
conda env create -n nexus python=3.10
conda activate nexus
```

Install Pytorch

```bash
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

Install submodules

```bash
pip install submodules/diff-gaussian-rasterization-confidence
pip install submodules/simple-knn
```

## Running

Taking LLFF as an example, the dataset folder structure is as follows:
```Shell
├── datasets
    ├── LLFF
        ├── scene
            ├── sparse
            ├── images 
            ├── images_8
            ├── 3_views
                ├── flow  
```

### LLFF

Download the model from Hugging Face ([link](https://huggingface.co/Yukinoo/NexusGS-llff)) and start training.

```sh
bash scripts/run_llff_hf.sh 0 
```

or

Download LLFF dataset: [Link](https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4).

Download the LLFF optical flow processed by [FlowFormer++](https://github.com/XiaoyuShi97/FlowFormerPlusPlus) from the [Link](https://drive.google.com/file/d/1DPr02s2wMznlNN0-qDOZRZohzmnnuanX/view?usp=drive_link).

Run using the following script:

```bash
sh scripts/run_llff.sh 0
```

### DTU & MipNeRF-360

We provide the results on the DTU and MipNeRF-360 datasets in the [link](https://drive.google.com/file/d/1dUNtsBPTBE2-W0jg4LnPkNTEOiCuwtDK/view?usp=drive_link).

## Hugging Face

The following code can be used to save the initial point cloud and camera parameters of the model.

``` python
# save pretrained
from scene.hf_gaussian_model import HFGaussianModel
model = HFGaussianModel(gaussians.get_xyz.shape[0], scene.cameras_extent)
model.load_from_gaussian_model(gaussians, scene)
for name, param in model.named_parameters():
    print(name, param.shape)
model.save_pretrained("hf_models/NexusGS-llff/fern")
```

We provide a script to load the model from Hugging Face and directly start training.

```sh
bash scripts/run_llff_hf.sh 0 
```

If you choose to download the model automatically, set `source_path` to the model repository name, e.g., `Yukinoo/NexusGS-llff`, and optionally specify the branch using `revision`, e.g., `fern`. If you're using a local model, simply set `source_path` to the local path of the model, e.g., `./hf_models/NexusGS-llff/fern`.

Alternatively, the following code can be used to manually load and utilize the model.

```python
from scene.hf_gaussian_model import HFGaussianModel
model = HFGaussianModel.from_pretrained("Yukinoo/NexusGS-llff", revision="fern")
```

## Citation
If you find our work useful for your project, please consider citing the following paper.


``` bibtex
@article{zheng2025nexusgs,
  title={NexusGS: Sparse View Synthesis with Epipolar Depth Priors in 3D Gaussian Splatting},
  author={Zheng, Yulong and Jiang, Zicheng and He, Shengfeng and Sun, Yandu and Dong, Junyu and Zhang, Huaidong and Du, Yong},
  journal={arXiv preprint arXiv:2503.18794},
  year={2025}
}
```

## Acknowledgement

Special thanks to the following awesome projects!

- [Gaussian-Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [FSGS](https://zehaozhu.github.io/FSGS/)
- [DNGaussian](https://fictionarry.github.io/DNGaussian/)
- [FlowFormer++](https://github.com/XiaoyuShi97/FlowFormerPlusPlus)