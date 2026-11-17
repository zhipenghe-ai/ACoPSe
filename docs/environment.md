# Environment

## Tested environment

The release was checked in a Conda environment on 64-bit Windows. GPU validation included model and loss parity, CUDA K-means, a one-epoch MIRFlickr run with the supplied features, and a small NUS-WIDE-format fixture. Full benchmark reproduction remains unverified.

| Component | Validated stack |
| --- | --- |
| Interpreter and pip | Python `3.7.16`, pip `22.3.1` |
| PyTorch and torchvision | `1.13.1+cu117`, `0.14.1+cu117` |
| CUDA and cuDNN | PyTorch CUDA build `11.7`, cuDNN `8.5.0` |
| GPU | NVIDIA GeForce RTX 4090, compute capability `8.9`, 24 GiB VRAM |
| Numeric and data packages | NumPy `1.21.5`, SciPy `1.7.3`, h5py `3.8.0` |
| Other direct dependencies | Lightly `1.4.3`, tqdm `4.64.1` |
| Additional installed packages | scikit-learn `1.0.2`, matplotlib `3.5.3`, munkres `1.1.4` |
| Selected indirect dependencies | lightly-utils `0.0.2`, PyTorch Lightning `1.9.5`, TorchMetrics `0.11.4`, Pillow `9.4.0` |

The MIRFlickr and NUS-WIDE launchers use the vendored PyTorch K-means. Their training and retrieval paths require CUDA, and `--GPU_ID` selects the device. The loader uses `np.float`, which is available in the NumPy version above. [Trainer](../my_opt.py) · [Data loader](../load_data.py)

## Proposed public setup for Windows with an NVIDIA GPU

Use a 64-bit Windows machine with an NVIDIA driver compatible with the CUDA 11.7 build. The commands include direct dependencies, selected Lightly dependencies, and other package versions from the validated stack. A clean installation with this recipe remains untested. The PyTorch wheel indexes list both CUDA 11.7 packages for CPython 3.7 on Windows x86-64. [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v1131) · [Torch wheels](https://download.pytorch.org/whl/cu117/torch/) · [Torchvision wheels](https://download.pytorch.org/whl/cu117/torchvision/)

```powershell
conda create -n acopse-repro python=3.7.16 pip=22.3.1
conda activate acopse-repro
python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install numpy==1.21.5 scipy==1.7.3 scikit-learn==1.0.2 h5py==3.8.0 matplotlib==3.5.3 tqdm==4.64.1 munkres==1.1.4 lightly==1.4.3 lightly-utils==0.0.2 pytorch-lightning==1.9.5 torchmetrics==0.11.4 pillow==9.4.0
python -m pip check
```

### Verify the setup

From the repository root, check that both launchers print their options and that the unit tests pass. These checks require no datasets or checkpoints. The final command verifies that PyTorch can execute a CUDA operation and prints the active runtime versions and device name.

```powershell
python --version
python main_mir.py --help
python main_nus.py --help
python -m unittest discover -s tests -v
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda); assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0), torch.ones(1, device='cuda').item())"
```

An Intel OpenMP runtime conflict during validation was resolved by setting `$env:MKL_THREADING_LAYER = 'SEQUENTIAL'` in PowerShell. If you encounter the same conflict, set it in your current PowerShell session before retrying the command.

For dataset layout and checkpoint commands, see the [evaluation guide](evaluation.md).
