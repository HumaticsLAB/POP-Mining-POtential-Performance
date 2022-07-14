
# POP: Mining POtential Performance of new fashion products via webly cross-modal query expansion

The official pytorch implementation of [GTM](https://github.com/HumaticsLAB/GTM-Transformer) additionals discussed in [POP: Mining POtential Performance of new fashion products via webly cross-modal query expansion](#)
paper.

Accepted as poster at the European Conference on Computer Vision @ ECCV2022 @ Tel-Aviv

## Installation

We suggest the use of VirtualEnv.

```bash

python3 -m venv pop_venv
source pop_venv/bin/activate
# pop_venv\Scripts\activate.bat # If you're running on Windows

pip install numpy pandas matplotlib opencv-python permetrics Pillow scikit-image scikit-learn scipy tqdm transformers fairseq wandb

pip install torch torchvision

# For CUDA11.1 (NVIDIA 3K Serie GPUs)
# Check official pytorch installation guidelines for your system
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

pip install pytorch-lightning

export INSTALL_DIR=$PWD

cd $INSTALL_DIR
git clone https://github.com/HumaticsLAB/GTM-Transformer.git
cd GTM-Transformer
mkdir ckpt
mkdir dataset
mkdir results

cd ..
git clone https://github.com/HumaticsLAB/POP-Mining-POtential-Performance.git
cd POP-Mining-POtential-Performance

cp -r utils ../GTM-Transformer/.
cp -r models ../GTM-Transformer/.
cp -r signals/* ../GTM-Transformer/dataset/.
cp train_POP.py ../GTM-Transformer/.
cp forecast_POP.py ../GTM-Transformer/.

unset INSTALL_DIR
```

## Dataset

**VISUELLE** dataset is publicly available to download [here](https://forms.gle/8Sk431AsEgCot9Kv5). Please download and extract it inside the root folder. A more accurate description of the dataset inside its [official page](https://humaticslab.github.io/forecasting/visuelle).  

**POP** signals are publicly available in signals folder. 

## Dataset

**VISUELLE** dataset is publicly available to download [here](https://forms.gle/cVGQAmxhHf7eRJ937). Please download and extract it inside the dataset folder.

## Training
To train the model of GTM-Transformer with POP signals please use the following scripts. Please check the arguments inside the script before launch.

```bash
python train_POP.py --data_folder dataset
```
## Inference
To evaluate the model of GTM-Transformer with POP signals please use the following script .Please check the arguments inside the script before launch.

```bash
python forecast_POP.py --data_folder dataset --ckpt_path ckpt/model.pth
```

## Citation

If you use **POP** signals or this paper implementation, please cite the following papers.

```
Coming Soon
```

If you use **VISUELLE** dataset or **GTM** implementation, please cite the following papers.


```
@misc{skenderi2021googled,
      title={Well Googled is Half Done: Multimodal Forecasting of New Fashion Product Sales with Image-based Google Trends}, 
      author={Geri Skenderi and Christian Joppi and Matteo Denitto and Marco Cristani},
      year={2021},
      eprint={2109.09824},
}
```
