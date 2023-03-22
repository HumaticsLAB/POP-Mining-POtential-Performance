
# POP: Mining POtential Performance of new fashion products via webly cross-modal query expansion

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pop-mining-potential-performance-of-new/new-product-sales-forecasting-on-visuelle)](https://paperswithcode.com/sota/new-product-sales-forecasting-on-visuelle?p=pop-mining-potential-performance-of-new)

The official pytorch implementation of [POP: Mining POtential Performance of new fashion products via webly cross-modal query expansion](https://arxiv.org/abs/2207.11001) paper. In this repository you find the **POP** signals and the [GTM](https://github.com/HumaticsLAB/GTM-Transformer) architecture used for forecasting with them.

Accepted as poster at the European Conference on Computer Vision (ECCV2022) in Tel-Aviv

## Installation

We suggest the use of VirtualEnv.

```bash

python3 -m venv pop_venv
source pop_venv/bin/activate
# pop_venv\Scripts\activate.bat # If you're running on Windows

pip install numpy pandas matplotlib opencv-python permetrics Pillow scikit-image scikit-learn scipy tqdm transformers==4.9.1 fairseq wandb

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

pip install pytorch-lightning

export INSTALL_DIR=$PWD

cd $INSTALL_DIR
git clone https://github.com/HumaticsLAB/POP-Mining-POtential-Performance.git
cd POP-Mining-POtential-Performance
mkdir ckpt
mkdir dataset
mkdir results

unset INSTALL_DIR
```

## Dataset

The **VISUELLE** dataset is publicly available to download [here](https://forms.gle/cVGQAmxhHf7eRJ937). Please download and extract it inside the root folder. A more accurate description of the dataset is available [in the official page](https://humaticslab.github.io/forecasting/visuelle).  

**POP** signals are publicly available in signals folder. 

## Training
To train the GTM-Transformer model with POP please use the following script. Please check the arguments inside the script before launch.

```bash
python train_POP.py --data_folder dataset
```
## Inference
To evaluate the GTM-Transformer model with POP use the following script. Please check the arguments inside the script before launch.

```bash
python forecast_POP.py --data_folder dataset --ckpt_path ckpt/model.pth
```

## Citation

If you use the **POP** pipeline, please cite the following paper.

```
@InProceedings{joppi2022,
      author="Joppi, Christian and Skenderi, Geri and Cristani, Marco",
      editor="Avidan, Shai and Brostow, Gabriel and Ciss{\'e}, Moustapha and Farinella, Giovanni Maria and Hassner, Tal",
      title="POP: Mining POtential Performance of New Fashion Products via Webly Cross-modal Query Expansion",
      booktitle="Computer Vision -- ECCV 2022",
      year="2022",
      publisher="Springer Nature Switzerland",
      pages="34--50",
      isbn="978-3-031-19839-7"
}
```

If you use the **VISUELLE** dataset or the **GTM** implementation, please cite the following paper.


```
@misc{skenderi2021googled,
      title={Well Googled is Half Done: Multimodal Forecasting of New Fashion Product Sales with Image-based Google Trends}, 
      author={Geri Skenderi and Christian Joppi and Matteo Denitto and Marco Cristani},
      year={2021},
      eprint={2109.09824},
}
```
