# DeepRelax

## Dataset
The datasets used in our work are publicly available and can be accessed from
- **XMnO Dataset** [1]: Available at [XMnO](https://zenodo.org/records/8081655).
- **MP Dataset** [1]: Available at [MPF.2021.2.8](https://figshare.com/articles/dataset/MPF_2021_2_8/19470599).
- **C2DB Dataset**: Available at [C2DB](https://cmr.fysik.dtu.dk/c2db/c2db.html).

Alternatively, you can download the raw data as well as the processed data from [link]. 

## Requirements
Required Python packages include:  
- `ase==3.22.1`
- `config==0.5.1`
- `lmdb==1.4.1`
- `matplotlib==3.7.2`
- `numpy==1.24.4`
- `pandas==2.1.3`
- `pymatgen==2023.5.10`
- `scikit_learn==1.3.0`
- `scipy==1.11.4`
- `torch==1.13.1`
- `torch_geometric==2.2.0`
- `torch_scatter==2.1.0`
- `tqdm==4.66.1`

Alternatively, install the environment using the provided YAML file at `./environment/environment.yaml`.

## Logger
For logging, we recommend using wandb. More details are available at https://wandb.ai/. Training logs and trained models are stored in the `./wandb` directory.

## Step-by-Step Guide

### Data Preprocessing
Download the data from [link] and unzip it. You can find the preprocessed data from cifs_xmno/train_DeepRelax, cifs_xmno/val_DeepRelax, and cifs_xmno/test_DeepRelax for the XMnO dataset.
  If you wish to preprocess them from scratch, run:
- `python preprocess_xmno.py --data_root your_data_path/cifs_xmno --num_workers 1` for the XMnO dataset.
- `python preprocess_mp.py --data_root /scratch/yangzd/materials/data/MPF.2021.2.8 --num_workers 1` for the MP dataset.
- `python preprocess_c2db.py --data_root /scratch/yangzd/materials/data/c2db --num_workers 1` for the C2DB dataset.

You can increase num_workers to increase processing speed.

## Acknowledgements
Some part of code in this project were adapted from [OCP](https://github.com/Open-Catalyst-Project/ocp). We gratefully acknowledge the contributions from this source. We also acknowledge Prof. Kristian Sommer Thygesen and Peder Lyngby for their generous provision of the C2DB database, complete with both initial and final structures. 
