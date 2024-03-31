# DeepRelax

## Dataset
The datasets used in our work are publicly available and can be accessed from
- **XMnO Dataset** [1]: Available at [XMnO](https://zenodo.org/records/8081655).
- **MP Dataset** [2]: Available at [MPF.2021.2.8](https://figshare.com/articles/dataset/MPF_2021_2_8/19470599).
- **C2DB Dataset** [3, 4, 5]: Available at [C2DB](https://cmr.fysik.dtu.dk/c2db/c2db.html).

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
To begin working with the datasets, first download the necessary files from the provided [link] and unzip them. You will find the preprocessed data in the following directories for each dataset:

- For the XMnO dataset: cifs_xmno/train_DeepRelax, cifs_xmno/val_DeepRelax, cifs_xmno/test_DeepRelax
- For the MP dataset: Similar directory structure as XMnO
- For the C2DB dataset: Please note that the C2DB dataset is available upon request. Contact the corresponding author of the C2DB dataset to obtain the files, including relaxed.db and unrelaxed.db. After successfully requesting the C2DB dataset, process it using `convert_c2db.py` available in this repository.

### Preprocessing Data from Scratch
If you prefer to preprocess the data from scratch, use the following commands, ensuring you replace your_data_path with the appropriate path to your data:

For the XMnO dataset:

`python preprocess_xmno.py --data_root your_data_path/cifs_xmno --num_workers 1`

For the MP dataset:

`python preprocess_mp.py --data_root your_data_path/MPF.2021.2.8 --num_workers 1`

For the C2DB dataset:

`python preprocess_c2db.py --data_root your_data_path/c2db --num_workers 1`

To increase the processing speed, you can adjust the --num_workers parameter to a higher value, depending on your system's capabilities

## Acknowledgements
Some part of code in this project were adapted from [OCP](https://github.com/Open-Catalyst-Project/ocp). We gratefully acknowledge the contributions from this source. We also acknowledge Prof. Kristian Sommer Thygesen and Peder Lyngby for their generous provision of the C2DB database, complete with both initial and final structures. 
