# Scalable Crystal Structure Relaxation Using an Iteration-Free Deep Generative Model with Uncertainty Quantification (DeepRelax)

## Dataset
Our research utilizes datasets that are publicly accessible. Details and access links for each dataset are provided below:
- **XMnO Dataset** [1]: Available at [XMnO](https://zenodo.org/records/8081655).
- **MP Dataset** [2]: Available at [MPF.2021.2.8](https://figshare.com/articles/dataset/MPF_2021_2_8/19470599).
- **C2DB Dataset** [3, 4, 5]: Available at [C2DB](https://cmr.fysik.dtu.dk/c2db/c2db.html).

For convenience, both raw and processed data from these datasets can also be downloaded from [Zenodo](https://zenodo.org/records/10899768). 

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
For logging, we recommend using Wandb. More details are available at https://wandb.ai/. Training logs and trained models are stored in the `./wandb` directory. The saved model can typically be found at ./wandb/run-xxx/files/model.pt, where xxx represents specific run information.

## Step-by-Step Guide

### Data Preprocessing
To begin working with the datasets, first download the necessary files from [Zenodo](https://zenodo.org/records/10899768) and unzip them. You will find the preprocessed data in the following directories for each dataset:

- For the XMnO dataset: cifs_xmno/train_DeepRelax, cifs_xmno/val_DeepRelax, cifs_xmno/test_DeepRelax
- For the MP dataset: Similar directory structure as XMnO
- For the C2DB dataset: Please note that the C2DB dataset is available upon request. Contact the corresponding author of the C2DB dataset to obtain the files, including relaxed.db and unrelaxed.db. After successfully requesting the C2DB dataset, process it using `convert_c2db.py` available in this repository.

### Preprocessing Data from Scratch
If you prefer to preprocess the data from scratch, use the following commands, ensuring you replace your_data_path with the appropriate path to your data:

For the XMnO dataset:

- `python preprocess_xmno.py --data_root your_data_path/cifs_xmno --num_workers 1`

For the MP dataset:

- `python preprocess_mp.py --data_root your_data_path/MPF.2021.2.8 --num_workers 1`

For the C2DB dataset:

- `python preprocess_c2db.py --data_root your_data_path/c2db --num_workers 1`

To increase the processing speed, you can adjust the --num_workers parameter to a higher value, depending on your system's capabilities.

### Train the Model
To initiate training of the DeepRelax model, execute the following commands. Make sure to substitute your_data_path with the actual path to your dataset:

For the XMnO dataset:
- `python train.py --data_root your_data_path/cifs_xmno --num_workers 4 --batch_size 32 --steps_per_epoch 800`

For the MP dataset:
- `python train.py --data_root your_data_path/MPF.2021.2.8 --num_workers 4  --batch_size 32 --steps_per_epoch 800`

For the C2DB dataset:
- `python train.py --data_root your_data_path/c2db --num_workers 4  --batch_size 32 --steps_per_epoch 100`

### Test the Model
To evaluate the DeepRelax model, specifically on the XMnO dataset, run the following command, replacing your_data_path and your_model_path with the appropriate paths:
- `python edg_solver.py --data_root your_data_path/cifs_xmno --model_path your_model_path/model.pt`

This process can be similarly applied to the other datasets. If you are using WandB for tracking experiments, the saved model can typically be found at ./wandb/run-xxx/files/model.pt, where xxx represents specific run information.

### Practical Application of DeepRelax through Transfer Learning
DeepRelax is optimally utilized via transfer learning. This approach allows you to leverage a pre-trained model and adapt it to your specific use case. Below, we outline a demonstration to guide you in transferring the trained model to your application.<br>
#### Organizing Your Data
First, ensure your data is structured as follows to facilitate processing:
- `custom/`
  - `train.csv`
  - `val.csv`
  - `test.csv`
  - `CIF/`
    - `data_1_unrelaxed.cif`
    - `data_1_relaxed.cif`
    - `data_2_unrelaxed.cif`
    - `data_2_relaxed.cif`
    - `...`

**Note**: The test set does not require relaxed structures. However, for training and validation sets, pairs of unrelaxed and relaxed structures are necessary.

Each .csv file should contain a column named atoms_id, with each row corresponding to the ID of a .cif file in your dataset. For example:<br>
| atoms_id    |
|-------------|
| data_1  |
| data_2  |
| data_3 |
| ...  |

When defining atoms_id, ensure it is consistent with the names of your .cif files to maintain integrity and facilitate seamless processing.

#### Preprocessing Your Data
To convert your .cif files into a format suitable for DeepRelax, use the following command, replacing your_data_path with the path to your custom directory:
- `python preprocess_custom.py --data_root your_data_path/custom --num_workers 1`

This command will process your .cif files and organize the output into two subdirectories (train_DeepRelax and val_DeepRelax) within the custom directory.

#### Applying Transfer Learning
After preprocessing your data, apply transfer learning to your custom dataset with the following command:
- `python train.py --data_root your_data_path/custom --num_workers 4 --batch_size 32 --steps_per_epoch 100 --transfer True`

Ensure to replace your_data_path with the appropriate path to where your custom directory is located.

#### Testing the Model
Refer to the Test the Model section previously discussed to evaluate the performance of your model trained with transfer learning on your custom dataset. Remember, you need pairs of unrelaxed and relaxed structures for evaluation. If your test set lacks relaxed structures, you can't directly evaluate performance but can predict relaxed structures as follows.

#### Predicting the Relaxed Structures
To predict relaxed structures and save them as .cif files:
- `python predict_relaxed_structure.py --data_root your_data_path/custom --model_path your_model_path/model.pt`

This script predicts the relaxed structure using record atoms_id in test.csv. Note that it is not need to provide relaxed structure for the test data. After running the prediction script, the predicted relaxed structures will be located in the ./predicted_structures directory within your project's root directory. This makes it easy to access and review the results of your model's predictions.

## Citation
If you find the DeepRelax model beneficial for your research, please include a citation to our paper. You can reference it as follows:<br>
@article{yang2024scaling,<br>
      &emsp; title={Scaling Crystal Structure Relaxation with a Universal Trustworthy Deep Generative Model},<br> 
      &emsp; author={Ziduo Yang and Yiming Zhao and Xiaoqing Liu and Xiuying Zhang and Yifan Li and Qiujie Lyu and Calvin Yu-Chian Chen and Lei Shen},<br>
      &emsp; journal={arXiv preprint arXiv:2404.00865},<br>
      &emsp; year={2024},<br>
}<br>

## Acknowledgements
Some part of code in this project were adapted from [OCP](https://github.com/Open-Catalyst-Project/ocp). We gratefully acknowledge the contributions from this source. We also acknowledge Prof. Kristian Sommer Thygesen and Peder Lyngby for their generous provision of the C2DB database, complete with both initial and final structures. 

## Reference
[1] Kim S, Noh J, Jin T, et al. A structure translation model for crystal compounds[J]. npj Computational Materials, 2023, 9(1): 142.<br>
[2] Chen C, Ong S P. A universal graph deep learning interatomic potential for the periodic table[J]. Nature Computational Science, 2022, 2(11): 718-728.<br>
[3] Haastrup S, Strange M, Pandey M, et al. The Computational 2D Materials Database: high-throughput modeling and discovery of atomically thin crystals[J]. 2D Materials, 2018, 5(4): 042002.<br>
[4] Gjerding M N, Taghizadeh A, Rasmussen A, et al. Recent progress of the computational 2D materials database (C2DB)[J]. 2D Materials, 2021, 8(4): 044002.<br>
[5] Lyngby P, Thygesen K S. Data-driven discovery of 2D materials by deep generative models[J]. npj Computational Materials, 2022, 8(1): 232.
