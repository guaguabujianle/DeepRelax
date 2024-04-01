# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import argparse
import time
import torch
import pandas as pd
from DeepRelax import DeepRelax
from torch_geometric.data import Batch
from ema import EMAHelper
from collections import defaultdict
from graph_utils import vector_norm
from graph_constructor import AtomsToGraphs
from ase.io import read, write
from utils import *
import warnings
warnings.filterwarnings("ignore")

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--data_root', type=str, default=None, help='data directory', required=True)
    parser.add_argument('--model_path', type=str, default=None, help='model path', required=True)

    args = parser.parse_args()
    data_root = args.data_root
    model_path = args.model_path 

    test_df = pd.read_csv(os.path.join(data_root, "test.csv"))

    device = torch.device('cuda:0')
    model = DeepRelax(hidden_channels=512, num_layers=4, num_rbf=128, cutoff=30.0, num_elements=118).to(device)

    ema_helper = EMAHelper(mu=0.999)
    ema_helper.register(model)
    ema_helper.load_state_dict(torch.load(model_path))
    ema_helper.ema(model)

    model = model.to(device)
    model.eval()
    
    a2g = AtomsToGraphs(
        radius=6,
        max_neigh=30,
        max_displace=20,
    )

    create_dir(['./predicted_structures'])
    
    performance_dict = defaultdict(list)
    for i, row in test_df.iterrows():
        atoms_id = row['atoms_id']
        cif_path = os.path.join( data_root, 'CIF', atoms_id + '_unrelaxed.cif')
        atoms_u = read(cif_path)
        data = a2g.convert_single(atoms_u)
        data = Batch.from_data_list([data])
        data = data.to(device)
        
        # Record the starting time
        start_time = time.time()

        with torch.no_grad():
            pred_dist_displace, pred_var_displace, pred_dist_relaxed, pred_var_relaxed, pred_cell = model(data)

        pos_u = data.pos_u.clone()

        target_dist_displace = pred_dist_displace 
        pred_var_displace = torch.clamp(pred_var_displace, -5, 2) 
        sigma_displace = pred_var_displace.exp()
        lower_bound_displace = torch.clamp(target_dist_displace - sigma_displace , min=0)
        upper_bound_displace = target_dist_displace + sigma_displace 

        target_dist_relaxed = pred_dist_relaxed
        pred_var_relaxed = torch.clamp(pred_var_relaxed, -5, 2) 
        sigma_relaxed = pred_var_relaxed.exp()
        lower_bound_relaxed = torch.clamp(target_dist_relaxed - sigma_relaxed , min=0)
        upper_bound_relaxed = target_dist_relaxed + sigma_relaxed

        neighbors = data.neighbors
        cell_u = data.cell_u
        cell_offsets = data.cell_offsets
        cell_offsets_unsqueeze = cell_offsets.unsqueeze(1).float()

        edge_index = data.edge_index
        j, i = edge_index

        edge_index_displace = edge_index[:, data.mask]
        j_d, i_d = edge_index_displace

        c = pred_cell.detach()
        c_r = c.repeat_interleave(neighbors, dim=0)
        offsets = (cell_offsets_unsqueeze @ c_r).squeeze(1)
        
        x = torch.tensor(pos_u.clone(), requires_grad=True)
        optimizer = torch.optim.Adam([x], lr=0.1)

        performance_list = []
        loss_list = []
        for _ in range(100):
            dist_displace = vector_norm(x[j_d] - pos_u[i_d])
            dist_relaxed = vector_norm(x[j] + offsets - x[i])

            dist_displace = torch.clamp(dist_displace, min=0, max=150)
            dist_relaxed = torch.clamp(dist_relaxed, min=0, max=150)

            loss_upper_displace = (dist_displace - upper_bound_displace).relu().mean()
            loss_lower_displace = (lower_bound_displace - dist_displace).relu().mean()

            loss_upper_relaxed = (dist_relaxed - upper_bound_relaxed).relu().mean()
            loss_lower_relaxed = (lower_bound_relaxed - dist_relaxed).relu().mean()

            loss = loss_upper_displace + loss_lower_displace + loss_upper_relaxed + loss_lower_relaxed
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        # Record the ending time
        end_time = time.time()

        # Calculate the time elapsed
        elapsed_time = end_time - start_time

        pred_cart_coords = x.detach().cpu()
        pred_cell = pred_cell.squeeze().detach().cpu()

        predicted_path = os.path.join('./predicted_structures', atoms_id + '_predicted.cif')

        atoms_u.set_positions(pred_cart_coords.numpy())
        atoms_u.set_cell(pred_cell.numpy())

        write(predicted_path, atoms_u)

# %%
