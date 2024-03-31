# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import argparse
import time
import torch
import pandas as pd
from DeepRelax import DeepRelax
from lmdb_dataset import TrajectoryLmdbDataset, collate_fn
import numpy as np
from torch.utils.data import DataLoader
from ema import EMAHelper
from collections import defaultdict
from graph_utils import vector_norm
from graph_utils import get_edge_dist_relaxed, get_edge_dist_unrelaxed
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

    dataset = data_root.split('/')[-1]

    test_set = TrajectoryLmdbDataset({"src": os.path.join(data_root, 'test_DeepRelax')})
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=4)

    device = torch.device('cuda:0')
    model = DeepRelax(hidden_channels=512, num_layers=4, num_rbf=128, cutoff=30.0, num_elements=118).to(device)

    ema_helper = EMAHelper(mu=0.999)
    ema_helper.register(model)
    ema_helper.load_state_dict(torch.load(model_path))
    ema_helper.ema(model)

    model = model.to(device)
    model.eval()

    performance_dict = defaultdict(list)
    for i, data in enumerate(test_loader):
        data = data.to(device)
        cif_id = data.cif_id[0]

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

        pred_cart_coords = x.detach()
        cart_coords_r = data.pos_r
        cart_coords_u = data.pos_u
        cell_u = data.cell_u
        cell_r = data.cell_r

        mae_cart_pos_Dummy = (cart_coords_u - cart_coords_r).abs().mean().item()
        mae_cart_pos_DeepRelax = (pred_cart_coords - cart_coords_r).abs().mean().item()

        mae_cell_Dummy = (cell_r - cell_u).abs().mean().item()
        mae_cell_DeepRelax = (cell_r - pred_cell).abs().mean().item()

        mae_volume_Dummy = (torch.linalg.det(cell_u) - torch.linalg.det(cell_r)).abs().item()
        mae_volume_DeepRelax = (torch.linalg.det(pred_cell) - torch.linalg.det(cell_r)).abs().item()

        dist_unrelaxed = get_edge_dist_unrelaxed(data)
        gt_dist_relaxed = get_edge_dist_relaxed(data)
        pred_dist_relaxed = target_dist_relaxed
        mae_dist_Dummy = (dist_unrelaxed - gt_dist_relaxed).abs().mean().item()
        mae_dist_DeepRelax = (pred_dist_relaxed - gt_dist_relaxed).abs().mean().item()

        performance_dict['cif_id'].append(data.cif_id[0])
        performance_dict['mae_cart_pos_Dummy'].append(mae_cart_pos_Dummy)
        performance_dict['mae_cart_pos_DeepRelax'].append(mae_cart_pos_DeepRelax)
        performance_dict['mae_cell_Dummy'].append(mae_cell_Dummy)
        performance_dict['mae_cell_DeepRelax'].append(mae_cell_DeepRelax)
        performance_dict['mae_volume_Dummy'].append(mae_volume_Dummy)
        performance_dict['mae_volume_DeepRelax'].append(mae_volume_DeepRelax)
        performance_dict['mae_dist_Dummy'].append(mae_dist_Dummy)
        performance_dict['mae_dist_DeepRelax'].append(mae_dist_DeepRelax)
        performance_dict['elapsed_time'].append(elapsed_time)

    create_dir(['./results'])
    performance_df = pd.DataFrame(performance_dict)
    performance_df.to_csv(f"./results/{dataset}.csv", index=False)

# %%
