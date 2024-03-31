# %%
import itertools
import numpy as np
import torch
from torch_geometric.data import Data

try:
    from pymatgen.io.ase import AseAtomsAdaptor
except Exception:
    pass

class AtomsToGraphs:
    """A class to help convert periodic atomic structures to graphs.
    """

    def __init__(
        self,
        radius=6,
        max_neigh=30,
        max_displace=20,
    ):
        self.radius = radius
        self.max_neigh = max_neigh
        self.max_displace = max_displace
        
    def _get_neighbors_pymatgen_intra_cell(self, atoms):
        _c_index, _n_index, n_distance, _offsets = [], [], [], []
        struct = AseAtomsAdaptor.get_structure(atoms)
        for i in range(len(struct)):
            for j in range(len(struct)):
                if j != i:
                    dist, jimage = struct[i].distance_and_image(struct[j], jimage=[0, 0, 0])
                    _c_index.append(i)
                    _n_index.append(j)
                    n_distance.append(dist)
                    _offsets.append(jimage)

        _c_index = np.array(_c_index)
        _n_index = np.array(_n_index)
        n_distance = np.array(n_distance)
        _offsets = np.array(_offsets)

        _nonmax_idx = []
        for i in range(len(atoms)):
            idx_i = (_c_index == i).nonzero()[0]
            # sort neighbors by distance, remove edges larger than max_displace
            idx_sorted = np.argsort(n_distance[idx_i])[: self.max_displace]
            _nonmax_idx.append(idx_i[idx_sorted])
        _nonmax_idx = np.concatenate(_nonmax_idx)

        _c_index = _c_index[_nonmax_idx]
        _n_index = _n_index[_nonmax_idx]
        n_distance = n_distance[_nonmax_idx]
        _offsets = _offsets[_nonmax_idx]

        edge_index = torch.from_numpy(np.stack([_n_index, _c_index])).long()
        n_distance = torch.from_numpy(np.array(n_distance)).float()
        offsets = torch.from_numpy(np.array(_offsets)).long()

        return edge_index, n_distance, offsets
    
    def _get_neighbors_pymatgen_inter_cell(self, atoms):
        """Preforms nearest neighbor search and returns edge index, distances,
        and cell offsets"""
        struct = AseAtomsAdaptor.get_structure(atoms)
        _c_index, _n_index, _offsets, n_distance = struct.get_neighbor_list(
            r=self.radius, numerical_tol=0, exclude_self=True
        )

        _nonmax_idx = []
        for i in range(len(atoms)):
            idx_i = (_c_index == i).nonzero()[0]
            # sort neighbors by distance, remove edges larger than max_neighbors
            idx_sorted = np.argsort(n_distance[idx_i])[: self.max_neigh]
            _nonmax_idx.append(idx_i[idx_sorted])
        _nonmax_idx = np.concatenate(_nonmax_idx)

        _c_index = _c_index[_nonmax_idx]
        _n_index = _n_index[_nonmax_idx]
        n_distance = n_distance[_nonmax_idx]
        _offsets = _offsets[_nonmax_idx]

        inter_cell_mask = ~(_offsets == np.array([0, 0, 0])).all(axis=-1)
        _c_index, _n_index, n_distance, _offsets = _c_index[inter_cell_mask], _n_index[inter_cell_mask], n_distance[inter_cell_mask], _offsets[inter_cell_mask]

        edge_index = np.vstack((_n_index, _c_index))

        return torch.LongTensor(edge_index), torch.FloatTensor(n_distance), torch.LongTensor(_offsets)
        
    def _get_neighbors_pymatgen(self, atoms):
        """Preforms nearest neighbor search and returns edge index, distances,
        and cell offsets"""
        struct = AseAtomsAdaptor.get_structure(atoms)
        _c_index, _n_index, _offsets, n_distance = struct.get_neighbor_list(
            r=self.radius, numerical_tol=0, exclude_self=True
        )

        _nonmax_idx = []
        for i in range(len(atoms)):
            idx_i = (_c_index == i).nonzero()[0]
            # sort neighbors by distance, remove edges larger than max_neighbors
            idx_sorted = np.argsort(n_distance[idx_i])[: self.max_neigh]
            _nonmax_idx.append(idx_i[idx_sorted])
        _nonmax_idx = np.concatenate(_nonmax_idx)

        _c_index = _c_index[_nonmax_idx]
        _n_index = _n_index[_nonmax_idx]
        n_distance = n_distance[_nonmax_idx]
        _offsets = _offsets[_nonmax_idx]

        return _c_index, _n_index, n_distance, _offsets
    
    def _reshape_features(self, c_index, n_index, n_distance, offsets):
        """Stack center and neighbor index and reshapes distances,
        takes in np.arrays and returns torch tensors"""
        edge_index = torch.LongTensor(np.vstack((n_index, c_index)))
        edge_distances = torch.FloatTensor(n_distance)
        cell_offsets = torch.LongTensor(offsets)

        # remove distances smaller than a tolerance ~ 0. The small tolerance is
        # needed to correct for pymatgen's neighbor_list returning self atoms
        # in a few edge cases.
        nonzero = torch.where(edge_distances >= 1e-8)[0]
        edge_index = edge_index[:, nonzero]
        edge_distances = edge_distances[nonzero]
        cell_offsets = cell_offsets[nonzero]

        return edge_index, edge_distances, cell_offsets

    def unwrap_cartesian_positions(self, coords_u, coords_r, cell_r):
        supercells = torch.FloatTensor(list(itertools.product((-1, 0, 1), repeat=3)))
        super_coords_r = coords_r.unsqueeze(1) + (supercells @ cell_r).unsqueeze(0)
        dists = torch.cdist(coords_u.unsqueeze(1), super_coords_r)
        image = dists.argmin(dim=-1).squeeze()
        cell_offsets = supercells[image]
        coords_r = coords_r + cell_offsets @ cell_r

        return coords_r
    
    def convert_single(
        self,
        atoms_u
    ):
        positions_u = torch.Tensor(atoms_u.get_positions())
        cell_u = torch.Tensor(atoms_u.get_cell())
        edge_index_intra_u, edge_distances_intra_u, cell_offsets_intra_u  = self._get_neighbors_pymatgen_intra_cell(atoms_u)
        edge_index_inter_u, edge_distances_inter_u, cell_offsets_inter_u = self._get_neighbors_pymatgen_inter_cell(atoms_u)

        edge_index = torch.cat([edge_index_intra_u, edge_index_inter_u], dim=-1)
        cell_offsets = torch.cat([cell_offsets_intra_u, cell_offsets_inter_u], dim=0)
        mask = torch.cat([torch.ones_like(edge_distances_intra_u), torch.zeros_like(edge_distances_inter_u)]).bool()
        atomic_numbers = torch.Tensor(atoms_u.get_atomic_numbers())
        natoms = positions_u.shape[0]
        pbc = torch.tensor(atoms_u.pbc)
        neighbors = edge_index.size(-1)

        data = Data(
            cell_u=cell_u.view(1, 3, 3),
            pos_u=positions_u,

            x=atomic_numbers,
            cell_offsets=cell_offsets,
            edge_index=edge_index,
            natoms=natoms,
            pbc=pbc,
            mask=mask,
            neighbors=neighbors,
        )     

        return data

    def convert_pairs(
        self,
        atoms_u,
        atoms_r
    ):
        """
        Convert a pair of atomic stuctures to a graph.
        """
        positions_u = torch.Tensor(atoms_u.get_positions())
        cell_u = torch.Tensor(atoms_u.get_cell())
        edge_index_intra_u, edge_distances_intra_u, cell_offsets_intra_u  = self._get_neighbors_pymatgen_intra_cell(atoms_u)
        edge_index_inter_u, edge_distances_inter_u, cell_offsets_inter_u = self._get_neighbors_pymatgen_inter_cell(atoms_u)

        positions_r = torch.Tensor(atoms_r.get_positions())
        cell_r = torch.Tensor(atoms_r.get_cell())
        unwrapped_positions_r = self.unwrap_cartesian_positions(positions_u, positions_r, cell_r)
        atoms_r.set_positions(unwrapped_positions_r)
        positions_r = torch.Tensor(atoms_r.get_positions()) # update position

        edge_index = torch.cat([edge_index_intra_u, edge_index_inter_u], dim=-1)
        cell_offsets = torch.cat([cell_offsets_intra_u, cell_offsets_inter_u], dim=0)
        mask = torch.cat([torch.ones_like(edge_distances_intra_u), torch.zeros_like(edge_distances_inter_u)]).bool()
        atomic_numbers = torch.Tensor(atoms_u.get_atomic_numbers())
        natoms = positions_u.shape[0]
        pbc = torch.tensor(atoms_u.pbc)

        data = Data(
            cell_u=cell_u.view(1, 3, 3),
            pos_u=positions_u,

            cell_r=cell_r.view(1, 3, 3),
            pos_r=positions_r,

            x=atomic_numbers,
            cell_offsets=cell_offsets,
            edge_index=edge_index,
            natoms=natoms,
            pbc=pbc,
            mask=mask
        )

        return data

# %%
