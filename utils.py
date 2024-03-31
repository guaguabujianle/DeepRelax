import os
import pickle
import torch
import itertools

def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path,i)  
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)

def write_pickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

def compute_cart_mean_absolute_displacement(coords_u, coords_r, cell_r):
    supercells = torch.FloatTensor(list(itertools.product((-1, 0, 1), repeat=3)))
    super_coords_r = coords_r.unsqueeze(1) + (supercells @ cell_r).unsqueeze(0)
    dists = torch.cdist(coords_u.unsqueeze(1), super_coords_r)
    image = dists.argmin(dim=-1).squeeze()
    cell_offsets = supercells[image]
    coords_r = coords_r + cell_offsets @ cell_r
    mad = (coords_r - coords_u).abs().mean()

    return mad

def compute_frac_mean_absolute_displacement(frac_coords_u, frac_coords_r):
    supercells = torch.FloatTensor(list(itertools.product((-1, 0, 1), repeat=3)))
    super_frac_coords_r = frac_coords_r.unsqueeze(1) + supercells.unsqueeze(0)
    dists = torch.cdist(frac_coords_u.unsqueeze(1), super_frac_coords_r)
    image = dists.argmin(dim=-1).squeeze()
    cell_offsets = supercells[image]
    frac_coords_r = frac_coords_r + cell_offsets
    mad = (frac_coords_r - frac_coords_u).abs().mean()
    
    return mad

def create_dir(dir_list):
    assert  isinstance(dir_list, list) == True
    for d in dir_list:
        if not os.path.exists(d):
            os.makedirs(d)

def save_model_dict(model, model_dir, msg):
    model_path = os.path.join(model_dir, msg + '.pt')
    torch.save(model.state_dict(), model_path)
    print("model has been saved to %s." % (model_path))

def load_model_dict(model, ckpt):
    model.load_state_dict(torch.load(ckpt))

class BestMeter(object):
    """Computes and stores the best value"""

    def __init__(self, best_type):
        self.best_type = best_type  
        self.count = 0      
        self.reset()

    def reset(self):
        if self.best_type == 'min':
            self.best = float('inf')
        else:
            self.best = -float('inf')

    def update(self, best):
        self.best = best
        self.count = 0

    def get_best(self):
        return self.best

    def counter(self):
        self.count += 1
        return self.count

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)

        return self.avg