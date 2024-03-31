# %%
import multiprocessing as mp
import os
import lmdb
import numpy as np
from tqdm import tqdm
from graph_constructor import AtomsToGraphs
from ase.io import read
import pandas as pd
import pickle
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings("ignore")

def write_data(mp_args):
    a2g, data_root, mp_ids, db_path, data_indices = mp_args
    db = lmdb.open(
        db_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    idx = 0
    for index in tqdm(data_indices, desc='Reading atoms objects', position=0, leave=True):
        mp_id = mp_ids[index]

        relaxed_path = os.path.join(data_root, 'CIF', f'{mp_id}_R.cif')
        if os.path.exists(relaxed_path):
            atoms_r = read(relaxed_path)
        else:
            continue

        unrelaxed_path = os.path.join(data_root, 'CIF', f'{mp_id}_U.cif')
        if os.path.exists(unrelaxed_path):
            atoms_u = read(unrelaxed_path)

            data = a2g.convert_pairs(atoms_u, atoms_r)
            data.cif_id = f'{mp_id}_U'

            txn = db.begin(write=True)
            txn.put(f"{idx}".encode("ascii"), pickle.dumps(data, protocol=-1))
            txn.commit()
            idx += 1
        else:
            continue

    # Save count of objects in lmdb.
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))
    txn.commit()

    db.sync()
    db.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--data_root', type=str, default=None, help='data directory', required=True)
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers')
    
    args = parser.parse_args()
    data_root = args.data_root
    num_workers = args.num_workers    

    train_df = pd.read_csv(os.path.join(data_root, 'train.csv'))
    val_df = pd.read_csv(os.path.join(data_root, 'val.csv'))
    test_df = pd.read_csv(os.path.join(data_root, 'test.csv'))
    
    train_mp_ids = train_df['mp_id']
    val_mp_ids = val_df['mp_id']
    test_mp_ids = test_df['mp_id']

    print("%d train samples" % len(train_mp_ids))
    print("%d val samples" % len(val_mp_ids))
    print("%d test samples" % len(test_mp_ids))

    a2g = AtomsToGraphs(
        radius=6,
        max_neigh=30,
        max_displace=20,
    )

    for dataset in ['train_DeepRelax', 'val_DeepRelax', 'test_DeepRelax']:
        if dataset == 'train_DeepRelax':
            mp_ids = train_mp_ids
            db_path = os.path.join(data_root, 'train_DeepRelax')
        elif dataset =='val_DeepRelax':
            mp_ids = val_mp_ids
            db_path = os.path.join(data_root, 'val_DeepRelax')
        elif dataset =='test_DeepRelax':
            mp_ids = test_mp_ids
            db_path = os.path.join(data_root, 'test_DeepRelax')

        data_len = len(mp_ids)
        print(f'{dataset}: {data_len}')

        data_indices = np.array(list(range(data_len)))
        save_path = Path(db_path)
        save_path.mkdir(parents=True, exist_ok=True)

        mp_db_paths = [
            os.path.join(save_path, "data.%04d.lmdb" % i)
            for i in range(num_workers)
        ]
        mp_data_indices = np.array_split(data_indices, num_workers)

        pool = mp.Pool(num_workers)
        mp_args = [
            (
                a2g,
                data_root,
                mp_ids,
                mp_db_paths[i],
                mp_data_indices[i]
            )
            for i in range(num_workers)
        ]

        pool.imap(write_data, mp_args)

        pool.close()
        pool.join()
