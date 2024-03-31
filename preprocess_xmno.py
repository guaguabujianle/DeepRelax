import os
import csv
import multiprocessing as mp
import lmdb
import numpy as np
from tqdm import tqdm
from graph_constructor import AtomsToGraphs
from ase.io import read
import pickle
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings("ignore")

def write_data(mp_args):
    data_root, a2g, id_prop_data, db_path, data_indices = mp_args
    db = lmdb.open(
        db_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    for i, index in enumerate(tqdm(data_indices, desc='Reading atoms objects', position=0, leave=True)):
        cif_id, target = id_prop_data[index]

        unrelaxed_path = os.path.join(data_root, cif_id.replace('relaxed', 'unrelaxed') + '.cif') 
        relaxed_path = os.path.join(data_root, cif_id + '.cif') 
        
        atoms_u = read(unrelaxed_path)
        atoms_r = read(relaxed_path)
        
        data = a2g.convert_pairs(atoms_u, atoms_r)
        data.cif_id = cif_id
        data.y = float(target)

        txn = db.begin(write=True)
        txn.put(f"{i}".encode("ascii"), pickle.dumps(data, protocol=-1))
        txn.commit()

    # Save count of objects in lmdb.
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(i+1, protocol=-1))
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

    train_id_prop_file = os.path.join(data_root, 'id_prop_train_all.csv')
    val_id_prop_file = os.path.join(data_root, 'id_prop_val_all.csv')
    test_id_prop_file = os.path.join(data_root, 'id_prop_test_all.csv')

    with open(train_id_prop_file) as f:
        reader = csv.reader(f)
        train_id_prop_data = [row for row in reader if row[0].split('_')[-1] == 'relaxed']

    with open(val_id_prop_file) as f:
        reader = csv.reader(f)
        val_id_prop_data = [row for row in reader if row[0].split('_')[-1] == 'relaxed']

    with open(test_id_prop_file) as f:
        reader = csv.reader(f)
        test_id_prop_data = [row for row in reader if row[0].split('_')[-1] == 'relaxed']

    a2g = AtomsToGraphs(
        radius=6,
        max_neigh=30,
        max_displace=20,
    )

    for dataset in ['train_DeepRelax', 'val_DeepRelax', 'test_DeepRelax']:
        if dataset == 'train_DeepRelax':
            id_prop_data = train_id_prop_data
            db_path = os.path.join(data_root,'train_DeepRelax')
        elif dataset == 'val_DeepRelax':
            id_prop_data = val_id_prop_data
            db_path = os.path.join(data_root,'val_DeepRelax')
        elif dataset == 'test_DeepRelax':
            id_prop_data = test_id_prop_data
            db_path = os.path.join(data_root,'test_DeepRelax')

        data_len = len(id_prop_data)
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
                data_root,
                a2g,
                id_prop_data,
                mp_db_paths[i],
                mp_data_indices[i]
            )
            for i in range(num_workers)
        ]

        pool.imap(write_data, mp_args)

        pool.close()
        pool.join()
