# %%
import os
import pickle as pk
import pandas as pd
import numpy as np
import json
import ase
from collections import defaultdict
import pymatgen
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from glob import glob
from ase.db import connect
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.io import write, read
import json
from pymatgen.io.ase import AseAtomsAdaptor

# Connect to the database
db_root = '/scratch/yangzd/materials/data/c2db'
cif_root = os.path.join(db_root, 'CIF')
db_relaxed_path = os.path.join(db_root, 'relaxed.db')
db_unrelaxed_path = os.path.join(db_root, 'unrelaxed.db')

with connect(db_unrelaxed_path) as conn:
    for i in range(1, len(conn) + 1):
        # Query the database to get all rows
        row = conn.get(i)
        folder = row.folder
        atoms_id = folder.split('/')[-1]

        # Extract relevant information
        struct_dict = row.data.get('structure.json')
        if struct_dict == None:
            struct_dict = row.data.get('structure_large_vacuum.json')
        if struct_dict == None:
            struct_dict = row.data.get('unrelaxed.json')
        
        atomic_numbers = np.array(struct_dict['1']['numbers'])
        positions = np.array(struct_dict['1']['positions'])
        cell = np.array(struct_dict['1']['cell'])

        # Convert atomic numbers to symbols
        symbols = [chemical_symbols[number] for number in atomic_numbers]

        # Create Atoms object
        atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=[True, True, False])

        write(os.path.join(cif_root, f'{atoms_id}_unrelaxed.cif'), atoms)

