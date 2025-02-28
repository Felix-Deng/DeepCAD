import os
import json
import numpy as np
import h5py
from joblib import Parallel, delayed
import sys
sys.path.append("..")
from cadlib.extrude import CADSequence
from cadlib.macro import *

DATA_ROOT = ""
RAW_DATA = os.path.join(DATA_ROOT, "cad_json_test")
SAVE_DIR = os.path.join(DATA_ROOT, "cad_vec_test")
print(RAW_DATA)
print(SAVE_DIR)
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

for sub_dir in os.listdir(RAW_DATA): 
    path = os.path.join(RAW_DATA, sub_dir)
    save_path = os.path.join(SAVE_DIR, sub_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    for cad_json in os.listdir(path): 
        data_path = os.path.join(path, cad_json)
        with open(data_path, 'r') as fp: 
            data = json.load(fp)
        
        try:
            cad_seq = CADSequence.from_dict(data)
            cad_seq.normalize()
            cad_seq.numericalize()
            cad_vec = cad_seq.to_vector(MAX_N_EXT, MAX_N_LOOPS, MAX_N_CURVES, MAX_TOTAL_LEN, pad=False)
        except Exception as e:
            print(e)
            print("failed:", data_path)
            continue 
        
        if type(cad_vec) == type(None): 
            print("none is returned:", data_path)
            continue
        elif MAX_TOTAL_LEN < cad_vec.shape[0]: # MAX_TOTAL_LEN = 60
            print("exceed length condition:", data_path, cad_vec.shape[0])
            continue
        
        name = data_path.split('/')[-1].split('.')[0]
        save_data_path = os.path.join(save_path, name + ".h5")
        print(save_data_path)
        with h5py.File(save_data_path, 'w') as fp:
            fp.create_dataset("vec", data=cad_vec, dtype=int)
        break 
        

"""
def process_one(data_id):
    json_path = os.path.join(RAW_DATA, data_id + ".json")
    with open(json_path, "r") as fp:
        data = json.load(fp)

    try:
        cad_seq = CADSequence.from_dict(data)
        cad_seq.normalize()
        cad_seq.numericalize()
        cad_vec = cad_seq.to_vector(MAX_N_EXT, MAX_N_LOOPS, MAX_N_CURVES, MAX_TOTAL_LEN, pad=False)

    except Exception as e:
        print("failed:", data_id)
        return

    if MAX_TOTAL_LEN < cad_vec.shape[0] or cad_vec is None:
        print("exceed length condition:", data_id, cad_vec.shape[0])
        return

    save_path = os.path.join(SAVE_DIR, data_id + ".h5")
    truck_dir = os.path.dirname(save_path)
    if not os.path.exists(truck_dir):
        os.makedirs(truck_dir)

    with h5py.File(save_path, 'w') as fp:
        fp.create_dataset("vec", data=cad_vec, dtype=np.int)


with open(RECORD_FILE, "r") as fp:
    all_data = json.load(fp)

Parallel(n_jobs=10, verbose=2)(delayed(process_one)(x) for x in all_data["train"])
Parallel(n_jobs=10, verbose=2)(delayed(process_one)(x) for x in all_data["validation"])
Parallel(n_jobs=10, verbose=2)(delayed(process_one)(x) for x in all_data["test"])
"""