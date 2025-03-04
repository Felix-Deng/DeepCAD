import os
import glob
import json
import h5py
import numpy as np
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Extend.DataExchange import read_step_file, write_step_file
import sys
# sys.path.append("..")
from cadlib.extrude import CADSequence
from cadlib.visualize import vec2CADsolid, create_CAD
from cadlib.macro import *
from utils.file_utils import ensure_dir


DATA_ROOT = ""
RAW_DATA = os.path.join(DATA_ROOT, "cad_json_test")
BREP_DIR = os.path.join(DATA_ROOT, "train_step") # input: current BRep model in STEP
VEC_DIR = os.path.join(DATA_ROOT, "train_vec") # output: ground-truth vector for next feature 

ensure_dir(VEC_DIR)
ensure_dir(BREP_DIR)

for sub_dir in os.listdir(RAW_DATA): # iterate through sub-folders 
    path = os.path.join(RAW_DATA, sub_dir)
    vec_save_path = os.path.join(VEC_DIR, sub_dir)
    brep_save_path = os.path.join(BREP_DIR, sub_dir)
    ensure_dir(vec_save_path)
    ensure_dir(brep_save_path)
    
    for file_name in os.listdir(path): # iterate through JSON files in sub-folder 
        name = file_name.split(".")[0]
        file_path = os.path.join(path, file_name)
        print(file_path)
        with open(file_path, 'r') as fp: 
            data = json.load(fp)
        
        # Load and retrieve full modelling sequence of the model 
        try: 
            cad_seq = CADSequence.from_dict(data)
            cad_seq.normalize() 
            cad_seq.numericalize() 
            cad_vec = cad_seq.to_vector(MAX_N_EXT, MAX_N_LOOPS, MAX_N_CURVES, MAX_TOTAL_LEN, pad=False)
        except Exception as e: 
            print("Faild:", file_path)
            continue
        
        if type(cad_vec) == type(None): 
            print("none is returned:", file_path)
            continue 
        elif MAX_TOTAL_LEN < cad_vec.shape[0]: # MAX_TOTAL_LEN = 60
            print("exceed length condition:", file_path, cad_vec.shape[0])
            continue
            
        # Break the model into modelling steps 
        modelling_steps = [] 
        current_step = [] 
        for vec in cad_vec: 
            current_step.append(vec)
            if vec[0] == 5: 
                current_step = np.concatenate([current_step, EOS_VEC[np.newaxis]], axis=0) # pad EOS token 
                modelling_steps.append(current_step)
                current_step = []
        
        # Generate training samples 
        for i in range(len(modelling_steps) - 1): 
            try: 
                # Training input BRep (.step)
                brep_input = vec2CADsolid(modelling_steps[i])
                write_step_file(brep_input, os.path.join(brep_save_path, name + "_{}.step".format(i)))
                # Training output vector (.h5)
                vec_output = modelling_steps[i + 1]
                with h5py.File(os.path.join(vec_save_path, name + "_{}.h5".format(i)), 'w') as fp: 
                    fp.create_dataset('vec', data=vec_output, dtype=int)
            except Exception as e: 
                print("Failed to save:", file_path)
