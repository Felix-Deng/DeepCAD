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
from utils.file_utils import ensure_dir


src_dir = "cad_json_test/0000"
print(src_dir)
file_format = "json" # either h5 or json 
out_paths = sorted(glob.glob(os.path.join(src_dir, "*.{}".format(file_format))))
save_dir = "cad_step_test/0000"
ensure_dir(save_dir)

filter_invalid = False 

for path in out_paths:
    print(path)
    try:
        if file_format == "h5":
            with h5py.File(path, 'r') as fp:
                out_vec = fp["out_vec"][:].astype(np.float)
                out_shape = vec2CADsolid(out_vec)
        else: # json 
            with open(path, 'r') as fp:
                data = json.load(fp)
            cad_seq = CADSequence.from_dict(data)
            cad_seq.normalize()
            out_shape = create_CAD(cad_seq)

    except Exception as e:
        print("load and create failed.")
        continue
    
    if filter_invalid:
        analyzer = BRepCheck_Analyzer(out_shape)
        if not analyzer.IsValid():
            print("detect invalid.")
            continue
    
    name = path.split("/")[-1].split(".")[0]
    save_path = os.path.join(save_dir, name + ".step")
    write_step_file(out_shape, save_path)

