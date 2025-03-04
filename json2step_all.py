import os
import glob
import json
import h5py
import numpy as np
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Extend.DataExchange import read_step_file, write_step_file
# sys.path.append("..")
from cadlib.extrude import CADSequence
from cadlib.visualize import vec2CADsolid, create_CAD
from utils.file_utils import ensure_dir

def json2step(src_dir: str, save_dir: str): 
    file_format = 'json' # either h5 or json 
    paths = sorted(glob.glob(os.path.join(src_dir, "*.{}".format(file_format))))
    ensure_dir(save_dir)

    filter_invalid = False

    for path in paths:        
        name = path.split("\\")[-1].split(".")[0]
        save_path = os.path.join(save_dir, name + ".step")
        
        skip_paths = [
            'cad_json\\0011\\00112828.json', 'cad_json\\0011\\00116212.json', 'cad_json\\0023\\00234799.json', 
            'cad_json\\0059\\00591942.json', 'cad_json\\0075\\00757721.json'
        ] # making the program stop for unknown reasons 
        if os.path.exists(save_path) or path in skip_paths: 
            continue
        
        print(path)
        try:
            if file_format == "h5":
                with h5py.File(path, 'r') as fp:
                    out_vec = fp["out_vec"][:].astype(np.float)
                    out_shape = vec2CADsolid(out_vec)
            else:
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
        try: 
            write_step_file(out_shape, save_path)
        except Exception as e: 
            print("write step file failed.")
            continue 

input_dir = 'cad_json'
output_dir = 'cad_step'
for path in os.listdir(input_dir): 
    json_dir = os.path.join(input_dir, path)
    step_dir = os.path.join(output_dir, path)
    json2step(json_dir, step_dir)
    