import pandas as pd
import json
import time
import os, shutil
from jsonmerge import merge as json_merge
import itertools

import sys
sys.path.append('..')

from os import listdir
from os.path import isfile, join

def generate_experiment_jsons():

    # read all JSON files from input template JSONs
    # outputs combined JSON files to output_json folder 

    _input_directory_ = "./jsons/input_json_templates"
    _output_directory_ = "./jsons/output_exp_jsons"

    if not os.path.exists(_input_directory_):
        raise Exception("Absent Input Directory !")
    
    if not os.path.exists(_output_directory_):
        os.makedirs(_output_directory_)
    else:
        # clear the contents of the directory
        shutil.rmtree(_output_directory_)
        os.makedirs(_output_directory_)
    
    json_templates = [f for f in listdir(_input_directory_) if isfile(join(_input_directory_, f))]
    
    _json_lists_ = []

    common_json_data = {}

    for temp in json_templates:

        temp_filename = join(_input_directory_, temp)
        template = open(temp_filename, 'r')

        _json_data_ = json.load(template)

        if isinstance(_json_data_, list):
            _json_lists_.append(_json_data_)
        elif isinstance(_json_data_, dict):
            for key, value in _json_data_.items():
                common_json_data[key] = value
        else:
            raise Exception("Invalid type of JSON Data ! Only List / Dict allowed ...")
        
    count = 0
    for combined_json in itertools.product(*_json_lists_):
        count += 1

        json_holder = common_json_data
        
        if not(combined_json and json_holder):
            continue

        for _dict_ in combined_json:
            json_holder = json_merge(_dict_, json_holder)

        new_json_filename = f"experiment_{count}.json"

        new_json_filename = join(_output_directory_, new_json_filename)
        with open(new_json_filename, 'w') as outfile:
            json.dump(json_holder, outfile)


if __name__ == "__main__":

    generate_experiment_jsons()