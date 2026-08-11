import glob
import json
import logging
import os
from datasets import load_dataset, load_metric, Dataset
import torch
import numpy as np
import random

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def get_last_checkpoint(dir):
    paths = list(glob.glob(os.path.join(dir, "checkpoint-*")))
    if len(paths) == 0:
        return dir
    last_path_value = paths[0].split("/")[-1].split("-")[1]
    last_path = paths[0]
    for path in paths:
        if int(path.split("/")[-1].split("-")[1]) > int(last_path_value):
            last_path = path
            last_path_value = path.split("/")[-1].split("-")[1]
    return last_path


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


import json


def read_dataset(filename):
    source_codes = []
    source_labels = []
    with open(filename) as f:
        for line in f:
            js = json.loads(line.strip())
            code = ' '.join(js['func'].split())
            label = int(js['target'])
            source_codes.append(code)
            source_labels.append(label)
    return source_codes, source_labels


class CodeDataset:
    def load(self, file_path, data_num=-1):
        dataset_dict = {'code': [], 'label': []}
        count = 0
        with open(file_path) as f:
            for line in f:
                js = json.loads(line.strip())
                dataset_dict['code'].append(' '.join(js['func'].split()))
                dataset_dict['label'].append(int(js['target']))
                count += 1
                if count == data_num:
                    break
        return Dataset.from_dict(dataset_dict)
