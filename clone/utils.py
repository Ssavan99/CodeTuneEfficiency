import json
import logging
import os
from datasets import load_dataset, load_metric, Dataset
import torch
import numpy as np
import random
import glob
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_last_checkpoint(dir):
    paths = list(glob.glob(os.path.join(dir, "checkpoint-*")))
    if len(paths) == 0:
        return None
    last_path_value = paths[0].split("/")[-1].split("-")[1]
    last_path = paths[0]
    for path in paths:
        if int(path.split("/")[-1].split("-")[1]) > int(last_path_value):
            last_path = path
            last_path_value = path.split("/")[-1].split("-")[1]
    return last_path


class CodeDataset:
    def __init__(self, data_dir):
        self.url_to_code = {}
        with open(data_dir + '/data.jsonl') as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                self.url_to_code[js['idx']] = js['func']

    def load(self, file_path, data_num=-1):
        dataset_dict = {'code1':[], 'code2':[], 'label':[]}
        count = 0
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                url1, url2, label = line.split('\t')
                if url1 not in self.url_to_code or url2 not in self.url_to_code:
                    continue

                count += 1
                dataset_dict['code1'].append(self.url_to_code[url1])
                dataset_dict['code2'].append(self.url_to_code[url2])
                dataset_dict['label'].append(int(label))
                if count == data_num:
                    break
        return Dataset.from_dict(dataset_dict)




