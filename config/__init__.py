import yaml, argparse
import os, re
import wandb
from .wandb_config import init_wandb
import datetime
from glob import glob
from .yaml_utils import read_yaml


CONFIG_ROOT = "./config"

class dict2class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])

def parser_argument():
    parser = argparse.ArgumentParser(description='Welcome to domain game')
    parser.add_argument('--phase', default='train', type=str, help='train | test | pseudo_gen | pseudo_train')
    parser.add_argument('--task', type=str, help='prostate | polyp | knee')
    parser.add_argument('--file', default='train.yaml', type=str, help='yaml file')
    # phase: "train"
    parser.add_argument('-m', type=str, default="")
    
    parser.add_argument('--resume', type=str, default="")
    parser.add_argument("--deviceid", type=int, default=0)
    parser.add_argument('--debug', action="store_true")

    # parser.add_argument('--sc', action="store_true", help='save self-contrastive')
    # parser.add_argument('--valid_only', action="store_true")
    # parser.add_argument('--valid_ratio', type=float, default=1)
    args = parser.parse_args()
    return args


def set_default(dict):
    DEFAULT_YAML = os.path.join(CONFIG_ROOT, "default.yaml")
    # Iterative read yaml and its root file
    DEFAULT_DICT = read_yaml(DEFAULT_YAML)

    # Get all default_dict elements that not in dict
    for key in DEFAULT_DICT.keys():
        if key not in dict.keys():
            dict[key] = DEFAULT_DICT[key]

    dict.setdefault('model_save_path', None)
    dict.setdefault("lora", False)
    dict.setdefault("mixup", False)

    return dict




def setup_experiment(opt):
    outputdir = f'./result/{opt["mission"]}'
    os.makedirs(outputdir, exist_ok=True)
    file_ocunt = len(list(glob(os.path.join(outputdir, '*'))))

    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y_%m_%d")

    exp_name =  f'{file_ocunt}_{formatted_datetime}_train_{opt["TAG"]}_{opt["m"]}'
    return exp_name


def get_config(path="./"):
    args = parser_argument()
    args.file = args.file.rsplit('.', 1)[0] + '.yaml'
    filename = os.path.join(CONFIG_ROOT, args.task, args.file)
    print("=== Reading yaml file:", filename)

    opt = read_yaml(filename)
    set_default(opt)

    opt['deviceid'] = args.deviceid
    opt['phase'] = args.phase
    opt['debug'] = args.debug

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.deviceid)
    opt['resume'] = args.resume
    
    if args.resume:
        opt['model_save_path'] = args.resume
        

    opt['m'] = args.m
    opt['experiment'] = setup_experiment(opt)

    init_wandb(opt)
    data = dict2class(opt)
    return data

