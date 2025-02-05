#!/usr/bin/env python3
# 2022 Ruchao Fan
# SPAPL

import os
import sys
import yaml
import json
import torch
import numpy as np
import torch.distributed as dist
import torch.backends.cudnn as cudnn

#add parent dir as env variable prior to running script
sys.path.append(os.environ['E2EASR']+'/src')
from tasks import CTCTask, ArtTask, CassNATTask, LMNATTask, LMNAT2Task, LMNAT3Task, HubertTask
from utils.parser import BaseParser

class Config():
    name = 'config'

def main():
    args = BaseParser().get_args()
    
    #extract config from args
    #rank - for multiGPU it is procid of each gpu
    if args.use_slurm:
        world_size = int(os.environ["WORLD_SIZE"])
        args.distributed = True if world_size > 1 else False
        if args.distributed:
            rank = int(os.environ['SLURM_PROCID'])
        else:
            rank = 0
        args.master_addr = os.environ["MASTER_ADDR"]
        args.port = os.environ["MASTER_PORT"]
    else:
        num_gpu = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        args.distributed = True if num_gpu > 1 else False
        rank = 0
        args.master_addr = "localhost"

    with open(args.train_config) as f:
        config = yaml.safe_load(f)

    with open(args.data_config) as f:
        #note: for hubert we use same variable names, but scp_path references wav.scp and not feats.scp
        #utt2numframes is also dummy variable and is not used
        data = yaml.safe_load(f)
        config['train_paths'] = [j for i, j in data['train_data_path'].items()]
        config['dev_paths'] = [j for i, j in data['dev_data_path'].items()]
        config['global_cmvn'] = data['global_cmvn'] #this file does not exist for hubert
        config['vocab_file'] = data['vocab_file']

    if rank == 0 and not os.path.isdir(args.exp_dir):
        os.makedirs(args.exp_dir)
    
    #add yml fields to args
    for key, val in config.items():
        setattr(args, key, val)
    #reset values in config from args, to prep for dumping as json file
    for var in vars(args):
        config[var] = getattr(args, var)

    if rank == 0:
        print("Experiment starts with config {}".format(json.dumps(config, sort_keys=True, indent=4)))
        json.dump(config, open(os.path.join(args.exp_dir, "config.yaml"), 'w'), sort_keys=True, indent=4)

    if args.use_specaug:
        specaug_conf = Config()
        for key, val in config["spec_aug"].items():
            setattr(specaug_conf, key, val)
        args.specaug_conf = specaug_conf
    else:
        args.specaug_conf = None

    if args.use_slurm:
        main_worker(rank, world_size, args)
    else:
        if args.distributed:
            import torch.multiprocessing as mp
            mp.spawn(main_worker, nprocs=num_gpu, args=(num_gpu, args))
        else:
            main_worker(0, 1, args)
        
def main_worker(rank, world_size, args, backend='nccl'):
    args.rank, args.world_size = rank, world_size
    if args.distributed:
        dist.init_process_group(backend=backend, init_method='tcp://{}:{}'.format(args.master_addr, args.port), world_size=world_size, rank=rank)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    use_cuda = args.use_gpu
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    task_dict = {"art": ArtTask, "cassnat": CassNATTask, "ctc": CTCTask, "lmnat": LMNATTask, 
                    "lmnat2": LMNAT2Task, "lmnat3": LMNAT3Task, "hubert": HubertTask}
    
    #based on task type, init model
    if args.task in task_dict:
        task = task_dict[args.task]("train", args)
    else:
        raise NotImplementedError
    
    #train model
    task.run(args)
      
if __name__ == '__main__':
    main()


