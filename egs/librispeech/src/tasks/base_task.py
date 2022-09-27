# 2022 Ruchao Fan
# SPAPL

import os
import torch
from data.speech_loader import SpeechDataset, DynamicDataset, SpeechDataLoader
from data.audio_loader import HubertDataset, HubertLoader

class BaseTask(object):
    def __init__(self, args):
        self.use_cuda = args.use_gpu

    def set_model(self, args):
        raise NotImplementedError   

    def load_model(self, args):
        #Either load from checkpoint, or load model defined in config file
        last_checkpoint = os.path.join(args.exp_dir, 'model.last.mdl')
        if os.path.exists(last_checkpoint):
            self.load_checkpoint(last_checkpoint, args.rank, args.use_gpu)
        else:
            self.load_pretrained_model(args.resume_model, args.rank)
    
        self.model_stats(args.rank, args.use_slurm, args.distributed)
    
    def load_pretrained_model(self, resume_model, rank):
        raise NotImplementedError

    def load_checkpoint(self, checkpoint, rank, use_cuda):
        #first load checkpoint and obtain model param names, then extract params, and then optimiser and epoch
        if rank == 0:
            print("Loading checkpoint from {}".format(checkpoint))
        checkpoint = torch.load(checkpoint, map_location='cpu')
        model_state = checkpoint["model_state"]
        
        for name, param in self.model.named_parameters():
            if name not in model_state:
                name = "module." + name
            param.data.copy_(model_state[name])
        
        self.optimizer.load_state_dict(checkpoint['optimizer'], rank, use_cuda)
        self._num_updates = self.optimizer._step
        self.start_epoch = checkpoint['epoch'] + 1

    def load_test_model(self, resume_model):
        #similar to above, but during testing we don't need optimiser and epoch stats
        if resume_model:
            print("Loading model from {}".format(resume_model))
            checkpoint = torch.load(resume_model, map_location='cpu')
            model_state = checkpoint["model_state"]
            for name, param in self.model.named_parameters():
                if name not in model_state:
                    name = "module." + name
                param.data.copy_(model_state[name])
    
    def model_stats(self, rank, use_slurm, distributed):
        if rank == 0:
            #///OG print(self.model)
            num_params, updated_params = 0, 0

            #count number of params and params that require gradient updates (numel - length of tensor)
            for name, param in self.model.named_parameters():
                num_params += param.numel()
                if param.requires_grad == True:
                    updated_params += param.numel()

            print("Number of parameters: {}, updated params: {}".format(num_params, updated_params))

            self.model_params = num_params
            self.updated_params = updated_params

        #if we use slurm, we effectively need rank for each individual gpu
        if use_slurm:
            local_rank = rank % torch.cuda.device_count()
        else:
            local_rank = rank

        if self.use_cuda:
            torch.cuda.set_device(local_rank)
            self.model = self.model.cuda(local_rank)

        if distributed:        
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], find_unused_parameters=True)

    def set_dataloader(self, args):

        if args.dataset_type=="HubertDataset":
            trainset = HubertDataset(self.vocab,args.train_paths,args)
            actual_bs = 1
            train_loader = HubertLoader(trainset, actual_bs, args.padding_idx, num_workers=args.load_data_workers, 
                                       distributed=args.distributed, shuffle=True)
            if args.rank == 0:
                print("Finish Loading training files. Number batches: {}".format(len(train_loader)))

            validset = HubertDataset(self.vocab, args.dev_paths, args)

            valid_loader = HubertLoader(validset, actual_bs, args.padding_idx, num_workers=args.load_data_workers, 
                                            distributed=False, shuffle=False)
            if args.rank == 0:
                print("Finish Loading dev files. Number batches: {}".format(len(valid_loader)))
        
        else:
            
            dataset_types = {"SpeechDataset": (SpeechDataset, args.batch_size), "DynamicDataset": (DynamicDataset, 1)}
            Dataset, actual_bs = dataset_types[args.dataset_type]

            #load train set and dataloader
            trainset = Dataset(self.vocab, args.train_paths, args)
            if args.use_cmvn:
                trainset._load_cmvn(args.global_cmvn)
            
            train_loader = SpeechDataLoader(trainset, actual_bs, args.padding_idx, num_workers=args.load_data_workers, 
                                        distributed=args.distributed, shuffle=True)
            if args.rank == 0:
                print("Finish Loading training files. Number batches: {}".format(len(train_loader)))

            #now load valid dataset
            args.use_specaug = False  #///OG specaug cannot be applied to valid
            validset = Dataset(self.vocab, args.dev_paths, args)
            if args.use_cmvn:
                validset._load_cmvn(args.global_cmvn)
            valid_loader = SpeechDataLoader(validset, actual_bs, args.padding_idx, num_workers=args.load_data_workers, 
                                            distributed=False, shuffle=False)
            if args.rank == 0:
                print("Finish Loading dev files. Number batches: {}".format(len(valid_loader)))

        self.train_loader = train_loader
        self.valid_loader = valid_loader
    
    def set_test_dataloader(self, args):
        #load test set
        args.use_specaug = False
        args.specaug_conf = None

        if args.dataset_type=="HubertDataset":
            testset = HubertDataset(self.vocab, args.test_paths, args)
            test_loader = HubertLoader(testset, args.batch_size, args.padding_idx, num_workers=args.load_data_workers, shuffle=False)
        else:
            testset = SpeechDataset(self.vocab, args.test_paths, args)
            if args.use_cmvn:
                testset._load_cmvn(args.global_cmvn)
            test_loader = SpeechDataLoader(testset, args.batch_size, args.padding_idx, num_workers=args.load_data_workers, shuffle=False)

        print("Finish Loading test files. Number batches: {}".format(len(test_loader)))

        self.test_loader = test_loader 

    def set_optimizer(self, args):
        raise NotImplementedError 
