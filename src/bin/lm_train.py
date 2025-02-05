#!/usr/bin/env python3
# 2020 Ruchao Fan

import os
import sys
import time
import yaml
import json
import torch
import argparse
import numpy as np
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.distributed import ReduceOp

sys.path.append(os.environ['E2EASR']+'/src')
import utils.util as util
from data.vocab import Vocab
from utils.optimizer import get_opt
from data.text_loader import TextDataset, TextDataLoader

class Config():
    name = 'config'

def main():
    parser = argparse.ArgumentParser(description="Configuration for training ctc-attention system")
   
    parser.add_argument("--exp_dir")
    parser.add_argument("--train_config")
    parser.add_argument("--data_config")
    parser.add_argument("--lm_type")
    parser.add_argument("--batch_size", default=32, type=int, help="Training minibatch size")
    parser.add_argument("--epochs", default=30, type=int, help="Number of training epochs")
    parser.add_argument("--save_epoch", default=20, type=int, help="Starting to save the model")
    parser.add_argument("--learning_rate", default=2e-4, type=float, help="Initial learning rate")
    parser.add_argument("--min_lr", default=1e-6, type=float, help="Minimal learning rate")
    parser.add_argument("--patience", default=2, type=int, help="Number of epochs without improvements")
    parser.add_argument("--end_patience", default=2, type=int, help="Number of epochs without improvements for early stop")
    parser.add_argument("--opt_type", default='normal', type=str, help="Type of optimizer, normal or noam")
    parser.add_argument("--anneal_lr_ratio", default=0.5, type=float, help="Learning rate decay ratio, used when opt_type='normal'")
    parser.add_argument("--weight_decay", default=0.00001, type=float, help="Weight decay in optimizer")
    parser.add_argument("--load_data_workers", default=0, type=int, help="Number of parallel data loaders")
    parser.add_argument("--resume_model", default='', type=str, help="The model path to resume")
    parser.add_argument("--print_freq", default=100, type=int, help="Number of iter to print")
    parser.add_argument("--seed", default=1, type=int, help="Random number seed")

    ##///OG 1. Parse and print config Main process
    args = parser.parse_args()
    with open(args.train_config) as f:
        config = yaml.safe_load(f)

    with open(args.data_config) as f:
        data = yaml.safe_load(f)
        config['train_paths'] = [j for i, j in data['train_data_path'].items()]
        config['dev_paths'] = [j for i, j in data['dev_data_path'].items()]
        config['vocab_file'] = data['vocab_file']

    for key, val in config.items():
        setattr(args, key, val)
    for var in vars(args):
        config[var] = getattr(args, var)
    print("Experiment starts with config {}".format(json.dumps(config, sort_keys=True, indent=4)))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    if not os.path.isdir(args.exp_dir):
        os.makedirs(args.exp_dir)

    num_gpu = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    args.distributed = True if num_gpu > 1 else False
    if args.distributed:
        import torch.multiprocessing as mp
        mp.spawn(main_worker, nprocs=num_gpu, args=(num_gpu, args))
    else:
        main_worker(0, 1, args)
        

def main_worker(rank, world_size, args, backend='nccl'):
    #combination or train_asr + task code
    args.rank, args.world_size = rank, world_size
    if args.distributed:
        dist.init_process_group(backend=backend, init_method='tcp://localhost:23456',
                                    world_size=world_size, rank=rank)

    ## ///OG 2. Define model and optimizer
    use_cuda = args.use_gpu
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    vocab = Vocab(args.vocab_file, args.rank)
    if args.lm_type == 'MLM':
        vocab.word2index['mask'] = vocab.n_words
        vocab.index2word[vocab.n_words] = 'mask'
        vocab.n_words += 1

    args.vocab_size = vocab.n_words
    from models.lm import make_model
    model = make_model(args)
    optimizer = get_opt(args.opt_type, model, args)
    
    if args.resume_model:
        if rank == 0:
            print("Loading model from {}".format(args.resume_model))
        checkpoint = torch.load(args.resume_model, map_location='cpu')
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if use_cuda:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()   
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0
    
    num_params = 0
    for name, param in model.named_parameters():
        num_params += param.numel()
    if args.rank == 0:
        print("Number of parameters: {}".format(num_params))
    if use_cuda:
        torch.cuda.set_device(args.rank)
        model = model.cuda(args.rank)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.rank])
    
    ##///OG 3. Define vocabulary and data loader
    trainset = TextDataset(vocab, args.train_paths, args)
    train_loader = TextDataLoader(trainset, args.batch_size, args.padding_idx, num_workers=args.load_data_workers, 
                                       distributed=args.distributed, shuffle=True)
    if args.rank == 0:
        print("Finish Loading training files. Number batches: {}".format(len(train_loader)))

    validset = TextDataset(vocab, args.dev_paths, args)
    valid_loader = TextDataLoader(validset, args.batch_size, args.padding_idx, num_workers=args.load_data_workers, 
                                        distributed=False, shuffle=False)
    if args.rank == 0:
        print("Finish Loading dev files. Number batches: {}".format(len(valid_loader)))
    
    criterion = torch.nn.NLLLoss(ignore_index=args.padding_idx, reduction='mean')
    
    ## ///OG 4. Start training iteratively
    best_loss = 100
    # ///OG This is used for noam early stop
    early_stop_patience = args.end_patience
    best_epoch = 0
    # ///OG This is used for noraml adam control
    if args.opt_type == 'normal':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.anneal_lr_ratio, 
                            patience=args.patience, min_lr=args.min_lr)

    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_loader.set_epoch(epoch)
        model.train()
        train_loss, train_acc = run_epoch(epoch, train_loader, model, criterion, args, optimizer, is_train=True)
        model.eval()
        with torch.no_grad():
            valid_loss, valid_acc  = run_epoch(epoch, valid_loader, model, criterion, args, is_train=False)
        
        temp_lr = optimizer.param_groups[0]['lr'] if args.opt_type == 'normal' else optimizer.optimizer.param_groups[0]['lr']
        if args.distributed:
            average_number = torch.Tensor([train_loss, valid_loss]).float().cuda(args.rank)
            torch.distributed.all_reduce(average_number, op=ReduceOp.SUM)
            train_loss, valid_loss = (average_number / args.world_size).cpu().numpy()
        if args.rank == 0:
            print("Epoch {} done, Train Loss: {:.4f}, Train Acc: {:.4f}, Valid Loss: {:.4f}, Valid Acc: {:.4f}, Current LR: {:4e}".format(epoch, train_loss, train_acc, valid_loss, valid_acc, temp_lr), flush=True)
        
        if args.opt_type == 'normal':
            scheduler.step(valid_loss)

        if epoch > args.save_epoch and args.rank == 0:
            output_file=args.exp_dir + '/model.' + str(epoch) + '.mdl'
            checkpoint = {'epoch': epoch, 'optimizer': optimizer.state_dict(),
                            'state_dict': model.state_dict()}
            torch.save(checkpoint, output_file)
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            if args.rank == 0:
                output_file=args.exp_dir + '/best_model.mdl'
                checkpoint = {'epoch': epoch, 'optimizer': optimizer.state_dict(),
                                'state_dict': model.state_dict()}
                torch.save(checkpoint, output_file)
        
        if epoch - best_epoch > early_stop_patience:
            if args.rank == 0:
                print("Early stop since valid_wer doesn't decrease")
            break

def subsequent_mask(size):
    ret = torch.ones(size, size, dtype=torch.uint8)
    return torch.tril(ret, out=ret).unsqueeze(0)

def run_epoch(epoch, dataloader, model, criterion, args, optimizer=None, is_train=True):
    batch_time = util.AverageMeter('Time', ':6.3f')
    losses = util.AverageMeter('Loss', ':.4e')
    accs = util.AverageMeter('Acc', ':.4f')
    token_speed = util.AverageMeter('TokenSpeed', ":.2f")
    progress = util.ProgressMeter(len(dataloader), batch_time, losses, accs, token_speed, prefix="Epoch: [{}]".format(epoch))
    
    end = time.time()
    
    for i, data in enumerate(dataloader):
        start = time.time()
        text, text_sizes, labels = data
        if args.lm_type == 'uniLM':
            tgt, tgt_label = text[:,:-1], text[:,1:]
            tgt_mask = (tgt != args.padding_idx).unsqueeze(1)
            tgt_mask_tril = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask)            
        elif args.lm_type == 'MLM':
            tgt, tgt_label = text, labels
            tgt_mask = (tgt != args.padding_idx).unsqueeze(1)
            tgt_mask_tril = tgt_mask

        tokens = (tgt_label != args.padding_idx).sum().item()
        if args.use_gpu:
            tgt, tgt_mask = tgt.cuda(), tgt_mask.cuda()
            tgt_mask_tril = tgt_mask_tril.cuda()
            tgt_label = tgt_label.cuda()
        
        lm_out = model(tgt, tgt_mask_tril)
        lm_pred = torch.max(lm_out, -1)[1]
        acc = ((lm_pred == tgt_label).masked_fill(tgt_mask.squeeze(1)==0, 0).sum().item()) / tokens
        accs.update(acc, tokens)

        # ///OG loss computation
        loss = criterion(lm_out.view(-1, lm_out.size(-1)), tgt_label.view(-1))
        losses.update(loss.item(), tokens)
        if is_train:
            loss = loss / args.accum_grad
            loss.backward()
            if i % args.accum_grad == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

        batch_time.update(time.time() - end)
        token_speed.update(tokens/(time.time()-start))

        if i % args.print_freq == 0 and args.rank == 0:
            progress.print(i)
    return losses.avg, accs.avg
    
if __name__ == '__main__':
    main()


