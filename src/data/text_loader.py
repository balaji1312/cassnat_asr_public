#!/usr/bin/env python3
# 2020 Ruchao Fan

import torch
import random
import numpy as np

from torch.utils.data import Dataset, DataLoader

class SingleSet(object):
    def __init__(self, vocab, data_path, max_len, rank):
        self.name = data_path['name']
        self.vocab = vocab
        self.max_len = max_len
        self.rank = rank
        text_path = data_path['text_path']
        self.items = self._load_label(text_path)

    def get_len(self):
        return len(self.items)

    def _load_label(self, lab_path):
        #return list of tokens
        all_text = []
        with open(lab_path, 'r') as fin:
            line = fin.readline()
            while line:
                line = line.strip().split(' ')
                if len(line) > self.max_len:
                    line = line[:self.max_len]
                all_text.append([self.vocab.word2index[word] if word in self.vocab.word2index else
                        self.vocab.word2index['unk'] for word in line])
                line = fin.readline()
        if self.rank == 0:
            print("Reading %d lines from %s" % (len(all_text), lab_path))
        return all_text

class TextDataset(Dataset):
    def __init__(self, vocab, data_paths, args):
        #similar to speechdataset, but for textstreams
        self.vocab = vocab
        self.max_len = args.max_len
        self.rank = args.rank
        self.type = args.lm_type
        self.data_streams = self._load_streams(data_paths)
        self.data_stream_sizes = [i.get_len() for i in self.data_streams]
        self.data_stream_cum_sizes = [self.data_stream_sizes[0]]
        for i in range(1, len(self.data_stream_sizes)):
            self.data_stream_cum_sizes.append(self.data_stream_cum_sizes[-1] + self.data_stream_sizes[i])

    def _load_streams(self, data_paths):
        data_streams = []
        for i in range(len(data_paths)):
            stream = SingleSet(self.vocab, data_paths[i], self.max_len, self.rank)
            data_streams.append(stream)
        return data_streams
                    
    def __getitem__(self, idx):
        stream_idx = -1
        for i in range(len(self.data_stream_cum_sizes)):
            if idx < self.data_stream_cum_sizes[i]:
                stream_idx = i
                break
        if stream_idx == -1:
            raise Exception('index exceed.')
        if stream_idx == 0:
            internal_idx = idx
        else:
            internal_idx = idx - self.data_stream_cum_sizes[stream_idx-1]
        
        text = self.data_streams[stream_idx].items[internal_idx]
        #if masked lm we need to call masking func
        if self.type == 'MLM':
            text, label = self.random_mask(text)
            label.insert(0, 0)
            label.append(0)
        elif self.type == "uniLM":
            label = None
        text.insert(0, self.vocab.word2index['sos'])
        text.append(self.vocab.word2index['eos'])

        return text, label

    def random_mask(self, text):
        #similar to bert masking
        tgt_input, label = [], []
        for i, token in enumerate(text):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                #///OG 80% randomly change token to mask token
                if prob < 0.8:
                    tgt_input.append(self.vocab.word2index['mask'])

                #///OG 10% randomly change token to random token
                elif prob < 0.9:
                    tgt_input.append(random.randrange(4, self.vocab.n_words-1))

                #///OG 10% randomly change token to current token
                else:
                    tgt_input.append(text[i])

                label.append(text[i])

            else:
                tgt_input.append(text[i])
                label.append(0)

        return tgt_input, label

    def __len__(self):
        return sum(self.data_stream_sizes)

class TextDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, padding_idx=-1, distributed=False, shuffle=False, num_workers=0, timeout=1000):
        #define sampling type, collate and set epoch
        self.padding_idx = padding_idx
        if distributed:
            base_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            self.base_sampler = base_sampler
        elif shuffle:
            base_sampler = torch.utils.data.RandomSampler(dataset)
        else:
            base_sampler = torch.utils.data.SequentialSampler(dataset)
        
        sampler = torch.utils.data.BatchSampler(base_sampler, batch_size, False)
        super(TextDataLoader, self).__init__(dataset, 
                                                batch_sampler=sampler,
                                                num_workers=num_workers,
                                                collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        #similar to just the text part in superviseloader
        text_max_length = max(len(x[0]) for x in batch)
        batch_size = len(batch)
        texts = torch.full([batch_size, text_max_length], self.padding_idx)
        labels = torch.full([batch_size, text_max_length], self.padding_idx)
        text_sizes = torch.zeros(batch_size)

        for x in range(batch_size):
            text, label = batch[x]
            text_length = len(text)
            texts[x].narrow(0, 0, text_length).copy_(torch.Tensor(text))
            if label is not None:
                labels[x].narrow(0, 0, text_length).copy_(torch.Tensor(label))
            text_sizes[x] = text_length - 2 #///OG substract sos and eos
        return texts.long(), text_sizes.long(), labels.long()

    def set_epoch(self, epoch):
        self.base_sampler.set_epoch(epoch)

