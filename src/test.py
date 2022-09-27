# import copy
# import math
# import editdistance as ed
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import os

# from typing import List, Tuple
# import numpy as np
# from torch.nn.parameter import Parameter
# from fairseq.models.hubert import HubertModel, HubertConfig
# from fairseq.checkpoint_utils import load_model_ensemble_and_task
# from fairseq.dataclass.utils import convert_namespace_to_omegaconf

# import logging



# from models.modules.norm import LayerNorm
# from models.modules.attention import MultiHeadedAttention, RelMultiHeadedAttention
# from models.modules.positionff import PositionwiseFeedForward
# from models.modules.embedding import PositionalEncoding, RelativePositionalEncoding, ConvEmbedding, TextEmbedding
# from models.modules.conformer_related import Swish, ConvModule
# from models.blocks import TrfEncoder, ConEncoder
# from models.blocks import ConSAD, ConMAD, TrfSAD, TrfMAD, ConAcExtra, TrfAcExtra
# from utils.ctc_prefix import logzero, logone, CTCPrefixScore
# from utils.loss import LabelSmoothing, KLDivLoss

# import sys


# # models, config, task_config = load_model_ensemble_and_task(['/data/balaji/workdir/cassnat_asr/egs/librispeech/exp/test/hubert_base_ls960.pt'])

# # model = models[0]

# # # print(len(models))

# # print(task_config.dictionaries)

# # print(sys.getsizeof(model.state_dict()), sys.getsizeof(config), sys.getsizeof(task_config))

# # print(model.state_dict().keys())


# import sys
# from types import ModuleType, FunctionType
# from gc import get_referents


# # Custom objects know their class.
# # Function objects seem to know way too much, including modules.
# # Exclude modules as well.
# BLACKLIST = type, ModuleType, FunctionType


# def getsize(obj):
#     """sum size of object & members."""
#     if isinstance(obj, BLACKLIST):
#         raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
#     seen_ids = set()
#     size = 0
#     objects = [obj]
#     while objects:
#         need_referents = []
#         for obj in objects:
#             if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
#                 seen_ids.add(id(obj))
#                 size += sys.getsizeof(obj)
#                 need_referents.append(obj)
#         objects = get_referents(*need_referents)
#     return size

# # print(getsize(model),getsize(config),getsize(task_config))

# # torch.load(model, map_location="cpu")

# # model_state = model["model_state"]

# # print(model_state)

# f = '/data/balaji/workdir/cassnat_asr/egs/librispeech/exp/test/hubert_base_ls960.pt'

# model = torch.load(f, map_location = 'cpu')

# for k in model:
#     print(k)

# # print(model['args'])
# # print(model['optimizer_history'])
# # print(model['task_state'])
# # print(model['last_optimizer_state'])
# # print(model['extra_state'])

# # print(model['model'])
# # print(type(model['args']))
# # cfg = convert_namespace_to_omegaconf(model["args"])

# # m = model['model']

# # count=0

# # for o in m:
# #     if count%2==1:
# #         print(o)
# #     count+=1

# # print(m.keys())

# import json
# import pickle

# # with open('/data/balaji/workdir/cassnat_asr/egs/librispeech/conf/hubert_task_conf.json', 'w') as f:
# #     json.dump(dict(model['task_state']), f, indent=2)

# # print(type(model['task_state']))
# # print(model['task_state'])

# # hubert_data = (model['args'],model['task_state'])

# # f = '/data/balaji/workdir/cassnat_asr/egs/librispeech/conf/hubert_conf'

# # hfile = open(f, 'ab')
      
# # # source, destination
# # pickle.dump(hubert_data, hfile)                     
# # hfile.close()


# # import fairseq.tasks as tasks

# # task = tasks.setup_task(cfg.task)

# # task.load_state_dict(model["task_state"])

# # # print(task.cfg.fine_tuning)

# # print(cfg.model.encoder_embed_dim)
# # # print('\n \n \n \n')
# # # print(task)

# # print(cfg)

# # f = '/data/balaji/workdir/cassnat_asr/egs/librispeech/conf/hubert_conf.json'
# # args = json.load(f)


# # cfg = convert_namespace_to_omegaconf(args)

# checkpoint = model['model']

# # print(checkpoint.keys())

# for param_tensor in checkpoint:
#     # print(param_tensor)
#     print(param_tensor, "\t", checkpoint[param_tensor].size())

# print(type(checkpoint['mask_emb']))

# a = {'a':1,'b':2}

# print(a.keys())

# for k in a:
#     print(k, a[k])

# import soundfile as sf

# wav_path = '/data/balaji/workdir/cassnat_asr/egs/librispeech/data/dev_other/wav.scp'

# new_wav = '/data/balaji/workdir/cassnat_asr/egs/librispeech/data/dev_other/wav_s.scp'


# with open(wav_path, 'r') as fin:
#     for line in fin:
#         cont = line.strip().split(' ')

#         path = cont[-2]

#         audio,_ = sf.read(path)

#         cont.append(str(len(audio)))

#         line_n  = ' '.join(cont)

#         line_n = line_n + "\n"

#         with open(new_wav, 'a') as gin:
#             gin.write(line_n)
import torch
bs = 4
ylens = torch.zeros(1000)
xmax = 500
log_probs = torch.zeros((4,500,1027))
path = torch.zeros((4,1000)).long()
batch_index = torch.arange(bs).type_as(ylens).unsqueeze(1)
seq_index = torch.arange(xmax).type_as(ylens).unsqueeze(1).unsqueeze(2)
log_probs_path = log_probs[seq_index, batch_index, path]

print(log_probs_path)