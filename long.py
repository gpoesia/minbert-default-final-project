import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from classifier import BertSentimentClassifier, SentimentDataset
from optimizer import AdamW
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data

# BertModel max seq len is 512: from tokenizer.py,
# PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
#     "bert-base-uncased": 512
# }

"""
1. Find better dataset if possible so don't have to config 
    - for if we don't want to truncate: https://huggingface.co/datasets/scientific_papers/viewer/pubmed/train
    - if we're cool truncating
    
2. Set up model using miniBERT embeds
3. make rouge scoring func

can we change other stuff in classifier? if yea, add a load state dict for model from .pt to line 267

Preproccess
    - First try truncating
    - If doesn't work, split into paragraphs of <= 512 tokens and apply again if need
    
"""