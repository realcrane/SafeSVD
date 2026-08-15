import torch
import torch.nn as nn
import json
import gc
from tqdm import tqdm
from My_helper import *
import copy
import os
from component.svd_llama import SVD_LlamaAttention, SVD_LlamaMLP
from component.svd_mistral import SVD_MistralAttention, SVD_MistralMLP
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch.nn.functional as F
import torch
import json
import gc
from tqdm import tqdm


def dual_comp(model, block_names):
  for block_name in block_names:
    block = model.get_submodule(block_name)
    