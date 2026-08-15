# =====================================================================
# Helper utilities (from whitener.py / input_collector.py)
# =====================================================================
import os
import copy
import json
import math
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from typing import Dict, Optional, Tuple
 
from VICON.src import utils
from VICON.src.dataset import all_datasets
import VICON.src.models as models
from VICON.src.trainer import Trainer
from MySVD.convert_attn import convert_attn_to_linear

def svd_whitener(Sigma, eps=1e-6):
    """SVD-based whitener for (near) PSD matrices. Returns L, L^{-T}."""
    s, U = torch.linalg.eigh((Sigma + Sigma.T) * 0.5)
    s = torch.clamp(s, min=0)
    sqrt_s = torch.sqrt(s + eps)
    L     = U * sqrt_s.unsqueeze(0)
    LinvT = U * (1.0 / sqrt_s).unsqueeze(0)
    return L, LinvT
