import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import math
import sys
import os

file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{file_dir}")
from APEX_models import AMP_model
from utils import *


pathogen_list = [
    "A. baumannii ATCC 19606",
    "E. coli ATCC 11775",
    "E. coli AIG221",
    "E. coli AIG222",
    "K. pneumoniae ATCC 13883",
    "P. aeruginosa PA01",
    "P. aeruginosa PA14",
    "S. aureus ATCC 12600",
    "S. aureus (ATCC BAA-1556) - MRSA",
    "vancomycin-resistant E. faecalis ATCC 700802",
    "vancomycin-resistant E. faecium ATCC 700221",
]

max_len = 52  # maximum seq length; 52 = start character + maximum peptide length (50 aa) + end character; longer peptides will be truncated
word2idx, idx2word = make_vocab()  # make amino acid vocabulary
# emb, AAindex_dict = AAindex('./aaindex1.csv', word2idx) #make amino acid embeddings


# Pretrained APEX models (8 in total) -- lazily loaded on first predict_APEX()
# call, onto CPU, instead of eagerly at import time. Import time is also
# multiprocessing "spawn" worker re-import time (every spawned worker
# re-imports this module transitively via config.py), so eager+GPU loading
# here meant every persistent pool worker loaded all 8 models onto its GPU
# whether or not it ever calls predict_APEX -- redundant across the whole
# pool and the direct cause of a CUDA OOM at milestone 40 of a real run.
# map_location="cpu": predict_APEX already calls .cuda() on each model
# right before use, so no GPU memory is touched until actually needed.
APEX_models: list | None = None
file_dir = os.path.dirname(os.path.abspath(__file__))
trained_models_dir = os.path.join(file_dir, "..", "apex", "trained_models")


def _load_apex_models() -> list:
    global APEX_models
    if APEX_models is None:
        models = []
        for a_model in glob.glob(os.path.join(trained_models_dir, "*")):
            # weights_only=False: these are pickled full AMP_model objects (not
            # just state dicts), from this repo's own trained_models/ -- trusted
            # source.
            model = torch.load(a_model, map_location="cpu", weights_only=False)
            model.eval()
            models.append(model)
        APEX_models = models
    return APEX_models


batch_size = 3000  # change according to your GPU memory


# Use pretrained APEX models to predict species-specific antimicrobial activity (i.e., minimum inhibitory concentration [MIC]; unit: uM)
# 8 pretrained APEX models are provided, and predictions are averaged
def predict_APEX(seq_list):
    apex_models = _load_apex_models()
    for ensemble_id in range(len(apex_models)):
        AMP_model = apex_models[ensemble_id].cuda().eval()

        data_len = len(seq_list)
        for i in range(int(math.ceil(data_len / float(batch_size)))):
            seq_batch = seq_list[i * batch_size : (i + 1) * batch_size]
            seq_rep = onehot_encoding(seq_batch, max_len, word2idx)  # make input
            X_seq = torch.LongTensor(seq_rep).cuda()

            AMP_pred_batch = AMP_model(X_seq).cpu().detach().numpy()  # make predictions
            AMP_pred_batch = (
                10 ** (6 - AMP_pred_batch)
            )  # transform back to MICs; When training the APEX models, MICs were transformed by: -np.log10(MICs/float(1000000))

            if i == 0:
                AMP_pred = AMP_pred_batch
            else:
                AMP_pred = np.vstack([AMP_pred, AMP_pred_batch])

        # sum up the predictions made by different APEX models
        if ensemble_id == 0:
            AMP_sum = AMP_pred
        else:
            AMP_sum += AMP_pred

    AMP_pred = AMP_sum / float(len(apex_models))  # average the predictions

    return AMP_pred


apex_wrapper = predict_APEX
apex_best_wrapper = lambda x: np.min(apex_wrapper(x), axis=1)
apex_mean_wrapper = lambda x: np.mean(apex_wrapper(x), axis=1)
apex_worst_wrapper = lambda x: np.max(apex_wrapper(x), axis=1)
