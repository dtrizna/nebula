import copy
import json
import os
import pathlib
import sys
# shap displays NumbaDeprecationWarning when importing -- supress them
import warnings

import matplotlib.pyplot as plt
import numpy as np
from numba.core.errors import NumbaDeprecationWarning

from scripts.attack.tgd import TokenGradientDescent

warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)

import shap
import torch
from torch import Tensor, sigmoid
import logging

logging.basicConfig(level=logging.INFO)

# correct path to repository root
if os.path.basename(os.getcwd()) == "scripts":
    REPOSITORY_ROOT = os.path.join(os.getcwd(), "..")
elif os.path.basename(os.getcwd()) == "adversarial":
    REPOSITORY_ROOT = os.path.join(os.getcwd(), "..", "..")
else:
    REPOSITORY_ROOT = os.getcwd()
sys.path.append(REPOSITORY_ROOT)

from nebula import PEDynamicFeatureExtractor
from nebula.models.attention import TransformerEncoderChunksOptionalEmbedding
from nebula.preprocessing import JSONTokenizerNaive

from nebula.misc import fix_random_seed

fix_random_seed(0)

DEVICE = "cpu"

def compute_score(llm, x, verbose=True):
    logit = llm(x)
    prob = sigmoid(logit)
    if verbose:
        print(f"\n[!!!] Probability of being malicious: {prob.item():.3f} | Logit: {logit.item():.3f}")
    return prob.item()


def load_tokenizer():
    with open(os.path.join(REPOSITORY_ROOT, "nebula", "objects",
                           "speakeasy_whitespace_50000_vocab.json")) as f:
        vocab = json.load(f)

    tokenizer = JSONTokenizerNaive(
        vocab_size=len(vocab),
        seq_len=512,
        vocab=vocab
    )
    return tokenizer


def tokenize_sample(report_path, encode=True):
    extractor = PEDynamicFeatureExtractor()
    filtered_report = extractor.filter_and_normalize_report(report_path)
    tokenizer = load_tokenizer()
    tokenized_report = tokenizer.tokenize(filtered_report)
    if encode:
        encoded_report = tokenizer.encode(tokenized_report, pad=True, tokenize=False)
        x = Tensor(encoded_report).long()
        return x
    return tokenized_report


def embed(llm_model, report, device="cpu"):
    src = tokenize_sample(report)
    s = llm_model.split(src)
    s = s.to(device)
    e = llm_model.embed(s)
    return e


def plot_shap_values(shap_values: np.ndarray, name: str):
    shap_values = shap_values.mean(axis=2)
    pos_idx = shap_values[0] >= 0
    neg_index = shap_values[0] < 0
    pos_shap = copy.deepcopy(shap_values)[0]
    pos_shap[neg_index] = 0
    neg_shap = copy.deepcopy(shap_values)[0]
    neg_shap[pos_idx] = 0
    x = range(512)
    plt.bar(x, pos_shap)
    plt.bar(x, neg_shap)
    plt.title(name)
    plt.show()


def load_model(skip_embedding=False):
    pretrained_model = os.path.join(REPOSITORY_ROOT, "nebula", "objects", "speakeasy_whitespace_50000_torch.model")
    with open(os.path.join(REPOSITORY_ROOT, "nebula", "objects", "speakeasy_whitespace_50000_vocab.json")) as f:
        vocab = json.load(f)
    model_config = {
        "vocab_size": len(vocab),
        "maxlen": 512,
        "chunk_size": 64,  # input splitting to chunks
        "dModel": 64,  # embedding & transformer dimension
        "nHeads": 8,  # number of heads in nn.MultiheadAttention
        "dHidden": 256,  # dimension of the feedforward network model in nn.TransformerEncoder
        "nLayers": 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        "numClasses": 1,  # binary classification
        "hiddenNeurons": [64],  # classifier ffnn dims
        "layerNorm": False,
        "dropout": 0.3,
        "mean_over_sequence": False,
        "norm_first": True,
        "skip_embedding": skip_embedding
    }
    state_dict = torch.load(pretrained_model, map_location='cpu')
    llm = TransformerEncoderChunksOptionalEmbedding(**model_config)
    llm.load_state_dict(state_dict)
    llm.eval()
    return llm


def analyse_folder(folder: pathlib.Path, llm_model, embed_baseline, name, plot=True, verbose=True):
    filepaths = []
    explanations_arr = []
    for i, report in enumerate(folder.glob("*.json")):
        print(report.name)
        x_embed = embed(llm_model, str(report))
        prob = compute_score(llm_model, x_embed, verbose=verbose)
        explainer = shap.GradientExplainer(model_no_embed, data=embed_baseline)
        explanations = explainer.shap_values(x_embed, nsamples=50)
        if plot:
            plot_shap_values(explanations, f"{name}_{i} : {prob:.3f}")
        explanations_arr.append(explanations)
        filepaths.append(report.name)
    return filepaths, explanations_arr


def compute_adv_exe_from_folder(folder: pathlib.Path, llm_model, verbose: bool = False):
    token_to_use = list(range(200, 1000))
    tgd_attack = TokenGradientDescent(model=llm_model, tokenizer=load_tokenizer(), step_size=32, steps=10,
                                      index_token_to_use=token_to_use, token_index_to_avoid=[0, 2], verbose=verbose,
                                      device=DEVICE)
    for i, report in enumerate(folder.glob("*.json")):
        print(report.name)
        x_embed = embed(llm_model, str(report), device=DEVICE)
        y = torch.sign(llm_model(x_embed))
        y = y.item()
        to_avoid = list(range(0, 250)) # faking impossible locations to manipulate
        final_x_adv, loss_seq, confidence_seq, x_path = tgd_attack(str(report), y, return_additional_info=True,
                                                                   input_index_locations_to_avoid=to_avoid)
        plt.plot(confidence_seq, label="conf")
        plt.plot(loss_seq, label="loss")
        plt.legend()
        plt.show()


model = load_model()
model_no_embed = load_model(skip_embedding=True)
model_no_embed = model_no_embed.to(device=DEVICE)
malware_folder = pathlib.Path(os.path.join(REPOSITORY_ROOT, "evaluation", "explanation", "malware"))
goodware_folder = pathlib.Path(os.path.join(REPOSITORY_ROOT, "evaluation", "explanation", "goodware"))

baseline_report = os.path.join(REPOSITORY_ROOT, r"emulation", "report_baseline.json")
x_baseline = tokenize_sample(baseline_report)
x_embed_baseline = embed(model, baseline_report)

compute_adv_exe_from_folder(malware_folder, model_no_embed, verbose=True)

# print(f"\n[!!!] MALWARE FOLDER")
# filepaths, explanations = analyse_folder(malware_folder, model_no_embed, x_embed_baseline, "malware")
#
# print(f"\n[!!!] GOODWARE FOLDER")
# _ = analyse_folder(goodware_folder, model_no_embed, x_embed_baseline, "goodware")
