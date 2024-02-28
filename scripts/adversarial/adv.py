import json
import os
import pathlib
import sys
# shap related imports
import warnings
from collections import Counter

import numpy as np
from numba.core.errors import NumbaDeprecationWarning

warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
import shap

import torch
import logging

logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt

# correct path to repository root
if os.path.basename(os.getcwd()) == "scripts":
    REPOSITORY_ROOT = os.path.join(os.getcwd(), "..")
elif os.path.basename(os.getcwd()) == "adversarial":
    REPOSITORY_ROOT = os.path.join(os.getcwd(), "..", "..")
else:
    REPOSITORY_ROOT = os.getcwd()
sys.path.append(REPOSITORY_ROOT)

from nebula.models.attention import TransformerEncoderOptionalEmbedding
from nebula.preprocessing import load_tokenizer, tokenize_sample
from nebula.misc.plots import plot_shap_values
from scripts.adversarial.tgd import TokenGradientDescent
from nebula.misc import fix_random_seed, compute_score

fix_random_seed(0)

DEVICE = "cpu"


def embed(llm_model, report, device="cpu"):
    src = tokenize_sample(report)
    return llm_model.embed_sample(src).to(device)


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
        "classifier_head": [64],  # classifier ffnn dims
        "layerNorm": False,
        "dropout": 0.3,
        "norm_first": True,
        "skip_embedding": skip_embedding
    }
    state_dict = torch.load(pretrained_model, map_location='cpu')
    llm = TransformerEncoderOptionalEmbedding(**model_config)
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


def compute_adv_exe_from_folder(
        folder: pathlib.Path,
        llm_model,
        verbose: bool = False,
        token_to_use=None,
        token_to_avoid=None,
        steps: int = 10,
        step_size: int = 32
):
    if token_to_use is None:
        token_to_use = list(range(200, 1000))
    if token_to_avoid is None:
        token_to_avoid = list(range(0, 3))
    tgd_attack = TokenGradientDescent(
        model=llm_model,
        tokenizer=load_tokenizer(),
        step_size=step_size,
        steps=steps,
        index_token_to_use=token_to_use,
        token_index_to_avoid=token_to_avoid,
        verbose=verbose,
        device=DEVICE
    )
    for i, report in enumerate(folder.glob("*.json")):
        print(report.name)
        original = tokenize_sample(str(report)).to(DEVICE)
        x_embed = embed(llm_model, str(report), device=DEVICE)
        y = torch.sign(llm_model(x_embed))
        y = y.item()
        to_avoid = list(range(0, 250))  # faking impossible locations to manipulate
        final_x_adv, loss_seq, confidence_seq, x_path = tgd_attack(
            str(report),
            y,
            return_additional_info=True,
            input_index_locations_to_avoid=to_avoid
        )
        # reconstructed = tgd_attack.reconstruct_tokens(original, final_x_adv, to_avoid)
        # normalized_report = tgd_attack.invert_tokenization(reconstructed)
        fig, ax = plt.subplots()

        ax.plot(confidence_seq, label="confidence")
        ax.scatter(range(len(confidence_seq)), confidence_seq)
        ax.set_ylabel("confidence")

        ax.plot(loss_seq, label="loss")
        ax.scatter(range(len(loss_seq)), loss_seq)
        ax.twinx().set_ylabel('loss')

        ax.set_xlabel("steps")
        ax.legend()
        plt.show()


if __name__ == "__main__":
    # MODELS

    model = load_model()
    model.to(DEVICE)
    model_no_embed = load_model(skip_embedding=True)
    model_no_embed.to(DEVICE)

    # REPORTS

    malware_folder = pathlib.Path(os.path.join(REPOSITORY_ROOT, "evaluation", "explanation", "malware"))
    goodware_folder = pathlib.Path(os.path.join(REPOSITORY_ROOT, "evaluation", "explanation", "goodware"))
    baseline_report = os.path.join(REPOSITORY_ROOT, r"emulation", "report_baseline.json")

    x_baseline = tokenize_sample(baseline_report)
    x_embed_baseline = embed(model, baseline_report)

    # ATTACK

    tokens_to_use = list(range(200, 1000))
    tokens_to_avoid = list(range(0, 3))

    compute_adv_exe_from_folder(
        malware_folder,
        model_no_embed,
        verbose=True,
        token_to_use=tokens_to_use,
        token_to_avoid=tokens_to_avoid,
        steps=20,
        step_size=32
    )
