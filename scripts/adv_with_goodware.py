import os
import sys
import json
import pathlib
from typing import List
from collections import Counter

# shap related imports
import warnings
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
from scripts.attack.tgd import TokenGradientDescent
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
        "hiddenNeurons": [64],  # classifier ffnn dims
        "layerNorm": False,
        "dropout": 0.3,
        "mean_over_sequence": False,
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
        token_to_use: List = list(range(200, 1000)),
        token_to_avoid = list(range(0, 3)),
        steps: int = 10,
        step_size: int = 32
):
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
        reconstructed = tgd_attack.reconstruct_tokens(original, final_x_adv, to_avoid)
        normalized_report = tgd_attack.invert_tokenization(reconstructed)
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

def load_goodware_reports(folder, nr=200):
    reports = []
    for i, report in enumerate(folder.glob("*.json")):
        if i > nr:
            break
        with open(report) as f:
            report = json.load(f)
        reports.append(report)
    return reports

def report_token_counter(reports):
    token_counter = Counter()
    for report in reports:
        tokens = tokenize_sample(report)
        token_counter.update(*tokens.tolist())
    return token_counter

if __name__ == "__main__":
    
    # MODELS
    logging.info(" [*] Loading models...")
    model = load_model()
    model.to(DEVICE)
    model_no_embed = load_model(skip_embedding=True)
    model_no_embed.to(DEVICE)
    logging.info(" [!] Models loaded.")

    # REPORTS
    logging.info(" [*] Loading reports...")
    malware_folder = pathlib.Path(os.path.join(REPOSITORY_ROOT, "evaluation", "explanation", "malware"))
    goodware_folder = pathlib.Path(os.path.join(REPOSITORY_ROOT, "evaluation", "explanation", "goodware"))
    baseline_report = os.path.join(REPOSITORY_ROOT, r"emulation", "report_baseline.json")
    x_baseline = tokenize_sample(baseline_report)
    x_embed_baseline = embed(model, baseline_report)
    logging.info(" [!] Reports loaded.")

    # ATTACK
    logging.info(" [*] Initializing an attack...")
    goodware_reports = load_goodware_reports(goodware_folder, nr=100)
    goodware_token_counter = report_token_counter(goodware_reports)
    
    TOP_N_TOKENS = 300
    SKIP_N_TOKENS = 100
    #tokens_to_use = list(range(200, 1000))
    #tokens_to_avoid = 
    n_frequent_tokens = [token for token, _ in goodware_token_counter.most_common(TOP_N_TOKENS)]
    tokens_to_use = n_frequent_tokens[SKIP_N_TOKENS:]
    tokens_to_avoid = n_frequent_tokens[:SKIP_N_TOKENS]
    # [0,1,2] are <pad>, <unk>, <mask> -- join with tokens_to_avoid
    tokens_to_avoid = list(set(list(range(0, 3)) + tokens_to_avoid))
    
    compute_adv_exe_from_folder(
            malware_folder,
            model_no_embed,
            verbose = True,
            token_to_use = tokens_to_use,
            token_to_avoid = tokens_to_avoid,
            steps = 20,
            step_size = 32
    )