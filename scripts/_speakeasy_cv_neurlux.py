import numpy as np
import logging
import os
import json
import time
from tqdm import tqdm
from sklearn.utils import shuffle

import sys
sys.path.extend(['..', '.'])
from nebula import ModelTrainer
from nebula.models.neurlux import NeurLuxModel
from nebula.models.neurlux.preprocessor import NeurLuxPreprocessor
from nebula.evaluation import CrossValidation
from nebula.misc import get_path, set_random_seed,clear_cuda_cache

from torch import cuda
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

def read_reports_from_root(root_repo, hashList, limit=None):
    reports_train = []
    for root, dirs, _ in os.walk(root_repo):
        for d in dirs:
            logging.warning(f"[*] Reading {d}...")
            files = os.listdir(os.path.join(root, d))[:limit]
            for file in tqdm(files):
                if file.endswith(".err"):
                    continue
                hhash = file.replace(".json", "")
                if hhash in hashList:
                    report_file = os.path.join(root, d, file)
                    with open(report_file) as f:
                        report = f.read()
                    reports_train.append(report)
    return reports_train

# supress UndefinedMetricWarning, which appears when a batch has only one class
import warnings
warnings.filterwarnings("ignore")
clear_cuda_cache()

runType = f"vocab50k_len512_ep6"
SCRIPT_PATH = get_path(type="script")
REPO_ROOT = os.path.join(SCRIPT_PATH, "..")
outputFolder = os.path.join(REPO_ROOT, "evaluation", "crossValidation", runType)
os.makedirs(outputFolder, exist_ok=True)

# loging setup
logFile = f"CrossValidationRun_{int(time.time())}.log"
logging.basicConfig(
    filename=os.path.join(outputFolder, logFile),
    level=logging.WARNING,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p'
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


# ============== REPORTING CONFIG
modelClass = NeurLuxModel
maxLens = [512, 2048]#, 4096]
for maxLen in maxLens:
    timestamp = int(time.time())
    runName = f"neurlux_BPE_maxLen_{maxLen}_6_epochs_{timestamp}"
    outputFolder = os.path.join(REPO_ROOT, "evaluation", "crossValidation", runType)
    outputFolder = os.path.join(outputFolder, runName)
    os.makedirs(outputFolder, exist_ok=True)

    # ============== Cross-Valiation CONFIG
    random_state = 1763
    set_random_seed(random_state)

    # transform above variables to dict
    run_config = {
        "train_limit": None,
        "nFolds": 3,
        "epochs": 6,
        "fprValues": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
        "rest": 30,
        "batchSize": 64,
        "optim_step_size": 3000,
        "random_state": random_state,
        "chunk_size": 64,
        "verbosity_batches": 100,
        "maxLen": maxLen,
        "vocab_size": 50000,
    }
    with open(os.path.join(outputFolder, f"run_config.json"), "w") as f:
        json.dump(run_config, f, indent=4)

    # ============== PreProcessing Reports NeurLux style
    # logging.warning("[*] Reading trainset reports...")
    # yHashes_train = json.load(open(os.path.join(REPO_ROOT, r"data\data_filtered\speakeasy_trainset_BPE\speakeasy_yHashes.json")))
    # root_train = os.path.join(REPO_ROOT, r"data\data_raw\windows_emulation_trainset")
    # reports_train = read_reports_from_root(root_train, yHashes_train, limit=run_config['train_limit'])
    # logging.warning("[!] Got {} training reports!".format(len(reports_train)))
    
    # logging.warning("[*] Reading testset reports...")
    # yHashes_test = json.load(open(os.path.join(REPO_ROOT, r"data\data_filtered\speakeasy_testset_BPE\speakeasy_yHashes.json")))
    # root_test = os.path.join(REPO_ROOT, r"data\data_raw\windows_emulation_testset")
    # reports_test = read_reports_from_root(root_test, yHashes_test, limit=run_config['train_limit'])
    # logging.warning("[!] Got {} test reports!".format(len(reports_test)))

    # logging.warning("[*] Training tokenizer...")
    # p = NeurLuxPreprocessor(
    #     vocab_size=run_config['vocab_size'],
    #     max_length=run_config['maxLen'],
    # )
    # p.train(reports_train)
    # x_train = p.preprocess_sequence(reports_train)
    # x_train = p.pad_sequence(x_train)
    # x_test = p.preprocess_sequence(reports_test)
    # x_test = p.pad_sequence(x_test)
    
    # using BPE arrays
    x_train = np.load(os.path.join(REPO_ROOT, rf"data\data_filtered\speakeasy_trainset_BPE\speakeasy_VocabSize_50000_maxLen_{maxLen}_x.npy"))
    x_test = np.load(os.path.join(REPO_ROOT, rf"data\data_filtered\speakeasy_testset_BPE\speakeasy_VocabSize_50000_maxLen_{maxLen}_x.npy"))
    y_train = np.load(os.path.join(REPO_ROOT, r"data\data_filtered\speakeasy_trainset_BPE\speakeasy_y.npy"))
    y_test = np.load(os.path.join(REPO_ROOT, r"data\data_filtered\speakeasy_testset_BPE\speakeasy_y.npy"))

    logging.warning("[!] Shape of padded training X and y: {} | {}".format(x_train.shape, y_train.shape))
    logging.warning("[!] Shape of padded test X and y: {} | {}".format(x_test.shape, y_test.shape))

    if run_config['train_limit']:
        x_train, y_train = shuffle(x_train, y_train, random_state=random_state)
        x_train = x_train[:run_config['train_limit']]
        y_train = y_train[:run_config['train_limit']]

    # dump train x and y
    np.save(os.path.join(outputFolder, f"x_train_limit_{run_config['train_limit']}.npy"), x_train)
    np.save(os.path.join(outputFolder, f"y_train_limit_{run_config['train_limit']}.npy"), y_train)
    # dump test x and y
    np.save(os.path.join(outputFolder, f"x_test.npy"), x_test)
    np.save(os.path.join(outputFolder, f"y_test.npy"), y_test)

    # dump vocab dict from preprocessor
    # with open(os.path.join(outputFolder, f"vocab.json"), "w") as f:
    #     json.dump(p.vocab, f, indent=4)
    # vocab = p.vocab
    with open(os.path.join(SCRIPT_PATH, r"..\nebula\objects\speakeasy_BPE_50000_vocab.json")) as f:
        vocab = json.load(f)
    vocabSize = len(vocab)
    logging.warning(f" [!] Loaded data and vocab. X train size: {x_train.shape}, vocab size: {vocabSize}")

    # =============== MODEL CONFIG
    modelArch = {
        "embedding_dim": 256,
        "vocab_size": vocabSize,
        "max_len": run_config["maxLen"],
    }
    # dump model config as json
    with open(os.path.join(outputFolder, f"model_config.json"), "w") as f:
        json.dump(modelArch, f, indent=4)

    device = "cuda" if cuda.is_available() else "cpu"
    modelInterfaceClass = ModelTrainer
    modelInterfaceConfig = {
        "device": device,
        "model": None, # will be set later within CrossValidation class
        "lossFunction": BCEWithLogitsLoss(),
        "optimizerClass": Adam,
        "optimizerConfig": {"lr": 2.5e-4},
        "optimSchedulerClass": "step",
        "optim_step_size": run_config["optim_step_size"],
        "outputFolder": None, # will be set later
        "batchSize": run_config["batchSize"],
        "verbosityBatches": run_config["verbosity_batches"],
    }

    # =============== CROSS-VALIDATION LOOPS
    logging.warning(f" [!] Using device: {device} | Dataset size: {len(x_train)}")

    metricFilePrefix = f"" 
    cv = CrossValidation(modelInterfaceClass, modelInterfaceConfig, modelClass, modelArch, outputFolder)
    cv.run_folds(
        x_train, 
        y_train, 
        epochs=run_config["epochs"], 
        folds=run_config["nFolds"], 
        fprValues=run_config["fprValues"], 
        random_state=random_state
    )
    cv.dump_metrics(prefix=metricFilePrefix)
    cv.log_avg_metrics()

    if run_config["rest"]:
        time.sleep(run_config["rest"])
