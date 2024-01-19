import logging
logging.basicConfig(level=logging.INFO)

import sys
sys.path.extend([".", ".."])
from nebula import Nebula
from nebula.misc import fix_random_seed
fix_random_seed(0)

# model setup
TOKENIZER = "bpe" # supports: ["bpe", "whitespace"]
nebula = Nebula(
    vocab_size = 50000, # pre-trained only for 50k
    seq_len = 512, # pre-trained only for 512
    tokenizer = TOKENIZER,
)

# 0. EMULATE IT: SKIP IF YOU HAVE JSON REPORT ALREADY
PE = r"C:\Windows\System32\calc.exe"
report = nebula.dynamic_analysis_pe_file(PE)
# 1. PREPROCESS EMULATED JSON REPORT AS ARRAY
x_arr = nebula.preprocess(report)
# 2. PASS THROUGH PYTORCH MODEL
prob = nebula.predict_proba(x_arr)

print(f"\n[!!!] Probability of being malicious: {prob:.3f}")

TRAIN_SAMPLE = False
if TRAIN_SAMPLE:
    # ===================
    # OPTIONAL !!!!!
    # ===================
    # TRAINING
    # ===================
    from torch import cuda
    from torch.optim import AdamW
    from torch.nn import BCEWithLogitsLoss
    from nebula import ModelTrainer

    TIME_BUDGET = 5 # minutes
    device = "cuda" if cuda.is_available() else "cpu"
    model_trainer_config = {
        "device": device,
        "model": nebula.model,
        "loss_function": BCEWithLogitsLoss(),
        "optimizer_class": AdamW,
        "optimizer_config": {"lr": 2.5e-4, "weight_decay": 1e-2},
        "optim_scheduler": None, # supports multiple LR schedulers
        "optim_step_budget": None,
        "outputFolder": "out",
        "batchSize": 96,
        "verbosity_n_batches": 100,
        "clip_grad_norm": 1.0,
        "n_batches_grad_update": 1,
        "time_budget": int(TIME_BUDGET*60) if TIME_BUDGET else None,
    }

    # 3. TRAIN
    model_trainer = ModelTrainer(**model_trainer_config)
    # model_trainer.train(x, y)
