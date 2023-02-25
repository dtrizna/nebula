import os
import sys
REPO_ROOT = r"../../.."
sys.path.append(REPO_ROOT)
import numpy as np
from tqdm import tqdm
from nebula.models.neurlux.preprocessor import NeurLuxPreprocessor
from nebula.models.neurlux import NeurLuxModel
from nebula import ModelTrainer

from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss

EMBEDDING_DIM = 256
VOCAB_SIZE = 10000
#MAX_LEN = 87000
MAX_LEN = 2048

# =============== READ REPORTS
print("reading reports...")
report_folders = os.listdir(os.path.join(REPO_ROOT, r"data\data_raw\windows_emulation_trainset"))
LIMIT = 500
reports = []
y = []
for report_folder in report_folders:
    if report_folder.endswith("7z"):
        continue
    report_folder = os.path.join(REPO_ROOT, r"data\data_raw\windows_emulation_trainset", report_folder)
    print("reading folder:", report_folder)
    files = os.listdir(report_folder)[0:LIMIT]
    for file in tqdm(files):
        # reading report
        report_file = os.path.join(report_folder, file)
        with open(report_file) as f:
            report = f.read()
        reports.append(report)
        
        # adding label
        if report_folder.endswith("report_clean"):
            y.append(0)
        else:
            y.append(1)

# =============== PREPROCESSS REPORTS
print("training tokenizer...")
p = NeurLuxPreprocessor(
    vocab_size=VOCAB_SIZE,
    max_length=MAX_LEN
)
p.train(reports)
X = p.preprocess_sequence(reports)
X = p.pad_sequence(X)
print("padded X:", X.shape)
y = np.array(y, dtype=np.float16)
print("y:", y.shape, y[0:10])

loss = BCEWithLogitsLoss()
optimizerConfig = {"lr": 2.5e-4}
model = NeurLuxModel(
    embedding_dim=EMBEDDING_DIM,
    vocab_size=VOCAB_SIZE,
    max_len=MAX_LEN
)
device = "cuda"
trainer = ModelTrainer(
    model=model,
    device=device,
    lossFunction=loss,
    optimizerConfig=optimizerConfig,
    optimizerClass=AdamW,
    batchSize=16
)
trainer.train(X, y, epochs=1)