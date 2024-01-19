import os

import sys
sys.path.extend([".", ".."])
import nebula
from nebula import PEDynamicFeatureExtractor
from nebula.models.neurlux import NeurLuxModel, NeurLuxPreprocessor
from nebula.misc import fix_random_seed
fix_random_seed(0)

from torch import Tensor
import torch

# MODEL SETUP
model = NeurLuxModel(
    embedding_dim=256, 
    vocab_size=10000, 
    seq_len=512, 
    num_classes=1
)
neurlux_model_file = os.path.join(os.path.join(os.path.dirname(nebula.__file__)), "objects", "neurlux_whitespace.model")
model.load_state_dict(torch.load(neurlux_model_file, map_location=torch.device('cpu')))
model.eval()

# SAMPLE PE FILE
emulator = PEDynamicFeatureExtractor()
report = emulator.emulate(path=r'C:\windows\system32\cmd.exe')
report = emulator.filter_and_normalize_report(report)

# PREPROCESS
preprocessor = NeurLuxPreprocessor(
    vocab_size=10000,
    max_length=512,
)
vocab_file = os.path.join(os.path.join(os.path.dirname(nebula.__file__)), "objects", "neurlux_whitespace_vocab_10000.json")
preprocessor.load_vocab(vocab_file)
x = preprocessor.preprocess(report)

# PREDICT
x = Tensor(x).long().unsqueeze(0)
print(x.shape)

y = model(x)
print(y)
print(torch.sigmoid(y))
