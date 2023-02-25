import os
import sys
sys.path.extend(['..', '../..', '.'])

from nebula.misc import get_path, clear_cuda_cache
from nebula import ModelTrainer
from nebula.models.attention import TransformerEncoderModel, TransformerEncoderChunks, TransformerEncoderChunksLM, TransformerEncoderModelLM

import numpy as np
from sklearn.utils import shuffle
import torch

script_path = get_path(type='script')
REPO_ROOT = os.path.join(script_path, '..', '..')

x = os.path.join(REPO_ROOT, r"data\data_filtered\speakeasy_trainset_BPE_50k\speakeasy_VocabSize_50000_maxLen_512_x.npy")
y = os.path.join(REPO_ROOT, r"data\data_filtered\speakeasy_trainset_BPE_50k\speakeasy_y.npy")
# shuffle and take 1000 samples
x = np.load(x)
y = np.load(y)
x, y = shuffle(x, y)
x = x[:1000]
y = y[:1000]

model1 = TransformerEncoderModel(
    vocab_size=50000,
    maxlen=512,
)
model2 = TransformerEncoderChunks(
    vocab_size=50000,
    maxlen=512,
)

modellm1 = TransformerEncoderModelLM(
    vocab_size=50000,
    maxlen=512,
)
modellm2 = TransformerEncoderChunksLM(
    vocab_size=50000,
    maxlen=512,
)

EPOCHS = 3
# ====================

model_trainer = ModelTrainer(
    model=model1,
    batchSize=32,
)
model_trainer.train(x,y, epochs=EPOCHS)
del model1, model_trainer
clear_cuda_cache()

model_trainer = ModelTrainer(
    model=model2,
    batchSize=32,
)
model_trainer.train(x,y, epochs=EPOCHS)
del model2, model_trainer
clear_cuda_cache()

# ====================

model_trainer = ModelTrainer(
    model=modellm1,
    batchSize=32,
)

model_trainer.train(x,y, epochs=EPOCHS)
example = modellm1.pretrain(torch.tensor(x[:32]).long().cuda())
print(example.shape, modellm1.__name__)
del modellm1, model_trainer
clear_cuda_cache()

model_trainer = ModelTrainer(
    model=modellm2,
    batchSize=32,
)

model_trainer.train(x,y, epochs=EPOCHS)
example = modellm2.pretrain(torch.tensor(x[:32]).long().cuda())
print(example.shape, modellm2.__name__)
del modellm2, model_trainer
clear_cuda_cache()
