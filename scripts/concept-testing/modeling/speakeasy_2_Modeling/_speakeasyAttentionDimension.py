import torch

import sys
sys.path.extend(['.', '..\\..'])
from nebula.models.attention import TransformerEncoderModel
from nebula.models.reformer import ReformerLM

model = TransformerEncoderModel(
    vocabSize=10000,
    maxLen=2048,
    classifier_head=[1024, 64]
).cuda()

x = torch.randint(0, 10000, (1, 2048)).cuda()
y = model(x).detach().cpu()
print(y.shape)

model2 = ReformerLM(
    vocabSize=10000,
    maxLen=2048,
    dim=512,
    depth=6,
).cuda()

x2 = torch.randint(0, 10000, (1, 2048)).cuda()
y2 = model2(x2).detach().cpu()
print(y2.shape)