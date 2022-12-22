import numpy as np
import torch
import logging
import os
import pickle
import sys
sys.path.extend(['..', '.', '../..'])
from transformers import ReformerConfig, ReformerModel, ReformerForSequenceClassification
from nebula import ModelInterface
from sklearn.utils import shuffle

outputFolder = r"C:\Users\dtrizna\Code\nebula\tests\speakeasy_2_Modeling\output"
# os.makedirs(outputFolder, exist_ok=True)
logFile = None # "transformer.log"
if logFile:
    logging.basicConfig(filename=os.path.join(outputFolder, logFile), level=logging.WARNING)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


# =============== LOAD DATA
train_limit = 500 # None
x_train = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_trainset\speakeasy_VocabSize_1500_maxLen_2048.npy"
x_train = np.load(x_train)
y_train = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_trainset\speakeasy_VocabSize_1500_maxLen_2048_y.npy"
y_train = np.load(y_train)

if train_limit:
    x_train, y_train = shuffle(x_train, y_train, random_state=42)
    x_train = x_train[:train_limit]
    y_train = y_train[:train_limit]

# some single sample

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

# Convert the numpy arrays to tensors
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

# Create the dataset object
dataset = CustomDataset(x_train, y_train)

# create torch.utils.data.Dataset from x_train and y_train
# trainSet = torch.utils.data.TensorDataset(
#                 torch.from_numpy(x_train).long(), 
#                 torch.from_numpy(y_train).float()
#             )
# trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=64, shuffle=True)
# batch = next(iter(trainLoader))[0]


logging.warning(f" [!] Dataset size: {len(x_train)}")

vocabPath = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_trainset\speakeasy_VocabSize_1500.pkl"
with open(vocabPath, "rb") as f:
    vocab = pickle.load(f)


# =============== DEFINE MODEL & ITS CONFIG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# config example here: 
# https://huggingface.co/docs/transformers/v4.25.1/en/model_doc/reformer#transformers.ReformerConfig
reformerConfig = {
    "vocab_size": len(vocab),
    "attention_head_size": 16,
    "num_buckets": 64,
    "max_position_embeddings": 2048,
    "axial_pos_shape": [32, 64],
    "axial_pos_embds_dim": [32, 32],
    "hidden_size": 64,
    "pad_token_id": vocab["<pad>"],
    "eos_token_id": vocab["<pad>"]
}
configuration = ReformerConfig(**reformerConfig)

#model = ReformerModel(configuration)#.to(device)
#outputs = model(batch)#.to(device))
#last_hidden_states = outputs.last_hidden_state
# output shape: [64, 2048, 128]

model = ReformerForSequenceClassification(configuration)#.to(device)

# with torch.no_grad():
#     outputs = model(batch)#.to(device))
# print(outputs.logits.shape)
#import pdb; pdb.set_trace()
# outputs.logits.shape: [64, 2]
# predicted_class_id = logits.argmax().item()

# ============================
# fitting 
from transformers import Trainer, TrainingArguments

# https://huggingface.co/docs/transformers/v4.25.1/en/main_classes/trainer#transformers.TrainingArguments
# modelConfig = {
#     "device": device,
#     "model": model,
#     "lossFunction": torch.nn.BCEWithLogitsLoss(),
#     "optimizer": torch.optim.Adam(model.parameters(), lr=0.001),
#     "outputFolder": None,
#     "verbosityBatches": 10
# }
args = TrainingArguments(
    output_dir=outputFolder,
    num_train_epochs=3
)

# https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer
trainer = Trainer(model, args=args, train_dataset=dataset, eval_dataset=None)
trainer.train()

## ERRORS OUT