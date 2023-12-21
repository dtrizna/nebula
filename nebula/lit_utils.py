import lightning as L
from lightning.pytorch.callbacks import TQDMProgressBar

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


class LitProgressBar(TQDMProgressBar):
    # to preserve progress bar after each epoch
    def on_train_epoch_end(self, *args, **kwargs):
        super().on_train_epoch_end(*args, **kwargs)
        print()


class LitPyTorchModel(L.LightningModule):
    def __init__(
            self,
            model,
            criterion=nn.BCEWithLogitsLoss(),
            # optimizer params
            optimizer="Adam",
            learning_rate=1e-3,
            weight_decay=1e-5,
            scheduler=None,
            # should be equal to number of batches processed during whole training 
            # so given the epochs: epoch * len(dataloader)
            scheduler_step_budget=100,
            # other params
            classes=2,
            f1_threshold=0.5
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_step_budget = scheduler_step_budget

        self.save_hyperparameters(ignore=["model", "criterion"])

        assert classes > 1, "classes must be greater than 1"
        if classes == 2:
            self.train_f1 = torchmetrics.F1Score(task="binary", threshold=f1_threshold)
            self.val_f1 = torchmetrics.F1Score(task="binary", threshold=f1_threshold)
            self.test_f1 = torchmetrics.F1Score(task="binary", threshold=f1_threshold)
        else:
            self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=classes, average='macro')
            self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=classes, average='macro',)
            self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=classes, average='macro')
        
        assert self.optimizer is not None, "optimizer must be specified"
        assert self.optimizer in ["Adam", "AdamW", "SGD"], "optimizer must be one of Adam, AdamW, SGD"

        assert self.scheduler in ["TriangularLR", "CosineAnnealingLR", "StepLR", None], \
            "scheduler must be one of TriangularLR, CosineAnnealingLR, StepLR, None"
        

    def configure_optimizers(self):
        if self.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )

        if self.scheduler is None:
            return self.optimizer
        
        if self.scheduler == "TriangularLR":
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=self.learning_rate,
                max_lr=self.learning_rate * 10,
                # half training rises, half decrease
                step_size_up=self.scheduler_step_budget // 2,
                cycle_momentum=False,
            )
        elif self.scheduler == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.scheduler_step_budget,
                eta_min=self.learning_rate / 10
            )
        elif self.scheduler == "StepLR":
            nr_of_steps = 5
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.scheduler_step_budget // nr_of_steps,
                gamma=0.5,
                verbose=True
            )

        return [self.optimizer], [self.scheduler]

    def forward(self, x):
        return self.model(x)
    
    def _shared_step(self, batch, threshold=0.5):
        # used by training_step, validation_step and test_step
        x, y_true = batch
        logits = self(x)
        loss = self.criterion(logits, y_true)
        y_pred = torch.sigmoid(logits)
        return loss, y_true, y_pred
    
    def training_step(self, batch, batch_idx):
        loss, y_true, y_pred = self._shared_step(batch)
        self.log('train_loss', loss, prog_bar=True) # logging loss for this mini-batch
        self.train_f1(y_pred, y_true)
        # counts: Counter(pred_labels.squeeze().cpu().tolist()).most_common()
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, y_true, y_pred = self._shared_step(batch)        
        self.log('val_loss', loss, prog_bar=True)
        self.val_f1(y_pred, y_true)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        loss, y_true, y_pred = self._shared_step(batch)
        self.log('test_loss', loss, prog_bar=True)
        self.test_f1(y_pred, y_true)
        self.log('test_f1', self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
