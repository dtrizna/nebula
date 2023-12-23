import os
import pickle
import numpy as np
from shutil import copyfile
from typing import Union, Any

from sklearn.metrics import roc_curve

import lightning as L
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from torch import nn
from torch import cat, sigmoid, optim
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
import torchmetrics


class LitProgressBar(TQDMProgressBar):
    # to preserve progress bar after each epoch
    def on_train_epoch_end(self, *args, **kwargs):
        super().on_train_epoch_end(*args, **kwargs)
        print()


class PyTorchLightningModel(L.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            learning_rate: float,
            fpr: float = 1e-4,
            scheduler: Union[None, str] = None,
            scheduler_step_budget: Union[None, int] = None
    ):
        # NOTE: scheduler_step_budget = epochs * len(train_loader)
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.fpr = fpr
        self.loss = BCEWithLogitsLoss()

        assert scheduler in [None, "onecycle", "cosine"], "Scheduler must be onecycle or cosine"
        if scheduler is not None:
            assert isinstance(scheduler_step_budget, int), "Scheduler step budget must be provided"
            print(f"[!] Scheduler: {scheduler} | Scheduler step budget: {scheduler_step_budget}")
        self.scheduler = scheduler
        self.scheduler_step_budget = scheduler_step_budget

        self.train_acc = torchmetrics.Accuracy(task='binary')
        self.train_f1 = torchmetrics.F1Score(task='binary', average='macro')
        self.train_auc = torchmetrics.AUROC(task='binary')
        self.train_tpr = self.get_tpr_at_fpr

        self.val_acc = torchmetrics.Accuracy(task='binary')
        self.val_f1 = torchmetrics.F1Score(task='binary', average='macro')
        self.val_auc = torchmetrics.AUROC(task='binary')
        self.val_tpr = self.get_tpr_at_fpr

        self.test_acc = torchmetrics.Accuracy(task='binary')
        self.test_f1 = torchmetrics.F1Score(task='binary', average='macro')
        self.test_auc = torchmetrics.AUROC(task='binary')
        self.test_tpr = self.get_tpr_at_fpr

        self.save_hyperparameters(ignore=["model"])
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        if self.scheduler is None:
            return optimizer

        if self.scheduler == "onecycle":
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=self.scheduler_step_budget
            )
        if self.scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_step_budget
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "step", # default: epoch
                "frequency": 1
            }
        }


    def get_tpr_at_fpr(self, predicted_logits, true_labels, fprNeeded=1e-4):
        predicted_probs = sigmoid(predicted_logits).cpu().detach().numpy()
        true_labels = true_labels.cpu().detach().numpy()
        fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
        if all(np.isnan(fpr)):
            return np.nan#, np.nan
        else:
            tpr_at_fpr = tpr[fpr <= fprNeeded][-1]
            #threshold_at_fpr = thresholds[fpr <= fprNeeded][-1]
            return tpr_at_fpr#, threshold_at_fpr

    def forward(self, x):
        return self.model(x)
    
    def _shared_step(self, batch):
        # used by training_step, validation_step and test_step
        x, y = batch
        y = y.unsqueeze(1)
        logits = self(x)
        loss = self.loss(logits, y)
        
        return loss, y, logits
    
    def training_step(self, batch, batch_idx):
        # NOTE: keep batch_idx -- lightning needs it
        loss, y, logits = self._shared_step(batch)
        self.log('train_loss', loss, prog_bar=True) # logging loss for this mini-batch
        self.train_acc(logits, y)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.train_f1(logits, y)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.train_auc(logits, y)
        self.log('train_auc', self.train_auc, on_step=False, on_epoch=True, prog_bar=True)
        train_tpr = self.train_tpr(logits, y, fprNeeded=self.fpr)
        self.log('train_tpr', train_tpr, on_step=False, on_epoch=True, prog_bar=True)
        learning_rate = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', learning_rate, on_step=False, on_epoch=True, prog_bar=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, y, logits = self._shared_step(batch)
        self.log('val_loss', loss)
        self.val_acc(logits, y)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.val_f1(logits, y)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.val_auc(logits, y)
        self.log('val_auc', self.val_auc, on_step=False, on_epoch=True, prog_bar=True)
        val_tpr = self.val_tpr(logits, y, fprNeeded=self.fpr)
        self.log('val_tpr', val_tpr, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, y, logits = self._shared_step(batch)
        self.log('test_loss', loss)
        self.test_acc(logits, y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.test_f1(logits, y)
        self.log('test_f1', self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.test_auc(logits, y)
        self.log('test_auc', self.test_auc, on_step=False, on_epoch=True, prog_bar=True)
        test_tpr = self.test_tpr(logits, y, fprNeeded=self.fpr)
        self.log('test_tpr', test_tpr, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return super().predict_step(batch[0], batch_idx, dataloader_idx)
        

def configure_trainer(
        name: str,
        log_folder: str,
        epochs: int,
        device: str = "cpu",
        # how many times to check val set within a single epoch
        val_check_times: int = 2,
        log_every_n_steps: int = 10,
        monitor_metric: str = "val_tpr",
        early_stop_patience: Union[None, int] = 5,
        lit_sanity_steps: int = 1
):
    model_checkpoint = ModelCheckpoint(
        monitor=monitor_metric,
        save_top_k=1,
        mode="max",
        verbose=False,
        save_last=True,
        filename="{epoch}-tpr{val_tpr:.4f}-f1{val_f1:.4f}-acc{val_cc:.4f}"
    )
    callbacks = [LitProgressBar(), model_checkpoint]

    if early_stop_patience is not None:
        early_stop = EarlyStopping(
            monitor=monitor_metric,
            patience=early_stop_patience,
            min_delta=0.0001,
            verbose=True,
            mode="max"
        )
        callbacks.append(early_stop)

    trainer = L.Trainer(
        num_sanity_val_steps=lit_sanity_steps,
        max_epochs=epochs,
        accelerator=device,
        devices=1,
        callbacks=callbacks,
        val_check_interval=1/val_check_times,
        log_every_n_steps=log_every_n_steps,
        logger=[
            CSVLogger(save_dir=log_folder, name=f"{name}_csv"),
            TensorBoardLogger(save_dir=log_folder, name=f"{name}_tb")
        ]
    )

    # Ensure folders for logging exist
    os.makedirs(os.path.join(log_folder, f"{name}_tb"), exist_ok=True)
    os.makedirs(os.path.join(log_folder, f"{name}_csv"), exist_ok=True)

    return trainer


def load_lit_model(
        model_file: str,
        pytorch_model: nn.Module,
        name: str,
        log_folder: str,
        epochs: int,
        device: str,
        lit_sanity_steps: int
):
    lightning_model = PyTorchLightningModel.load_from_checkpoint(checkpoint_path=model_file, model=pytorch_model)
    trainer = configure_trainer(name, log_folder, epochs, device=device, lit_sanity_steps=lit_sanity_steps)
    return trainer, lightning_model


def train_lit_model(
        X_train_loader: DataLoader,
        X_test_loader: DataLoader,
        pytorch_model: nn.Module,
        name: str,
        log_folder: str,
        epochs: int = 10,
        learning_rate: float = 1e-3,
        scheduler: Union[None, str] = None,
        scheduler_budget: Union[None, int] = None,
        model_file: Union[None, str] = None,
        device: str = "cpu",
        lit_sanity_steps: int = 1,
        early_stop_patience: int = 5
):
    lightning_model = PyTorchLightningModel(model=pytorch_model, learning_rate=learning_rate, scheduler=scheduler, scheduler_step_budget=scheduler_budget)
    trainer = configure_trainer(name, log_folder, epochs, device=device, lit_sanity_steps=lit_sanity_steps, early_stop_patience=early_stop_patience)

    print(f"[*] Training '{name}' model...")
    trainer.fit(lightning_model, X_train_loader, X_test_loader)

    if model_file is not None:
        # copy best checkpoint to the LOGS_DIR for further tests
        last_version_folder = [x for x in os.listdir(os.path.join(log_folder, name + "_csv")) if "version" in x][-1]
        checkpoint_path = os.path.join(log_folder, name + "_csv", last_version_folder, "checkpoints")
        best_checkpoint_name = [x for x in os.listdir(checkpoint_path) if x != "last.ckpt"][0]
        copyfile(os.path.join(checkpoint_path, best_checkpoint_name), model_file)

    return trainer, lightning_model


def predict_lit_model(
        loader: DataLoader, 
        trainer: L.Trainer, 
        lightning_model: PyTorchLightningModel, 
        decision_threshold: int = 0.5, 
        dump_logits: bool = False
) -> np.ndarray:
    """Get scores out of a loader."""
    y_pred_logits = trainer.predict(model=lightning_model, dataloaders=loader)
    y_pred = sigmoid(cat(y_pred_logits, dim=0)).numpy()
    y_pred = np.array([1 if x > decision_threshold else 0 for x in y_pred])
    if dump_logits:
        assert isinstance(dump_logits, str), "Please provide a path to dump logits: dump_logits='path/to/logits.pkl'"
        pickle.dump(y_pred_logits, open(dump_logits, "wb"))
    return y_pred
