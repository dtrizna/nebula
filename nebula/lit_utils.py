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
from torch import cat, sigmoid, save, load, optim
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
import torchmetrics

from .data_utils import create_dataloader


class LitProgressBar(TQDMProgressBar):
    # to preserve progress bar after each epoch
    def on_train_epoch_end(self, *args, **kwargs):
        super().on_train_epoch_end(*args, **kwargs)
        print()


class PyTorchLightningModel(L.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            learning_rate: float = 1e-3,
            fpr: float = 1e-4,
            scheduler: Union[None, str] = None,
            scheduler_step_budget: Union[None, int] = None
            # NOTE: scheduler_step_budget = epochs * len(train_loader)
    ):
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
        

class LitTrainerWrapper:
    def __init__(
        self,
        pytorch_model: nn.Module,
        name: str,
        log_folder: str,
        model_file: str = None,
        # training config
        epochs: int = 10,
        learning_rate: float = 1e-3,
        device: str = "cpu",
        val_check_times: int = 2,
        log_every_n_steps: int = 10,
        lit_sanity_steps: int = 1,
        monitor_metric: str = "val_tpr",
        early_stop_patience: Union[None, int] = 5,
        early_stop_min_delta: float = 0.0001,
        scheduler: Union[None, str] = None,
        # data config
        batch_size: int = 1024,
        dataloader_workers: int = 4,
        random_state: int = 42,
        verbose: bool = False,
        skip_trainer_init: bool = False
    ):
        self.pytorch_model = pytorch_model
        self.lit_model = None
        self.model_file = model_file

        self.name = name
        self.log_folder = log_folder
        os.makedirs(self.log_folder, exist_ok=True)
        print(f"[!] Logging to {self.log_folder}")
        
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device
        self.val_check_times = val_check_times
        self.log_every_n_steps = log_every_n_steps
        self.lit_sanity_steps = lit_sanity_steps
        self.monitor_metric = monitor_metric
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta
        self.scheduler = scheduler

        self.batch_size = batch_size
        self.dataloader_workers = dataloader_workers

        self.verbose = verbose
        self.random_state = random_state
        L.seed_everything(self.random_state)
        if not skip_trainer_init:
            self.setup_trainer()

        # TODO: provide option to setup time
        self.training_time = None
        # assert (max_time is None) or (max_epochs is None), "only either 'max_time' or 'max_epochs' can be set"
        # assert (max_time is not None) or (max_epochs is not None), "at least one of 'max_time' or 'max_epochs' should be set"
        # assert max_time is None or isinstance(max_time, dict),\
        #     """max_time must be None or dict, e.g. {"minutes": 2, "seconds": 30}"""
        # NOTE: above format is what L.Trainer expects to receive as max_time parameter
        

    def setup_trainer(self):
        model_checkpoint = ModelCheckpoint(
            monitor=self.monitor_metric,
            save_top_k=1,
            mode="max",
            verbose=self.verbose,
            save_last=True,
            filename="{epoch}-tpr{val_tpr:.4f}-f1{val_f1:.4f}-acc{val_cc:.4f}"
        )
        callbacks = [LitProgressBar(), model_checkpoint]

        if self.early_stop_patience is not None:
            early_stop = EarlyStopping(
                monitor=self.monitor_metric,
                patience=self.early_stop_patience,
                min_delta=self.early_stop_min_delta,
                verbose=self.verbose,
                mode="max"
            )
            callbacks.append(early_stop)

        self.trainer = L.Trainer(
            num_sanity_val_steps=self.lit_sanity_steps,
            max_epochs=self.epochs,
            accelerator=self.device,
            devices=1,
            callbacks=callbacks,
            val_check_interval=1/self.val_check_times,
            log_every_n_steps=self.log_every_n_steps,
            logger=[
                CSVLogger(save_dir=self.log_folder, name=f"{self.name}_csv"),
                TensorBoardLogger(save_dir=self.log_folder, name=f"{self.name}_tb")
            ]
        )

        # Ensure folders for logging exist
        os.makedirs(os.path.join(self.log_folder, f"{self.name}_tb"), exist_ok=True)
        os.makedirs(os.path.join(self.log_folder, f"{self.name}_csv"), exist_ok=True)


    def load_lit_model(
            self,
            model_file: str,
    ):
        self.model_file = model_file
        self.lit_model = PyTorchLightningModel.load_from_checkpoint(
            checkpoint_path=self.model_file,
            model=self.pytorch_model
        )
        

    def train_lit_model(
            self,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader = None,
    ):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.scheduler_budget = self.calculate_scheduler_step_budget(
            max_epochs=self.epochs,
            max_time=self.training_time
        )
        if self.lit_model is None:
            self.lit_model = PyTorchLightningModel(
                model=self.pytorch_model,
                learning_rate=self.learning_rate,
                scheduler=self.scheduler,
                scheduler_step_budget=self.scheduler_budget,
            )

        print(f"[*] Training '{self.name}' model...")
        self.trainer.fit(self.lit_model, self.train_dataloader, self.val_dataloader)

        if self.model_file is not None:
            self.save_lit_model()


    def predict_lit_model(
            self,
            loader: DataLoader, 
            decision_threshold: int = 0.5, 
            dump_logits: Union[bool, str] = False
    ) -> np.ndarray:
        assert self.lit_model is not None,\
            "[-] lightning_model isn't instantiated: either .train_lit_model() or .load_list_model()"
        """Get scores out of a loader."""
        y_pred_logits = self.trainer.predict(model=self.lit_model, dataloaders=loader)
        y_pred = sigmoid(cat(y_pred_logits, dim=0)).numpy()
        y_pred = np.array([1 if x > decision_threshold else 0 for x in y_pred])
        if dump_logits:
            assert isinstance(dump_logits, str), "Please provide a path to dump logits: dump_logits='path/to/logits.pkl'"
            pickle.dump(y_pred_logits, open(dump_logits, "wb"))
        return y_pred     


    def save_lit_model(self, model_name: str = None):
        # TODO: verify if this works properly, do not see model file
        if model_name is None:
            model_name = self.model_file
        assert model_name is not None, "Please provide a model name"

        # copy the best checkpoint
        last_version_folder = [x for x in os.listdir(os.path.join(self.log_folder, self.name + "_csv")) if "version" in x][-1]
        checkpoint_path = os.path.join(self.log_folder, self.name + "_csv", last_version_folder, "checkpoints")
        best_checkpoint_name = [x for x in os.listdir(checkpoint_path) if x != "last.ckpt"][0]
        copyfile(os.path.join(checkpoint_path, best_checkpoint_name), self.model_file)


    def save_torch_model(self, model_name: str = None):
        if model_name is None:
            model_name = self.model_file
        assert model_name is not None, "Please provide a model name"

        save(self.pytorch_model, model_name)
        print(f"[!] Saved model to {model_name}")
    

    def create_dataloader(
            self,
            X: np.ndarray,
            y: np.ndarray,
            batch_size: int = None,
            dataloader_workers: int = None,
            shuffle: bool = False
    ):
        if batch_size is None:
            batch_size = self.batch_size
        else:
            self.batch_size = batch_size
        if dataloader_workers is None:
            dataloader_workers = self.dataloader_workers
        else:
            self.dataloader_workers = dataloader_workers

        dataloader = create_dataloader(
                X=X,
                y=y,
                batch_size=self.batch_size,
                shuffle=shuffle,
                workers=self.dataloader_workers
        )
        return dataloader
    

    def calculate_scheduler_step_budget(
        self,
        max_time: dict = None,
        max_epochs: int = None
    ) -> int:
        if max_epochs is not None:
            total_batches = max_epochs * len(self.train_dataloader)
        if max_time is not None:
            # TODO: does lightning provide a way to get the number of batches from time?
            raise NotImplementedError("calculate_scheduler_step_budget for max_time is not implemented yet")
        return total_batches
    