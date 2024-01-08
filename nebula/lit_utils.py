import os
import pickle
import numpy as np
from time import time
from shutil import copyfile
from typing import Union, Any, Callable, Optional

from sklearn.metrics import roc_curve

import lightning as L
from lightning.lite.utilities.seed import seed_everything
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint, EarlyStopping, ModelSummary
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
import torchmetrics

from .data_utils import create_dataloader


class LitProgressBar(TQDMProgressBar):
    def on_train_epoch_end(self, *args, **kwargs):
        super().on_train_epoch_end(*args, **kwargs)
        # to preserve progress bar after each epoch
        print()


class PyTorchLightningModelBase(L.LightningModule):
    def __init__(
            self,
            model: torch.nn.Module,
            learning_rate: float = 1e-3,
            fpr: float = 1e-4,
            scheduler: Union[None, str] = None,
            scheduler_step_budget: Union[None, int] = None,
            # NOTE: scheduler_step_budget = epochs * len(train_loader)
            loss: Callable = BCEWithLogitsLoss(),
    ):
        super().__init__()

        self.model = model
        self.causal = hasattr(self.model, "causal_attention") and self.model.causal_attention
        self.learning_rate = learning_rate
        self.fpr = fpr
        self.loss = loss

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
        self.train_recall = torchmetrics.Recall(task='binary')
        self.train_precision = torchmetrics.Precision(task='binary')

        self.val_acc = torchmetrics.Accuracy(task='binary')
        self.val_f1 = torchmetrics.F1Score(task='binary', average='macro')
        self.val_auc = torchmetrics.AUROC(task='binary')
        self.val_tpr = self.get_tpr_at_fpr
        self.val_recall = torchmetrics.Recall(task='binary')
        self.val_precision = torchmetrics.Precision(task='binary')

        # NOTE: not using .test(), don't want to have these rudimentary metrics to drag over
        # self.test_acc = torchmetrics.Accuracy(task='binary')
        # self.test_f1 = torchmetrics.F1Score(task='binary', average='macro')
        # self.test_auc = torchmetrics.AUROC(task='binary')
        # self.test_tpr = self.get_tpr_at_fpr
        # self.test_recall = torchmetrics.Recall(task='binary')
        # self.test_precision = torchmetrics.Precision(task='binary')

        # self.save_hyperparameters(ignore=["model"])
    
    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Callable, optimizer_idx: int):
        # https://pytorch-lightning.readthedocs.io/en/1.3.8/benchmarking/performance.html#zero-grad-set-to-none-true
        optimizer.zero_grad(set_to_none=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        if self.scheduler is None:
            return optimizer

        if self.scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=self.scheduler_step_budget
            )
        if self.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_step_budget,
                # NOTE: minimum learning rate, should be ~= learning_rate/10 per Chinchilla
                eta_min=self.learning_rate/10
            )
        print(f"[!] Setting up {self.scheduler} scheduler with step budget {self.scheduler_step_budget}")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "step", # default: epoch
                "frequency": 1
            }
        }

    def get_tpr_at_fpr(
            self,
            predicted_logits: torch.Tensor,
            true_labels: torch.Tensor,
            fprNeeded: float = 1e-4,
            return_thresholds: bool = False
    ):
        predicted_probs = torch.sigmoid(predicted_logits).cpu().detach()
        true_labels = true_labels.cpu().detach()
        try :
            fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
        except ValueError: 
            # when multi-label 'ValueError: multilabel-indicator format is not supported'
            return (torch.nan, torch.nan) if return_thresholds else torch.nan
        if all(np.isnan(fpr)):
            return (torch.nan, torch.nan) if return_thresholds else torch.nan
        else:
            tpr_at_fpr = tpr[fpr <= fprNeeded][-1]
            threshold_at_fpr = thresholds[fpr <= fprNeeded][-1]
            return (tpr_at_fpr, threshold_at_fpr) if return_thresholds else tpr_at_fpr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def _shared_step(self, batch: torch.Tensor):
        # used by training_step, validation_step and test_step
        x, y = batch
        logits = self.forward(x)

        if y.ndim == 2 and logits.ndim == 3: # e.g. autoregressive pre-training
            logits, y = logits.view(-1, logits.size(-1)), y.view(-1)
        elif y.ndim == 1: # binary classification: (batch_size,) => (batch_size, 1)
            y = y.unsqueeze(-1)

        loss = self.loss(logits, y)
        # NOTE: by returning only loss here we avoid memory leaks
        self.logits = logits.detach().cpu()
        self.y = y.detach().cpu()
        return loss
    
    def training_step(self, batch: torch.Tensor, batch_idx):
        # NOTE: keep batch_idx -- lightning needs it
        loss = self._shared_step(batch)
        self.log('train_loss', loss, prog_bar=True)
        learning_rate = self.optimizers().param_groups[0]['lr']
        self.log('lr', learning_rate, on_step=True, prog_bar=False)
        self.log('memory', float(torch.cuda.memory_allocated()), on_step=True, prog_bar=True)
        self.train_f1(self.logits, self.y)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        train_tpr = self.train_tpr(self.logits, self.y, fprNeeded=self.fpr)
        self.log('train_tpr', train_tpr, on_step=False, on_epoch=True, prog_bar=True)
        self.train_acc(self.logits, self.y)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.train_auc(self.logits, self.y)
        self.log('train_auc', self.train_auc, on_step=False, on_epoch=True, prog_bar=False)
        self.train_recall(self.logits, self.y)
        self.log('train_recall', self.train_recall, on_step=False, on_epoch=True, prog_bar=False)
        self.train_precision(self.logits, self.y)
        self.log('train_precision', self.train_precision, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def predict_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return super().predict_step(batch[0], batch_idx, dataloader_idx)

    # NOTE: not using .test(), don't want to have these rudimentary metrics to drag over
    # def test_step(self, batch, batch_idx):
    #     # NOTE: keep batch_idx -- lightning needs it
    #     loss, y, logits = self._shared_step(batch)
    #     self.log('test_loss', loss)
    #     self.test_f1(logits, y)
    #     self.log('test_f1', self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
    #     test_tpr = self.test_tpr(logits, y, fprNeeded=self.fpr)
    #     self.log('test_tpr', test_tpr, on_step=False, on_epoch=True, prog_bar=True)
    #     self.test_acc(logits, y)
    #     self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
    #     self.test_auc(logits, y)
    #     self.log('test_auc', self.test_auc, on_step=False, on_epoch=False, prog_bar=False)
    #     self.test_recall(logits, y)
    #     self.log('test_recall', self.test_recall, on_step=False, on_epoch=False, prog_bar=False)
    #     self.test_precision(logits, y)
    #     self.log('test_precision', self.test_precision, on_step=False, on_epoch=False, prog_bar=False)
    #     return loss


class PyTorchLightningModel(PyTorchLightningModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def validation_step(self, batch: (torch.Tensor, torch.Tensor), batch_idx):
        # NOTE: keep batch_idx -- lightning needs it
        # loss, y, logits = self._shared_step(batch)
        loss = self._shared_step(batch)
        self.log('val_loss', loss)
        self.val_f1(self.logits, self.y)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        val_tpr = self.val_tpr(self.logits, self.y, fprNeeded=self.fpr)
        self.log('val_tpr', val_tpr, on_step=False, on_epoch=True, prog_bar=True)
        self.val_acc(self.logits, self.y)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.val_auc(self.logits, self.y)
        self.log('val_auc', self.val_auc, on_step=False, on_epoch=True, prog_bar=False)
        self.val_recall(self.logits, self.y)
        self.log('val_recall', self.val_recall, on_step=False, on_epoch=True, prog_bar=False)
        self.val_precision(self.logits, self.y)
        self.log('val_precision', self.val_precision, on_step=False, on_epoch=True, prog_bar=False)
        return loss    


class PyTorchLightningModelLM(PyTorchLightningModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(loss = CrossEntropyLoss(), *args, **kwargs)
        assert hasattr(self.model, "pretrain"), "This model does not have a 'pretrain' method."
        self.forward = self.model.pretrain
    
    def training_step(self, batch: torch.Tensor, batch_idx):
        # NOTE: keep batch_idx -- lightning needs it
        loss = self._shared_step(batch)
        self.log('train_loss', loss, prog_bar=True)
        learning_rate = self.optimizers().param_groups[0]['lr']
        self.log('lr', learning_rate, on_step=True, prog_bar=True)
        self.log('memory', float(torch.cuda.memory_allocated()), on_step=True, prog_bar=True)
        return loss


class LitTrainerWrapper:
    def __init__(
        self,
        pytorch_model: torch.nn.Module,
        name: Optional[str] = None,
        log_folder: Optional[str] = None,
        # training config
        epochs: int = None,
        learning_rate: float = 1e-3,
        device: str = "cpu",
        val_check_times: int = 2,
        log_every_n_steps: int = 10,
        lit_sanity_steps: int = 1,
        monitor_metric: str = "val_tpr",
        monitor_mode: str = "max",
        early_stop_patience: Union[None, int] = 5,
        early_stop_min_delta: float = 0.0001,
        # efficient training strategies
        scheduler: Union[None, str] = None,
        accumulate_grad_batches: Optional[int] = None,
        gradient_clip_val: Optional[float] = None,
        # https://lightning.ai/docs/pytorch/stable/advanced/speed.html#mixed-precision-16-bit-training
        # lit: Double precision (64), full precision (32), half precision (16) or bfloat16 precision (bf16) for TPUS
        precision: Optional[int] = 32,
        # data config
        batch_size: int = 1024,
        dataloader_workers: int = 4,
        random_state: int = 42,
        verbose: bool = False,
        skip_trainer_init: Optional[bool] = False
    ):
        self.pytorch_model = pytorch_model
        self.lit_model = None

        if name is None:
            self.name = f"{self.pytorch_model.__class__.__name__}_{int(time())}"
        else:
            self.name = name
        self.log_folder = log_folder
        
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device
        self.val_check_times = val_check_times
        self.log_every_n_steps = log_every_n_steps
        self.lit_sanity_steps = lit_sanity_steps
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta
        self.scheduler = scheduler
        self.scheduler_budget = None
        self.accumulate_grad_batches = accumulate_grad_batches
        self.gradient_clip_val = gradient_clip_val
        assert precision in [16, 32, 64]
        self.precision = precision

        self.batch_size = batch_size
        self.dataloader_workers = dataloader_workers

        self.verbose = verbose
        self.random_state = random_state
        
        # TODO: provide option to setup time
        self.training_time = None
        # assert (max_time is None) or (max_epochs is None), "only either 'max_time' or 'max_epochs' can be set"
        # assert (max_time is not None) or (max_epochs is not None), "at least one of 'max_time' or 'max_epochs' should be set"
        # assert max_time is None or isinstance(max_time, dict),\
        #     """max_time must be None or dict, e.g. {"minutes": 2, "seconds": 30}"""
        # NOTE: above format is what L.Trainer expects to receive as max_time parameter        
       
        if not skip_trainer_init:
            self.setup_trainer()


    def setup_callbacks(self):
        if "val" in self.monitor_metric:
            ckpt_filename = "{epoch}-{val_tpr:.4f}-{val_f1:.4f}-{val_cc:.4f}"
        else:
            ckpt_filename = "{epoch}-{train_loss:.4f}"
        model_checkpoint = ModelCheckpoint(
            monitor=self.monitor_metric,
            mode=self.monitor_mode,
            save_top_k=1,
            save_last=True,
            filename=ckpt_filename,
            verbose=self.verbose,
        )
        callbacks = [LitProgressBar(), model_checkpoint]
        if self.verbose:
            summary = ModelSummary(max_depth=-1)
            callbacks.append(summary)

        if self.early_stop_patience is not None:
            early_stop = EarlyStopping(
                monitor=self.monitor_metric,
                patience=self.early_stop_patience,
                min_delta=self.early_stop_min_delta,
                verbose=self.verbose,
                mode="max"
            )
            callbacks.append(early_stop)
        return callbacks


    def setup_trainer(self):
        seed_everything(self.random_state)
        callbacks = self.setup_callbacks()

        if self.log_folder is None:
            self.log_folder = f"./out_{self.name}_{int(time())}"
        try:
            os.makedirs(self.log_folder, exist_ok=True)
        except ValueError as ex:
            print(self.log_folder)
            raise ex
            
        print(f"[!] Logging to {self.log_folder}")

        self.trainer = L.Trainer(
            max_epochs=self.epochs,
            accelerator=self.device,
            devices=1,
            callbacks=callbacks,
            val_check_interval=1/self.val_check_times,
            log_every_n_steps=self.log_every_n_steps,
            num_sanity_val_steps=self.lit_sanity_steps,
            enable_model_summary=False, # added manually with different depth
            accumulate_grad_batches=self.accumulate_grad_batches,
            gradient_clip_val=self.gradient_clip_val,
            precision=self.precision,
            logger=[
                CSVLogger(save_dir=self.log_folder, name=f"{self.name}_csv"),
                TensorBoardLogger(save_dir=self.log_folder, name=f"{self.name}_tb")
            ]
        )

        # Ensure folders for logging exist
        os.makedirs(os.path.join(self.log_folder, f"{self.name}_tb"), exist_ok=True)
        os.makedirs(os.path.join(self.log_folder, f"{self.name}_csv"), exist_ok=True)


    def load_lit_model(self, model_file: str = None):
        assert model_file is not None, "Please provide a model file"
        self.lit_model = PyTorchLightningModel.load_from_checkpoint(
            checkpoint_path=model_file,
            model=self.pytorch_model
        )


    def load_torch_model(self, model_file: str = None):
        assert model_file is not None, "Please provide a model file"
        self.pytorch_model = torch.load(model_file)
        # NOTE: you have to reset self.lit_model after this
        #  if lit_model is already initialized, then load state dict directly:
        # self.lit_model.model.load_state_dict(state_dict)


    def setup_lit_model(self):
        if self.scheduler is not None and self.scheduler_budget is None:
            self.calculate_scheduler_step_budget(max_epochs=self.epochs, max_time=self.training_time)
        self.lit_model = PyTorchLightningModel(
                model=self.pytorch_model,
                learning_rate=self.learning_rate,
                scheduler=self.scheduler,
                scheduler_step_budget=self.scheduler_budget,
            )


    def fit(self, *args, **kwargs):
        assert self.trainer is not None, "Please setup trainer first"
        self.trainer.fit(*args, **kwargs)


    def train_lit_model(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader = None,
    ):
        assert self.trainer is not None, "Please setup trainer first"
        self.train_loader = train_loader
        self.val_loader = val_loader

        if self.lit_model is None:
            self.setup_lit_model()

        print(f"[*] Training '{self.name}' model...")
        self.trainer.fit(self.lit_model, self.train_loader, self.val_loader)


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
        y_pred = torch.sigmoid(torch.cat(y_pred_logits, dim=0)).numpy()
        y_pred = np.array([1 if x > decision_threshold else 0 for x in y_pred])
        if dump_logits:
            assert isinstance(dump_logits, str), "Please provide a path to dump logits: dump_logits='path/to/logits.pkl'"
            pickle.dump(y_pred_logits, open(dump_logits, "wb"))
        return y_pred     


    def save_lit_model(self, model_file: str = None, how="best"):
        assert how in ["best", "last"], "how must be either 'best' or 'last'"
        if how == "best":
          checkpoint_path = self.trainer.checkpoint_callback.best_model_path
        if how == "last":
          checkpoint_path = self.trainer.checkpoint_callback.last_model_path
        basename = os.path.basename(checkpoint_path)
        
        if model_file is None: # case when no model file is specified
            basename = f"{int(time())}_epoch_{self.trainer.current_epoch}_{basename}.ckpt"
            model_file = os.path.join(self.log_folder, basename)
        if os.path.exists(model_file): # be sure not to override existing models
            basename = f"{int(time())}_epoch_{self.trainer.current_epoch}_{basename}.ckpt"
            model_file = os.path.join(self.log_folder, basename)
        
        if os.path.exists(checkpoint_path): 
            copyfile(checkpoint_path, model_file)
            print(f"[!] Saved Ligthining model to {model_file}")
        else:
            print(f"[-] Cannot locate lit model checkpoint...")


    def save_torch_model(self, model_file: str = None):
        if model_file is None: # case when no model file is specified
            basename = f"{int(time())}_epoch_{self.trainer.current_epoch}.torch"
            model_file = os.path.join(self.log_folder, basename)
        if os.path.exists(model_file): # be sure not to override existing models
            basename = f"{int(time())}_epoch_{self.trainer.current_epoch}_{os.path.basename(model_file)}"
            model_file = os.path.join(self.log_folder, basename)
        
        torch.save(self.pytorch_model, model_file)
        print(f"[!] Saved PyTorch model to {model_file}")
    

    def create_dataloader(
            self,
            X: Union[np.ndarray, torch.Tensor],
            y: Union[np.ndarray, torch.Tensor],
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
        assert (max_time is None) or (max_epochs is None), "only either 'max_time' or 'max_epochs' can be set"
        assert (max_time is not None) or (max_epochs is not None), "at least one of 'max_time' or 'max_epochs' should be set"

        accumulate_grad_batches = 1 if self.accumulate_grad_batches is None else self.accumulate_grad_batches
        total_batches = 0
        if max_epochs is not None:
            total_batches = int(np.ceil(max_epochs * len(self.train_loader) / accumulate_grad_batches))
        if max_time is not None:
            # TODO: Implement logic for max_time if needed
            raise NotImplementedError("calculate_scheduler_step_budget for max_time is not implemented yet")

        self.scheduler_budget = total_batches
