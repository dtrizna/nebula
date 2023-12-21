import numpy as np
import time
import os

from torch.utils.data import DataLoader, TensorDataset
from torch import from_numpy, save
from sklearn.model_selection import KFold

import lightning as L
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from ..lit_utils import LitProgressBar, LitPyTorchModel


class LitCrossValidation(object):
    def __init__(self,
                    model_class,
                    model_config,
                    # cross validation config
                    folds=3,
                    dump_data_splits=True,
                    dump_models=True,
                    # dataloader config
                    batch_size=32,
                    dataloader_workers=8,
                    # auxiliary
                    random_state=42,
                    out_folder="./cv_logs"
                ):
        self.model_class = model_class
        self.model_config = model_config

        # cross validation config
        assert folds > 1, "folds must be greater than 1"
        self.folds = folds
        self.dump_data_splits = dump_data_splits
        self.dump_models = dump_models

        # dataloader config
        self.batch_size = batch_size
        self.dataloader_workers = dataloader_workers

        # auxiliary
        self.random_state = random_state
        L.seed_everything(self.random_state)

        self.out_folder = out_folder
        os.makedirs(self.out_folder, exist_ok=True)

    def build_dataloader(self, 
                        X: np.ndarray,
                        y: np.ndarray = None,
                        shuffle: bool = True) -> DataLoader:
        y = y.reshape(-1, 1) if len(y.shape) == 1 else y
        assert X.shape[0] == y.shape[0], "X and y must have the same number of rows"
        dataset = TensorDataset(from_numpy(X), from_numpy(y).float())        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.dataloader_workers,
            # NOTE: these are important for lightning to iterate quickly
            persistent_workers=True,
            pin_memory=True
        )
        return dataloader

    def dump_dataloaders(self, outfolder: str = None) -> None:
        if outfolder is None:
            outfolder = os.path.join(self.out_folder, "data_splits", f"version_{self.fold}")
        os.makedirs(outfolder, exist_ok=True)
        train_outfile = os.path.join(outfolder, f"train_dataloader.torch")
        save(self.train_dataloader, train_outfile)
        val_outfile = os.path.join(outfolder, f"val_dataloader.torch")
        save(self.val_dataloader, val_outfile)

    def print_tensor_shapes(self) -> None:
        y_dim = list(self.train_dataloader.dataset.tensors[1].shape)
        x_dim = list(self.train_dataloader.dataset.tensors[0].shape)
        msg = f"[!] Dataset shapes: x_dim={x_dim}, y_dim={y_dim} | "
        msg += "train size: {len(self.train_dataloader.dataset)} | "
        msg += "val size: {len(self.val_dataloader.dataset)}"
        print(msg)
        
    def save_model(self, model_outfolder: str = None) -> None:
        if model_outfolder is None:
            model_outfolder = os.path.join(self.out_folder, "models", f"version_{self.fold}")
        os.makedirs(model_outfolder, exist_ok=True)
        model_outfile = os.path.join(model_outfolder, f"model.torch")
        save(self.torch_model, model_outfile)
        print(f"[!] Saved model to {model_outfile}")

    def calculate_scheduler_step_budget(self,
                                        max_time: dict = None,
                                        max_epochs: int = None) -> int:
        if max_epochs is not None:
            total_batches = max_epochs * len(self.train_dataloader)
        if max_time is not None:
            # TODO: does lightning provide a way to get the number of batches from time?
            raise NotImplementedError("calculate_scheduler_step_budget for max_time is not implemented yet")
        return total_batches

    def run(self,
            X: np.array,
            y: np.array,
            max_time: dict = None,
            max_epochs: int = None
    ) -> None:
        assert (max_time is None) or (max_epochs is None), "only either 'max_time' or 'max_epochs' can be set"
        assert (max_time is not None) or (max_epochs is not None), "at least one of 'max_time' or 'max_epochs' should be set"
        assert max_time is None or isinstance(max_time, dict),\
            """max_time must be None or dict, e.g. {"minutes": 2, "seconds": 30}"""

        kf = KFold(
            n_splits=self.folds,
            shuffle=True,
            random_state=self.random_state
        )
        kf.get_n_splits(X)

        for fold, (train_index, val_index) in enumerate(kf.split(X)):
            self.fold = fold
            print(f"[*] Fold {self.fold}/{self.folds-1}...")

            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            self.train_dataloader = self.build_dataloader(X=X_train, y=y_train)
            self.val_dataloader = self.build_dataloader(X=X_val, y=y_val, shuffle=False)
            self.print_tensor_shapes()
            if self.dump_data_splits:
                self.dump_dataloaders()

            trainer = L.Trainer(
                max_time=max_time,
                max_epochs=max_epochs,
                accelerator="gpu",
                devices=1,
                deterministic=True,
                callbacks=[LitProgressBar()],
                logger=[
                    CSVLogger(save_dir=self.out_folder, name=f"logger_csv"),
                    TensorBoardLogger(save_dir=self.out_folder, name=f"logger_tb")
                ],
                
                # --- calculating validation values ---
                # A: if float: check validation values every X epoch, e.g. 0.5 half an epoch
                # A: if int: check every X batches, e.g. 100 batches
                val_check_interval=0.5,
                # B: default: 1, check as well after every epoch
                check_val_every_n_epoch=1, 
                
                # --- how often (in batches) to log train and validation metrics ---
                log_every_n_steps=50 # default: 50
            )

            self.torch_model = self.model_class(**self.model_config)
            self.litmodel = LitPyTorchModel(
                model=self.torch_model,
                optimizer="Adam",
                # scheduler="StepLR",
                # scheduler_step_budget=
                #     self.calculate_scheduler_step_budget(max_time, max_epochs)
            )

            trainer.fit(
                model=self.litmodel,
                train_dataloaders=self.train_dataloader,
                val_dataloaders=self.val_dataloader
            )
            if self.dump_models:
                self.save_model()
