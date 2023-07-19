import numpy as np
import time
import os

from torch.utils.data import DataLoader, TensorDataset
from torch import from_numpy
from sklearn.model_selection import KFold

import lightning as L
from lightning.pytorch.loggers import CSVLogger
from ..lit_utils import LitProgressBar, LitPyTorchModel


class LitCrossValidation(object):
    def __init__(self,
                    model_class,
                    model_config,
                    # cross validation config
                    folds=3,
                    dump_data_splits=True,
                    # dataloader config
                    batch_size=32,
                    dataloader_workers=8,
                    # auxiliary
                    random_state=42,
                    log_folder="./cv_logs"
                ):
        self.model_class = model_class
        self.model_config = model_config

        # cross validation config
        assert folds > 1, "folds must be greater than 1"
        self.folds = folds
        self.dump_data_splits = dump_data_splits

        # dataloader config
        self.batch_size = batch_size
        self.dataloader_workers = dataloader_workers

        # auxiliary
        self.random_state = random_state
        self.log_folder = log_folder

        L.seed_everything(self.random_state)
        
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

    def dump_split(self,
            X_train: np.array,
            y_train: np.array,
            X_val: np.array,
            y_val: np.array,
            timestamp: int
    ) -> None:
        split_name = f"dataset_splits_{timestamp}.npz"
        np.savez_compressed(
            os.path.join(self.log_folder, split_name),
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val)

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

        # Iterate over the folds
        for i, (train_index, val_index) in enumerate(kf.split(X)):
            print(f"[*] Fold {i+1}/{self.folds}...")
            timestamp = int(time.time())

            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            print(f"[!] Dataset shapes: X_train={X_train.shape}, y_train={y_train.shape}, X_val={X_val.shape}, y_val={y_val.shape}")

            if self.dump_data_splits:
                self.dump_split(X_train, y_train, X_val, y_val, timestamp)

            self.train_dataloader = self.build_dataloader(X=X_train, y=y_train)
            self.val_dataloader = self.build_dataloader(X=X_val, y=y_val, shuffle=False)

            trainer = L.Trainer(
                max_time=max_time,
                max_epochs=max_epochs,
                accelerator="gpu",
                devices=1,
                deterministic=True,
                callbacks=[LitProgressBar()],
                logger=CSVLogger(save_dir=self.log_folder, name=f"csv_logger_{timestamp}"),
                log_every_n_steps=10 # default: 50
            )

            model = self.model_class(**self.model_config)
            litmodel = LitPyTorchModel(
                model=model,
                optimizer="Adam",
                # scheduler="StepLR",
                # scheduler_step_budget=
                #     self.calculate_scheduler_step_budget(max_time, max_epochs)
            )

            trainer.fit(
                model=litmodel,
                train_dataloaders=self.train_dataloader,
                val_dataloaders=self.val_dataloader
            )
