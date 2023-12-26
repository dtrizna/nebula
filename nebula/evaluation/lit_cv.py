import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from torch import save
from ..lit_utils import LitTrainerWrapper

class LitCrossValidation(LitTrainerWrapper):
    def __init__(
            self,
            folds=3,
            dump_data_splits=True,
            dump_models=True,
            *args,
            **kwargs
        ):
        super().__init__(skip_trainer_init=True, *args, **kwargs)
        
        # cross validation config
        assert folds > 1, "folds must be greater than 1, otherwise use LitTrainerWrapper directly"
        self.folds = folds
        self.fold = None

        self.dump_data_splits = dump_data_splits
        self.dump_models = dump_models

        self.pytorch_initial_state = self.pytorch_model.state_dict()
        self.train_dataloader = None
        self.val_dataloader = None


    def dump_dataloaders(self, outfolder: str = None) -> None:
        assert self.train_dataloader is not None, "train_dataloader is None"
        assert self.val_dataloader is not None, "val_dataloader is None"

        if outfolder is None:
            outfolder = self.log_folder

        train_outfile = os.path.join(outfolder, f"fold_{self.fold}_train_dataloader.torch")
        save(self.train_dataloader, train_outfile)

        val_outfile = os.path.join(outfolder, f"fold_{self.fold}_val_dataloader.torch")
        save(self.val_dataloader, val_outfile)


    def print_tensor_shapes(self) -> None:
        assert self.train_dataloader is not None, "train_dataloader is None"
        assert self.val_dataloader is not None, "val_dataloader is None"
        y_dim = list(self.train_dataloader.dataset.tensors[1].shape)
        x_dim = list(self.train_dataloader.dataset.tensors[0].shape)
        msg = f"[!] Dataset shapes: x_dim={x_dim}, y_dim={y_dim} | "
        msg += f"train size: {len(self.train_dataloader.dataset)} | "
        msg += f"val size: {len(self.val_dataloader.dataset)}"
        print(msg)


    def run(self,
            x: np.ndarray,
            y: np.ndarray,
            print_fold_scores: bool = True
    ) -> None:

        kf = KFold(
            n_splits=self.folds,
            shuffle=True,
            random_state=self.random_state
        )
        kf.get_n_splits(x)

        for fold, (train_index, val_index) in enumerate(kf.split(x)):
            self.fold = fold
            print(f"[*] Fold {self.fold+1}/{self.folds}...")
            self.setup_trainer()

            X_train, y_train  = x[train_index], y[train_index]
            self.train_dataloader = self.create_dataloader(
                X=X_train,
                y=y_train,
                shuffle=True
            )
            X_val, y_val = x[val_index], y[val_index]
            self.val_dataloader = self.create_dataloader(
                X=X_val,
                y=y_val,
                shuffle=False,
            )
            self.print_tensor_shapes()
            if self.dump_data_splits:
                self.dump_dataloaders()

            self.pytorch_model.load_state_dict(self.pytorch_initial_state)
            self.train_lit_model(
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader
            )
            if self.dump_models:
                self.save_torch_model(model_file=os.path.join(self.log_folder, f"fold_{self.fold}_model.torch"))
                self.save_lit_model(model_file=os.path.join(self.log_folder, f"fold_{self.fold}_lit_model.ckpt"))

            if print_fold_scores:
                y_train_pred = self.predict_lit_model(self.train_dataloader)
                f1_train = f1_score(y_train, y_train_pred)
                y_val_pred = self.predict_lit_model(self.val_dataloader)
                f1_val = f1_score(y_val, y_val_pred)
                print(f"[*] Fold {self.fold+1}/{self.folds} | train f1: {f1_train:.4f} | val f1: {f1_val:.4f}")
