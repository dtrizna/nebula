# TODO

- Q: Why such huge delays between epochs in lightnings trainer.fit()?
  A: Dataloaders... Their config has to have:

    ```python
    DataLoader(
        ...
        persistent_workers=True,
        pin_memory=True
    )
    ```

- Q: No correct values for loss and F1
  A: B'cause you have to use proper loss -- BCEWithLogitsLoss() for our case.

- Q: Analyzie logging -- is everything provided?
  A: Yes, model config in hparams.yaml, metrics in metrics.csv, checkpoints in checkpoints/
    TODO: tensorboard logs in lightning_logs/

- Q: Learn how:
  - schedulers+lightning work;
  - how to compute step budget given max_time?
  A: ...
