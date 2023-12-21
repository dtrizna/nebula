# Benchmark library API

## 1. Existing modeling scheme

### Preprocessing example

```python
import nebula

x, y = nebula.preprocess_file(
    preprocessing_type: str,
    data_type: str,
    file_path: str,
    file_label: int,
)

x, y = nebula.preprocess_folder(
    preprocessing: str,
    data_type: str,
    folder_path: str,
    folder_label: list[int],
)
```

### Modeling example

```python
import nebula

metrics = nebula.evaluate(
    training_params: dict,
    model_type: str,
    model_params: dict,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
)
```

`training_params` example:

```json
{
    "batch_size": 64,
    "epochs": 10, // or "training_budget": 10 # min
    "learning_rate": 0.001,
    "optimizer": "adam",
    "cv_splits": 5,
    "split_ratio": 0.2,
    "loss": "categorical_crossentropy",
    "metrics": ["accuracy", "f1", "precision", "recall", "auc"],
    "output_folder": "./output",
}
```

`model_params` example -- depends on `model_type`:

```json
{
    "encoder_layers": 2,
    "attention_chunk_size": 64,
    "attention_num_heads": 4,
    "attention_head_size": 64,
    "attention_mlp_dims": [256, 128],
}
```

## 2. New modeling scheme

```python

nebula.evaluate(
    training_params: dict,
    model: torch.nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
)
```
