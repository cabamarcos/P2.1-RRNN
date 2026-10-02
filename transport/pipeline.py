"""Train on train/validation, select by macro F1, then evaluate the sealed test split."""

import hashlib
import importlib.metadata
import json
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

from transport import CLASS_NAMES
from transport.data import SensorScaler, load_data, partition, track_groups


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    strategy: str = "track"
    epochs: int = 80
    patience: int = 10
    batch_size: int = 128

    def __post_init__(self):
        if min(self.epochs, self.patience, self.batch_size) < 1:
            raise ValueError("Épocas, paciencia y tamaño de lote deben ser positivos.")


def tensorflow():
    # Keep CPU execution small and reproducible on Windows and CI runners.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
    import tensorflow as tf

    tf.config.experimental.enable_op_determinism()
    return tf


def metrics(truth, predicted):
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "per_class": classification_report(
            truth,
            predicted,
            labels=range(5),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=range(5)).tolist(),
    }


def dataset(tf, x, y, config, weights=None, shuffle=False):
    values = (x, y) if weights is None else (x, y, weights[y])
    ds = tf.data.Dataset.from_tensor_slices(values)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    options.threading.max_intra_op_parallelism = 1
    ds = ds.with_options(options)
    if shuffle:
        ds = ds.shuffle(len(y), seed=config.seed, reshuffle_each_iteration=True)
    return ds.batch(config.batch_size).prefetch(1)


def train_network(tf, x, y, config, weighted):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config.seed)
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(x["train"].shape[1],)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(5, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    weights = compute_class_weight("balanced", classes=np.arange(5), y=y["train"])
    train = dataset(
        tf,
        x["train"],
        y["train"],
        config,
        weights=weights.astype(np.float32) if weighted else None,
        shuffle=True,
    )
    validation = dataset(tf, x["validation"], y["validation"], config)
    history = model.fit(
        train,
        validation_data=validation,
        epochs=config.epochs,
        verbose=0,
        shuffle=False,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.patience,
                min_delta=0.001,
                restore_best_weights=True,
            )
        ],
    )
    predictions = np.argmax(model(x["validation"], training=False).numpy(), axis=1)
    return model, history.history, metrics(y["validation"], predictions), weights.tolist()


def run_experiment(data_path, output_path, config=None):
    config = config or TrainConfig()
    frame = load_data(data_path)
    parts = partition(frame, config.seed, config.strategy)
    output = Path(output_path)
    # Avoid silently replacing a previous experiment or using its test result to select models.
    output.mkdir(parents=True, exist_ok=False)
    scaler = SensorScaler.fit(frame.iloc[parts["train"]])
    x = {name: scaler.transform(frame.iloc[rows]) for name, rows in parts.items()}
    y = {name: frame.label.iloc[rows].to_numpy(dtype=np.int64) for name, rows in parts.items()}
    tf = tensorflow()
    candidates = {}
    best_model, best_name, best_score = None, None, -1.0
    for name, weighted in [("MLP", False), ("MLP ponderado", True)]:
        print(f"Entrenando {name}...", flush=True)
        model, history, validation, weights = train_network(tf, x, y, config, weighted)
        candidates[name] = {
            "validation": validation,
            "history": history,
            "class_weights": weights if weighted else [1.0] * 5,
            "epochs_run": len(history["loss"]),
        }
        score = validation["macro_f1"]
        print(f"F1 macro de validación: {score:.4f}", flush=True)
        if score > best_score:
            best_model, best_name, best_score = model, name, score
    baselines = {
        "Clase mayoritaria": DummyClassifier(strategy="most_frequent"),
        "Regresión logística": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=config.seed
        ),
    }
    test_results = {}
    for name, model in baselines.items():
        model.fit(x["train"], y["train"])
        test_results[name] = metrics(y["test"], model.predict(x["test"]))
    probabilities = best_model(x["test"], training=False).numpy()
    predictions = np.argmax(probabilities, axis=1)
    test_results[best_name] = metrics(y["test"], predictions)
    best_model.save(output / "model.keras")
    scaler.save(output / "scaler.json")
    splits = {name: rows.tolist() for name, rows in parts.items()}
    write_json(output / "splits.json", splits)
    test_rows = pd.DataFrame({"row": parts["test"], "actual": y["test"], "predicted": predictions})
    for i in range(5):
        test_rows[f"p{i}"] = probabilities[:, i]
    test_rows.to_csv(output / "test_predictions.csv", index=False)
    tracks = track_groups(frame)
    overlaps = {}
    for name, group in [("tracks", tracks), ("users", frame.user_id)]:
        sets = {part: set(group.iloc[rows]) for part, rows in parts.items()}
        overlaps[name] = {
            "train_test": len(sets["train"] & sets["test"]),
            "train_validation": len(sets["train"] & sets["validation"]),
            "validation_test": len(sets["validation"] & sets["test"]),
        }
    report = {
        "config": asdict(config),
        "dataset_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "rows": len(frame),
        "features": scaler.features,
        "split_sizes": {name: len(rows) for name, rows in parts.items()},
        "class_counts": {name: np.bincount(y[name], minlength=5).tolist() for name in parts},
        "group_overlap": overlaps,
        "selected_model": best_name,
        "selection_metric": "validation macro F1",
        "candidates": candidates,
        "test": test_results,
        "python": platform.python_version(),
        "versions": {
            package: importlib.metadata.version(package)
            for package in ["tensorflow", "keras", "scikit-learn", "numpy", "pandas", "matplotlib"]
        },
    }
    write_json(output / "metrics.json", report)
    plot_report(report, output)
    write_markdown(report, output)
    return report


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def predict(model_path, data_path, output_path=None):
    directory = Path(model_path)
    frame = pd.read_csv(data_path)
    scaler = SensorScaler.load(directory / "scaler.json")
    x = scaler.transform(frame)
    if len(frame) == 0:
        raise ValueError("El CSV de predicción está vacío.")
    tf = tensorflow()
    model = tf.keras.models.load_model(directory / "model.keras", compile=False, safe_mode=True)
    probabilities = model(x, training=False).numpy()
    labels = np.argmax(probabilities, axis=1)
    result = pd.DataFrame({"label": labels, "class": [CLASS_NAMES[i] for i in labels]})
    for i in range(5):
        result[f"p{i}"] = probabilities[:, i]
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
    return result


def plot_report(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = report["selected_model"]
    matrix = np.array(report["test"][selected]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.imshow(matrix, cmap="Blues")
    ax.set(
        xticks=range(5),
        yticks=range(5),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicción",
        ylabel="Clase real",
        title=f"{selected} · test por trayectorias"
        if report["config"]["strategy"] == "track"
        else f"{selected} · test",
    )
    for i in range(5):
        for j in range(5):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > matrix.max() / 2 else "#15283b",
            )
    fig.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for name, candidate in report["candidates"].items():
        ax.plot(range(1, candidate["epochs_run"] + 1), candidate["history"]["val_loss"], label=name)
    ax.set(
        xlabel="Época", ylabel="Entropía cruzada de validación", title="Selección con validación"
    )
    ax.legend()
    fig.savefig(output / "validation_loss.png", dpi=160)
    plt.close(fig)


def write_markdown(report, output):
    lines = [
        "# Resultado reproducible",
        "",
        f"Semilla: {report['config']['seed']}. "
        f"Partición: **{report['config']['strategy']}**. "
        f"Modelo seleccionado con F1 macro de validación: **{report['selected_model']}**.",
        "",
        "| Modelo | Accuracy test | F1 macro test |",
        "|---|---:|---:|",
    ]
    for name, result in report["test"].items():
        lines.append(f"| {name} | {result['accuracy']:.4f} | {result['macro_f1']:.4f} |")
    lines.extend(
        [
            "",
            "![Matriz de confusión](confusion_matrix.png)",
            "",
            "![Pérdida de validación](validation_loss.png)",
            "",
            "Las trayectorias y los usuarios compartidos se detallan en metrics.json. "
            "La partición por trayectoria permite usuarios comunes entre conjuntos; "
            "no mide la generalización a personas nuevas.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
