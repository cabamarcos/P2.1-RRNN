"""Validate, partition and scale sensor data without fitting on the holdout."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

from transport import CLASS_NAMES

IDENTIFIERS = ["user_id", "track_id"]


def load_data(path):
    frame = pd.read_csv(path)
    required = {*IDENTIFIERS, "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Faltan columnas: {sorted(required - set(frame.columns))}")
    features = frame.drop(columns=[*IDENTIFIERS, "label"])
    if features.shape[1] == 0 or len(frame) == 0:
        raise ValueError("El dataset necesita filas y atributos de sensores.")
    try:
        values = features.to_numpy(dtype=float)
        labels = frame["label"].to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Los sensores y las etiquetas deben ser numéricos.") from exc
    if not np.isfinite(values).all() or not np.isfinite(labels).all():
        raise ValueError("No se admiten valores vacíos, NaN o infinitos.")
    if frame[IDENTIFIERS].isna().any().any():
        raise ValueError("Los identificadores no pueden estar vacíos.")
    if not np.isin(labels, np.arange(len(CLASS_NAMES))).all():
        raise ValueError("label debe ser un entero entre 0 y 4.")
    if set(labels.astype(int)) != set(range(len(CLASS_NAMES))):
        raise ValueError("El entrenamiento necesita las cinco clases.")
    return frame


def track_groups(frame):
    return frame["user_id"].astype(str) + ":" + frame["track_id"].astype(str)


def partition(frame, seed=42, strategy="track"):
    """Approximately 53.3/13.3/33.3%; grouped ratios are measured in groups."""
    indices = np.arange(len(frame))
    if strategy == "row":
        train, test = train_test_split(
            indices, test_size=1 / 3, stratify=frame.label, random_state=seed
        )
        train, validation = train_test_split(
            train, test_size=0.2, stratify=frame.label.iloc[train], random_state=seed
        )
    elif strategy in {"track", "user"}:
        groups = track_groups(frame) if strategy == "track" else frame.user_id.astype(str)
        first = GroupShuffleSplit(n_splits=1, test_size=1 / 3, random_state=seed)
        remaining, test = next(first.split(indices, groups=groups))
        second = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_local, validation_local = next(second.split(remaining, groups=groups.iloc[remaining]))
        train, validation = remaining[train_local], remaining[validation_local]
    else:
        raise ValueError("La partición debe ser track, user o row.")
    parts = {"train": train, "validation": validation, "test": test}
    for name, rows in parts.items():
        if set(frame.label.iloc[rows]) != set(range(len(CLASS_NAMES))):
            raise ValueError(f"Faltan clases en {name}; usa otra semilla o más datos.")
    return parts


@dataclass
class SensorScaler:
    features: list[str]
    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, frame):
        sensors = frame.drop(columns=[*IDENTIFIERS, "label"])
        scaler = StandardScaler().fit(sensors)
        return cls(list(sensors.columns), scaler.mean_, scaler.scale_)

    def transform(self, frame):
        missing = set(self.features) - set(frame.columns)
        if missing:
            raise ValueError(f"Faltan sensores: {sorted(missing)}")
        try:
            values = frame[self.features].to_numpy(dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError("Los sensores deben ser numéricos.") from exc
        with np.errstate(over="ignore", invalid="ignore"):
            scaled = ((values - self.center) / self.scale).astype(np.float32)
        if not np.isfinite(scaled).all():
            raise ValueError("Los sensores deben ser finitos y estar dentro del rango numérico.")
        return scaled

    def save(self, path):
        Path(path).write_text(
            json.dumps(
                {
                    "features": self.features,
                    "center": self.center.tolist(),
                    "scale": self.scale.tolist(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        features = data["features"]
        center, scale = np.array(data["center"]), np.array(data["scale"])
        if (
            len(set(features)) != len(features)
            or len(features) == 0
            or center.shape != (len(features),)
            or scale.shape != center.shape
            or not np.isfinite(center).all()
            or not np.isfinite(scale).all()
            or (scale <= 0).any()
        ):
            raise ValueError("El fichero de normalización no es válido.")
        return cls(features, center, scale)
