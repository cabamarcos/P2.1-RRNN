import json

import numpy as np
import pandas as pd
import pytest

from transport.data import SensorScaler, load_data, partition, track_groups


@pytest.fixture
def frame():
    rng = np.random.default_rng(5)
    return pd.DataFrame(
        {
            "user_id": np.repeat(np.arange(30), 20),
            "track_id": np.repeat(np.arange(60), 10),
            "speed": rng.normal(size=600),
            "constant": np.ones(600),
            "label": np.tile(np.arange(5), 120),
        }
    )


@pytest.mark.parametrize("strategy", ["track", "user", "row"])
def test_partitions_are_disjoint_and_reproducible(frame, strategy):
    parts = partition(frame, strategy=strategy)
    repeated = partition(frame, strategy=strategy)
    all_rows = np.concatenate(list(parts.values()))
    assert sorted(all_rows) == list(range(len(frame)))
    assert len(set(all_rows)) == len(frame)
    for name in parts:
        np.testing.assert_array_equal(parts[name], repeated[name])
        assert set(frame.label.iloc[parts[name]]) == set(range(5))
    if strategy != "row":
        groups = track_groups(frame) if strategy == "track" else frame.user_id
        sets = [set(groups.iloc[rows]) for rows in parts.values()]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])


def test_scaler_fits_training_only_and_roundtrips(frame, tmp_path):
    parts = partition(frame)
    scaler = SensorScaler.fit(frame.iloc[parts["train"]])
    assert scaler.features == ["speed", "constant"]
    train = scaler.transform(frame.iloc[parts["train"]])
    np.testing.assert_allclose(train.mean(axis=0), 0, atol=1e-6)
    assert (train[:, 1] == 0).all()
    holdout = frame.iloc[parts["test"]].copy()
    holdout["speed"] = 1e4
    before = scaler.center.copy()
    assert scaler.transform(holdout)[:, 0].min() > 1000
    np.testing.assert_array_equal(before, scaler.center)
    path = tmp_path / "scaler.json"
    scaler.save(path)
    np.testing.assert_array_equal(SensorScaler.load(path).transform(frame), scaler.transform(frame))


@pytest.mark.parametrize(
    "change", ["nan", "inf", "bad_label", "missing_id", "text", "absent_class"]
)
def test_invalid_datasets_fail(frame, tmp_path, change):
    if change in {"nan", "inf"}:
        frame.loc[0, "speed"] = np.nan if change == "nan" else np.inf
    elif change == "bad_label":
        frame["label"] = frame.label.astype(float)
        frame.loc[0, "label"] = 0.5
    elif change == "missing_id":
        frame = frame.drop(columns="track_id")
    elif change == "text":
        frame["speed"] = "not a number"
    else:
        frame = frame[frame.label != 4]
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_data(path)


def test_load_rejects_missing_identifier_values(frame, tmp_path):
    frame.loc[0, "user_id"] = np.nan
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="identificadores"):
        load_data(path)


def test_inference_checks_features_and_scaler(frame, tmp_path):
    scaler = SensorScaler.fit(frame)
    with pytest.raises(ValueError, match="Faltan sensores"):
        scaler.transform(frame.drop(columns="speed"))
    frame.loc[0, "speed"] = np.inf
    with pytest.raises(ValueError, match="finitos"):
        scaler.transform(frame)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"features": ["speed"], "center": [0], "scale": [0]}))
    with pytest.raises(ValueError, match="no es válido"):
        SensorScaler.load(path)


def test_unknown_split(frame):
    with pytest.raises(ValueError, match="partición"):
        partition(frame, strategy="unknown")
