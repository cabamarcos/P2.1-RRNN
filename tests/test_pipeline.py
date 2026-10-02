import json

import numpy as np
import pandas as pd
import pytest

from transport.pipeline import TrainConfig, predict, run_experiment


def test_training_serialization_and_frozen_holdout(tmp_path):
    rng = np.random.default_rng(9)
    labels = np.tile(np.arange(5), 60)
    frame = pd.DataFrame(
        {
            "user_id": np.repeat(np.arange(30), 10),
            "track_id": np.repeat(np.arange(60), 5),
            "speed": labels + rng.normal(0, 0.1, len(labels)),
            "turns": rng.normal(size=len(labels)),
            "label": labels,
        }
    )
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    output = tmp_path / "experiment"
    report = run_experiment(path, output, TrainConfig(epochs=2, patience=1, batch_size=32))
    assert report["group_overlap"]["tracks"]["train_test"] == 0
    assert report["selected_model"] in report["candidates"]
    assert report["selection_metric"] == "validation macro F1"
    split = json.loads((output / "splits.json").read_text())
    frame.iloc[split["test"]].drop(columns="label").to_csv(tmp_path / "holdout.csv", index=False)
    predictions = predict(output, tmp_path / "holdout.csv", tmp_path / "predictions.csv")
    recorded = pd.read_csv(output / "test_predictions.csv")
    np.testing.assert_array_equal(predictions.label, recorded.predicted)
    np.testing.assert_allclose(predictions[[f"p{i}" for i in range(5)]].sum(axis=1), 1, atol=1e-6)
    matrix = report["test"][report["selected_model"]]["confusion_matrix"]
    assert np.sum(matrix) == len(split["test"])
    assert (output / "confusion_matrix.png").stat().st_size > 1000
    assert (output / "report.md").exists()
    with pytest.raises(FileExistsError):
        run_experiment(path, output)


@pytest.mark.parametrize("argument", ["epochs", "patience", "batch_size"])
def test_invalid_training_config(argument):
    with pytest.raises(ValueError, match="positivos"):
        TrainConfig(**{argument: 0})
