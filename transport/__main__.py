import argparse

from transport.pipeline import TrainConfig, predict, run_experiment


def main():
    parser = argparse.ArgumentParser(description="Clasificación de medios de transporte con MLP.")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Entrenar, comparar y guardar el modelo seleccionado")
    train.add_argument("--data", default="dataRRNN2.csv")
    train.add_argument("--output", default="runs/track-42")
    train.add_argument("--split", choices=["track", "user", "row"], default="track")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--patience", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=128)
    inference = commands.add_parser("predict", help="Clasificar un CSV con un modelo guardado")
    inference.add_argument("--model", required=True)
    inference.add_argument("--data", required=True)
    inference.add_argument("--output", default="runs/predictions.csv")
    args = parser.parse_args()
    try:
        if args.command == "train":
            report = run_experiment(
                args.data,
                args.output,
                TrainConfig(
                    seed=args.seed,
                    strategy=args.split,
                    epochs=args.epochs,
                    patience=args.patience,
                    batch_size=args.batch_size,
                ),
            )
            chosen = report["selected_model"]
            print(
                f"Guardado en {args.output}. {chosen}: "
                f"F1 macro test = {report['test'][chosen]['macro_f1']:.4f}"
            )
        else:
            result = predict(args.model, args.data, args.output)
            print(f"Guardadas {len(result)} predicciones en {args.output}")
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
