"""Execute both current notebooks; historical archive is excluded."""

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="python3")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for name in ["P2.1_RRNN.ipynb", "RRNN2.ipynb"]:
        path = root / name
        notebook = nbformat.read(path, as_version=4)
        NotebookClient(
            notebook,
            timeout=180,
            kernel_name=args.kernel,
            resources={"metadata": {"path": str(root)}},
        ).execute()
        if args.save:
            nbformat.write(notebook, path)
        print(f"OK: {name}", flush=True)


if __name__ == "__main__":
    main()
