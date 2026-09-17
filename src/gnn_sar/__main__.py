"""``python -m gnn_sar`` — same entry point as the ``gnn-sar`` script."""

from gnn_sar.training.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
