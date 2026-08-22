"""Deprecated shim — use `musiclm-train` (or `python -m musiclm.training.cli`)."""
from musiclm.training.cli import main

if __name__ == "__main__":
    main()
