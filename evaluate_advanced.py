"""Deprecated shim — use `musiclm-eval` (or `python -m musiclm.evaluation.cli`)."""
from musiclm.evaluation.cli import main

if __name__ == "__main__":
    main()
