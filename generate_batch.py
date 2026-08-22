"""Deprecated shim — use `musiclm-batch` (or `python -m musiclm.inference.batch`)."""
from musiclm.inference.batch import main

if __name__ == "__main__":
    main()
