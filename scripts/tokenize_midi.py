"""Deprecated shim — use `musiclm-tokenize` (or `python -m musiclm.data.preprocess`)."""
from musiclm.data.preprocess import main

if __name__ == "__main__":
    main()
