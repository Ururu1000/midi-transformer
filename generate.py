"""Deprecated shim — use `musiclm-generate` (or `python -m musiclm.inference.cli`)."""
from musiclm.inference.cli import main

if __name__ == "__main__":
    main()
