# Symbolic Music Language Model

[License: MIT](LICENSE)

An **AI-powered autoregressive symbolic music generator** built with a custom LLaMA-style Transformer architecture in PyTorch. The model is trained on classical piano MIDI corpora (MAESTRO v3.0.0 and GiantMIDI-Piano) using BPE-compressed REMI tokenization, Classifier-Free Guidance (CFG), and modern sampling techniques to compose composer-conditioned classical piano music.

---

## Key Features

- **LLaMA-Style Transformer Architecture**: implemented from scratch — **Rotary Position Embeddings (RoPE)**, **SwiGLU** feed-forward networks, **RMSNorm**, and weight-tied embedding heads.
- **Block-Diagonal Attention Masking**: bin-packs multiple MIDI pieces into fixed-length 4096-token rows with `doc_ids` masks to prevent cross-piece attention leakage without wasting compute on padding.
- **BPE-Compressed REMI Tokenization**: customized `ComposerREMI` with **Byte-Pair Encoding (2,048 vocabulary)**, 12 positions per beat (triplets & rubato), rests, chords, and pitch-shift augmentation (−6 to +5 semitones).
- **Composer-Conditioned Generation**: steering via **Classifier-Free Guidance (CFG)** across top composers (*Chopin*, *Bach*, *Beethoven*, *Liszt*, *Debussy*, *Rachmaninoff*, …).
- **Advanced Sampling & Inference**: KV-caching, Min-P sampling, pitch-only repetition penalties that protect the rhythmic grid, optional seeding for reproducible runs.
- **Statistical Evaluation Suite**: 6 metric families compared against a validation corpus (see below).

---

## Repository Structure

```text
ai-music-project/
├── pyproject.toml            # Package metadata, pinned deps, ruff & pytest config
├── src/musiclm/
│   ├── config.py             # Paths + ModelConfig / TrainConfig / GenerateConfig
│   ├── model.py              # MusicTransformer (RoPE, SwiGLU, RMSNorm, KV-cache)
│   ├── data/
│   │   ├── tokenizer.py      # ComposerREMI + vocabulary helpers (miditok adapter)
│   │   ├── preprocess.py     # Dataset prep: metadata, filters, BPE training, tokenization
│   │   └── dataset.py        # PackedMusicDataset (bin-packing + doc_ids)
│   ├── training/
│   │   ├── trainer.py        # Training loop (AMP, grad accumulation, W&B, early stopping)
│   │   └── cli.py            # musiclm-train entry point
│   ├── inference/
│   │   ├── sampler.py        # Generation engine (KV-cache, CFG, min-p, rep penalty)
│   │   ├── cli.py            # musiclm-generate entry point
│   │   └── batch.py          # musiclm-batch preset runner
│   ├── evaluation/
│   │   ├── metrics.py        # Statistical metrics suite
│   │   └── cli.py            # musiclm-eval entry point
│   └── app.py                # Gradio web UI (musiclm-app)
├── tests/                    # pytest suite (mask, packing, sampler, metrics, tokenizer)
├── scripts/
│   ├── download_data.sh      # Fetch MAESTRO (+ GiantMIDI instructions)
│   └── upload_to_hf.py       # Publish checkpoint/tokenizer to Hugging Face Hub
├── checkpoints/              # Trained weights (not source-controlled)
└── data/                     # raw_midi / processed / generated (not source-controlled)
```

---

## Quick Start

### 1. Installation

Requires Python 3.9+:

```bash
git clone https://github.com/Ururu1000/music-generator.git
cd music-generator

python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate

pip install -e .                  # core library + CLIs
pip install -e '.[app]'           # + Gradio web app
pip install -e '.[dev]'           # + pytest/ruff for development
```

### 2. Dataset Preparation

```bash
bash scripts/download_data.sh     # downloads MAESTRO v3.0.0 into data/raw_midi/
musiclm-tokenize                  # quality gates, BPE vocab (~2048), token tensors
```

This produces `data/processed/tokens_train.pt`, `tokens_val.pt` and the tokenizer (with the learned BPE model serialized inside).

### 3. Training

```bash
musiclm-train                     # resume_mode=weights by default
```

Useful flags:

```bash
musiclm-train --resume-mode none --epochs 50    # fresh training run
musiclm-train --resume-mode full                # restore optimizer/scheduler/RNG too
musiclm-train --no-wandb                        # disable experiment tracking
```

Defaults (overridable via CLI): `pack_seq_len` 4096 · batch 8 × 16 accumulation = effective 128 · LR 2e-4 cosine with 5 warmup epochs · CFG dropout 15% · early-stop patience 5. Metrics log to Weights & Biases automatically.

### 4. Music Generation

```bash
musiclm-generate --composer "Frédéric Chopin" --length 1024 --seed 42 \
                 --output data/generated/chopin.mid
```

Sampling defaults (`src/musiclm/config.py::GenerateConfig`):

| Parameter | Default | Meaning |
|---|---|---|
| `temperature` | 0.92 | Sampling temperature |
| `min_p` | 0.02 | Min-P confidence threshold |
| `cfg_scale` | 1.2 | Classifier-Free Guidance strength (1.0 = off) |
| `repetition_penalty` | 1.0 | Pitch-only repetition suppression (window 64) |
| `seed` | none | Set it for reproducible sampling |

Bulk A/B sampling across composers:

```bash
musiclm-batch --output-dir data/generated
```

Play the `.mid` files in any MIDI player or DAW (Ableton, FL Studio, GarageBand).

### 5. Interactive Web App

```bash
musiclm-app
```

Gradio UI: pick a composer, tweak sliders, generate, preview audio, download MIDI.

### 6. Quantitative Evaluation

```bash
musiclm-eval data/generated data/raw_midi/validation --csv evaluation_summary.csv
```

Compares generated vs validation corpora across **6 metric families**:

- **Scale Consistency**: Krumhansl-Schmuckler key detection + in-key note ratio
- **Pitch Class Entropy**: Shannon entropy of 12-bin pitch-class histograms
- **Pitch Range**: semitone span statistics
- **Polyphony Rate & Note Density**: plus KL divergence vs validation distributions
- **Groove Consistency**: Inter-Onset Interval (IOI) statistics and histogram KLD
- **Structural Compression**: LZ77 compression ratios of pitch-duration patterns

---

## Model Architecture Details

| Parameter     | Value  | Description                   |
| ------------- | ------ | ----------------------------- |
| `vocab_size`  | ~2,048 | BPE Tokenizer Vocabulary      |
| `d_model`     | 768    | Hidden Embedding Dimension    |
| `nhead`       | 12     | Attention Heads               |
| `num_layers`  | 24     | Transformer Decoder Blocks    |
| `d_ff`        | 3,072  | SwiGLU Hidden Dimension       |
| `max_seq_len` | 4,096  | Context Window Length         |

## Development

```bash
pytest                            # run the test suite (CPU-only, fast)
ruff check src/ tests/            # lint
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
