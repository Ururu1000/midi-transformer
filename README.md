# Symbolic Music Language Model

[License: MIT](LICENSE)
[Python 3.10+](https://www.python.org/downloads/)
[PyTorch](https://pytorch.org/)

An advanced **AI-powered Autoregressive Symbolic Music Generator** built with a custom LLaMA-style Transformer architecture in PyTorch. The model is trained on classical piano MIDI corpora (MAESTRO v3.0.0 and GiantMIDI-Piano) using BPE-compressed REMI tokenization, Classifier-Free Guidance (CFG), and state-of-the-art sampling techniques to compose high-quality, composer-conditioned classical piano music.

---

## Key Features

- **LLaMA-Style Transformer Architecture**: Implemented from scratch featuring **Rotary Position Embeddings (RoPE)**, **SwiGLU** feed-forward networks, **RMSNorm**, and weight-tied embedding heads.
- **Block-Diagonal Attention Masking**: Bin-packs multiple MIDI pieces into fixed-length 4096-token sequence rows with custom attention masks (`doc_ids`) to prevent cross-piece attention leakage without wasting compute on zero-padding.
- **BPE-Compressed REMI Tokenization**: Customized `ComposerREMI` representation with **Byte-Pair Encoding (2,048 vocabulary size)**, 12 positions per beat (triplets & rubato support), rests, chords, and pitch-shift data augmentations ($-6$ to $+5$ semitones).
- **Composer-Conditioned Generation**: Steering via **Classifier-Free Guidance (CFG)** across top classical composers (e.g., *Frédéric Chopin*, *Johann Sebastian Bach*, *Ludwig van Beethoven*, *Franz Liszt*, *Claude Debussy*, *Sergei Rachmaninoff*, etc.).
- **Advanced Sampling & Inference**:
  - **KV-Caching** for fast step-by-step autoregressive decoding.
  - **Min-P Sampling** to eliminate low-probability long-tail tokens without distoring confidence scaling.
  - **Pitch-Only Repetition Penalties** to prevent melodic stagnation without destroying rhythmic grid integrity (`Bar`, `Position`, `Tempo`).
- **Statistical Evaluation Suite**: Evaluates generated MIDI files against ground truth corpora using **Pitch Class Histograms (PCH)**, **KL Divergence**, note density, and active duration metrics.

---



## Repository Structure

```text
ai-music-project/
├── model.py                # PyTorch MusicTransformer architecture (RoPE, SwiGLU, RMSNorm, KV-Cache)
├── train.py                # Training loop (AMP, Gradient Accumulation, W&B, Early Stopping)
├── generate.py             # Inference engine (KV-Cache decoding, CFG, Min-P, Repetition Penalty)
├── generate_batch.py       # Batch generation script for bulk sampling across composers
├── evaluate.py             # Quantitative evaluation (KL Divergence, Pitch Class Histograms, Note Density)
├── scripts/
│   └── tokenize_midi.py    # MIDI dataset preprocessing, filtering, BPE tokenization & augmentation
├── checkpoints/            # Directory for trained model weights (e.g., model_best.pt)
├── data/
│   ├── raw_midi/           # Datasets (MAESTRO v3.0.0 & GiantMIDI-Piano)
│   ├── processed/          # BPE tokenizer, composer mappings, tokenized PyTorch tensors
│   └── generated/          # Output generated MIDI files
└── LICENSE                 # MIT License
```

---



## Quick Start



### 1. Installation & Setup

Ensure you have Python 3.10+ and PyTorch installed:

```bash
git clone https://github.com/Ururu1000/midi-transformer.git
cd midi-transformer

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch miditok pretty_midi pandas scipy wandb
```



### 2. Dataset Preparation & Tokenization

Download the [MAESTRO v3.0.0 dataset](https://magenta.tensorflow.org/datasets/maestro) or GiantMIDI into `data/raw_midi/`, then run the tokenization pipeline:

```bash
python scripts/tokenize_midi.py
```

This pipeline cleans MIDI files, applies quality gates (polyphony & note density limits), builds the BPE vocabulary, transposes audio across 12 semitones, and saves tokenized PyTorch tensors (`tokens_train.pt`, `tokens_val.pt`).

---



## Training

To launch model training:

```bash
python train.py
```

Key training parameters (configured in `train.py`):

- `PACK_SEQ_LEN`: 4096 tokens
- `BATCH_SIZE`: 8 (Effective batch size ~56 via gradient accumulation)
- `LEARNING_RATE`: 2e-4 with Cosine Warmup Scheduler
- `CFG_DROP_PROB`: 15% probability of dropping composer tokens for CFG learning
- `EARLY_STOP_PATIENCE`: 5 validation epochs

Training loss and evaluation metrics will log automatically to **Weights & Biases (W&B)**.

---



## Music Generation

Generate new classical piano compositions using a trained checkpoint:

```bash
python generate.py
```



### Sampling Parameters (`generate.py`)

```python
COMPOSER = "Frédéric Chopin"   # Composer steering prompt
GENERATION_LENGTH = 1024       # Output sequence length
TEMPERATURE = 0.95             # Sampling temperature
MIN_P = 0.03                   # Min-P confidence threshold
CFG_SCALE = 1.2                # Classifier-Free Guidance strength
PENALTY_WINDOW = 64            # Pitch repetition penalty context window
```

To generate samples across multiple composers concurrently:

```bash
python generate_batch.py
```

Generated `.mid` files will be saved in `data/generated/`. You can play them using any standard MIDI player or Digital Audio Workstation (DAW) like Ableton, FL Studio, or GarageBand.

---



## Quantitative Evaluation

Evaluate the musical quality of generated samples against the validation dataset:

```bash
python evaluate.py
```

The script calculates:

- **Pitch Class Histogram (PCH) Distribution**: Measuring tonal center adherence.
- **KL Divergence**: Statistical similarity between generated pitch distributions and classical compositions.
- **Note Density & Active Duration**: Ensuring realistic note rates without silent gaps or excessive clutter.

---



## Model Architecture Details


| Parameter     | Value  | Description                   |
| ------------- | ------ | ----------------------------- |
| `vocab_size`  | ~2,048 | BPE Tokenizer Vocabulary      |
| `d_model`     | 768    | Hidden Embedding Dimension    |
| `nhead`       | 12     | Multi-Head Attention Heads    |
| `num_layers`  | 24     | Transformer Decoder Blocks    |
| `d_ff`        | 3,072  | SwiGLU Hidden Layer Dimension |
| `max_seq_len` | 4,096  | Context Window Length         |


---



## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.