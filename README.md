# LLM from Scratch
 
> A learning implementation of Transformer building blocks, with a trainable PyTorch decoder-only language model.
 
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-in%20progress-orange)
 
---
**Goals**
 
- Implement each building block of the Transformer *without* relying on `nn.Transformer` or pre-built attention layers.
- Validate every component against reference behavior with unit tests.
- Train a small decoder-only language model end to end on a modest dataset.
- Keep the code readable and well-commented so it works as a learning resource.
---
## Architecture
 
The project follows the original *Attention Is All You Need* design.
 
```
Raw text
   │
   ▼
Tokenizer (byte-level BPE)  ──►  token ids
   │
   ▼
Token Embedding + Positional Encoding
   │
   ▼
┌──────────────────────────────────┐
│  Transformer Block  (× N)        │
│  ├─ Multi-Head Self-Attention    │
│  │    (causal mask in decoder)   │
│  ├─ Add & LayerNorm              │
│  ├─ Position-wise Feed-Forward   │
│  └─ Add & LayerNorm              │
└──────────────────────────────────┘
   │
   ▼
Linear projection  ──►  softmax  ──►  next-token probabilities
```
 
| Variant | Attention | Used for |
|---|---|---|
| Encoder block | Bidirectional self-attention | Understanding / embeddings |
| Decoder block | Masked self-attention (+ cross-attention in encoder-decoder setups) | Text generation |
 
---

## Run the model

Run the focused checks with:

```bash
python -m unittest test_transformer.py
```

`TransformerLM` accepts token IDs with shape `(batch_size, sequence_length)` and returns next-token logits for every position. For training, shift the input and target sequences by one token and apply cross-entropy to the logits. `generate` provides autoregressive sampling.

The runnable language model uses differentiable PyTorch layers. The NumPy modules remain standalone reference implementations of individual Transformer components.

## Train on the Hugging Face corpora

Install the dataset loader, stream a balanced sample from both datasets, and train:

```powershell
uv sync
.\.venv\Scripts\python.exe prepare_hf_corpus.py
.\.venv\Scripts\python.exe train.py train --epochs 10
.\.venv\Scripts\python.exe train.py generate --prompt "<fr> il était une fois" --max-new-tokens 40
```

The preparation script downloads raw UTF-8 files from `roettger/eighteenth_century_french_novels` and `agentlans/high-quality-english-sentences`, avoiding Windows locale-dependent decoding. By default it selects about 100,000 whitespace tokens from each dataset and prefixes rows with `<fr>` or `<en>`; use `--tokens-per-dataset 0` to include every row. The English training archive is about 85 MB compressed. The French collection is historical French, while the English set is modern sentence data, so this combined corpus trains a small multilingual model rather than a model specialized in one language. Training now learns a 4,096-token byte-level BPE vocabulary, which can represent rare and unseen words as subword pieces. Prompt with `<fr>` or `<en>` to steer the language. The new tokenizer is saved inside the checkpoint; the previous word-level `hf_model.pt` remains loadable but uses its original tokenizer and model weights.

To generate English, use a prompt beginning with `<en>`. The old synthetic-data generator remains available for a quick offline smoke test, but it is no longer the default training corpus.
