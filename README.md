# LLM from Scratch
 
> A ground-up implementation of the Transformer built with NumPy.
 
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
Tokenizer (BPE)  ──►  token ids
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
