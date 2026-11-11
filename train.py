import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tokenization import SimpleTokenizer, SubwordTokenizer
from transformer import TransformerLM


class NextTokenDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, context_length: int) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if tokens.numel() <= context_length:
            raise ValueError("The corpus must contain more tokens than context_length")
        self.tokens = tokens
        self.context_length = context_length

    def __len__(self) -> int:
        return (self.tokens.numel() - 1) // self.context_length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.context_length
        end = start + self.context_length
        return self.tokens[start:end], self.tokens[start + 1 : end + 1]


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(name)


def train(args: argparse.Namespace) -> None:
    text = args.data.read_text(encoding="utf-8")
    tokenizer = SubwordTokenizer.train(text, args.vocab_size)
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    dataset = NextTokenDataset(tokens, args.context_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    config = {
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "max_seq_len": args.context_length,
        "dropout": args.dropout,
    }
    model = TransformerLM(vocab_size=tokenizer.vocab_size, **config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    print(
        f"Training on {tokens.numel()} subword tokens "
        f"(vocabulary={tokenizer.vocab_size}) using {device}",
        flush=True,
    )

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        correct_tokens = 0
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, tokenizer.vocab_size), targets.reshape(-1)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()
            correct_tokens += (logits.detach().argmax(dim=-1) == targets).sum().item()

        print(
            f"epoch {epoch + 1}/{args.epochs} "
            f"loss={total_loss / total_tokens:.4f} "
            f"accuracy={correct_tokens / total_tokens:.2%}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "config": config,
        "tokenizer_json": tokenizer.to_str(),
    }
    torch.save(checkpoint, args.output)
    print(f"Saved checkpoint to {args.output}")


def generate(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if "tokenizer_json" in checkpoint:
        tokenizer = SubwordTokenizer.from_str(checkpoint["tokenizer_json"])
    else:
        word_to_id = checkpoint["word_to_id"]
        tokenizer = SimpleTokenizer()
        tokenizer.word_to_id = word_to_id
        tokenizer.id_to_word = {
            token_id: word for word, token_id in word_to_id.items()
        }
        tokenizer.vocab_size = len(word_to_id)

    model = TransformerLM(vocab_size=tokenizer.vocab_size, **checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        raise ValueError("The prompt must contain at least one word")
    token_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = model.generate(
        token_ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature
    )
    print(tokenizer.decode(generated[0].cpu().tolist()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or run the scratch Transformer LM")
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="Train from a UTF-8 text corpus")
    train_parser.add_argument("--data", type=Path, default=Path("data/hf_corpus.txt"))
    train_parser.add_argument("--output", type=Path, default=Path("checkpoints/bpe_model.pt"))
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--context-length", type=int, default=64)
    train_parser.add_argument("--vocab-size", type=int, default=4096)
    train_parser.add_argument("--d-model", type=int, default=32)
    train_parser.add_argument("--num-heads", type=int, default=4)
    train_parser.add_argument("--num-layers", type=int, default=2)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train_parser.set_defaults(run=train)

    generate_parser = commands.add_parser("generate", help="Generate text from a checkpoint")
    generate_parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/bpe_model.pt"))
    generate_parser.add_argument("--prompt", required=True)
    generate_parser.add_argument("--max-new-tokens", type=int, default=20)
    generate_parser.add_argument("--temperature", type=float, default=0.8)
    generate_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    generate_parser.set_defaults(run=generate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.run(args)


if __name__ == "__main__":
    main()