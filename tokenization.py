from typing import List, Dict

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers


class SubwordTokenizer:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    @classmethod
    def train(cls, text: str, vocab_size: int) -> "SubwordTokenizer":
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.normalizer = normalizers.Sequence(
            [normalizers.NFKC(), normalizers.Lowercase()]
        )
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[UNK]"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tokenizer.train_from_iterator([text], trainer=trainer)
        return cls(tokenizer)

    @classmethod
    def from_str(cls, serialized: str) -> "SubwordTokenizer":
        return cls(Tokenizer.from_str(serialized))

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def to_str(self) -> str:
        return self.tokenizer.to_str()

class SimpleTokenizer:
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """
        #initialize special tokens
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for i, token in enumerate(special_tokens):
            self.word_to_id[token] = i 
            self.id_to_word[i] = token

        #lowercase and remove whitespace
        unique_words = set()
        for text in texts:
            words = text.lower().split()
            unique_words.update(words)

        # sorted and assign idx
        for word in sorted(unique_words):
            idx = len(self.word_to_id)
            self.word_to_id[word] = idx
            self.id_to_word[idx] = word
            
        self.vocab_size = len(self.word_to_id)

    def encode(self, text: str) -> List[int]:
        """
        transform word to token id
        """
        words = text.lower().split()
        unk_id = self.word_to_id[self.unk_token]
        return [self.word_to_id.get(word, unk_id) for word in words]

    def decode(self, ids: List[int]) -> str:
        """convert token id to word"""
        return " ".join(self.id_to_word.get(idx, self.unk_token) for idx in ids)