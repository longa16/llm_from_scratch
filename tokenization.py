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