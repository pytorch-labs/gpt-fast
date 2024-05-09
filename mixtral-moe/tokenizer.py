import os
import sentencepiece as spm
import tiktoken
from tiktoken.load import load_tiktoken_bpe
from pathlib import Path
from typing import Dict
from transformers import GPT2TokenizerFast

class TokenizerInterface:
    def __init__(self, model_path):
        self.model_path = os.path.join(model_path, "tokenizer.model")

    def encode(self, text):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def decode(self, tokens):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def bos_id(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def eos_id(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

class SentencePieceWrapper(TokenizerInterface):
    def __init__(self, model_path):
        super().__init__(model_path)
        assert os.path.isfile(self.model_path), str(self.model_path)
        self.processor = spm.SentencePieceProcessor(str(self.model_path))

    def encode(self, text):
        return self.processor.EncodeAsIds(text)

    def decode(self, tokens):
        return self.processor.DecodeIds(tokens)

    def bos_id(self):
        return self.processor.bos_id()

    def eos_id(self):
        return self.processor.eos_id()

class DBRXTokenizeWrapper(TokenizerInterface):
    def __init__(self, model_path):
        super().__init__(model_path)
        vocab_file = os.path.join(model_path.parent, "vocab.json")
        merges_file = os.path.join(model_path.parent, "merges.txt")
        tokenizer_file = os.path.join(model_path.parent, "tokenizer.json")
        self.processor = GPT2TokenizerFast(vocab_file, merges_file, tokenizer_file)

    def encode(self, text):
        return self.processor.encode(text)

    def decode(self, tokens):
        return self.processor.decode(tokens)

    def bos_id(self):
        return self.processor.bos_token_id

    def eos_id(self):
        return self.processor.eos_token_id

class TiktokenWrapper(TokenizerInterface):
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: Dict[str, int]

    num_reserved_special_tokens = 256

    pat_str = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""  # noqa: E501

    def __init__(self, model_path):
        super().__init__(model_path)
        assert os.path.isfile(self.model_path), str(self.model_path)
        mergeable_ranks = load_tiktoken_bpe(str(self.model_path))
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
            "<|endoftext|>",
            "<|pad|>",
        ]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=Path(self.model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        # BOS / EOS token IDs
        self._bos_id: int = self.special_tokens["<|endoftext|>"]
        self._eos_id: int = self.special_tokens["<|endoftext|>"]

    def encode(self, text):
        return self.model.encode(text)

    def decode(self, tokens):
        return self.model.decode(tokens)

    def bos_id(self):
        return self._bos_id

    def eos_id(self):
        return self._eos_id

def get_tokenizer(tokenizer_model_path, model_name):
    """
    Factory function to get the appropriate tokenizer based on the model name.

    Args:
    - tokenizer_model_path (str): The file path to the tokenizer model.
    - model_name (str): The name of the model, used to determine the tokenizer type.

    Returns:
    - TokenizerInterface: An instance of a tokenizer.
    """
    if "Llama-3" in str(model_name):
        return TiktokenWrapper(tokenizer_model_path)
    elif "dbrx" in str(model_name):
        return DBRXTokenizeWrapper(tokenizer_model_path)
    else:
        return SentencePieceWrapper(tokenizer_model_path)
