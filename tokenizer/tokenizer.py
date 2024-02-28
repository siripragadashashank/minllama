"""
LLaMA tokenizer based on SentencePiece
"""
import argparse
from typing import List

# sentencepiece import
from sentencepiece import SentencePieceProcessor

SPM_TOKENIZER = "tokenizer.model"


class LLaMATokenizer:
    def __init__(self, max_len=None, spm_model_path=None):
        """
        :param max_len: Max length for a token sequence
        :param spm_model_path: Path to load an existing SPM Model
        """
        # check if path exists or use the default tokenizer in directory
        model_path = spm_model_path if spm_model_path else SPM_TOKENIZER

        self.max_len = max_len
        self.spm_model = SentencePieceProcessor(model_file=model_path)
        self.model_path = model_path

        # deal with gnarly SPM special tokens
        self.vocab_size: int = self.spm_model.vocab_size()
        self.bos_id: int = self.spm_model.bos_id()
        self.eos_id: int = self.spm_model.eos_id()
        # overwrite the default pad_id token
        self.pad_id: int = self.spm_model.piece_to_id("<0x00>")

    def encode(self, inp: str, bos: bool = None, eos: bool = None) -> List[int]:
        assert type(inp) is str
        tokens = self.spm_model.encode(inp)
        # deal with max length
        if self.max_len is not None and len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        return tokens

    def decode(self, tokens: List[int]) -> str:
        return self.spm_model.decode(tokens)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--spm-model-path", type=str, help="path to custom tokenizer")
    args = parser.parse_args()

    str_to_encode = "SentencePiece is a tricky tokenizer to work with!"
    tokenizer_model = LLaMATokenizer(args.spm_model_path)
    toks = tokenizer_model.encode(str_to_encode)
    print(toks)


