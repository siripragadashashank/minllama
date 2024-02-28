"""
LLaMA tokenizer based on SentencePiece
"""
import os
import struct
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
        # overwrite the default pad_id token to 0
        self.pad_id: int = self.spm_model.piece_to_id()







if __name__ == '__main__':
    