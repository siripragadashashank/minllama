import os
import sys
import unittest
from pathlib import Path
from tokenizer import LLaMATokenizer


class TestLLaMATokenizer(unittest.TestCase):
    """
    a primitive Test for LLaMA tokenizer
    """

    def test_encode_decode(self):
        # hacky import for SPM tokenizer
        tokenizer_path = Path(__file__).parent.parent
        tok_path = os.path.join(tokenizer_path, 'tokenizer', 'tokenizer.model')
        str_to_encode = "SentencePiece is a tricky tokenizer to work with!"
        tokenizer = LLaMATokenizer(spm_model_path=tok_path)
        tokens = tokenizer.encode(str_to_encode)
        str_output = tokenizer.decode(tokens)
        self.assertEqual(str_output, str_to_encode)


