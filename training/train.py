from doctest import Example
import pathlib
from typing import Any, Mapping, Iterator
import enum
import functools


# We import JAX and some related packages.
# import chex
# import jax
# import jax.numpy as jnp
# import optax

import torch

# We will use tensorflow to handle the dataset
import tensorflow as tf
import tensorflow_datasets as tfds

# Finally, we import Recurrentgemma.
import sentencepiece as spm
from recurrentgemma import jax as recurrentgemma



class Tokenizer():
    
    def __init__(self, spm_processor):
        super().__init__()
        self._spm_processor = spm_processor
    
    @property
    def pad_id(self):
        return self._spm_processor.pad_id()
    
    def tokenize(
        self,
        input,
        add_eos=True
    ):
        
        int_list = self._spm_processor.EncodeAsIds(input)
        if add_eos:
            int_list.append(self._spm_processor.eos_id())
        return int_list

    def tokenize_tf_op(
        self,
        str_tensor: tf.Tensor,
        prefix: str = '',
        suffix: str = '',
        add_eos: bool = True,
    ) -> tf.Tensor:
        """Tensforflow operator for the tokenize function."""
        encoded = tf.numpy_function(
            self.tokenize,
            [str_tensor, prefix, suffix, add_eos],
            tf.int32)
        encoded.set_shape([None])
        return encoded

    def to_string(self, tokens) -> str:
        """Convert an array of tokens to a string."""
        ## TODO running this line will probably error out since spp expects a list of ints
        return self._spm_processor.DecodeIds(tokens)


if __name__ == "__main__":
    vocab = spm.SentencePieceProcessor()
    vocab.load("../model/tokenizer.model")
    
    tokeninzer = Tokenizer(vocab)

    temp = tokeninzer.tokenize("HEllo world")
    
    print(temp)
    
    temp = tokeninzer.to_string(temp)
    
    print(temp)
    