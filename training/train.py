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

from attr import dataclass
from numpy import dtype
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
    
    
@dataclass   
class TrainingInput:
    # Input tokens given to the model
    input_tokens: torch.Tensor

    # A mask that determines which tokens contribute to the target loss
    # calculation.
    target_mask: torch.Tensor

class DatasetSplit(enum.Enum):
  TRAIN = 'train'
  VALIDATION = 'valid'
  
class DatasetBuilder:
     
    N_ITEMS = 0
    BUFFER_SIZE_SHUFFLE = 10_000
     
    def __init__(self,
                tokenizer: Tokenizer,
                max_seq_len: int):
        """Constructor.

        Args:
        tokenizer: Gemma tokenizer to use.
        max_seq_len: size of each sequence in a given batch.
        """
        self.tokenizer = tokenizer
        
        self._base_data = {
        # DatasetSplit.TRAIN: tfds.load("mtnt/en-fr",split="train"),
        # DatasetSplit.VALIDATION: tfds.load("mtnt/en-fr",split="valid"),
        }
        self._max_seq_len = max_seq_len
        

    def _tokenize_source(self, example: tf.Tensor):
        """Tokenization function for the source."""
        return self._tokenizer.tokenize_tf_op(
            example, prefix=self.TRANSLATION_PREFIX, suffix=self.TRANSLATION_SUFFIX,
            add_eos=False
        )

    def _tokenize_destination(self, example: tf.Tensor):
        """Tokenization function for the French translation."""
        return self._tokenizer.tokenize_tf_op(example, add_eos=True)
    

    def _pad_up_to_max_len(self,
        input_tensor: tf.Tensor,
        pad_value: int | bool,
    ) -> tf.Tensor:
        """Pad the given tensor up to sequence length of a batch."""
        seq_len = tf.shape(input_tensor)[0]
        to_pad = tf.maximum(self._max_seq_len - seq_len, 0)
        return tf.pad(
        input_tensor, [[0, to_pad]], mode='CONSTANT', constant_values=pad_value,
        )
        
        
    def _to_training_input(
        self,
        src_tokens: torch.Tensor,
        dst_tokens: torch.Tensor,
    ) -> TrainingInput:
        """Build a training input from a tuple of source and destination tokens."""

        # The input sequence fed to the model is simply the concatenation of the
        # source and the destination.
        tokens = tf.concat([src_tokens, dst_tokens], axis=0)

        # We want to prevent the model from updating based on the source (input)
        # tokens. To achieve this, we add a target mask to each input.
        q_mask = tf.zeros_like(src_tokens, dtype=tf.bool)
        a_mask = tf.ones_like(dst_tokens, dtype=tf.bool)
        mask = tf.concat([q_mask, a_mask], axis=0)

        # If the output tokens sequence is smaller than the target sequence size,
        # then we pad it with pad tokens.
        tokens = self._pad_up_to_max_len(tokens, self._tokenizer.pad_id)

        # We don't want to perform the backward on the pad tokens.
        mask = self._pad_up_to_max_len(mask, False)

        return TrainingInput(input_tokens=tokens, target_mask=mask)


class Sampler:
    
    def __init__(self,
                 model,
                 tokenizer: Tokenizer,
                 device,
                 greedy_sampling=True,
                 ):
        self.model = model
        self.vocab = tokenizer
        self.device = device
        self.greedy_sampling = greedy_sampling
        self._eos_token = torch.tensor([self.vocab.eos_id()], device=self.device)
        self.tokenizer = tokeninzer
        
    def __call__(self,
                 img_path,
                 input_string,
                 total_generation_steps,
                 end_sampling_at_eos_token: bool = True,
                 ):
        
        in_tokens = self.tokenizer.tokenize(input_string)
        
        in_len = torch.Tensor(len(in_tokens), device=self.device, dtype=torch.int32) + 1
        

if __name__ == "__main__":
    vocab = spm.SentencePieceProcessor()
    vocab.load("../model/tokenizer.model")
    
    tokeninzer = Tokenizer(vocab)

    db = DatasetBuilder(tokenizer=tokeninzer, max_seq_len=100)
    
    device = "cuda:1"
    
    params = torch.load("./model/2b-it.pt")
    params = {k: v.to(device=device) for k, v in params.items()}
    
    config = recurrentgemma.GriffinConfig.from_torch_params(
        params,
        preset=recurrentgemma.Preset.RECURRENT_GEMMA_2B_V1,
    )
    
    model = recurrentgemma.Griffin(config, device=device, dtype=torch.bfloat16)
    
    model.load_state_dict(params, strict=False)
    
    sampler = Sampler(model=model, tokeninzer=tokeninzer, device=device)
    
    