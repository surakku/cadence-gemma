from doctest import Example
import pathlib
from typing import Any, Mapping, Iterator
import enum
import functools

import json

import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

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
from datasets import load_dataset

# Finally, we import Recurrentgemma.
import sentencepiece as spm
from recurrentgemma import torch as recurrentgemma
def tokenize_source(tokenizer, example: tf.Tensor):
  return tokenizer.tokenize_tf_op(
      example,
      add_eos=False
  )
  
def tokenize_destination(tokenizer, example: tf.Tensor):
  return tokenizer.tokenize_tf_op(example, add_eos=True)

def load_json_dataset(json_file):
    # Load JSON data from file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract features and labels from JSON data (example)
    imgs = [entry['image'] for entry in data]
    qs = [entry['question'] for entry in data]
    ans = [entry['answers'][0]['answer'] for entry in data]
    # Create TensorFlow Dataset from extracted features and labels
    dataset = tf.data.Dataset.from_tensor_slices((imgs, qs, ans))
    

    return dataset



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
        int_list = self._spm_processor.EncodeAsIds(str(input))
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
            [str_tensor, add_eos],
            tf.int32)
        encoded.set_shape([None])
        return encoded

    def to_string(self, tokens) -> str:
        """Convert an array of tokens to a string."""
        ## TODO running this line will probably error out since spp expects a list of ints
        return self._spm_processor.DecodeIds(tokens)
    
    
@dataclass   
class TrainingInput:
    
    image: str
    # Input tokens given to the model
    input_tokens: torch.Tensor

    # A mask that determines which tokens contribute to the target loss
    # calculation.
    target_mask: torch.Tensor

class DatasetSplit(enum.Enum):
  TRAIN = 'train'
  VALIDATION = 'valid'
  
class DatasetBuilder:
     
    N_ITEMS = {DatasetSplit.TRAIN: 20_000, DatasetSplit.VALIDATION: 0}
    BUFFER_SIZE_SHUFFLE = 6_000
     
    def __init__(self,
                tokenizer: Tokenizer,
                max_seq_len: int):
        """Constructor.

        Args:
        tokenizer: Gemma tokenizer to use.
        max_seq_len: size of each sequence in a given batch.
        """
        self._tokenizer = tokenizer
        
        self._base_data = {
            DatasetSplit.TRAIN: load_dataset("../data/anno", data_files="train.json"),
        # DatasetSplit.VALIDATION: tfds.load("mtnt/en-fr",split="valid"),
        }
        self._max_seq_len = max_seq_len

    def _tokenize_source(self, example: tf.Tensor):
        """Tokenization function for the source."""
        return self._tokenizer.tokenize_tf_op(
            example,
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
        image,
        src_tokens: torch.Tensor,
        dst_tokens: torch.Tensor,
    ) -> TrainingInput:
        """Build a training input from a tuple of source and destination tokens."""

        # The input sequence fed to the model is simply the concatenation of the
        # source and the destination.
        tokens = torch.concat([src_tokens, dst_tokens], axis=0)

        # We want to prevent the model from updating based on the source (input)
        # tokens. To achieve this, we add a target mask to each input.
        q_mask = torch.zeros_like(src_tokens, dtype=torch.bool)
        a_mask = torch.ones_like(dst_tokens, dtype=torch.bool)
        mask = torch.concat([q_mask, a_mask], axis=0)

        # If the output tokens sequence is smaller than the target sequence size,
        # then we pad it with pad tokens.
        tokens = self._pad_up_to_max_len(tokens, self._tokenizer.pad_id)

        # We don't want to perform the backward on the pad tokens.
        mask = self._pad_up_to_max_len(mask, False)

        return TrainingInput(image=image, input_tokens=tokens, target_mask=mask)
    
    def get_train_dataset(self, batch_size: int, num_epochs: int):
        """Build the training dataset."""

        # Tokenize each sample
        ds = self._base_data[DatasetSplit.TRAIN]["train"]
        
        inputs = []
        
        for x in ds:
            q_tokens = self._tokenizer.tokenize(x['question'], add_eos=False)
            a_tokens = self._tokenizer.tokenize(x["answers"][0]["answer"], add_eos=True)
            img = x["image"]
            
            train_input = self._to_training_input(img,torch.as_tensor(q_tokens, dtype=torch.int32), torch.as_tensor(a_tokens, dtype=torch.int32))
            inputs.append(train_input)
        
        # Remove the samples which are too long
        # ds = ds.filter(lambda x: torch.shape(x.input_tokens)[0] <= self._max_seq_len)

        # Shuffle the dataset
        np.random.shuffle(inputs)
        # Repeat if necessary
        inputs = inputs * num_epochs

        # Build batches
        inputs = [inputs[i:i + batch_size] for i in range(0, len(inputs), batch_size)]
        return inputs

    def get_validation_dataset(self, batch_size: int):
        """Build the validation dataset."""

        # Same as the training dataset, but no shuffling and no repetition
        ds = self._base_data[DatasetSplit.VALIDATION].map(
            lambda x : (self._tokenize_source(x['src']),
                        self._tokenize_destination(x['dst']))
        )
        ds = ds.map(lambda x, y: self._to_training_input(x, y))
        ds = ds.filter(lambda x: tf.shape(x.input_tokens)[0] <= self._max_seq_len)
        ds = ds.batch(batch_size, drop_remainder=True)
        return ds
        

if __name__ == "__main__":
    vocab = spm.SentencePieceProcessor()
    vocab.load("../model/tokenizer.model")
    
    tokeninzer = Tokenizer(vocab)
    
    device = "cuda:1"
    
    # ds = load_dataset("json",data_files="../data/anno/train.json", split="train")
    
    # print(ds[:1])
    # db = DatasetBuilder(tokenizer=tokeninzer, max_seq_len=100)
        
    # params = torch.load("./model/2b-it.pt")
    # params = {k: v.to(device=device) for k, v in params.items()}
    
    # config = recurrentgemma.GriffinConfig.from_torch_params(
    #     params,
    #     preset=recurrentgemma.Preset.RECURRENT_GEMMA_2B_V1,
    # )
    
    # model = recurrentgemma.Griffin(config, device=device, dtype=torch.bfloat16)
    
    # model.load_state_dict(params, strict=False)
    
    # sampler = Sampler(model=model, tokeninzer=tokeninzer, device=device)
    
    ds_builder = DatasetBuilder(tokeninzer, max_seq_len=30)
    ds = ds_builder.get_train_dataset(3, 1)
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
    
    
    path_checkpoint = "../model/2b-it.pt"
    
    device = torch.device('cuda:1')
    print(f"Loading the parameters from {path_checkpoint} into {device}")
    params = torch.load(path_checkpoint)
    params = {k: v.to(device=device) for k, v in params.items()}
    print("Parameters loaded.")
    # Create a sampler with the right param shapes.
    config = recurrentgemma.GriffinConfig.from_torch_params(
        params,
        preset=recurrentgemma.Preset.RECURRENT_GEMMA_2B_V1,
    )
    model = recurrentgemma.Griffin(config, device=device, dtype=torch.bfloat16)
    model.load_state_dict(params, strict=False)
    
    sampler = recurrentgemma.Sampler(model=model, vocab=vocab)
    
    print(len(ds[0]))
    
    output = sampler(["Tell me about this thing."], 100, img_path="/homes/jkobza/projects/recurrentgemma_experiments/recurrentgemma/vit/img_tests/dog.jpg")
    print(output)
    