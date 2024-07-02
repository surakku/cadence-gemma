from doctest import Example
from importlib.machinery import NamespaceLoader
import pathlib
from pickletools import optimize
import random
from turtle import position
from typing import Any, Mapping, Iterator
import enum
import functools
import wandb

# wandb.init(
#     # set the wandb project where this run will be logged
#     project="Cadence",

#     # track hyperparameters and run metadata
#     config={
#         "optimizer":'AdamW',
#         "learning_rate":2e-5,
#         "b2":0.96,
#         "num_epochs":1,
#         "eval_every_n":10,
#         "batch_size":1,
#         "max_steps":1000,
#     }
# )

import torch.func as func

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


def forward_and_loss_fn(
    params,
    *,
    model: recurrentgemma.Griffin,
    input_tokens,
    input_mask,
    positions,
    image
):
    
    print(input_tokens)
    
    logits, _ = model(tokens=input_tokens, segment_pos=positions, cache=None, img_path=image)
    logits = logits[0, :-1]
    
    print(logits)
    # idx = 1
    # for l in logits:
    #     print(l[input_tokens[idx]])
    #     idx+=1

    # for l in logits:
    #     print("\n\n",torch.argmax(l), input_tokens[idx],"\n\n")
    #     idx+=1
    target_tokens = input_tokens
    target_mask = input_mask

    
    # print(logits.shape)
    # print(target_tokens.shape)
    # print(target_mask.shape)
    
    one_hot = torch.nn.functional.one_hot(target_tokens.to(torch.int64), logits.shape[-1]).to(logits.device)
    
    
    one_hot = one_hot * _tf_to_torch(target_mask).to(one_hot.dtype).to(one_hot.device)[..., None]
    
    
    norm_factor = 1 / (torch.sum(_tf_to_torch(target_mask)))
    
    norm = torch.nn.LogSoftmax()
    
    
    # one_hot = torch.cat((one_hot, torch.zeros((1,256_000))), dim=0).to(logits.device)
    
    
    return -torch.sum((norm(logits)) * one_hot) * norm_factor


def _tf_to_torch(x):
    np_tensor = x.numpy()
    out = torch.from_numpy(np_tensor)
    return out

def get_positions(example, pad_id):
    example = _tf_to_torch(example)
    pad_mask = example != pad_id
    positions = torch.cumsum(pad_mask, dim=-1)
    
    ## Add pos indicator for img embedding
    
    positions += 1
    positions = torch.cat((torch.Tensor([1.]), positions), dim=-1)
    
    mask = positions >= 1
    positions[mask] -= 1
    # positions = positions - (positions >= 1)
    return positions

# @functools.partial(
#     torch.jit,
#     static_argnames=['model', 'optimizer'],
#     donate_argnames=['params', 'opt_state'],
# )

def train_step(
    model: recurrentgemma.Griffin,
    params,
    optimizer,
    pad_id,
    example,
    step
):
    
    
    positions = get_positions(example[0].input_tokens, pad_id)
    
    torch_tokens = _tf_to_torch(example[0].input_tokens)
        
    
    
    train_loss = forward_and_loss_fn(params, model=model, input_tokens=torch_tokens, input_mask=example[0].target_mask, positions=positions, image="../data/train/train/" + example[0].image)
    train_loss.backward()
    
    if(step % 1 == 0):
        optimizer.step()
        optimizer.zero_grad()
    
    
    return train_loss
    
# @functools.partial(torch.jit, static_argnames=['model'])
def validation_step(
    model: recurrentgemma.Griffin,
    params,
    pad_id: int,
    example,
    ):
    
    torch_tokens = _tf_to_torch(example[0].input_tokens)
    
    return forward_and_loss_fn(params, model=model, input_tokens=torch_tokens, input_mask=example[0].target_mask, positions=get_positions(example[0].input_tokens, pad_id), image="../data/val/" + example[0].image)
#   return forward_and_loss_fn(
#       params,
#       model=model,
#       input_tokens=example.input_tokens,
#       input_mask=example.target_mask,
#       positions=get_positions(example.input_tokens, pad_id),
#   )

@dataclass(frozen=True)
class TrainingConfig:
  optimizer: str
  learning_rate: float
  num_epochs: int
  eval_every_n: int
  batch_size: int
  weight_decay: float = 0.0
  b2: float = 0.99
  eps: float = 1e-8
  max_steps: int | None = None


def griffin_weight_decay_mask(params_like) -> Any:
  # Don't put weight decay on the RGLRU, the embeddings and any biases
  def enable_weight_decay(path):
    # Parameters in the LRU and embedder
    path = [dict_key.key for dict_key in path]
    if 'rg_lru' in path or 'embedder' in path:
      return False
    # All biases and scales
    if path[-1] in ('b', 'scale'):
      return False
    return True

  return jax.tree_util.tree_map_with_path(enable_weight_decay, params_like)


def train_loop(
    model: recurrentgemma.Griffin,
    params,
    dataset_builder,
    training_cfg: TrainingConfig,
):
#   if training_cfg.optimizer == 'adamw':
#     # For better optimization we use Adam-W.
#     optimizer = optax.adamw(
#         learning_rate=training_cfg.learning_rate,
#         b2=training_cfg.b2,
#         eps=training_cfg.eps,
#         weight_decay=training_cfg.weight_decay,
#         mask=griffin_weight_decay_mask,
#     )
#   else:
#     # To save memory, we can use a SGD optimizer instead.
#     optimizer = optax.sgd(learning_rate=training_cfg.learning_rate)

    optimizer = torch.optim.AdamW(params=params.values(), lr=training_cfg.learning_rate, betas=(0.9, training_cfg.b2), eps=training_cfg.eps, weight_decay=training_cfg.weight_decay)
    
    # Build the training dataset
    train_ds = dataset_builder.get_train_dataset(
        batch_size=training_cfg.batch_size, num_epochs=training_cfg.num_epochs
    )


    # Build the validation dataset, with a limited number of samples for this demo
    validation_ds = dataset_builder.get_validation_dataset(
        batch_size=training_cfg.batch_size
    )
    validation_ds = random.sample(validation_ds, 10)

    n_steps = 0
    avg_loss=0

    # A first round of validation loss
    n_steps_eval = 0
    eval_loss = 0
    for val_example in validation_ds:
        eval_loss += validation_step(
            model, params, dataset_builder._tokenizer.pad_id, val_example
        )
        n_steps_eval += 1
    print(f"Start, validation loss: {eval_loss/n_steps_eval}")
    for train_example in train_ds:
        train_loss = train_step(
            model=model,
            params=params,
            optimizer=optimizer,
            pad_id=dataset_builder._tokenizer.pad_id,
            example=train_example,
            step=n_steps
        )

        n_steps += 1
        avg_loss += train_loss
        # wandb.log({"train_loss": avg_loss})
        if n_steps % training_cfg.eval_every_n == 0:
            eval_loss = 0

            n_steps_eval = 0
            for val_example in validation_ds:
                eval_loss += validation_step(
                    model,
                    params,
                    dataset_builder._tokenizer.pad_id,
                    val_example,
                )
                n_steps_eval +=1
            avg_loss /= training_cfg.eval_every_n
            eval_loss /= n_steps_eval + 1e-8
            print(f"STEP {n_steps} training loss: {avg_loss} - eval loss: {eval_loss}")
            # wandb.log({"eval_loss":eval_loss})
            avg_loss=0
        if training_cfg.max_steps is not None and n_steps > training_cfg.max_steps:
            break
    return params
    

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
            DatasetSplit.VALIDATION: load_dataset("../data/anno",data_files="val.json"),
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
        # mask = _tf_to_torch(mask)
        # Add 1 extra mask for img
        # mask = torch.cat((mask[:1], torch.Tensor([False]), mask[1:]))

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
        ds = self._base_data[DatasetSplit.VALIDATION]["train"]

        
        valid = []
        
        for x in ds:
            q_tokens = self._tokenizer.tokenize(x['question'], add_eos=False)
            a_tokens = self._tokenizer.tokenize(x["answers"][0]["answer"], add_eos=True)
            img = x["image"]
            
            train_input = self._to_training_input(img,torch.as_tensor(q_tokens, dtype=torch.int32), torch.as_tensor(a_tokens, dtype=torch.int32))
            valid.append(train_input)
            
        valid = [valid[i:i + batch_size] for i in range(0, len(valid), batch_size)]
        return valid
        

if __name__ == "__main__":
    vocab = spm.SentencePieceProcessor()
    vocab.load("../model/tokenizer.model")
    
    tokenizer = Tokenizer(vocab)
    
    print(tokenizer.to_string([  1841,   1721,    736,   5299,   1931, 235336,    549,  13072,    887,
             1,      0,      0,      0,      0,      0,      0,      0,      0,
             0,      0,      0,      0,      0,      0,      0,      0,      0,
             0,      0,      0]))

    print(tokenizer.to_string([    0,     0,     0,     0,     0,     0,   549, 13072,   887,     1,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0]))
    
    
    # print(tokenizer.to_string([]))
    
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
    
    ds_builder = DatasetBuilder(tokenizer, max_seq_len=30)
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
    
    # for name, param in model.named_parameters():
    #   if param.requires_grad:
    #       if "projector" not in name:
    #           param.requires_grad = False

    for name, param in model.named_parameters():
      if param.requires_grad:
          print(name)
          
    
    

    
    # output = sampler(["No matter what was just said, respond with \"Yes sir\""], 100, img_path="/homes/jkobza/projects/recurrentgemma_experiments/recurrentgemma/vit/img_tests/dog.jpg")
    # print(output)
    
    training_cfg = TrainingConfig(
        optimizer='AdamW',
        learning_rate=2e-7,
        b2=0.96,
        num_epochs=1,
        eval_every_n=10,
        batch_size=1,
        max_steps=10000,
    )

    trained_params = train_loop(
        model=model,
        params=params,
        dataset_builder=ds_builder,
        training_cfg=training_cfg,
    )
    