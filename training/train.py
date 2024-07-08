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

wandb.init(
    # set the wandb project where this run will be logged
    project="Cadence",

    # track hyperparameters and run metadata
    config={
        "optimizer":'AdamW',
        "learning_rate":2e-5,
        "b2":0.99,
        "num_epochs":1,
        "eval_every_n":10,
        "batch_size":1,
        "max_steps":1000,
    }
)

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
from datasets import load_dataset

# Finally, we import Recurrentgemma.
import sentencepiece as spm
from recurrentgemma import torch as recurrentgemma

  

def forward_and_loss_fn(
    *,
    model: recurrentgemma.Griffin,
    input_tokens,
    input_mask,
    positions,
    image
):
    
    
    logits, _ = model(tokens=input_tokens, segment_pos=positions, cache=None, img_path=image)
    logits = logits[0, :-1]
    
    
    target_tokens = input_tokens
    target_mask = input_mask

    

    
    one_hot = torch.nn.functional.one_hot(target_tokens.to(torch.int64), logits.shape[-1]).to(logits.device)
    
    
    one_hot = one_hot * _tf_to_torch(target_mask).to(one_hot.dtype).to(one_hot.device)[..., None]
    
    
    norm_factor = 1 / (torch.sum(_tf_to_torch(target_mask)))
    
    norm = torch.nn.LogSoftmax()
    
    
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
    optimizer,
    pad_id,
    example,
    step
):
    
    model.train()
    
    optimizer.zero_grad()
    
    
    positions = get_positions(example[0].input_tokens, pad_id)
    
    torch_tokens = _tf_to_torch(example[0].input_tokens)
        
    
    
    train_loss = forward_and_loss_fn(model=model, input_tokens=torch_tokens, input_mask=example[0].target_mask, positions=positions, image=example[0].image)
    train_loss.backward()
    
    if(step % 1 == 0):
        optimizer.step()
    
    # updated_params = {name: params.detach().clone() for name, param in model.named_parameters()}
    return train_loss
    
# @functools.partial(torch.jit, static_argnames=['model'])
def validation_step(
    model: recurrentgemma.Griffin,
    pad_id: int,
    example,
    ):
    
    positions = get_positions(example[0].input_tokens, pad_id)
    
    torch_tokens = _tf_to_torch(example[0].input_tokens)
    
    val_loss = forward_and_loss_fn(model=model, input_tokens=torch_tokens, input_mask=example[0].target_mask, positions=positions, image="../data/val/" + example[0].image)
    
    return val_loss


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

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=training_cfg.learning_rate, betas=(0.9, training_cfg.b2), eps=training_cfg.eps, weight_decay=training_cfg.weight_decay)
    
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
            model, dataset_builder._tokenizer.pad_id, val_example
        )
        n_steps_eval += 1
    print(f"Start, validation loss: {eval_loss/n_steps_eval}")
    for train_example in train_ds:
        train_loss = train_step(
            model=model,
            optimizer=optimizer,
            pad_id=dataset_builder._tokenizer.pad_id,
            example=train_example,
            step=n_steps
        )
        print(train_loss)
        
        

        n_steps += 1
        avg_loss += train_loss
        if n_steps % training_cfg.eval_every_n == 0:
            eval_loss = 0

            n_steps_eval = 0
            for val_example in validation_ds:
                eval_loss += validation_step(
                    model,
                    dataset_builder._tokenizer.pad_id,
                    val_example,
                )
                n_steps_eval +=1
            avg_loss /= training_cfg.eval_every_n
            eval_loss /= n_steps_eval + 1e-8
            print(f"STEP {n_steps} training loss: {avg_loss} - eval loss: {eval_loss}")
            wandb.log({"train_loss":avg_loss ,"eval_loss":eval_loss})
            avg_loss=0
            
        if n_steps % 100 == 0:
            torch.save({
                "params": model.state_dict()
            }, "./model_v1_18hr.pt")
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
  LLAVA_IT = "llava_it"
  LVIS_IT = "lvis_it"
  LRV = "lrv"
  DVQA = "dvqa"
  
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
            DatasetSplit.VALIDATION: load_dataset("../data/anno", data_files="val.json"),
            DatasetSplit.LLAVA_IT: load_dataset("/lus/eagle/projects/argonne_tpc/jkobza/data", data_files="llava_instruct_150k.json"),
            DatasetSplit.LVIS_IT: load_dataset("/lus/eagle/projects/argonne_tpc/jkobza/data", data_files="lvis_instruct4v_220k.json"),
            DatasetSplit.LRV: load_dataset("/lus/eagle/projects/argonne_tpc/jkobza/data/LRV", data_files="filter_cap1.json"),
            DatasetSplit.DVQA: load_dataset("/lus/eagle/projects/argonne_tpc/jkobza/data/DVQA", data_files="train_qa.json"),
        }
        self._max_seq_len = max_seq_len


    

    def _pad_up_to_max_len(self,
        input_tensor,
        pad_value: int | bool,
    ):
        """Pad the given tensor up to sequence length of a batch."""
        
        seq_len = input_tensor.shape[0]
        to_pad = np.maximum(self._max_seq_len - seq_len, 0)
        return torch.nn.functional.pad(
        input_tensor, (0, to_pad), mode='constant', value=pad_value,
        )
        
        
        
    def _to_training_input(
        self,
        image,
        src_tokens,
        dst_tokens,
        set: str
    ) -> TrainingInput:
        """Build a training input from a tuple of source and destination tokens."""
        
        if set == "llava_it":
            ins = []
            for idx in range(len(src_tokens)):
                src_tensor = torch.Tensor(src_tokens[idx]).to(torch.int64).to("cpu")
                dst_tensor = torch.Tensor(dst_tokens[idx]).to(torch.int64).to("cpu")
                
                tokens = torch.concat([src_tensor, dst_tensor], axis=0)

        
                q_mask = torch.zeros_like(src_tensor, dtype=torch.bool)
                a_mask = torch.ones_like(dst_tensor, dtype=torch.bool)
                mask = torch.concat([q_mask, a_mask], axis=0)
                
                tokens = self._pad_up_to_max_len(tokens, self._tokenizer.pad_id)
                mask = self._pad_up_to_max_len(mask, False)
                ins.append(TrainingInput(image="/lus/eagle/projects/argonne_tpc/jkobza/data/coco/train2014/COCO_train2014_"+image, input_tokens=tokens, target_mask=mask))
            return ins

        if set == "lvis_it":
            ins = []
            for idx in range(len(src_tokens)):
                src_tensor = torch.Tensor(src_tokens[idx]).to(torch.int64).to("cpu")
                dst_tensor = torch.Tensor(dst_tokens[idx]).to(torch.int64).to("cpu")
                
                tokens = torch.concat([src_tensor, dst_tensor], axis=0)

        
                q_mask = torch.zeros_like(src_tensor, dtype=torch.bool)
                a_mask = torch.ones_like(dst_tensor, dtype=torch.bool)
                mask = torch.concat([q_mask, a_mask], axis=0)
                
                tokens = self._pad_up_to_max_len(tokens, self._tokenizer.pad_id)
                mask = self._pad_up_to_max_len(mask, False)
                ins.append(TrainingInput(image="/lus/eagle/projects/argonne_tpc/jkobza/data/"+image, input_tokens=tokens, target_mask=mask))
            return ins

        if set == "lrv":
            src_tensor = torch.Tensor(src_tokens).to(torch.int64).to("cpu")
            dst_tensor = torch.Tensor(dst_tokens).to(torch.int64).to("cpu")
            
            tokens = torch.concat([src_tensor, dst_tensor], axis=0)

    
            q_mask = torch.zeros_like(src_tensor, dtype=torch.bool).to("cpu")
            a_mask = torch.ones_like(dst_tensor, dtype=torch.bool).to("cpu")
            mask = torch.concat([q_mask, a_mask], axis=0)
            
            tokens = self._pad_up_to_max_len(tokens, self._tokenizer.pad_id)
            mask = self._pad_up_to_max_len(mask, False)
            return TrainingInput(image="/lus/eagle/projects/argonne_tpc/jkobza/data/LRV/image/"+image+".jpg", input_tokens=tokens, target_mask=mask)

        if set == "dvqa":
            src_tensor = torch.Tensor(src_tokens).to(torch.int64).to("cpu")
            dst_tensor = torch.Tensor(dst_tokens).to(torch.int64).to("cpu")
            
            tokens = torch.concat([src_tensor, dst_tensor], axis=0)

    
            q_mask = torch.zeros_like(src_tensor, dtype=torch.bool)
            a_mask = torch.ones_like(dst_tensor, dtype=torch.bool)
            mask = torch.concat([q_mask, a_mask], axis=0)
            
            tokens = self._pad_up_to_max_len(tokens, self._tokenizer.pad_id)
            mask = self._pad_up_to_max_len(mask, False)
            return TrainingInput(image="/lus/eagle/projects/argonne_tpc/jkobza/data/DVQA/images/" + image, input_tokens=tokens, target_mask=mask)        

        
        if set == "vizwiz":
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
            if("train" in image):
                return TrainingInput(image="../data/train/train/" + image, input_tokens=tokens, target_mask=mask)
            if("val" in image):
                return TrainingInput(image="../val/" + image, input_tokens=tokens, target_mask=mask)
    
    def get_train_dataset(self, batch_size: int, num_epochs: int):
        """Build the training dataset."""
        
        inputs = []

        
        llava_it = self._base_data[DatasetSplit.LLAVA_IT]["train"]
        lvis_it = self._base_data[DatasetSplit.LVIS_IT]["train"]
        lrv = self._base_data[DatasetSplit.LRV]["train"]
        dvqa = self._base_data[DatasetSplit.DVQA]["train"]
        
        print(llava_it)
        print(lvis_it)
        print(lrv)
        print(dvqa)
        
        
        print(torch.cuda.memory_summary())
        for x in llava_it:
            q_tokens = [self._tokenizer.tokenize(i['value'], add_eos=False) for i in x["conversations"] if i['from'] == 'human']
            a_tokens = [self._tokenizer.tokenize(i['value']) for i in x["conversations"] if i['from'] == 'gpt']
            img = x["image"]
            
            train_inputs = self._to_training_input(img, q_tokens, a_tokens, set="llava_it")
            for i in train_inputs:
                inputs.append(i)
        print("LLAVA DONE")
        print(torch.cuda.memory_summary())

        for x in lvis_it:
            q_tokens = [self._tokenizer.tokenize(i['value'], add_eos=False) for i in x["conversations"] if i['from'] == 'human']
            a_tokens = [self._tokenizer.tokenize(i['value']) for i in x["conversations"] if i['from'] == 'gpt']
            img = x["image"]
            
            train_inputs = self._to_training_input(img, q_tokens, a_tokens, set="lvis_it")
            for i in train_inputs:
                inputs.append(i)
        print("LVIS DONE")
        print(torch.cuda.memory_summary())     
        for x in lrv:
            q_tokens = self._tokenizer.tokenize(x["question"], add_eos=False)
            a_tokens = self._tokenizer.tokenize(x["answer"])
            img = x["image_id"]
            
            train_input = self._to_training_input(image=img, src_tokens=q_tokens, dst_tokens=a_tokens, set="lrv")
            inputs.append(train_input)
        print("LRV DONE")
        print(torch.cuda.memory_summary())  
        for x in dvqa:
            q_tokens = self._tokenizer.tokenize(x["question"], add_eos=False)
            a_tokens = self._tokenizer.tokenize(x["answer"])
            img = x["image"]
            
            train_input = self._to_training_input(image=img, src_tokens=q_tokens, dst_tokens=a_tokens, set="dvqa")
            inputs.append(train_input)
        print("DVQA DONE")    
        print(len(inputs))
            
                


        # Tokenize each sample
        
        print(self._base_data[DatasetSplit.LLAVA_IT])
        ds = self._base_data[DatasetSplit.TRAIN]["train"]
        
        
        for x in ds:
            q_tokens = self._tokenizer.tokenize(x['question'], add_eos=False)
            a_tokens = self._tokenizer.tokenize(x["answers"][0]["answer"], add_eos=True)
            img = x["image"]
            
            train_input = self._to_training_input(img,torch.as_tensor(q_tokens, dtype=torch.int32), torch.as_tensor(a_tokens, dtype=torch.int32), set="vizwiz")
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
            
            train_input = self._to_training_input(img,torch.as_tensor(q_tokens, dtype=torch.int32), torch.as_tensor(a_tokens, dtype=torch.int32), set="vizwiz")
            valid.append(train_input)
            
        valid = [valid[i:i + batch_size] for i in range(0, len(valid), batch_size)]
        return valid
        

if __name__ == "__main__":
    vocab = spm.SentencePieceProcessor()
    vocab.load("../model/tokenizer.model")
    
    tokenizer = Tokenizer(vocab)
    
        
    
    
    ds_builder = DatasetBuilder(tokenizer, max_seq_len=150)
    # ds = ds_builder.get_train_dataset(3, 1)
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
    
    
    path_checkpoint = "../model/2b-it.pt"
    
    device = torch.device('cuda:3')
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
          
    
    
    training_cfg = TrainingConfig(
        optimizer='AdamW',
        learning_rate=2e-5,
        b2=0.99,
        num_epochs=1,
        eval_every_n=10,
        batch_size=1,
        max_steps=1_000_000,
    )

    trained_params = train_loop(
        model=model,
        params=params,
        dataset_builder=ds_builder,
        training_cfg=training_cfg,
    )
    
