import random
from typing import Any, Mapping, Iterator
import enum
import functools
import wandb
from pickle import dump
import sys
import pathlib
from torch2jax import j2t, t2j
import re
from jax.tree_util import register_pytree_node
from jax.tree_util import register_pytree_node_class


from recurrentgemma.vit import VisionEncoder

from memory_profiler import profile
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="Cadence",

#     # track hyperparameters and run metadata
#     config={
#         "optimizer":'AdamW',
#         "learning_rate":2e-5,
#         "b2":0.99,
#         "num_epochs":1,
#         "eval_every_n":10,
#         "batch_size":1,
#         "max_steps":1000,
#     }
# )

import torch.func as func

import time

import json

import numpy as np
from torch.utils.data.dataloader import DataLoader

from accelerate import Accelerator
accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=4)

import os

import chex
import jax
import jax.numpy as jnp
import optax

from attr import dataclass
from numpy import dtype
import torch

# We will use tensorflow to handle the dataset
from datasets import load_dataset

# Finally, we import Recurrentgemma.
import sentencepiece as spm
from recurrentgemma import jax as recurrentgemma

import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

# from ..vit import VisionEncoder

# jax.config.update('jax_log_compiles', True)

@chex.dataclass(frozen=True)
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

@chex.dataclass(frozen=True)
class TrainingInput:
    
    # Input tokens given to the model
    input_tokens: jax.Array

    # A mask that determines which tokens contribute to the target loss
    # calculation.
    target_mask: jax.Array
    
    image: str
    
# def flattenInput(tree):
#     children = (tree.count,)
#     return (children)
    
# def unflattenInput(children):
#     return TrainingInput(*children)
    
# register_pytree_node(
#     TrainingInput,
#     flattenInput,
#     unflattenInput
# )
    
    
    


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
        
        
        for x in llava_it:
            q_tokens = [self._tokenizer.tokenize(i['value'], add_eos=False) for i in x["conversations"] if i['from'] == 'human']
            a_tokens = [self._tokenizer.tokenize(i['value']) for i in x["conversations"] if i['from'] == 'gpt']
            img = x["image"]
            idx = x['id']
            
            train_inputs = self._to_training_input(img, q_tokens, a_tokens, set="llava_it")
            inputs.extend(train_inputs)
        print("LLAVA DONE")

        # for x in lvis_it:
        #     q_tokens = [self._tokenizer.tokenize(i['value'], add_eos=False) for i in x["conversations"] if i['from'] == 'human']
        #     a_tokens = [self._tokenizer.tokenize(i['value']) for i in x["conversations"] if i['from'] == 'gpt']
        #     img = x["image"]
        #     idx = x['id']
            
        #     train_inputs = self._to_training_input(img, q_tokens, a_tokens, set="lvis_it")
        #     for i in train_inputs:
        #         inputs.append(i)
        # print("LVIS DONE")
        # for x in lrv:
        #     q_tokens = self._tokenizer.tokenize(x["question"], add_eos=False)
        #     a_tokens = self._tokenizer.tokenize(x["answer"])
        #     img = x["image_id"]

            
        #     train_input = self._to_training_input(image=img, src_tokens=q_tokens, dst_tokens=a_tokens, set="lrv")
        #     inputs.append(train_input)
        # print("LRV DONE")
        # for x in dvqa:
        #     print(x)
        #     q_tokens = self._tokenizer.tokenize(x["question"], add_eos=False)
        #     a_tokens = self._tokenizer.tokenize(x["answer"])
        #     img = x["image"]
            
        #     train_input = self._to_training_input(image=img, src_tokens=q_tokens, dst_tokens=a_tokens, set="dvqa")
        #     inputs.append(train_input)
        # print("DVQA DONE")    
        print(len(inputs))
        
                


        # Tokenize each sample
        
        print(self._base_data[DatasetSplit.LLAVA_IT])
        ds = self._base_data[DatasetSplit.TRAIN]["train"]
        
        
        # for x in ds:
        #     q_tokens = self._tokenizer.tokenize(x['question'], add_eos=False)
        #     a_tokens = self._tokenizer.tokenize(x["answers"][0]["answer"], add_eos=True)
        #     img = x["image"]
            
        #     train_input = self._to_training_input(img,torch.as_tensor(q_tokens, dtype=torch.int32), torch.as_tensor(a_tokens, dtype=torch.int32), set="vizwiz")
        #     inputs.append(train_input)
        
        # Remove the samples which are too long
        # ds = ds.filter(lambda x: torch.shape(x.input_tokens)[0] <= self._max_seq_len)

        # Shuffle the dataset
        np.random.shuffle(inputs)
        # Repeat if necessary
        inputs = inputs * num_epochs

        # Build batches
        inputs = [i for i in inputs if i.input_tokens.shape[-1]<=self._max_seq_len]
        inputs = [inputs[i:i + batch_size] for i in range(0, len(inputs), batch_size)]
        inputs = {i: obj for i, obj in enumerate(inputs)}
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
            # match = re.search(r'\d+', img)
            # idx = int(match.group()) if match else None
            
            train_input = self._to_training_input(img, torch.as_tensor(q_tokens, dtype=torch.int32), torch.as_tensor(a_tokens, dtype=torch.int32), set="vizwiz")
            valid.append(train_input)
            
        valid = [valid[i:i + batch_size] for i in range(0, len(valid), batch_size)]
        valid = {i: obj for i, obj in enumerate(valid)}
        return valid
    
    
    def _to_training_input(
        self,
        image,
        src_tokens,
        dst_tokens,
        set: str,
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
                # print(type(img_id), img_id, type({"llava_it": img_id}), {"llava_it": img_id})
                # ins.append(TrainingInput(image="/lus/eagle/projects/argonne_tpc/jkobza/data/coco/train2014/COCO_train2014_"+image, input_tokens=tokens, target_mask=mask))
                ins.append(TrainingInput(image="/lus/eagle/projects/argonne_tpc/jkobza/data/coco/train2014/COCO_train2014_"+image, input_tokens=t2j(tokens), target_mask=t2j(mask)))
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
                ins.append(TrainingInput(image="/lus/eagle/projects/argonne_tpc/jkobza/data/"+image, input_tokens=t2j(tokens), target_mask=t2j(mask)))
                # print(img_id)
                # ins.append(TrainingInput(image={"lvis_it": int(img_id)}, input_tokens=tokens, target_mask=mask))
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
            # return TrainingInput(image="/lus/eagle/projects/argonne_tpc/jkobza/data/LRV/image/"+image+".jpg", input_tokens=tokens, target_mask=mask)
            return TrainingInput(image="/lus/eagle/projects/argonne_tpc/jkobza/data/LRV/image/"+image+".jpg", input_tokens=t2j(tokens), target_mask=t2j(mask))

        if set == "dvqa":
            src_tensor = torch.Tensor(src_tokens).to(torch.int64).to("cpu")
            dst_tensor = torch.Tensor(dst_tokens).to(torch.int64).to("cpu")
            
            tokens = torch.concat([src_tensor, dst_tensor], axis=0)

    
            q_mask = torch.zeros_like(src_tensor, dtype=torch.bool)
            a_mask = torch.ones_like(dst_tensor, dtype=torch.bool)
            mask = torch.concat([q_mask, a_mask], axis=0)
            
            tokens = self._pad_up_to_max_len(tokens, self._tokenizer.pad_id)
            mask = self._pad_up_to_max_len(mask, False)
            return TrainingInput(image="/lus/eagle/projects/argonne_tpc/jkobza/data/DVQA/images/" + image, input_tokens=t2j(tokens), target_mask=t2j(mask))        

        
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
                return TrainingInput(image="../data/train/train/" + image, input_tokens=t2j(tokens), target_mask=t2j(mask))
            if("val" in image):
                return TrainingInput(image="../data/val/" + image, input_tokens=t2j(tokens), target_mask=t2j(mask))



def forward_and_loss_fn(
    params,
    *,
    model: recurrentgemma.Griffin,
    input_tokens: jax.Array,            # Shape [B, L]
    input_mask: jax.Array,              # Shape [B, L]
    positions: jax.Array,               # Shape [B, L]
    image: jax.Array
) -> jax.Array:
    """Foward pass and loss function.

    Args:
        params: model's input parameters.
        model: gemma transformer model to call.
        input_tokens: input tokens sequence, shape [B, L].
        input_mask: tokens to ignore when computing the loss, shape [B, L].
        positions: relative position of each token, shape [B, L].

    Returns:
        Softmax cross-entropy loss for the next-token prediction task.
    """

    # Convert torch inputs into numpy then jax array 




    # Foward pass on the input data.
    # No attention cache is needed here.
    logits, _ = model.apply(
        {"params": params},
        input_tokens,
        positions,
        None,              # Attention cache is None.
        image=image
    )

    # Exclude the last step as it does not appear in the targets.
    logits = logits[0, :-1]

    # Similarly, the first token cannot be predicteds.
    target_tokens = input_tokens[1:]
    target_mask = input_mask[1:]

    # Convert the target labels into one-hot encoded vectors.
    one_hot = jax.nn.one_hot(target_tokens, logits.shape[-1])

    # Don't update on unwanted tokens.
    one_hot = one_hot * target_mask.astype(one_hot.dtype)[...,None]

    # Normalisation factor.
    norm_factor = 1 / (jnp.sum(target_mask) + 1e-8)

    # Extend one hot for img logits
    
    one_hot = jnp.concatenate([jnp.zeros((729, logits.shape[-1])), one_hot])

    # Return the nll loss.
    return -jnp.sum(jax.nn.log_softmax(logits) * one_hot) * norm_factor






Params = dict[str, Any]

def get_positions(example: jax.Array, pad_id : int) -> jax.Array:
    """Builds the position vector from the given tokens."""
    pad_mask = example != pad_id
    positions = jnp.cumsum(pad_mask, axis=-1)
    # Subtract one for all positions from the first valid one as they are
    # 0-indexed
    positions = positions - (positions >= 1)
    return positions


@functools.partial(
    jax.jit,
    static_argnames=['model', 'optimizer'],
    donate_argnames=['params', 'opt_state'],
)
def train_step(
    model: recurrentgemma.Griffin,
    params: Params,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    pad_id: int,
    input_tokens: jax.Array,
    input_mask: jax.Array,
    img_embed: jax.Array
    ) -> tuple[jax.Array, Params, optax.OptState]:
    """Train step.

    Args:
        model: gemma transformer model.
        params: model's input parameters.
        optimizer: optax optimizer to use.
        opt_state: input optimizer's state.
        pad_id: id of the pad token.
        example: input batch.

    Returns:
        Training loss, updated parameters, updated optimizer state.
    """

    positions = get_positions(input_tokens, pad_id)

    # Forward and backward passes
    train_loss, grads = jax.value_and_grad(forward_and_loss_fn)(
        params,
        model=model,
        input_tokens=input_tokens,
        input_mask=input_mask,
        positions=positions,
        image=img_embed
    )
    print("Pre optim")
    # Update the parameters
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    jax.profiler.save_device_memory_profile("optim.prof")
    print("Post optim")

    return train_loss, params, opt_state

@functools.partial(jax.jit, static_argnames=['model'])
def validation_step(
    model: recurrentgemma.Griffin,
    params: Params,
    pad_id: int,
    input_tokens: jax.Array,
    input_mask: jax.Array,
    img_embed: jax.Array
    ) -> jax.Array:
    return forward_and_loss_fn(
        params,
        model=model,
        input_tokens=input_tokens,
        input_mask=input_mask,
        positions=get_positions(input_tokens, pad_id),
        image=img_embed
    )
    

  
def griffin_weight_decay_mask(params_like: optax.Params) -> Any:
    # Don't put weight decay on the RGLRU, the embeddings and any biases
    def enable_weight_decay(path: list[str], _: Any) -> bool:
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
    params: Params,
    dataset_builder: DatasetBuilder,
    training_cfg: TrainingConfig):
    if training_cfg.optimizer == 'adamw':
        # For better optimization we use Adam-W.
        optimizer = optax.adamw(
            learning_rate=training_cfg.learning_rate,
            b2=training_cfg.b2,
            eps=training_cfg.eps,
            weight_decay=training_cfg.weight_decay,
            mask=griffin_weight_decay_mask,
        )
    else:
        # To save memory, we can use a SGD optimizer instead.
        optimizer = optax.sgd(learning_rate=training_cfg.learning_rate)

    opt_state = jax.jit(optimizer.init)(params)

    # Build the training dataset
    train_ds = dataset_builder.get_train_dataset(
        batch_size=training_cfg.batch_size, num_epochs=training_cfg.num_epochs
    )



    # Build the validation dataset, with a limited number of samples for this demo
    validation_ds = dataset_builder.get_validation_dataset(
        batch_size=training_cfg.batch_size
    )
    
    sampled_keys = random.sample(list(validation_ds.keys()), 10)

    # Create a new dictionary using the sampled keys
    val_dict = {key: validation_ds[key] for key in sampled_keys}
    train_dict = {key: train_ds[key] for key in sampled_keys}

    n_steps = 0
    avg_loss=0
    
    # vision_enc = VisionEncoder()


    # A first round of validation loss
    n_steps_eval = 0
    eval_loss = 0
    
    vis_enc = VisionEncoder()
    
    for val_example in val_dict.keys():
        torch_emb = torch.squeeze(vis_enc(val_dict[val_example][0].image))
        input_tokens = jax.device_put(val_dict[val_example][0].input_tokens, jax.devices("gpu")[0])
        target_mask = jax.device_put(val_dict[val_example][0].target_mask, jax.devices("gpu")[0])
        img_embed = t2j(torch_emb).astype(jnp.bfloat16)
        eval_loss += validation_step(
            model, params, dataset_builder._tokenizer.pad_id, input_tokens, target_mask, img_embed
        )
        n_steps_eval += 1
    print(f"Start, validation loss: {eval_loss/n_steps_eval}")
    jax.profiler.save_device_memory_profile("val.prof")
    for train_example in train_dict.keys():        
        input_tokens = jax.device_put(train_dict[train_example][0].input_tokens, jax.devices("gpu")[0])
        target_mask = jax.device_put(train_dict[train_example][0].target_mask, jax.devices("gpu")[0])
        img_embed = t2j(torch_emb).astype(jnp.bfloat16)
        print("stepping")
        train_loss, params, opt_state = train_step(
            model=model,
            params=params,
            optimizer=optimizer,
            opt_state=opt_state,
            pad_id=dataset_builder._tokenizer.pad_id,
            input_tokens=input_tokens,
            input_mask=target_mask,
            img_embed=img_embed
            )
        n_steps += 1
        avg_loss += train_loss
        print(train_loss)
        if n_steps % training_cfg.eval_every_n == 0:
            eval_loss = 0

            n_steps_eval = 0
            val_iterator = validation_ds.as_numpy_iterator()
            for val_example in val_iterator:
                eval_loss += validation_step(
                    model,
                    params,
                    dataset_builder._tokenizer.pad_id,
                    jax.device_put(val_example, jax.devices("gpu")[0]), ## Might have to make this sharding instead of explicit
                )
                n_steps_eval +=1
            avg_loss /= training_cfg.eval_every_n
            eval_loss /= n_steps_eval
            print(f"STEP {n_steps} training loss: {avg_loss} - eval loss: {eval_loss}")
            avg_loss=0
        if training_cfg.max_steps is not None and n_steps > training_cfg.max_steps:
            break
        return params


        
        

            
    
    
    
if __name__ == "__main__":
    

    vocab = spm.SentencePieceProcessor()
    vocab.Load("../model/flax/tokenizer.model")
    
    weights_dir = pathlib.Path("/home/jkobza/cadence/cadence-gemma/model/flax")
    ckpt_path = weights_dir / "2b-it"
    vocab_path = weights_dir / 'tokenizer.model'
    
    preset = recurrentgemma.Preset.RECURRENT_GEMMA_2B_V1
    
    params =  recurrentgemma.load_parameters(ckpt_path, "single_device")
    
    
    key = jax.random.PRNGKey(0)
    params['vl_connector'] = {'ffw_up': {'w': jax.random.uniform(key, (1, 2176, 4000), dtype=jnp.bfloat16),'b': jax.random.uniform(key, (1, 1, 1, 4000), dtype=jnp.bfloat16)},
                              'ffw_down': {'bias': jax.random.uniform(key, (2560,), dtype=jnp.bfloat16), 'kernel': jax.random.uniform(key, (4000, 2560), dtype=jnp.bfloat16)}}
    model_config = recurrentgemma.GriffinConfig.from_flax_params_or_variables(params, preset=preset)
    model = recurrentgemma.Griffin(model_config)


    
    
    SEQ_SIZE = 300
    tokenizer = Tokenizer(vocab)
    dataset_builder= DatasetBuilder(tokenizer, SEQ_SIZE)
    training_cfg = TrainingConfig(
        optimizer='adamw',
        learning_rate=1e-5,
        b2=0.96,
        num_epochs=1,
        eval_every_n=20,
        batch_size=1,
        max_steps=1000,
    )

    trained_params = train_loop(
        model=model,
        params=params,
        dataset_builder=dataset_builder,
        training_cfg=training_cfg,
    )