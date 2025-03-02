import random
from typing import Any, Mapping, Iterator
import enum
import functools
from pickle import dump
import sys
import pathlib
from torch2jax import j2t, t2j



from recurrentgemma.vit import VisionEncoder

from memory_profiler import profile

import mlflow






import numpy as np



import chex
import jax
import jax.numpy as jnp
import optax

import torch

# We will use tensorflow to handle the dataset
from datasets import load_dataset

# Finally, we import Recurrentgemma.
import sentencepiece as spm
from recurrentgemma import jax as recurrentgemma


jax.config.update("jax_traceback_filtering", "off")

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
    freeze_llm: bool = False

@chex.dataclass(frozen=True)
class TrainingInput:
    
    # Input tokens given to the model
    input_tokens: jax.Array

    # A mask that determines which tokens contribute to the target loss
    # calculation.
    target_mask: jax.Array
    
    image: str
    
    
    
    


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
        int_list = [self._spm_processor.bos_id()]
        int_list.extend(self._spm_processor.EncodeAsIds(str(input)))
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
    USER_TOKEN = 1645
    MODEL_TOKEN = 2516
    START_TOKEN = 106
    END_TOKEN = 107
     
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
            q_tokens = [self._tokenizer.tokenize(recurrentgemma.common.apply_it_formatter(i['value']), add_eos=False) for i in x["conversations"] if i['from'] == 'human']
            a_tokens = [self._tokenizer.tokenize(i['value']+"<end_of_turn>\n") for i in x["conversations"] if i['from'] == 'gpt']
            img = x["image"]
            idx = x['id']
            
            train_inputs = self._to_training_input(img, q_tokens, a_tokens, set="llava_it")
            inputs.extend(train_inputs)
        print("LLAVA DONE")

        # for x in lvis_it:
        #     q_tokens = [self._tokenizer.tokenize(recurrentgemma.common.apply_it_formatter(i['value']), add_eos=False) for i in x["conversations"] if i['from'] == 'human']
        #     a_tokens = [self._tokenizer.tokenize(i['value']+"<end_of_turn>\n") for i in x["conversations"] if i['from'] == 'gpt']
        #     img = x["image"]
        #     idx = x['id']
            
        #     train_inputs = self._to_training_input(img, q_tokens, a_tokens, set="lvis_it")
        #     inputs.extend(train_inputs)
        # print("LVIS DONE")
        # for x in lrv:
        #     q_tokens = self._tokenizer.tokenize(recurrentgemma.common.apply_it_formatter(x["question"]), add_eos=False)
        #     a_tokens = self._tokenizer.tokenize(x["answer"]+"<end_of_turn>\n")
        #     img = x["image_id"]

            
        #     train_inputs = self._to_training_input(image=img, src_tokens=q_tokens, dst_tokens=a_tokens, set="lrv")
        #     inputs.append(train_inputs)
        # print("LRV DONE")
        # for x in dvqa:
        #     q_tokens = self._tokenizer.tokenize(recurrentgemma.common.apply_it_formatter(x["question"]), add_eos=False)
        #     a_tokens = self._tokenizer.tokenize(x["answer"]+"<end_of_turn>\n")
        #     img = x["image"]
            
        #     train_inputs = self._to_training_input(image=img, src_tokens=q_tokens, dst_tokens=a_tokens, set="dvqa")
        #     inputs.append(train_inputs)
        # print("DVQA DONE")
        print(f"Number of training examples: {len(inputs)}")
        
                

        # Shuffle the dataset
        np.random.shuffle(inputs)
        # Repeat if necessary
        inputs = inputs * num_epochs

        inputs = [i for i in inputs if i.input_tokens.shape[-1]<=self._max_seq_len] # Ensure no inputs are too long
        inputs = [inputs[i:i + batch_size] for i in range(0, len(inputs), batch_size)] # Batch inputs
        
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
            
            train_input = self._to_training_input(img, torch.as_tensor(q_tokens, dtype=torch.int32), torch.as_tensor(a_tokens, dtype=torch.int32), set="vizwiz")
            valid.append(train_input)
            
        valid = [valid[i:i + batch_size] for i in range(0, len(valid), batch_size)]

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
        image: image embeddings.

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
    logits = logits[:, :-1]

    # Similarly, the first token cannot be predicted.
    target_tokens = input_tokens[:, 1:]
    target_mask = input_mask[:, 1:]

    # Convert the target labels into one-hot encoded vectors.
    one_hot = jax.nn.one_hot(target_tokens, logits.shape[-1])

    # Don't update on unwanted tokens.
    one_hot = one_hot * target_mask.astype(one_hot.dtype)[...,None]

    # Normalisation factor.
    norm_factor = 1 / (jnp.sum(target_mask) + 1e-8)

    # Extend one hot for img logits
    one_hot = jnp.concatenate([jnp.zeros((logits.shape[0], 729, logits.shape[-1]), dtype=jnp.bfloat16), one_hot], axis=1)

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
    img_embed: jax.Array,
    ) -> tuple[jax.Array, Params, optax.OptState]:
    """Train step.

    Args:
        model: gemma transformer model.
        params: model's input parameters.
        optimizer: optax optimizer to use.
        opt_state: input optimizer's state.
        pad_id: id of the pad token.
        input_tokens: input text tokens.
        input_mask: masking for the input tokens.
        img_embed: image embeddings.

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
    # Update the parameters according to train policy

    grads, opt_state = optimizer.update(grads, opt_state, params)

    params = optax.apply_updates(params, grads)

    return train_loss, params, opt_state


@functools.partial(
    jax.jit,
    static_argnames=['model', 'optimizer'],
    donate_argnames=['params', 'opt_state'],
)
def frozen_train_step(
    model: recurrentgemma.Griffin,
    params: Params,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    pad_id: int,
    input_tokens: jax.Array,
    input_mask: jax.Array,
    img_embed: jax.Array,
    ) -> tuple[jax.Array, Params, optax.OptState]:
    """Train step.

    Args:
        model: gemma transformer model.
        params: model's input parameters.
        optimizer: optax optimizer to use.
        opt_state: input optimizer's state.
        pad_id: id of the pad token.
        input_tokens: input text tokens.
        input_mask: masking for the input tokens.
        img_embed: image embeddings.

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
    # Update the parameters according to train policy
    grads, opt_state = optimizer.update(grads['vl_connector'], opt_state, params['vl_connector'])

    params['vl_connector'] = optax.apply_updates(params['vl_connector'], grads)


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
    training_cfg: TrainingConfig,
    ):
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

    if training_cfg.freeze_llm:
        opt_state = jax.jit(optimizer.init)(params['vl_connector'])
    else:
        opt_state = jax.jit(optimizer.init)(params)
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
    
    vis_enc = VisionEncoder()
    
    print(train_ds[500][0].input_tokens, train_ds[500][0].image)
    
    with mlflow.start_run():
        for val_example in validation_ds:
            torch_emb = [torch.squeeze(vis_enc(i.image)) for i in val_example]
            input_tokens = [jax.device_put(i.input_tokens, jax.devices("gpu")[0]) for i in val_example] ## Make dynamic for DDP
            input_tokens = jnp.stack(input_tokens, axis=0)
            target_mask = jnp.stack([jax.device_put(i.target_mask, jax.devices("gpu")[0]) for i in val_example], axis=0)
            img_embed = jnp.stack([t2j(i).astype(jnp.bfloat16) for i in torch_emb], axis=0)
            eval_loss += validation_step(
                model, params, dataset_builder._tokenizer.pad_id, input_tokens, target_mask, img_embed
            )
            n_steps_eval += 1
        print(f"Start, validation loss: {eval_loss/n_steps_eval}")
        mlflow.log_metric("val_loss", eval_loss/n_steps_eval, step=0)
        for train_example in train_ds:
            torch_emb = [torch.squeeze(vis_enc(i.image)) for i in train_example]
            input_tokens = [jax.device_put(i.input_tokens, jax.devices("gpu")[0]) for i in train_example] ## Make dynamic for DDP
            input_tokens = jnp.stack(input_tokens, axis=0)
            target_mask =  jnp.stack([jax.device_put(i.target_mask, jax.devices("gpu")[0]) for i in train_example], axis=0)
            img_embed = jnp.stack([t2j(i).astype(jnp.bfloat16) for i in torch_emb], axis=0)
            if training_cfg.freeze_llm:
                train_loss, params, opt_state = frozen_train_step(
                    model=model,
                    params=params,
                    optimizer=optimizer,
                    opt_state=opt_state,
                    pad_id=dataset_builder._tokenizer.pad_id,
                    input_tokens=input_tokens,
                    input_mask=target_mask,
                    img_embed=img_embed,
                    )
            else:
                train_loss, params, opt_state = train_step(
                    model=model,
                    params=params,
                    optimizer=optimizer,
                    opt_state=opt_state,
                    pad_id=dataset_builder._tokenizer.pad_id,
                    input_tokens=input_tokens,
                    input_mask=target_mask,
                    img_embed=img_embed,
                    )
            n_steps += 1
            avg_loss += train_loss
            # mlflow.log_metric("train_loss", train_loss, step=n_steps)
            print(train_loss)
            if n_steps % training_cfg.eval_every_n == 0:
                eval_loss = 0

                n_steps_eval = 0
                for val_example in validation_ds:
                    torch_emb = [torch.squeeze(vis_enc(i.image)) for i in val_example]
                    input_tokens = [jax.device_put(i.input_tokens, jax.devices("gpu")[0]) for i in val_example] ## Make dynamic for DDP
                    input_tokens = jnp.stack(input_tokens, axis=0)
                    target_mask = jnp.stack([jax.device_put(i.target_mask, jax.devices("gpu")[0]) for i in val_example], axis=0)
                    img_embed = jnp.stack([t2j(i).astype(jnp.bfloat16) for i in torch_emb], axis=0)
                    eval_loss += validation_step(
                        model, params, dataset_builder._tokenizer.pad_id, input_tokens, target_mask, img_embed
                    )
                    n_steps_eval += 1
                avg_loss /= training_cfg.eval_every_n
                eval_loss /= n_steps_eval
                print(f"STEP {n_steps} training loss: {avg_loss} - eval loss: {eval_loss}")
                mlflow.log_metric("train_loss", avg_loss, step=n_steps)
                mlflow.log_metric("val_loss", eval_loss, step=n_steps)
                avg_loss=0
            if training_cfg.max_steps is not None and n_steps > training_cfg.max_steps:
                break
    return params


        


            
    
    
    
if __name__ == "__main__":
    from absl import flags
    
    _STEPS = flags.DEFINE_integer(
    "steps",
    15_000,
    help="Number of training steps to run",
)
    

    vocab = spm.SentencePieceProcessor()
    vocab.Load("../model/flax/tokenizer.model")
    
    weights_dir = pathlib.Path("/home/jkobza/cadence/cadence-gemma/model/flax")
    ckpt_path = weights_dir / "2b-it"
    vocab_path = weights_dir / 'tokenizer.model'
    
    preset = recurrentgemma.Preset.RECURRENT_GEMMA_2B_V1
    
    params =  recurrentgemma.load_parameters(ckpt_path, "single_device")
    
    
    key = jax.random.PRNGKey(0)
    up_init = jax.nn.initializers.variance_scaling(
        scale=1.0,
        mode="fan_in",
        distribution="normal",
        in_axis=[1],
    )
    down_init = jax.nn.initializers.variance_scaling(
        scale=1.0,
        mode="fan_in",
        distribution="normal",
    )

    params['vl_connector'] = {'ffw_up': {'w': up_init(key, (1, 2176, 4000), dtype=jnp.bfloat16),'b': jnp.zeros((1, 1, 1, 4000), dtype=jnp.bfloat16)},
                              'ffw_down': {'bias': jnp.zeros((2560,), dtype=jnp.bfloat16), 'kernel': down_init(key, (4000, 2560), dtype=jnp.bfloat16)}}

    model_config = recurrentgemma.GriffinConfig.from_flax_params_or_variables(params, preset=preset)
    model = recurrentgemma.Griffin(model_config)

    
    
    SEQ_SIZE = 300
    tokenizer = Tokenizer(vocab)
    
    print(tokenizer.to_string([2, 106, 1645, 108, 1841, 603, 573, 8957, 1154, 575, 573, 2416, 235336, 107, 108, 106, 2516, 108, 2, 651, 8957, 575, 573, 2416, 603, 30619, 235269, 948, 603, 8884, 604, 15494, 152551, 235265, 107, 108, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]))
    
    dataset_builder= DatasetBuilder(tokenizer, SEQ_SIZE)
    frozen_training_cfg = TrainingConfig(
        optimizer='adamw',
        learning_rate=1e-5,
        b2=0.96,
        num_epochs=1,
        eval_every_n=20,
        batch_size=7,
        max_steps=4_000,
        freeze_llm=True
    )

    training_cfg = TrainingConfig(
        optimizer='adamw',
        learning_rate=1e-5,
        b2=0.96,
        num_epochs=1,
        eval_every_n=20,
        batch_size=7,
        max_steps=4_000,
        freeze_llm=False
    )
    

    
    mlflow.set_tracking_uri(uri="http://98.214.168.64:5000")
    mlflow.set_experiment("Cadence-Jax-1")

    # jax.debug.breakpoint()
    trained_params = train_loop(
        model=model,
        params=params,
        dataset_builder=dataset_builder,
        training_cfg=frozen_training_cfg,
    )
    
    trained_params = train_loop(
        model=model,
        params=trained_params,
        dataset_builder=dataset_builder,
        training_cfg=training_cfg,
    )
    
    
    recurrentgemma.utils.save_parameters("/home/jkobza/cadence/cadence-gemma/training/2b-mm", trained_params)
    
    ##TODO
    """
    2 stage approach has found no noticable difference
    
    1. Increase batch size - DONE
    2. Train for much longer
    3. Parralellize??
    """