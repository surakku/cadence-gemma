from absl import app
from absl import flags
from recurrentgemma import torch as recurrentgemma
import torch

import sentencepiece as spm
from typing import Sequence

from typing import Generic, NamedTuple, TypeVar

import dataclasses
from recurrentgemma import common


Cache = TypeVar("Cache")


@dataclasses.dataclass
class SamplingState(Generic[Cache]):
  """Internal sampling state.

  Attributes:
    tokens_buffer: Fixed-size buffer for accumulating the output tokens.
    step: The number of the current decoding step.
    total_steps: Total number of sampling steps.
    positions: The position of the latest token in the sequence.
    cache: Model state for conditioning the model on autoregressively.
    done: Whether decoding is done on the current sequence.
    logits_buffer: Fixed-size buffer for accumulating the output logits.
  """
  tokens_buffer: torch.Tensor
  step: torch.Tensor
  total_steps: torch.Tensor
  positions: torch.Tensor
  cache: Cache
  done: torch.Tensor
  logits_buffer: torch.Tensor | None = None


class SamplerOutput(NamedTuple):
  """Output of the sampler.

  Attributes:
    text: Decoded samples from the model.
    logits: Per-step logits used during sampling.
    tokens: Tokens corresponding to the generated samples.
  """

  text: list[str]
  logits: list[torch.Tensor]
  tokens: list[torch.Tensor]


class Sampler:
    
    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def vocab_size(self) -> int:
        return self.model.config.vocab_size
    
    def __init__(
        self,
        model,
        vocab,
        greedy_sample=False
    ):
        
        self.model = model
        self.vocab = vocab
        self.greedy_sample = greedy_sample
        self._eos_token = torch.tensor([self.vocab.eos_id()], device=self.device)
        


    def apply_model(
        self,
        tokens,
        segment_pos,
        cache=None,
        return_logits=True,
        return_cache=True,
        img_path: str| None = None
    ):
        return self.model(
            tokens=tokens,
            segment_pos=segment_pos,
            cache=cache,
            return_logits=return_logits,
            return_cache=return_cache,
            img_path=img_path
        )
        
    
    def _sample_from_logits(
        self,
        logits,
        ):
            """Samples from the logits categorical distribution."""
            if self.greedy_sample:
                return torch.argmax(logits, dim=-1)
            else:
                return torch.distributions.Categorical(logits=logits).sample()

    def _sample_step(
        self,
        sampler_state,
        end_sampling_at_eos_token,
    ):
        
        step = sampler_state.step
        tokens_buffer = sampler_state.tokens_buffer
        logits_buffer = sampler_state.logits_buffer
        
        last_token = sampler_state.tokens_buffer[:, step][:, None]
        logits, cache = self.apply_model(
            tokens=last_token,
            segment_pos=sampler_state.positions,
            cache=sampler_state.cache,
            return_logits=True,
            return_cache=True,
        )
    
        next_token = self._sample_from_logits(logits[:, 0])
        tokens_buffer[:, step + 1] = next_token
        
        if logits_buffer is not None:
            logits_buffer[:, step + 1] = logits[:, 0]
            
        if end_sampling_at_eos_token:
            print("EOS")
            # done_now = torch.equal(next_token, self._eos_token)
            done_now = False
        else:
            done_now = False
            
        return SamplingState(
            tokens_buffer=tokens_buffer,
            step=step + 1,
            total_steps=sampler_state.total_steps,
            positions=sampler_state.positions + 1,
            cache=cache,
            done=sampler_state.done | done_now,
            logits_buffer=logits_buffer,
        )


    def tokenize(self, input_string: str) -> torch.Tensor:
        """Tokenizes the input string."""
        input_string = common.apply_it_formatter(input_string)

        input_ids = self.vocab.EncodeAsIds(input_string)
        input_ids = torch.tensor(
            [self.vocab.bos_id()] + input_ids,
            dtype=torch.int32,
            device=self.device,
        )
        return input_ids
    
    def _sample_fn(
        self,
        sampler_state: SamplingState,
        end_sampling_at_eos_token: bool = True,
    ) -> SamplingState:
        """Internal sampling function (to be jitted)."""
        # This is -1, since we make the first sampling from the prompt.
        while (
            (sampler_state.step < sampler_state.total_steps - 1) &
            torch.any(torch.logical_not(sampler_state.done))
        ):
            sampler_state = self._sample_step(
                sampler_state, end_sampling_at_eos_token
                )

        return sampler_state
    
    
    def _prompt_processing_fn(
        self,
        tokens,
        input_lengths,
        total_generation_steps: int,
        return_logits: bool,
        echo: bool,
        img_path: str
        ) -> SamplingState:
        
        factory_kwargs = dict(device=self.device, dtype=torch.int32)
        batch_size, prompt_length = tokens.shape
    
        positions = torch.arange(prompt_length, **factory_kwargs)
        positions = torch.repeat_interleave(positions[None], batch_size, dim=0)
        positions = positions - prompt_length + input_lengths[:, None]
        positions = torch.clip(positions, min=-1)
        
        # Actual prompt processing.
        if total_generation_steps == 0:
        # No sampling.
            prev_logits, cache = self.apply_model(
                tokens=tokens,
                segment_pos=positions,
                cache=None,
                return_logits=return_logits and echo,
                return_cache=False,
                img_path=img_path
            )
            logits = None

        elif prompt_length == 1:
            # Just a single BOS token.
            logits, cache = self.apply_model(
                tokens=tokens,
                segment_pos=positions,
                cache=None,
                return_logits=return_logits,
                return_cache=True,
                img_path=img_path
            )
            prev_logits = logits[:, :0]
            
        else:
            print(positions[:, :-1])
            prev_logits, cache = self.apply_model(
                tokens=tokens[:, :-1],
                segment_pos=positions[:, :-1],
                cache=None,
                return_logits=return_logits and echo,
                return_cache=True,
                img_path=img_path
            )
            
            logits, cache = self.apply_model(
                tokens=tokens[:, -1:],
                segment_pos=positions[:, -1:],
                cache=cache,
                return_logits=True,
                return_cache=total_generation_steps > 1,
            )
            
            tokens_buffer = torch.full(
                (batch_size, total_generation_steps),
                self.vocab.pad_id(),
                **factory_kwargs,
            )
            
            if logits is not None:
            # Sample the next token and update the tokens buffer.
                next_token = self._sample_from_logits(logits[:, 0])
                tokens_buffer[:, 0] = next_token

            if return_logits:
                # Logits buffer for samples.
                logits_buffer = torch.zeros(
                    (batch_size, total_generation_steps, self.vocab_size),
                    dtype=self.dtype, device=self.device,
                )

                if logits is not None:
                    # Updated the logits buffer with the ones used for the next token.
                    logits_buffer[:, 0] = logits[:, 0]
            else:
                logits_buffer = None
            
            step = torch.tensor(0, **factory_kwargs)
            total_steps = torch.tensor(total_generation_steps, **factory_kwargs)
            
            if echo:
            # Append the tokens to start of the token buffer.
                tokens_buffer = torch.concatenate([tokens, tokens_buffer], dim=1)

                if return_logits:
                    if logits is None:
                    # No sampling, so all logits are coming from the prompt.
                        logits_buffer = prev_logits
                    else:
                    # Append the logits from the prompt to the start of the logits buffer.
                        all_logits = [prev_logits, logits, logits_buffer]
                        logits_buffer = torch.concatenate(all_logits, dim=1)

                # Update the step and the total steps accordingly.
                step = step + prompt_length
                total_steps = total_steps + prompt_length
                
        return SamplingState(
            tokens_buffer=tokens_buffer,
            step=step,
            total_steps=total_steps,
            positions=positions[:, -1:] + 1,
            cache=cache,
            done=torch.zeros((batch_size,), dtype=torch.bool),
            logits_buffer=logits_buffer,
        )


    def _get_padded_tokens(
      self,
      tokens: Sequence[torch.Tensor],
        ):
        """Returns an array of padded tokens."""
        max_input_length = max(len(input_ids) for input_ids in tokens)

        pad_values = [
            torch.full(
                [max_input_length - len(input_ids)],
                self.vocab.pad_id(),
                dtype=input_ids.dtype,
                device=self.device,
            )
            for input_ids in tokens
        ]

        padded_tokens = [
            torch.concatenate([pad, input_ids], dim=0)
            for input_ids, pad in zip(tokens, pad_values)
        ]
        padded_tokens = torch.stack(padded_tokens, dim=0)
        return padded_tokens
    
    @torch.no_grad
    def __call__(
        self,
        input_strings: Sequence[str],
        total_generation_steps: int,
        echo: bool = False,
        return_logits: bool = False,
        end_sampling_at_eos_token: bool = True,
        img_path: str | None = None
        ):
        """Samples a completion of the input string.

        Args:
        input_strings: input prompts to feed to the model for sampling.
        total_generation_steps: number of generation steps. will correspond to the
            longest prompt in the batch.
        echo: whether to return the prompt as part of the output sample.
        return_logits: whether to return per-step logits used during generation.
        end_sampling_at_eos_token: Whether to stop sampling for every sequence if
            the model produces an EOS token.

        Returns:
        sampler_output: A SamplerOutput object containing the generated samples.
        """

        if total_generation_steps < 0:
            raise ValueError("total_generation_steps must be at least 0.")

        # Create a batched array from inputs.
        all_input_ids = [self.tokenize(x) for x in input_strings]
        
        input_lengths = torch.tensor(
            [len(input_ids) for input_ids in all_input_ids],
            device=self.device,
            dtype=torch.int32,
        )
        
        padded_tokens = self._get_padded_tokens(all_input_ids)
        _, pad_length = padded_tokens.shape
        pad_lengths = pad_length - input_lengths
        
        sampling_state = self._prompt_processing_fn(
            padded_tokens,
            input_lengths,
            total_generation_steps,
            return_logits,
            echo,
            img_path=img_path
        )

        if total_generation_steps > 1:
            sampling_state = self._sample_fn(
                sampling_state,
                end_sampling_at_eos_token,
            )
            
        tokens = [
            tokens[l:]
            for tokens, l in zip(sampling_state.tokens_buffer, pad_lengths)
        ]
        
        if return_logits:
            logits = [
                logits[l:]
                for logits, l in zip(sampling_state.logits_buffer, pad_lengths)
            ]
        else:
            logits = []

            return SamplerOutput(
                text=[self.vocab.DecodeIds(seq.tolist()) for seq in tokens],
                tokens=tokens,
                logits=logits,
            )



_PATH_CHECKPOINT = flags.DEFINE_string(
    "path_checkpoint", None, required=True, help="Path to checkpoint."
)
_PATH_TOKENIZER = flags.DEFINE_string(
    "path_tokenizer", None, required=True, help="Path to tokenizer."
)
_TOTAL_GENERATION_STEPS = flags.DEFINE_integer(
    "total_sampling_steps",
    500,
    help="Maximum number of step to run when decoding.",
)
_STRING_TO_SAMPLE = flags.DEFINE_string(
    "string_to_sample",
    "What is this an image of?",
    help="Input string to sample.",
)

_PATH_IMAGE = flags.DEFINE_string(
    "path_image",
    None,
    required=True,
    help="Input image to sample.",
)

def _load_and_sample(
    *,
    path_checkpoint: str,
    path_tokenizer: str,
    input_string: str,
    total_generation_steps: int,
    img_path: str,
    ):
        """Loads and samples a string from a checkpoint."""
    #   device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device('cuda:1')
        print(f"Loading the parameters from {path_checkpoint} into {device}")
        params = torch.load(path_checkpoint)
        print(params["params"].keys())
        params = {k: v.to(device=device) for k, v in params["params"].items()}
        print("Parameters loaded.")
        # Create a sampler with the right param shapes.
        config = recurrentgemma.GriffinConfig.from_torch_params(
            params,
            preset=recurrentgemma.Preset.RECURRENT_GEMMA_2B_V1,
        )
        vocab = spm.SentencePieceProcessor()
        vocab.Load(path_tokenizer)
        model = recurrentgemma.Griffin(config, device=device, dtype=torch.bfloat16)
        model.load_state_dict(params, strict=False)
        sampler = Sampler(model=model, vocab=vocab)
        sampler_output = sampler(
            input_strings=[input_string],
            total_generation_steps=total_generation_steps,
            img_path = img_path
        )

        print(f"Input string: {input_string}")
        print(f"Sampled string: {sampler_output.text}")
  
def main(argv: Sequence[str]) -> None:

  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  _load_and_sample(
      path_checkpoint=_PATH_CHECKPOINT.value,
      path_tokenizer=_PATH_TOKENIZER.value,
      input_string=_STRING_TO_SAMPLE.value,
      total_generation_steps=_TOTAL_GENERATION_STEPS.value,
      img_path=_PATH_IMAGE.value
  )


if __name__ == "__main__":
  app.run(main)