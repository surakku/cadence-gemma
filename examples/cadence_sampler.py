from absl import app
from absl import flags
from recurrentgemma import torch as recurrentgemma
import torch

import sentencepiece as spm
from typing import Sequence



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
        sampler = recurrentgemma.Sampler(model=model, vocab=vocab)
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