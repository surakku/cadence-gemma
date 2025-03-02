# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
r"""An example showing how to load a checkpoint and sample from it.

Getting Started with RecurrentGemma Sampling:

Prerequisites:

1. Download your RecurrentGemma checkpoint: Choose the desired checkpoint and download it.
2. Get the Gemma tokenizer: Download the tokenizer file required for your model.
3. Install RecurrentGemma: Follow the straightforward instructions in the README to install the RecurrentGemma repository.

Ready to Sample!

Here's how to run the sampling_jax.py script:

python sampling_jax.py --path_checkpoint=${PATH_TO_THE_GEMMA_CHECKPOINT} \
    --path_tokenizer=${PATH_TO_THE_GEMMA_TOKENIZER} \
    --string_to_sample="Where is Paris?"
"""
from typing import Sequence

from absl import app
from absl import flags
from recurrentgemma import jax as recurrentgemma

import sentencepiece as spm

_PATH_CHECKPOINT = flags.DEFINE_string(
    "path_checkpoint", None, required=True, help="Path to checkpoint."
)
_PATH_TOKENIZER = flags.DEFINE_string(
    "path_tokenizer", None, required=True, help="Path to tokenizer."
)
_TOTAL_GENERATION_STEPS = flags.DEFINE_integer(
    "total_sampling_steps",
    128,
    help="Maximum number of step to run when decoding.",
)
_STRING_TO_SAMPLE = flags.DEFINE_string(
    "string_to_sample",
    "What animal is this?",
    help="Input string to sample.",
)

_IMAGE_TO_SAMPLE = flags.DEFINE_string(
    "image_to_sample",
    "/home/jkobza/cadence/cadence-gemma/recurrentgemma/vit/img_tests/dutch.jpg",
    help="Input image to sample."
)


def _load_and_sample(
    *,
    path_checkpoint: str,
    path_tokenizer: str,
    input_string: str,
    total_generation_steps: int,
    input_image: str
) -> None:
  """Loads and samples a string from a checkpoint."""
  print(f"Loading the parameters from {path_checkpoint}")
  params = recurrentgemma.load_parameters(path_checkpoint, "single_device")
  print("Parameters loaded.")
  # Create a sampler with the right param shapes.
  vocab = spm.SentencePieceProcessor()
  vocab.Load(path_tokenizer)
  config = recurrentgemma.GriffinConfig.from_flax_params_or_variables(
      params,
      preset=recurrentgemma.Preset.RECURRENT_GEMMA_2B_V1,
  )
  model = recurrentgemma.Griffin(config)
  sampler = recurrentgemma.ModalSampler(model=model, vocab=vocab, params=params["params"], is_it_model=True)
  sampled_output = sampler(
      input_strings=[input_string],
      total_generation_steps=total_generation_steps,
      img_path=input_image
  )

  print(f"Input string: {input_string}")
  print(f"Input image: {input_image}")
  print(f"Sampled string: {sampled_output.text}")


def main(argv: Sequence[str]) -> None:

  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  _load_and_sample(
      path_checkpoint=_PATH_CHECKPOINT.value,
      path_tokenizer=_PATH_TOKENIZER.value,
      input_string=_STRING_TO_SAMPLE.value,
      total_generation_steps=_TOTAL_GENERATION_STEPS.value,
      input_image=_IMAGE_TO_SAMPLE.value
  )


if __name__ == "__main__":
  app.run(main)
