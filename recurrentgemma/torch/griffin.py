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
"""Griffin model."""
from typing import Literal, overload


from recurrentgemma import common
from recurrentgemma.torch import array_typing as at
from recurrentgemma.torch import layers
from recurrentgemma.torch import modules
import torch
from torch import bfloat16, nn
from torch.utils import checkpoint

from ..vit import VisionEncoder
from ..projector import MLPProjector


Cache = dict[str, modules.ResidualBlockCache]


class Griffin(nn.Module):
  """Griffin model - https://arxiv.org/abs/2402.19427."""

  def __init__(
      self,
      config: common.GriffinConfig,
      gradient_checkpointing: bool = True,
      device: str | torch.device | None = None,
      dtype: torch.dtype | None = None,
  ):
    """Initializes the Griffin model.

    Args:
      config: The Griffin config.
      gradient_checkpointing: Whether to apply gradient checkpointing on every
        residual block.
      device: On what device to initialize parameters. Needed to allow for
        initializing the module without parameter initialzation.
      dtype: What dtype to use for initialziation.
    """
    super().__init__()
    self.config = config
    self.gradient_checkpointing = gradient_checkpointing
    print("\n\nCADENCE\n\n")
    
    self.vis_encoder = VisionEncoder(device=device)
    
    self.projector = MLPProjector(device=device)

    self.embedder = modules.Embedder(
        vocab_size=self.config.vocab_size,
        embed_dim=self.config.width,
        scale_by_sqrt_dim=self.config.embeddings_scale_by_sqrt_dim,
        device=device,
        dtype=dtype,
    )

    self.blocks = nn.ModuleList([
        modules.ResidualBlock(
            width=self.config.width,
            mlp_expanded_width=self.config.mlp_expanded_width,
            num_heads=self.config.num_heads,
            attention_window_size=self.config.attention_window_size,
            temporal_block_type=block_type,
            lru_width=self.config.lru_width,
            final_w_init_variance_scale=2.0 / self.config.num_layers,
            device=device,
            dtype=dtype,
        )
        for block_type in self.config.block_types
    ])
    self.final_norm = layers.RMSNorm(
        width=self.config.width, device=device, dtype=dtype
    )
    
    
  def reset_parameters(self) -> None:
    """Resets the parameters of the module."""
    self.embedder.reset_parameters()
    for block in self.blocks:
      block.reset_parameters()
    self.final_norm.reset_parameters()

  @overload
  def forward(
      self,
      tokens: at.Tokens,
      segment_pos: at.SegmentPos,
      cache: Cache | None = None,
      return_logits: Literal[False] = False,
      return_cache: Literal[False] = False,
  ) -> tuple[None, None]:
    ...

  @overload
  def forward(
      self,
      tokens: at.Tokens,
      segment_pos: at.SegmentPos,
      cache: Cache | None = None,
      return_logits: Literal[False] = False,
      return_cache: Literal[True] = True,
  ) -> tuple[None, Cache]:
    ...

  @overload
  def forward(
      self,
      tokens: at.Tokens,
      segment_pos: at.SegmentPos,
      cache: Cache | None = None,
      return_logits: Literal[True] = True,
      return_cache: Literal[False] = False,
  ) -> tuple[at.TokenLogits, None]:
    ...

  @overload
  def forward(
      self,
      tokens: at.Tokens,
      segment_pos: at.SegmentPos,
      cache: Cache | None = None,
      return_logits: Literal[True] = True,
      return_cache: Literal[True] = True,
  ) -> tuple[at.TokenLogits, Cache]:
    ...

  @at.typed
  def forward(
      self,
      tokens: at.Tokens,
      segment_pos: at.SegmentPos,
      cache: Cache | None = None,
      return_logits: bool = True,
      return_cache: bool = True,
      img_path: str | None = None
  ) -> tuple[at.TokenLogits | None, Cache | None]:
    """Calls Griffin.

    Args:
      tokens: Sequence of input tokens.
      segment_pos: Positions of each token in the sequence.
      cache: Optiona for the model.
      return_logits: Whether to compute and return the logits.
      return_cache: Whether to compute and return the updated cache.

    Returns:
      Output of the model together with the updated cache. If `cache` is None
      than the returned updated cache is empty initialized and filled in from
      the input sequence.
    """
    if not return_logits and not return_cache:
      return None, None

    if(tokens.shape[0] != 1):
        tokens = tokens[None, :]
        
    input_emb = self.embedder.encode(tokens)
    x = input_emb
    

    if 0 in segment_pos and img_path:
        dog = self.vis_encoder(img_path)
        dog = self.projector(dog).to(bfloat16)
        dog = dog.to(x.device)
        x = torch.cat((dog, x), dim=1)
        segment_pos+=729
        seg_extended = [torch.tensor([[0]], device=segment_pos.device, dtype=segment_pos.dtype),
                        torch.cumsum(torch.ones((1, 728), device=segment_pos.device, dtype=segment_pos.dtype), dim=-1),
                        segment_pos]
        segment_pos = torch.cat(seg_extended, dim=-1)
        print(segment_pos)


    new_cache = {}
    for i, block in enumerate(self.blocks):
      block_name = f"blocks.{i}"
      block_cache = None if cache is None else cache[block_name]
      if self.gradient_checkpointing:
        x, new_cache[block_name] = checkpoint.checkpoint(
            block,
            x,
            segment_pos,
            block_cache,
            return_cache,
            use_reentrant=False,
            determinism_check="none",
        )
      else:
        x, new_cache[block_name] = block(x, segment_pos, block_cache)

    if not return_logits:
      return None, new_cache

    x = self.final_norm(x)
    logits = self.embedder.decode(x)

    c = self.config.logits_soft_cap
    if c:
      logits = nn.functional.tanh(logits / c) * c

    if not return_cache:
      return logits, None

    return logits, new_cache

  def init_cache(
      self,
      batch_size: int,
      dtype: torch.dtype,
  ) -> Cache:
    """Initializes an empty cache for the model."""
    cache = {}
    for i, block_type in enumerate(self.config.block_types):
      cache[f"blocks.{i}"] = modules.ResidualBlock.init_cache(
          batch_size=batch_size,
          width=self.config.width,
          num_heads=self.config.num_heads,
          attention_window_size=self.config.attention_window_size,
          temporal_block_type=block_type,
          dtype=dtype,
          lru_width=self.config.lru_width,
      )
    return cache