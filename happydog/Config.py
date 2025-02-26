from transformers import PretrainedConfig
from typing import List

class LLMConfig(PretrainedConfig):
     model_type="happydog"
     def __init__(self,
                  dim:int=512,
                  n_layers: int= 8,
                  n_heads: int=8,
                  n_kv_heads: int=8,
                  vocab_size: int=6400,
                  hidden_dim: int=None,
                  multiple_of: int=64,
                  norm_eps: float=1e-5,
                  max_seq_len: int=1024,
                  rope_theta: int=1e6,
                  dropout: float=0.0,
                  ):
          self.dim = dim
          self.n_layers=n_layers
          self.n_heads=n_heads
          self.n_kv_heads=n_kv_heads
          self.vocab_size=vocab_size
          self.hidden_dim=hidden_dim
          self.multiple_of=multiple_of
          self.norm_eps=norm_eps
          self.max_seq_len=max_seq_len
          self.rope_theta=rope_theta
          self.dropout=dropout