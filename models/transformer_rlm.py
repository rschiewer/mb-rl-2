import copy
from enum import Enum

import torch
from transformers import GPT2Model, GPT2Config

from rl_model import RLModel


class TransformerModel(RLModel):

    def __init__(self, config: dict, state_embedding: torch.nn.Module = None, action_embedding: torch.nn.Module = None):
        super(TransformerModel, self).__init__(config)

        gpt2_conf = GPT2Config()
        gpt2_conf['n_layer'] = 6
        gpt2_conf['n_head'] = 8
        gpt2_conf['n_embed'] = config['d_embed']
        self._transformer = GPT2Model(config=gpt2_conf)

        if state_embedding is None:
            state_embedding = torch.nn.Linear(config['d_state'], config['d_embed'])
        else:
            state_embedding = copy.deepcopy(state_embedding)

        if action_embedding is None:
            action_embedding = torch.nn.Linear(config['d_action'], config['d_embed'])
        else:
            action_embedding = copy.deepcopy(action_embedding)

        self._embed_state = state_embedding
        self._embed_action = action_embedding

    def forward(self, trajectory: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        pass