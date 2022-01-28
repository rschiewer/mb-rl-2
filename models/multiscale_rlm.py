from typing import Dict, Iterable, Union, Tuple

import torch
import torch.nn.functional as F

from rl_model import RLModel


class old_DeterministicRecurrentModel(torch.nn.Module):

    def __init__(self,
                 d_state: int,
                 d_action: int,
                 d_hidden: int,
                 n_layers: int = 1):
        super(old_DeterministicRecurrentModel, self).__init__()

        self.d_state = d_state
        self.d_actions = d_action
        self.d_hidden = d_hidden
        self.layers = torch.nn.LSTM(d_state + d_action, d_hidden, num_layers=n_layers, batch_first=True)

    def forward(self,
                s: torch.Tensor,
                a: torch.Tensor,
                h: Tuple[torch.Tensor, torch.Tensor] = None):
        assert len(s.shape) == len(a.shape) == 2, ('Expected input with shape (batch_size, dim), ',
                                                               f'found {s.shape}, {a.shape}')
        n_batch = s.shape[0]
        if h is None:
            h = (torch.zeros(n_batch, self.d_hidden), torch.zeros(n_batch, self.d_hidden))

        x = torch.concat([s, a], dim=-1)
        x, h = self.layers(x, h)

        return x, h


def add_time_dim(*xs: torch.Tensor, batch_first: bool = True):
    i_unsqueeze = 1 if batch_first else 0
    unsqueezed = [x.unsqueeze(i_unsqueeze) for x in xs]
    if len(unsqueezed) == 1:
        unsqueezed = unsqueezed[0]
    return unsqueezed


def remove_time_dim(*xs: torch.Tensor, batch_first: bool = True):
    i_unsqueeze = 1 if batch_first else 0
    squeezed = [x.squeeze(i_unsqueeze) for x in xs]
    if len(squeezed) == 1:
        squeezed = squeezed[0]
    return squeezed


def make_time_constant(*xs: torch.Tensor, n_timesteps: int, batch_first: bool = True):
    if batch_first:
        consts = [x.unsqueeze(1).expand(x.shape[0], n_timesteps, *x.shape[1:]) for x in xs]
    else:
        consts = [x.unsqueeze(0).expand(n_timesteps, *x.shape) for x in xs]
    if len(consts) == 1:
        consts = consts[0]
    return consts


class DeterministicRecurrentModel(torch.nn.Module):

    def __init__(self,
                 *d_inputs: int,
                 d_hidden: int,
                 n_layers: int = 1,
                 batch_first: bool = True):
        super(DeterministicRecurrentModel, self).__init__()

        self.d_inputs = d_inputs
        self.d_hidden = d_hidden
        self.batch_first = batch_first
        self.layers = torch.nn.LSTM(sum(d_inputs), d_hidden, num_layers=n_layers, batch_first=batch_first)

    def forward(self,
                *xs: torch.Tensor,
                h: Tuple[torch.Tensor, torch.Tensor] = None):
        n_batch = xs[0].shape[0]
        if h is None:
            h = (torch.zeros(n_batch, self.d_hidden), torch.zeros(n_batch, self.d_hidden))

        x = torch.concat(xs, dim=-1)
        x, h = self.layers(x, h)

        return x, h


class SamplingModel(torch.nn.Module):

    def __init__(self,
                 d_input: int,
                 lws: Iterable[int] = None):
        super(SamplingModel, self).__init__()

        if lws is None:
            lws = []

        self.d_input = d_input
        self.lws = (d_input, *lws, 2)
        self.layers = [torch.nn.Linear(lw_in, lw_out) for lw_in, lw_out in zip(self.lws, self.lws[1:])]

    def forward(self, x: torch.Tensor):
        for l in self.layers:
            x = l(x)
            x = torch.nn.functional.relu(x)
        x_dist = torch.distributions.Normal(x[..., 0], x[..., 1])

        return x_dist


class SingleStepModel(torch.nn.Module):

    def __init__(self,
                 d_macro_state: int,
                 d_macro_action: int,
                 d_state: int,
                 d_action: int,
                 d_hidden: int,
                 batch_first: bool = True):
        super(SingleStepModel, self).__init__()

        # the deterministic model receives s, a, marco_s, macro_a, macro_s_next
        self.det_mdl = DeterministicRecurrentModel(d_state, d_action, d_macro_state, d_macro_action, d_macro_state,
                                                   d_hidden=d_hidden, n_layers=3, batch_first=batch_first)
        # the sampling model receives the output of the deterministic model and no additional input
        self.sampling_mdl = SamplingModel(d_hidden, (64, 64))

        self.d_macro_state = d_macro_state
        self.d_macro_action = d_macro_action
        self.d_state = d_state
        self.d_action = d_action
        self.d_hidden = d_hidden

    def forward(self,
                s: torch.Tensor,
                a: torch.Tensor,
                macro_s: torch.Tensor,
                macro_a: torch.Tensor,
                macro_s_next: torch.Tensor,
                h: torch.Tensor,
                batch_first: bool = True):
        s, a, macro_s, macro_a, macro_s_next = add_time_dim(s, a, macro_s, macro_a, macro_s_next,
                                                            batch_first=batch_first)

        s_next_det, h = self.det_mdl(s, a, macro_s, macro_a, macro_s_next, h=h)
        s_next_dist = self.sampling_mdl(remove_time_dim(s_next_det, batch_first=batch_first))

        return s_next_det, s_next_dist, h


class AbstractModel(torch.nn.Module):

    def __init__(self,
                 d_mac):
        super(AbstractModel, self).__init__()



class MultiscaleRLModel(RLModel):

    def __init__(self, config: Dict):
        super(MultiscaleRLModel, self).__init__(config)


    def forward(self, trajectory: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
