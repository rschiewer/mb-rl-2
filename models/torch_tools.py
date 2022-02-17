from typing import Tuple, Union, Iterable

import torch


class RecurrentBlock(torch.nn.Module):

    def __init__(self,
                 *d_inputs: int,
                 d_hidden: int,
                 n_layers: int = 1,
                 batch_first: bool = True):
        super(RecurrentBlock, self).__init__()

        self.d_inputs = d_inputs
        self.d_hidden = d_hidden
        self.n_rec_layers = n_layers
        self.batch_first = batch_first
        self.layer_list = torch.nn.LSTM(sum(d_inputs), d_hidden, num_layers=n_layers, batch_first=batch_first)

    def forward(self,
                *xs: torch.Tensor,
                h: Tuple[torch.Tensor, torch.Tensor] = None):
        n_batch = xs[0].shape[0]
        if h is None:
            h = (torch.zeros(n_batch, self.d_hidden), torch.zeros(n_batch, self.d_hidden))

        x = torch.concat(xs, dim=-1)
        x, h = self.layer_list(x, h)

        return x, h


class GaussianBlock(torch.nn.Module):

    def __init__(self,
                 *d_inputs: int,
                 lws: Union[Iterable[int], int]):
        super(GaussianBlock, self).__init__()

        if type(lws) is int:
            lws = [lws]

        self.d_inputs = d_inputs
        self.lws = (sum(d_inputs), *lws[:-1], lws[-1] * 2)  # double last layer width to have parameters for loc and scale
        self.layer_list = torch.nn.ModuleList([torch.nn.Linear(lw_in, lw_out)
                                               for lw_in, lw_out in zip(self.lws, self.lws[1:])])

    def forward(self,
                *xs: torch.Tensor):
        x = torch.concat(xs, dim=-1)
        for l in self.layer_list[:-1]:  # only activations on inner layers
            x = l(x)
            x = torch.nn.functional.relu(x)
        x = self.layer_list[-1](x)  # no activation on last layer

        loc, scale = torch.tensor_split(x, 2, dim=-1)
        scale = torch.nn.functional.relu(scale) + 1e-5  # scale should always be positive

        x_dist = torch.distributions.Normal(loc, scale)

        return x_dist


class FeedforwardBlock(torch.nn.Module):

    def __init__(self,
                 *d_inputs: int,
                 lws: Union[Iterable[int], int] = None):
        super(FeedforwardBlock, self).__init__()

        if lws is None:
            lws = []
        elif type(lws) is int:
            lws = [lws]

        self.d_inputs = d_inputs
        self.lws = (sum(d_inputs), *lws)
        self.layer_list = torch.nn.ModuleList([torch.nn.Linear(lw_in, lw_out)
                                               for lw_in, lw_out in zip(self.lws, self.lws[1:])])

    def forward(self,
                *xs: torch.Tensor):
        x = torch.concat(xs, dim=-1)
        for l in self.layer_list:
            x = l(x)
            x = torch.nn.functional.relu(x)

        return x


def add_time_dim(*xs: torch.Tensor,
                 batch_first: bool = True):
    i_unsqueeze = 1 if batch_first else 0
    unsqueezed = [x.unsqueeze(i_unsqueeze) for x in xs]
    if len(unsqueezed) == 1:
        unsqueezed = unsqueezed[0]
    return unsqueezed


def remove_time_dim(*xs: torch.Tensor,
                    batch_first: bool = True):
    i_unsqueeze = 1 if batch_first else 0
    squeezed = [x.squeeze(i_unsqueeze) for x in xs]
    if len(squeezed) == 1:
        squeezed = squeezed[0]
    return squeezed


def make_time_constant(*xs: torch.Tensor,
                       n_timesteps: int,
                       batch_first: bool = True):
    if batch_first:
        consts = [x.unsqueeze(1).expand(x.shape[0], n_timesteps, *x.shape[1:]) for x in xs]
    else:
        consts = [x.unsqueeze(0).expand(n_timesteps, *x.shape) for x in xs]
    if len(consts) == 1:
        consts = consts[0]
    return consts