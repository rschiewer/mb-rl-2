from __future__ import annotations

from typing import Tuple, Sequence, Optional, List, Any, Dict

import torch
from torch.nn import ModuleList

from mdm.models.building_blocks import InputEncoder, OutputDecoder
from mdm.utils.torch_tools import layers_with_activation as lwa

RSSMStateType = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class RSSMCell(torch.nn.Module):

    def __init__(self,
                 d_z: int,
                 d_h: int,
                 d_a: int,
                 o_encoder: 'InputEncoder',
                 o_decoder: 'OutputDecoder',
                 r_decoder: 'OutputDecoder',
                 term_decoder: 'OutputDecoder',
                 d_s_embedding: int = None,
                 n_hidden_layers: int = 1,
                 hidden_dropout: float = 0.1,
                 epsilon: float = 0.01,
                 z_prior_lws: Sequence[int] = (32, 32),
                 z_post_lws: Sequence[int] = (32, 32),
                 s_embedding_lws: Sequence[int] = (32, 32),
                 layer_norm: bool = False,
                 activation: str = 'relu',
                 rnn_type: str = 'lstm',
                 latent_dist: str = 'normal',
                 n_latent_categories: int = 32,
                 name: str = 'rssm_cell'):
        super().__init__()

        # if latent_dist != 'normal':
        #    raise NotImplementedError('Check z_dist, z_dist_params, z_sample and z_mode methods first!')

        # if latent_dist == 'normal':
        #    assert o_decoder.d_x_encoded == d_z + d_h
        #    assert r_decoder.d_x_encoded == d_z + d_h
        #    assert term_decoder.d_x_encoded == d_z + d_h
        # elif latent_dist == 'categorical':
        #    assert o_decoder.d_x_encoded == d_z * n_latent_categories + d_h
        #    assert r_decoder.d_x_encoded == d_z * n_latent_categories + d_h
        #    assert term_decoder.d_x_encoded == d_z * n_latent_categories + d_h

        self.d_z = d_z
        self.d_h = d_h
        self.d_a = d_a
        self.d_s_embedding = d_s_embedding if d_s_embedding is not None else d_h + d_z
        self.o_encoder = o_encoder
        self.o_decoder = o_decoder
        self.r_decoder = r_decoder
        self.term_decoder = term_decoder
        self.n_hidden_layers = n_hidden_layers
        self.epsilon = epsilon
        self.layer_norm = layer_norm
        self.activation = activation
        self.rnn_type = rnn_type
        self.latent_dist = latent_dist

        self.n_latent_categories = n_latent_categories
        if latent_dist == 'normal':
            d_z_out = d_z * 2
            d_z_smpl = d_z
        elif latent_dist == 'bernoulli':
            d_z_out = d_z
            d_z_smpl = d_z
        elif latent_dist == 'categorical':
            d_z_out = d_z * self.n_latent_categories
            d_z_smpl = d_z * self.n_latent_categories
        else:
            raise ValueError(f'Unknown latent distribution type: {latent_dist}')
        self.d_z_smpl = d_z_smpl
        self.d_z_out = d_z_out

        z_prior_lws = (d_h, *z_prior_lws, d_z_out)
        z_post_lws = (d_h + self.d_o_encoded, *z_post_lws, d_z_out)

        d_det_core = d_z_smpl + d_a
        # need both to satisfy torch script
        self._lstm = ModuleList([torch.nn.LSTMCell(d_det_core, hidden_size=d_h)]
                                + [torch.nn.LSTMCell(d_h, hidden_size=d_h) for _ in range(n_hidden_layers - 1)])
        self._gru = ModuleList([torch.nn.GRUCell(d_det_core, hidden_size=d_h)]
                               + [torch.nn.GRUCell(d_h, hidden_size=d_h) for _ in range(n_hidden_layers - 1)])

        if self.rnn_type == 'lstm':
            for p in self._gru.parameters():
                p.requires_grad = False
        else:
            for p in self._lstm.parameters():
                p.requires_grad = False

        self._z_prior = torch.nn.Sequential(lwa(z_prior_lws, activation, layer_norm=layer_norm, name=f'{name}_z_prior'))
        self._z_post = torch.nn.Sequential(lwa(z_post_lws, activation, layer_norm=layer_norm, name=f'{name}_z_post'))

        if d_s_embedding == d_h + d_z:
            self.s_embedding = lambda x: x
        else:
            s_embedding_lws = (d_h + d_z_smpl, *s_embedding_lws, d_s_embedding)
            self.s_embedding = torch.nn.Sequential(lwa(s_embedding_lws, activation, layer_norm=layer_norm,
                                                       name=f'{name}_s_embedding'))

    @property
    def o_shape(self):
        return self.o_encoder.s_x_orig

    @property
    def d_o_encoded(self):
        return self.o_encoder.d_x_encoded

    @torch.jit.export
    @torch.no_grad()
    def init_state(self,
                   d_batch: int,
                   device: torch.device) -> RSSMStateType:
        z = self.zero_z(d_batch, device)
        h = self.zero_h(d_batch, device)
        rnn_state = self.zero_rnn_state(d_batch, device)
        mock = torch.zeros(d_batch, self.d_z_out, device=device)
        z_prior_params = self.z_dist_params(mock)
        z_post_params = self.z_dist_params(mock)
        s_embedding = self.zero_s_embedding(d_batch, device)
        # return z, z_prior_params, z_post_params, rnn_state[:, 0, :]  # for gru
        return h, z, z_prior_params, z_post_params, rnn_state, s_embedding

    @torch.jit.export
    @torch.no_grad()
    def zero_s_embedding(self,
                         d_batch: int,
                         device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, self.d_s_embedding, device=device)

    @torch.jit.export
    @torch.no_grad()
    def zero_z(self,
               d_batch: int,
               device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, self.d_z_smpl, device=device)

    @torch.jit.export
    @torch.no_grad()
    def zero_h(self,
               d_batch: int,
               device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, self.d_h, device=device)

    @torch.jit.export
    @torch.no_grad()
    def zero_o(self,
               d_batch: int,
               device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, self.d_o_encoded, device=device)

    @torch.jit.export
    @torch.no_grad()
    def zero_o_dist(self,
                    d_batch: int,
                    device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, self.d_z_out, device=device)

    @torch.jit.export
    @torch.no_grad()
    def zero_a(self,
               d_batch: int,
               device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, self.d_a, device=device)

    @torch.jit.export
    @torch.no_grad()
    def zero_r(self,
               d_batch: int,
               device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, 1, device=device)

    @torch.jit.export
    @torch.no_grad()
    def zero_term(self,
                  d_batch: int,
                  device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, 1, device=device)

    @torch.jit.export
    @torch.no_grad()
    def zero_rnn_state(self,
                       d_batch: int,
                       device: torch.device) -> torch.Tensor:
        if self.rnn_type == 'lstm':
            return torch.zeros(d_batch, self.n_hidden_layers, self.d_h, 2, device=device)
        else:
            return torch.zeros(d_batch, self.n_hidden_layers, self.d_h, device=device)

    def _lstm_forward(self,
                      inp: torch.Tensor,
                      last_det_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h, c = last_det_state.unbind(-1)
        h_next, c_next = [], []
        inp_layer = inp
        for i_l, layer in enumerate(self._lstm):
            h_layer, c_layer = h[:, i_l], c[:, i_l]
            h_layer_next, c_layer_next = layer(inp_layer, (h_layer, c_layer))
            inp_layer = h_layer_next
            h_next.append(h_layer_next)
            c_next.append(c_layer_next)
        h_next = torch.stack(h_next, 1)
        c_next = torch.stack(c_next, 1)
        next_rnn_state = torch.stack([h_next, c_next], dim=-1).contiguous()
        h_out = h_next[:, -1]
        return h_out, next_rnn_state

    def _gru_forward(self,
                     inp: torch.Tensor,
                     last_rnn_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_next = self._gru[0](inp, last_rnn_state[:, 0])
        return h_next, h_next.unsqueeze(1)

    def imagine(self,
                a: torch.Tensor,
                last_z: torch.Tensor,
                last_rnn_state: torch.Tensor,
                sample: bool = True):
        # z = torch.flatten(last_state['z'], start_dim=1)  # in case of categorical latents this is necessary
        z = torch.flatten(last_z, start_dim=1)  # in case of categorical latents this is necessary
        # z_embed = self._z_embed_net(last_state['z'])

        # if torch.isnan(z_embed).any():
        #    raise RuntimeError(f'Invalid NAN embedded z in imagine: {z_embed}')

        inp = torch.concat([z, a], dim=-1)
        if self.rnn_type == 'lstm':
            h, next_rnn_state = self._lstm_forward(inp, last_rnn_state)
        else:
            h, next_rnn_state = self._gru_forward(inp, last_rnn_state)

        # if torch.isnan(h).any() or torch.isinf(h).any():
        #    raise RuntimeError(f'Invalid h in imagine: {h}')

        z_prior_params = self._z_prior(h)
        z_prior_params = self.z_dist_params(z_prior_params)
        if sample:
            z_smpl = self.z_sample(z_prior_params)
        else:
            z_smpl = self.z_mode(z_prior_params)

            # if torch.isnan(z_smpl).any() or torch.isinf(z_smpl).any():
        #    raise RuntimeError(f'Invalid z in imagine: {z_smpl}')

        return h, z_smpl, z_prior_params, z_prior_params, next_rnn_state

    def observe(self,
                a: torch.Tensor,
                o_enc: torch.Tensor,
                last_z: torch.Tensor,
                last_rnn_state: torch.Tensor,
                sample: bool = True):
        h, z_smpl, z_prior_params, _, next_rnn_state = self.imagine(a, last_z, last_rnn_state, sample)

        # if torch.isnan(o_enc).any():
        #    raise RuntimeError(f'Invalid NAN encoded observation in observe: {o_enc}')

        z_post_params = self._z_post(torch.concat([h, o_enc], dim=-1))
        z_post_params = self.z_dist_params(z_post_params)

        # overwrite sample from prior with sample from posterior
        if sample:
            z_smpl = self.z_sample(z_post_params)
        else:
            z_smpl = self.z_mode(z_post_params)

        return h, z_smpl, z_prior_params, z_post_params, next_rnn_state

    def forward(self,
                a: torch.Tensor,
                o_enc: Optional[torch.Tensor] = None,
                last_state: Optional[RSSMStateType] = None,
                use_posterior: bool = True,
                sample_state: bool = True) -> RSSMStateType:
        if torch.any(a > 1.0) or torch.any(a < -1.0):
            raise ValueError('Found invalid actions outside of [-1, 1] interval')
        # if o is None and last_state is None:
        #    raise ValueError('Need at least "o" or "last_state"')
        # if o is None and use_posterior:
        #    raise ValueError('Can\'t use posterior if no ground truth data is provided')

        d_batch = a.shape[0]
        if last_state is None:
            last_state = self.init_state(d_batch, a.device)
        if o_enc is None:
            o_enc = torch.zeros(d_batch, self.o_encoder.d_x_encoded, device=a.device)

        if torch.isnan(a).any() or torch.isinf(a).any():
            raise RuntimeError(f'Invalid action in imagine: {a}')
        if torch.isnan(last_state[1]).any() or torch.isinf(last_state[1]).any():
            raise RuntimeError(f'Invalid last state in imagine: {last_state[1]}')

        if use_posterior:
            ret = self.observe(a, o_enc, last_state[1], last_state[4], sample_state)
        else:
            ret = self.imagine(a, last_state[1], last_state[4], sample_state)
        h, z_smpl, z_prior_params, z_post_params, next_rnn_state = ret

        s = torch.concat([h, z_smpl], dim=-1)
        s = self.s_embedding(s)
        return h, z_smpl, z_prior_params, z_post_params, next_rnn_state, s

    @torch.jit.export
    def scan(self,
             a: torch.Tensor,
             o_enc: Optional[torch.Tensor] = None,
             start_state: Optional[RSSMStateType] = None,
             posterior_steps: int = 0,
             sample_state: bool = True) -> List[RSSMStateType]:
        d_time, d_batch = a.shape[:2]

        if o_enc is None:
            o_enc = torch.zeros(d_time, d_batch, self.o_encoder.d_x_encoded, device=a.device)
        elif posterior_steps < d_time:  # necessary to avoid index error in the loop below if o_enc is too short
            pad = torch.zeros(d_time - posterior_steps, d_batch, self.o_encoder.d_x_encoded, device=a.device)
            o_enc = torch.concat([o_enc, pad], dim=0)

        state_mem: List[RSSMStateType] = []
        state = start_state
        for t, a_t in enumerate(a):
            use_posterior = t <= posterior_steps
            state = self(a[t], o_enc[t], state, use_posterior, sample_state)
            state_mem.append(state)

        return state_mem

    def z_dist_params(self,
                      net_output: torch.Tensor):
        if self.latent_dist == 'normal':
            mu, logvar = torch.tensor_split(net_output, 2, -1)
            sigma = torch.nn.functional.softplus(logvar) + self.epsilon
            z_dist = torch.stack([mu, sigma], dim=-1)
        elif self.latent_dist == 'bernoulli':
            probs = torch.nn.functional.sigmoid(net_output)
            z_dist = probs
        else:  # categorical
            net_output = net_output.reshape((net_output.shape[0], self.d_z, self.n_latent_categories))
            probs = torch.nn.functional.softmax(net_output, dim=-1)
            # This ensures that the kl divergence and log probabilities stay well behaved
            # see https://github.com/ray-project/ray/blob/0b0431cad08cb56ce09921f47903eda525dd3e21/rllib/algorithms/dreamerv3/tf/models/components/representation_layer.py#L105C9-L105C79
            probs = 0.99 * probs + 0.01 * (1.0 / self.n_latent_categories)
            z_dist = probs.reshape(probs.shape[0], self.d_z * self.n_latent_categories)
        return z_dist

    def z_sample(self,
                 dist_params: torch.Tensor):
        if self.latent_dist == 'normal':
            mu, sigma = dist_params.unbind(-1)
            z_smpl = mu + torch.rand_like(sigma) * sigma
        elif self.latent_dist == 'bernoulli':
            # TODO: continuous bernoulli?
            z_smpl = torch.bernoulli(dist_params) + dist_params - dist_params.detach()
        else:  # categorical
            probs_reshaped = dist_params.reshape(dist_params.shape[0] * self.d_z, self.n_latent_categories)
            indices = torch.multinomial(probs_reshaped, 1, True)
            indices = indices.squeeze(-1)  # remove redundant extra dim coming from generating only one sample
            z_smpl = torch.nn.functional.one_hot(indices, self.n_latent_categories).to(dist_params)
            z_smpl = z_smpl + probs_reshaped - probs_reshaped.detach()  # straight-through gradient
            z_smpl = z_smpl.reshape(dist_params.shape[0], self.d_z * self.n_latent_categories)
            # z_smpl = torch.distributions.OneHotCategorical(probs=dist_params['probs']).sample()
            # z_smpl = z_smpl + dist_params['probs'] - dist_params['probs'].detach()
            # z_smpl = torch.flatten(z_smpl, start_dim=-2, end_dim=-1)
        return z_smpl

    def z_mode(self,
               dist_params: torch.Tensor):
        if self.latent_dist == 'normal':
            mu, sigma = dist_params.unbind(-1)
            z_smpl = mu
        elif self.latent_dist == 'bernoulli':
            z_smpl = torch.round(dist_params) + dist_params - dist_params.detach()
        else:  # categorical
            probs_reshaped = dist_params.reshape(dist_params.shape[0] * self.d_z, self.n_latent_categories)
            z_smpl = torch.argmax(probs_reshaped, dim=-1)
            z_smpl = torch.nn.functional.one_hot(z_smpl, num_classes=self.n_latent_categories)
            z_smpl = z_smpl + probs_reshaped - probs_reshaped.detach()  # straight-through gradient
            z_smpl = z_smpl.reshape(dist_params.shape[0], self.d_z * self.n_latent_categories)
        return z_smpl

    @torch.jit.ignore
    def z_dist(self,
               dist_params: torch.Tensor):
        if self.latent_dist == 'normal':
            mu, sigma = dist_params.unbind(-1)
            z_dist = torch.distributions.Normal(loc=mu, scale=sigma)
            z_dist = torch.distributions.Independent(z_dist, 1)
        elif self.latent_dist == 'bernoulli':
            z_dist = torch.distributions.Bernoulli(probs=dist_params)
            z_dist = torch.distributions.Independent(z_dist, 1)
        else:  # categorical
            z_dist = torch.distributions.OneHotCategorical(probs=dist_params)
        return z_dist

    @torch.jit.export
    def decode(self,
               s: torch.Tensor,
               sample: bool,
               reconstruct_observation: bool):
        if s.ndim == 2:
            d_batch = 0
        else:
            d_batch = 1
        if reconstruct_observation:
            o_dist, o_smpl = self.o_decoder(s, sample)
        else:
            o_dist, o_smpl = self.zero_o_dist(d_batch, s.device), self.zero_o(d_batch, s.device)
        r_dist, r_smpl = self.r_decoder(s, sample)
        # r_smpl = torch.nn.functional.tanh(r_smpl)
        term_dist, term_smpl = self.term_decoder(s, sample)

        return {'o': o_smpl, 'o_dist': o_dist, 'r': r_smpl, 'r_dist': r_dist, 'terminal': term_smpl,
                'terminal_dist': term_dist}


@torch.jit.script
def rssm_stack_states(h: List[torch.Tensor],
                      z: List[torch.Tensor],
                      z_prior: List[torch.Tensor],
                      z_post: List[torch.Tensor],
                      rnn_state: List[torch.Tensor],
                      s_embedding: List[torch.Tensor]):
    return (torch.stack(h).contiguous(),
            torch.stack(z).contiguous(),
            torch.stack(z_prior).contiguous(),
            torch.stack(z_post).contiguous(),
            torch.stack(rnn_state).contiguous(),
            torch.stack(s_embedding).contiguous())


@torch.jit.ignore
def rssm_stack_state_list(states: List[RSSMStateType]):
    return [list(x) for x in zip(*states)]


@torch.jit.script
def rssm_detach_state(h: torch.Tensor,
                      z: torch.Tensor,
                      z_prior: torch.Tensor,
                      z_post: torch.Tensor,
                      rnn_state: torch.Tensor,
                      s_embedding: torch.Tensor):
    return h.detach(), z.detach(), z_prior.detach(), z_post.detach(), rnn_state.detach(), s_embedding.detach()


@torch.jit.script
def rssm_state_seq_to_batch(h: List[torch.Tensor],
                            z: List[torch.Tensor],
                            z_prior: List[torch.Tensor],
                            z_post: List[torch.Tensor],
                            rnn_state: List[torch.Tensor],
                            s_embedding: List[torch.Tensor]):
    return (torch.concat(h, dim=0).contiguous(),
            torch.concat(z, dim=0).contiguous(),
            torch.concat(z_prior, dim=0).contiguous(),
            torch.concat(z_post, dim=0).contiguous(),
            torch.concat(rnn_state, dim=0).contiguous(),
            torch.concat(s_embedding, dim=0).contiguous())


@torch.jit.script
def rssm_state_keys() -> Tuple[str, str, str, str, str, str]:
    return 'h', 'z', 'z_prior', 'z_post', 'rnn_state', 's_embedding'


@torch.jit.ignore
def rssm_add_labels(seq: Sequence[Any, Any, Any, Any, Any, Any]) -> Dict[str, Any]:
    keys = rssm_state_keys()
    return {keys[0]: seq[0], keys[1]: seq[1], keys[2]: seq[2], keys[3]: seq[3], keys[4]: seq[4], keys[5]: seq[5]}


@torch.jit.script
def rssm_remove_labels(state: Dict[str, Any]) -> Tuple[Any, Any, Any, Any, Any, Any]:
    keys = rssm_state_keys()
    return state[keys[0]], state[keys[1]], state[keys[2]], state[keys[3]], state[keys[4]], state[keys[5]]
