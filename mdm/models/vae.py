from typing import Sequence

import torch

from mdm.utils.torch_tools import layers_with_activation as lwa, masked_mean


class VAE(torch.nn.Module):

    def __init__(self,
                 d_x: int,
                 d_z: int,
                 latent_dist: str,
                 encoder_lws: Sequence[int],
                 decoder_lws: Sequence[int],
                 activation: str,
                 layer_norm: bool,
                 beta: float,
                 n_latent_categories: int = 0,
                 n_output_categories: int = 0,
                 epsilon: float = 0.1,
                 output_dist: str = None):
        super().__init__()

        # infer final encoder layer widths depending on d_x, d_z and latent_dist
        if latent_dist == 'normal':
            encoder_lws = [d_x, *encoder_lws, d_z * 2]
            d_z_smpl = d_z
        elif latent_dist == 'categorical':
            assert n_latent_categories > 0, f'n_latent_categories must be greater than 0 but is {n_latent_categories}'
            encoder_lws = [d_x, *encoder_lws, d_z * n_latent_categories]
            d_z_smpl = d_z * n_latent_categories
        else:
            raise ValueError(f'Unknown latent_dist: {latent_dist}')

        # infer final decoder layer widths depending on d_x and output_dist
        if output_dist is None:
            decoder_lws = [d_z_smpl, *decoder_lws, d_x]
        elif output_dist == 'normal':
            decoder_lws = [d_z_smpl, *decoder_lws, d_x * 2]
        elif output_dist == 'bernoulli':
            decoder_lws = [d_z_smpl, *decoder_lws, d_x]
        elif output_dist == 'categorical':
            assert n_output_categories > 0, f'n_output_categories must be greater than 0 but is {n_output_categories}'
            decoder_lws = [d_z_smpl, *decoder_lws, d_x * n_output_categories]
        else:
            raise ValueError(f'Unknown output dist: {output_dist}')

        # build encoder and decoder models and store constructor arguments
        self.encoder = torch.nn.Sequential(lwa(encoder_lws, activation, layer_norm=layer_norm, name='vae_encoder'))
        self.decoder = torch.nn.Sequential(lwa(decoder_lws, activation, layer_norm=layer_norm, name='vae_decoder'))
        self.d_z = d_z
        self.d_z_smpl = d_z_smpl
        self.latent_dist = latent_dist
        self.beta = beta
        self.n_latent_categories = n_latent_categories
        self.epsilon = epsilon
        self.output_dist = output_dist

    def z_dist(self,
               z_dist_params: torch.Tensor):
        if self.latent_dist == 'normal':
            mu, logvar = torch.tensor_split(z_dist_params, 2, -1)
            sigma = torch.nn.functional.softplus(logvar) + self.epsilon
            return torch.distributions.Normal(loc=mu, scale=sigma)
        elif self.latent_dist == 'categorical':
            params = z_dist_params.reshape(*z_dist_params.shape[:-1], self.d_z, self.n_latent_categories)
            return torch.distributions.OneHotCategorical(logits=params)

    def z_sample(self,
                 z_dist_params: torch.Tensor):
        z_dist = self.z_dist(z_dist_params)
        if self.latent_dist == 'normal':
            return z_dist.rsample()
        else:  # categorical
            z_dist_params = z_dist_params.reshape(*z_dist_params.shape[:-1], self.d_z, self.n_latent_categories)
            smpl = z_dist.sample() + z_dist_params - z_dist_params.detach()  # straight-through gradients
            return smpl

    def z_mode(self,
               z_dist_params: torch.Tensor):
        if self.latent_dist == 'normal':
            mu, _ = torch.tensor_split(z_dist_params, 2, -1)
            return mu
        else:  # categorical
            max_val, max_idx = torch.max(z_dist_params, dim=-1, keepdim=True)
            z_smpl = torch.zeros_like(z_dist_params)
            z_smpl.scatter_(-1, max_idx, 1)
            return z_smpl + z_dist_params - z_dist_params.detach()  # straight-through gradients

    def x_dist(self,
               x_dist_params: torch.Tensor):
        if self.output_dist is None:
            return x_dist_params
        elif self.output_dist == 'normal':
            mu, logvar = torch.tensor_split(x_dist_params, 2, -1)
            sigma = torch.nn.functional.softplus(logvar) + self.epsilon
            return torch.distributions.Normal(loc=mu, scale=sigma)
        elif self.output_dist == 'categorical':
            params = x_dist_params.reshape(*x_dist_params.shape[:-1], self.d_z, self.n_latent_categories)
            return torch.distributions.OneHotCategorical(logits=params)
        else:  # bernoulli
            return torch.distributions.Bernoulli(logits=x_dist_params)

    def x_sample(self,
                 x_dist_params: torch.Tensor):
        x_dist = self.x_dist(x_dist_params)
        if self.output_dist is None:
            return x_dist_params
        elif self.output_dist == 'normal':
            return x_dist.rsample()
        elif self.output_dist == 'categorical':
            x_dist_params = x_dist_params.reshape(*x_dist_params.shape[:-1], self.d_z, self.n_latent_categories)
            return x_dist.sample() + x_dist_params - x_dist_params.detach()  # straight-through gradients
        else:  # bernoulli
            return x_dist.sample() + x_dist_params - x_dist_params.detach()  # straight-through gradients

    def x_mode(self,
               x_dist_params: torch.Tensor):
        if self.output_dist is None:
            return x_dist_params
        elif self.output_dist == 'normal':
            mu, _ = torch.tensor_split(x_dist_params, 2, -1)
            return mu
        elif self.output_dist == 'categorical':
            max_val, max_idx = torch.max(x_dist_params, dim=-1, keepdim=True)
            x_smpl = torch.zeros_like(x_dist_params)
            x_smpl.scatter_(-1, max_idx, 1)
            return x_smpl + x_dist_params - x_dist_params.detach()  # straight-through gradients
        else:  # bernoulli
            return torch.round(torch.sigmoid(x_dist_params)) + x_dist_params - x_dist_params.detach()

    def encode(self,
               x: torch.Tensor,
               sample: bool):
        z_dist_params = self.encoder(x)

        if sample:
            z_smpl = self.z_sample(z_dist_params)
        else:
            z_smpl = self.z_mode(z_dist_params)

        return z_dist_params, z_smpl

    def prior_dist(self,
                   z_smpl: torch.Tensor):
        if self.latent_dist == 'normal':
            mu = torch.zeros_like(z_smpl)
            sigma = torch.ones_like(z_smpl)
            return torch.distributions.Normal(loc=mu, scale=sigma)
        elif self.latent_dist == 'categorical':
            logits = torch.ones_like(z_smpl)
            return torch.distributions.OneHotCategorical(logits=logits)

    def decode(self,
               z_smpl: torch.Tensor,
               sample: bool):
        if self.latent_dist == 'categorical':
            z_smpl = z_smpl.reshape(*z_smpl.shape[:-2], z_smpl.shape[-2] * z_smpl.shape[-1])
        x_rec_dist_params = self.decoder(z_smpl)

        if sample:
            x_rec_smpl = self.x_sample(x_rec_dist_params)
        else:
            x_rec_smpl = self.x_mode(x_rec_dist_params)

        return x_rec_dist_params, x_rec_smpl

    def forward(self,
                x: torch.Tensor,
                sample: bool):
        z_dist_params, z_smpl = self.encode(x, sample=sample)
        x_rec_dist_params, x_rec_smpl = self.decode(z_smpl, sample=sample)
        return z_dist_params, z_smpl, x_rec_dist_params, x_rec_smpl

    def eval_step(self,
                  x: torch.Tensor,
                  mask: torch.Tensor = None):
        z_dist_params, z_smpl, x_rec_dist_params, x_rec_smpl = self(x, sample=True)

        if self.output_dist is None:
            rec_loss = (x - x_rec_smpl) ** 2
        else:
            rec_loss = -self.x_dist(x_rec_dist_params).log_prob(x)
        rec_loss = masked_mean(rec_loss, mask)

        prior = self.prior_dist(z_smpl)
        posterior = self.z_dist(z_dist_params)
        kl_loss = torch.distributions.kl_divergence(posterior, prior)
        kl_loss = masked_mean(kl_loss, mask)

        loss = rec_loss + self.beta * kl_loss

        rec_mae = torch.abs(x - x_rec_smpl)
        rec_mae = masked_mean(rec_mae, mask)

        return {'total': loss, 'rec': rec_loss, 'kl': kl_loss, 'monitoring_rec_mae': rec_mae}