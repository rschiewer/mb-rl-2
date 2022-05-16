from typing import Union, Sequence, Dict, Any, Callable

import torch

from mdm.utils.torch_tools import *
from mdm.models.dynamics_model import DynamicsModel


class RSSM(torch.nn.Module, DeviceMixin):

    forbidden_names = ('s', 'a', 'h')

    def __init__(self,
                 det_core: RecurrentBlock,
                 prior_core: GaussianBlock,
                 post_core: GaussianBlock,
                 **pred_heads: Union[FeedforwardBlock, GaussianBlock, ContinuousBernoulliBlock]):
        super(RSSM, self).__init__()

        for head_name in pred_heads:
            if head_name in self.forbidden_names:
                raise ValueError(f'Found prediction head name {head_name}, which is forbidden! '
                                 f'Choose head names different from: {self.forbidden_names}.')

        self.pred_heads = pred_heads
        self.det_core = det_core
        self.prior_core = prior_core
        self.post_core = post_core
        self.d_heads = {head_name: head.lws[-1] for head_name, head in pred_heads.items()}

    @property
    def d_s(self):
        return self.det_core.d_inputs[0]

    @property
    def d_a(self):
        return self.det_core.d_inputs[1]

    @property
    def d_context(self):
        return self.det_core.d_inputs[2]

    def gen_init_values(self, d_batch: int):
        device = self.device
        s_init = torch.zeros(d_batch, self.d_s, dtype=torch.float32, device=device)
        a_init = torch.zeros(d_batch, self.d_a, dtype=torch.float32, device=device)
        context_init = torch.zeros(d_batch, self.d_context, dtype=torch.float32, device=device)
        h_init = self.det_core.gen_h_placeholder(d_batch)
        heads_init = {head_name: torch.zeros(d_batch, d_head, dtype=torch.float32, device=device)
                      for head_name, d_head in self.d_heads}

        return {'s': s_init, 'h': h_init}, heads_init

    def forward(self,
                s: torch.Tensor,
                a: torch.Tensor,
                context: torch.Tensor,
                h: Any,
                **heads_ground_truth: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        s, a, context = add_time_dim(s, a, context, batch_first=self.det_core.batch_first)
        s_next_det, h = self.det_core(s, a, context, h=h)
        s_next_det = remove_time_dim(s_next_det, batch_first=self.det_core.batch_first)

        s_next_prior = self.prior_core(s_next_det)
        x_in = (s_next_prior, s_next_det)

        if heads_ground_truth:
            if heads_ground_truth.keys() != self.pred_heads.keys():
                raise ValueError(f'Expected ground truth data for the following heads: {self.pred_heads.keys()} '
                                 f'but got: {heads_ground_truth.keys()}')
            s_next_post = self.post_core(s_next_det, *heads_ground_truth.values())
            x_in = (s_next_post, s_next_det)
        else:
            s_next_post = None

        out_heads = {head_name: head(x_in) for head_name, head in self.pred_heads.items()}

        return {'s_next_prior': s_next_prior, 's_next_post': s_next_post, 'h': h}, out_heads

    def forward_with_loss(self,
                          s: torch.Tensor,
                          a: torch.Tensor,
                          h: Any,
                          context: torch.Tensor,
                          **heads_ground_truth: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor],
                                                                       Dict[str, torch.Tensor]]:
        out_internals, out_heads = self(s, a, h, context, **heads_ground_truth)

        loss = {head_name: torch.mean(head_out - heads_ground_truth[head_name] ** 2)
                    for head_name, head_out in out_heads.items()}
        loss['kl_s'] = torch.distributions.kl_divergence(out_internals['s_next_post'], out_internals['s_next_prior'])

        return out_internals, out_heads, loss


class MultiscaleDynamicsModelMK2(DynamicsModel):

    def __init__(self,
                 abstraction_levels: Dict[str, RSSM],
                 model_step_sizes: Dict[str, int]):
        super(MultiscaleDynamicsModelMK2, self).__init__()

        if abstraction_levels.keys() != model_step_sizes.keys():
            raise ValueError(f'Keys in arguments abstraction_levels and model_step_sizes should be the same, '
                             f'but are: {abstraction_levels.keys()} and {model_step_sizes.keys()}')

        #self.step_sizes = {level_name: abstraction_level[1] for level_name, abstraction_level in abstraction_levels.items()}
        self.abstraction_levels = abstraction_levels
        self.model_step_sizes = model_step_sizes


    def forward(self,
                s_start: torch.Tensor,
                a: torch.Tensor):
        d_batch, n_steps = a.shape[:2]
        n_start_states = s_start.shape[1]
        device = self.device

        # memory for all prediction levels
        pred_mem = {level_name: {head_name: [] for head_name in model.pred_heads.keys()}
                    for level_name, model in self.abstraction_levels.items()}

        # init current values for all prediction levels
        current_values = {level_name: model.gen_init_values(d_batch)
                          for level_name, model in self.abstraction_levels.items()}



        
