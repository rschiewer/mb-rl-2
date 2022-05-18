from typing import Union, Sequence, Dict, Any, Callable
from abc import ABC, abstractmethod

import torch

from mdm.utils.torch_tools import *
from mdm.models.dynamics_model import DynamicsModel


class AbstractActionModel(torch.nn.Module, DeviceMixin):

    def __init__(self,
                 d_action: int,
                 n_abstract_steps: int,
                 d_abstract_action: int,
                 lws: tuple = (64, 64)):
        super(AbstractActionModel, self).__init__()

        self.flatten_layer = torch.nn.Flatten(start_dim=1)
        self.det_mdl = FeedforwardBlock(d_action * n_abstract_steps, lws=(*lws, d_abstract_action))

    def forward(self, actions: torch.Tensor):
        # actions.shape = (d_batch, n_abstract_steps, d_action)
        macro_action = self.flatten_layer(actions)
        macro_action = self.det_mdl(macro_action)
        macro_action = torch.tanh(macro_action)
        #macro_action = torch.softmax(macro_action, dim=-1)
        #macro_action = F.gumbel_softmax(macro_action, hard=True)
        return macro_action


class RSSMBase(torch.nn.Module, DeviceMixin, ABC):

    forbidden_names = ('s', 'a', 'h', 's_next_prior', 's_next_post')

    def __init__(self,
                 det_core: RecurrentBlock,
                 prior_core: GaussianBlock,
                 post_core: GaussianBlock):
        super(RSSMBase, self).__init__()

        self.det_core = det_core
        self.prior_core = prior_core
        self.post_core = post_core

    @property
    def d_s(self):
        return self.det_core.d_inputs[0]

    @property
    def d_a(self):
        return self.det_core.d_inputs[1]

    @property
    def d_s_high_level(self):
        return self.det_core.d_inputs[2]

    def gen_init_values(self, d_batch: int):
        device = self.device
        s_init = torch.zeros(d_batch, self.d_s, dtype=torch.float32, device=device)
        a_init = torch.zeros(d_batch, self.d_a, dtype=torch.float32, device=device)
        s_high_level_init = torch.zeros(d_batch, self.d_s_high_level, dtype=torch.float32, device=device)
        h_init = self.det_core.gen_h_placeholder(d_batch)

        return {'s': s_init, 'a': a_init, 'h': h_init}

    @abstractmethod
    def forward_with_loss(self,
                          *args,
                          **kwargs) -> Any:
        pass


class RSSM(RSSMBase):

    def __init__(self,
                 det_core: RecurrentBlock,
                 prior_core: GaussianBlock,
                 post_core: GaussianBlock,
                 **pred_heads: Union[FeedforwardBlock, GaussianBlock, ContinuousBernoulliBlock]):
        super(RSSM, self).__init__(det_core, prior_core, post_core)

        for head_name in pred_heads:
            if head_name in self.forbidden_names:
                raise ValueError(f'Found prediction head name {head_name}, which is forbidden! '
                                 f'Choose head names different from: {self.forbidden_names}.')

        self.pred_heads = torch.ModuleDict(**pred_heads)
        self.d_heads = {head_name: head.lws[-1] for head_name, head in pred_heads.items()}

    def gen_init_values(self, d_batch: int):
        device = self.device
        internal_init = super(RSSM, self).gen_init_values(d_batch)
        heads_init = {head_name: torch.zeros(d_batch, d_head, dtype=torch.float32, device=device)
                      for head_name, d_head in self.d_heads}

        return internal_init, heads_init

    def forward(self,
                s: torch.Tensor,
                a: torch.Tensor,
                h: Any,
                s_high_level: torch.Tensor,
                **heads_ground_truth: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        s, a, s_high_level = add_time_dim(s, a, s_high_level, batch_first=self.det_core.batch_first)
        s_next_det, h = self.det_core(s, a, s_high_level, h=h)
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
                          s_high_level: torch.Tensor,
                          **heads_ground_truth: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor],
                                                                       Dict[str, torch.Tensor]]:
        out_internals, out_heads = self(s, a, h, s_high_level, **heads_ground_truth)

        out_heads = {head_name: y_hat.rsample() for head_name, y_hat in out_heads}
        loss = {head_name: torch.mean(y_hat - heads_ground_truth[head_name] ** 2)
                for head_name, y_hat in out_heads.items()}
        loss['kl_s'] = torch.distributions.kl_divergence(out_internals['s_next_post'], out_internals['s_next_prior'])

        return out_internals, out_heads, loss


class HiddenRSSM(RSSMBase):

    def forward(self,
                s: torch.Tensor,
                a: torch.Tensor,
                h: Any,
                s_high_level: torch.Tensor,
                h_low_level: torch.Tensor) -> Dict[str, torch.Tensor]:
        s, a, s_high_level = add_time_dim(s, a, s_high_level, batch_first=self.det_core.batch_first)
        s_next_det, h = self.det_core(s, a, s_high_level, h=h)
        s_next_det = remove_time_dim(s_next_det, batch_first=self.det_core.batch_first)

        s_next_prior = self.prior_core(s_next_det)
        s_next_post = self.post_core(s_next_det, h_low_level)

        return {'s_next_prior': s_next_prior, 's_next_post': s_next_post, 'h': h}

    def forward_with_loss(self,
                          s: torch.Tensor,
                          a: torch.Tensor,
                          h: Any,
                          s_high_level: torch.Tensor,
                          h_low_level: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        out_internals = self(s, a, h, s_high_level, h_low_level)

        loss = {'kl_s': torch.distributions.kl_divergence(out_internals['s_next_post'], out_internals['s_next_prior'])}

        return out_internals, loss


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
        """
        1) Get batch shape and time steps
        2) Init memories for all outputs
        3) init all variables used in the prediction process
        4) do the big for-loop
            4.1) check for each level if a prediction step is due
            4.2) if so, update the respective variables
            4.3) store a copy of the updated variables
        """

        d_batch, n_steps = a.shape[:2]
        n_s_start = s_start.shape[1]
        device = self.device

        # memory for all prediction levels
        pred_mem = {level_name: {head_name: [] for head_name in model.pred_heads.keys()}
                    for level_name, model in self.abstraction_levels.items()}

        # init current values for all prediction levels
        current_values = {level_name: model.gen_init_values(d_batch)
                          for level_name, model in self.abstraction_levels.items()}



        
