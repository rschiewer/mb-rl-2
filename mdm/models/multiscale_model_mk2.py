from typing import Dict, Any, OrderedDict
from abc import ABC, abstractmethod

from mdm.models.building_blocks import AbstractActionModel, RSSM
from mdm.utils.torch_tools import *
from mdm.models.dynamics_model import DynamicsModel


class RSSMBase(torch.nn.Module, DeviceMixin, ABC):

    forbidden_names = ('s', 'a', 'h', 's_prior', 's_post', 'h_low_level')

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


class RSSM_(RSSMBase):

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

    def gen_init_values(self,
                        d_batch: int):
        device = self.device
        internal_init = super(RSSM, self).gen_init_values(d_batch)
        heads_init = {head_name: torch.zeros(d_batch, d_head, dtype=torch.float32, device=device)
                      for head_name, d_head in self.d_heads}

        internal_init.update(heads_init)

        return internal_init

    def forward(self,
                s: torch.Tensor,
                a: torch.Tensor,
                h: Any,
                context_high_level: torch.Tensor = None,
                posterior_input: List[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        if context_high_level:
            s, a, context_high_level = add_time_dim(s, a, context_high_level, batch_first=self.det_core.batch_first)
            s_next_det, h = self.det_core(s, a, context_high_level, h=h)
        else:
            s, a = add_time_dim(s, a, batch_first=self.det_core.batch_first)
            s_next_det, h = self.det_core(s, a, h=h)
        s_next_det = remove_time_dim(s_next_det, batch_first=self.det_core.batch_first)

        s_next_prior = self.prior_core(s_next_det)
        x_in = (s_next_prior, s_next_det)

        # do we want to predict a posterior?
        if posterior_input:
            s_next_post = self.post_core(s_next_det, *posterior_input)
            x_in = (s_next_post, s_next_det)
        else:
            s_next_post = None

        out_heads = {head_name: head(x_in) for head_name, head in self.pred_heads.items()}
        out = {'s_prior': s_next_prior, 's_post': s_next_post, 'h': h}
        out.update(out_heads)

        return out

    def forward_with_loss(self,
                          s: torch.Tensor,
                          a: torch.Tensor,
                          h: Any,
                          context_high_level: torch.Tensor,
                          posterior_input: List[torch.Tensor],
                          heads_ground_truth: Dict[str, torch.Tensor]
                          ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        out_internals, out_heads = self(s, a, h, context_high_level, posterior_input)
        out_heads = {head_name: y_hat.rsample() for head_name, y_hat in out_heads}
        loss = {head_name: torch.mean((y_hat - heads_ground_truth[head_name]) ** 2)
                for head_name, y_hat in out_heads.items()}
        loss['kl_s'] = torch.distributions.kl_divergence(out_internals['s_post'], out_internals['s_prior'])

        return out_internals, out_heads, loss


class HiddenRSSM_(RSSMBase):

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
                 primitive_model: RSSM,
                 abstract_model: RSSM,
                 abstract_action_model: AbstractActionModel,
                 abstract_step_size: int):
        super().__init__()
        self.primitive_model = primitive_model
        self.abstract_model = abstract_model
        self.abstract_action_model = abstract_action_model
        self.abstract_step_size = abstract_step_size
        self.d_state = primitive_model.d_state
        self.d_action = primitive_model.d_action
        self.d_reward = primitive_model.d_reward
        self.d_observation = primitive_model.d_observation
        self.d_abstract_state = abstract_model.d_state
        self.d_abstract_action = abstract_model.d_action
        self.d_abstract_reward = abstract_model.d_reward

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
        actions_binned = bin_every_k_steps(a, self.macro_step_size, device)

        # memory for primitive predictions
        prim_obs = []
        prim_rews = []
        prim_terms = []
        prim_priors = []
        prim_posts = []
        prim_h = None
        prim_o = None
        prim_s = None

        abstr_obs = []
        abstr_rews = []
        abstr_terms = []
        abstr_priors = []
        abstr_posts = []
        abstr_h = None


        # memory for abstract predictions
        abstract_predictions = {head_name: [] for head_name in self.abstract_model.d_heads}
        abstract_priors = []
        abstract_posteriors = []

        # init current value placeholders
        current_primitive = self.primitive_model.gen_init_values(d_batch)
        current_abstract = self.abstract_model.gen_init_values(d_batch)

        for t in range(n_steps):
            if t % self.abstract_step_size == 0 and t > 0:
                # prepare input for next abstract state prediction
                h_flat = self._filter_h(current_primitive['h'])
                current_abstract['a'] = self.macro_action_model(actions_binned[:, t // self.abstract_step_size - 1])

                # do prediction
                pred = self.abstract_model(s=current_abstract['s'], a=current_abstract['a'],
                                           h=current_abstract['h'], posterior_input=[h_flat])

                # update current_abstract for next prediction
                current_abstract['s'] = pred['s_post'].rsample()
                current_abstract['h'] = pred['h']

                # store things
                abstract_priors.append(pred['s_prior'])
                abstract_posteriors.append(pred['s_post'])
                for head_name, head_mem in abstract_predictions.items():
                    head_mem.append(pred[head_name])

            # TODO: to this for primitive model as well


    def _filter_h(self, h: Tuple[torch.Tensor, torch.Tensor]):
        h = torch.concat(h, dim=0)  # concat h and c tensors of LSTM along the layer dimension, this is arbitrary
        h = torch.transpose(h, 0, 1)  # bring batch dimension to front
        h = torch.flatten(h, start_dim=1)  # fold h/c/layer dimension into d_hidden
        return h


class MultiscaleDynamicsModelMK3(DynamicsModel):

    def __init__(self,
                 abstraction_levels: OrderedDict[str, RSSM],
                 model_step_sizes: OrderedDict[str, int]):
        super(MultiscaleDynamicsModelMK2, self).__init__()

        if abstraction_levels.keys() != model_step_sizes.keys():
            raise ValueError(f'Keys in arguments abstraction_levels and model_step_sizes should be the same, '
                             f'but are: {abstraction_levels.keys()} and {model_step_sizes.keys()}')

        # self.step_sizes = {level_name: abstraction_level[1] for level_name, abstraction_level in abstraction_levels.items()}
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
