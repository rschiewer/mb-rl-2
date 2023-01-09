from typing import Dict

import torch.nn.functional

from mdm.utils.torch_tools import FuzzyDeviceMixin
from mdm.models.dynamics_model import DynamicsModel
from mdm.models.building_blocks import *


class RnnBaselineModel(DynamicsModel, FuzzyDeviceMixin):

    def __init__(self,
                 d_h: int,
                 d_a: int,
                 obs_encoder: 'InputEncoder',
                 obs_decoder: 'OutputDecoder',
                 r_decoder: 'GaussianDecoder',
                 term_decoder: 'BinomialDecoder',
                 n_hidden_layers: int = 1,
                 hidden_dropout: float = 0.1,
                 rnn_type: str = 'lstm',
                 train_multistep_predictions: bool = False,
                 multistep_prediction_stride: int = 1,
                 stochastic_outputs: bool = True):
        super(RnnBaselineModel, self).__init__()
        self.d_a = d_a
        self.obs_encoder = obs_encoder
        self.obs_decoder = obs_decoder
        self.r_decoder = r_decoder
        self.term_decoder = term_decoder
        self.stochastic_outputs = stochastic_outputs
        self.train_multistep_predictions = train_multistep_predictions
        self.multistep_prediction_stride = multistep_prediction_stride

        if rnn_type == 'lstm':
            rnn_constr = torch.nn.LSTM
        elif rnn_type == 'gru':
            rnn_constr = torch.nn.GRU
        else:
            raise ValueError(f'Unsupported rnn type: {rnn_type}')

        d_det_core = obs_encoder.d_x_encoded + d_a + 2
        self._rnn = rnn_constr(d_det_core, hidden_size=d_h, num_layers=n_hidden_layers, batch_first=False,
                               dropout=hidden_dropout)

    def forward(self,
                o: torch.Tensor,
                a: torch.Tensor,
                r: torch.Tensor,
                term: torch.Tensor,
                n_warmup: int,
                rnn_state_start: torch.Tensor = None,
                sample: bool = True):
        o = o[:-1]
        a = a[1:]
        r = r[:-1]
        term = term[:-1]

        n_time_steps = a.shape[0]

        o_enc = self.obs_encoder(o[:n_warmup])
        inp = torch.cat([o_enc, a[:n_warmup], r[:n_warmup], term[:n_warmup]], dim=-1)
        pred_warmup, rnn_state = self._rnn(inp, rnn_state_start)
        _, o_warmup = self.obs_decoder(pred_warmup, sample=sample)
        _, r_warmup = self.r_decoder(pred_warmup, sample=sample)
        _, term_warmup = self.term_decoder(pred_warmup, sample=sample)

        o_last = o_warmup[-1].unsqueeze(0)  # add back time dim
        r_last = r_warmup[-1].unsqueeze(0)
        term_last = term_warmup[-1].unsqueeze(0)
        o, r, term = [], [], []
        for t in range(n_warmup, n_time_steps):
            a_t = a[t].unsqueeze(0)
            o_enc = self.obs_encoder(o_last)
            inp = torch.cat([o_enc, a_t, r_last, term_last], dim=-1)

            pred, rnn_state = self._rnn(inp, rnn_state)

            _, o_last = self.obs_decoder(pred, sample=sample)
            _, r_last = self.r_decoder(pred, sample=sample)
            _, term_last = self.term_decoder(pred, sample=sample)

            o.append(o_last)
            r.append(r_last)
            term.append(term_last)
        o_final = torch.cat([o_warmup] + o, dim=0)
        r_final = torch.cat([r_warmup] + r, dim=0)
        term_final = torch.cat([term_warmup] + term, dim=0)

        return {'prim_o': o_final,
                'prim_a': a,
                'prim_r': r_final,
                'prim_term': term_final}

    def train_step(self,
                   o_ground_truth: torch.Tensor,
                   a_ground_truth: torch.Tensor,
                   r_ground_truth: torch.Tensor,
                   term_ground_truth: torch.Tensor,
                   mask: torch.Tensor,
                   optimizer: torch.optim.Optimizer) -> Dict[str, torch.Tensor]:
        optimizer.zero_grad(set_to_none=True)
        losses = self.eval_step(o_ground_truth, a_ground_truth, r_ground_truth, term_ground_truth, mask)
        losses['total'].backward()
        optimizer.step()
        return losses

    def eval_step(self,
                   o_ground_truth: torch.Tensor,
                   a_ground_truth: torch.Tensor,
                   r_ground_truth: torch.Tensor,
                   term_ground_truth: torch.Tensor,
                   mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        n_time_steps = a_ground_truth.shape[0]

        pred_teacher_forcing = self(o_ground_truth,  a_ground_truth, r_ground_truth,
                                    term_ground_truth, n_time_steps, sample=self.stochastic_outputs)
        loss = self.calc_loss(pred_teacher_forcing, o_ground_truth, r_ground_truth, term_ground_truth, mask)

        if self.train_multistep_predictions:
            for t in range(1, n_time_steps, self.multistep_prediction_stride):
                pred_warmup = self(o_ground_truth, a_ground_truth, r_ground_truth, term_ground_truth, t,
                                   sample=self.stochastic_outputs)
                loss_warmup = self.calc_loss(pred_warmup, o_ground_truth, r_ground_truth, term_ground_truth, mask)
                for k in loss:
                    loss[k] += loss_warmup[k]
            for k in loss:
                loss[k] /= n_time_steps // self.multistep_prediction_stride

        return loss

    def calc_loss(self,
                  pred,
                  o_ground_truth: torch.Tensor,
                  r_ground_truth: torch.Tensor,
                  term_ground_truth: torch.Tensor,
                  mask: torch.Tensor):
        mask = 1 - mask
        o_loss = torch.nn.functional.mse_loss(pred['prim_o'], o_ground_truth[1:], reduction='none')
        o_loss = torch.mean(o_loss * mask[:-1])
        r_loss = torch.nn.functional.mse_loss(pred['prim_r'], r_ground_truth[1:], reduction='none')
        r_loss = torch.mean(r_loss * mask[:-1])
        term_loss = torch.nn.functional.mse_loss(pred['prim_term'], term_ground_truth[1:], reduction='none')
        term_loss = torch.mean(term_loss * mask[:-1])

        o_mae = torch.mean(torch.abs(pred['prim_o'] - o_ground_truth[1:]) * mask[:-1])
        r_mae = torch.mean(torch.abs(pred['prim_r'] - r_ground_truth[1:]) * mask[:-1])
        term_mae = torch.mean(torch.abs(pred['prim_term'] - term_ground_truth[1:]) * mask[:-1])

        total = o_loss + r_loss + term_loss

        return {'total': total, 'prim_o': o_loss, 'prim_r': r_loss, 'prim_term': term_loss,
                'monitoring_prim_o': o_mae, 'monitoring_prim_r': r_mae, 'monitoring_prim_term': term_mae}







