import unittest

from mdm.models.building_blocks import *
from mdm.models.rssm_cell import RSSMCell
from mdm.utils.utils import filter_mem_state_seq_to_batch
from mdm.utils.torch_tools import stack_dists, masked_mean


class RSSMTest(unittest.TestCase):

    def _gen_state_seq(self,
                       rssm: RSSMCell,
                       d_batch: int,
                       seq_len: int):
        seq_mem = {k: [] for k in self.rssm_cell.init_state(d_batch, 'cpu')}
        for i in range(seq_len):
            z = torch.full((d_batch, rssm.d_z), fill_value=i, dtype=torch.float32, device='cpu')
            z_dist = torch.distributions.Normal(loc=z, scale=torch.ones_like(z))
            rnn_state = rssm.zero_rnn_state(d_batch, 'cpu')
            if isinstance(rnn_state, tuple):
                rnn_state = torch.fill(rnn_state[0], i), torch.fill(rnn_state[1], i)
            else:
                rnn_state = torch.fill(rnn_state, i)
            seq_mem['z'].append(z)
            seq_mem['z_dist'].append(z_dist)
            seq_mem['z_prior'].append(z_dist)
            seq_mem['z_post'].append(z_dist)
            seq_mem['rnn_state'].append(rnn_state)

        return seq_mem

    def _gen_seq(self,
                 rssm: RSSMCell,
                 d_batch: int,
                 seq_len: int):
        seq_mem = self._gen_state_seq(rssm=rssm, d_batch=d_batch, seq_len=seq_len)
        n_time_steps = len(seq_mem['z'])
        d_batch = seq_mem['z'][0].shape[0]
        seq_mem['a'] = [torch.full((d_batch, rssm.d_a), fill_value=i) for i in range(n_time_steps)]
        seq_mem['terminal'] = [torch.full((d_batch, 1), fill_value=0) for _ in range(n_time_steps)]

        return seq_mem

    def setUp(self) -> None:
        d_o = 10
        d_a = 2

        d_det = 15
        d_stoch = 17
        d_state = d_det + d_stoch
        n_hidden_layers = 4
        rnn_type = 'gru'

        o_enc = MLPEncoder(s_x_orig=(d_o,), d_x_encoded=d_state, lws=[2, 2], activation='relu', layer_norm=True)
        o_dec = GaussianDecoder(s_x_orig=(d_o,), d_x_encoded=d_state, lws=[3, 3], activation='relu', layer_norm=False,
                                epsilon=0.1)
        r_dec = GaussianDecoder(s_x_orig=(d_o,), d_x_encoded=d_state, lws=[5, 5], activation='relu', layer_norm=False,
                                epsilon=0.1)
        term_dec = BinomialDecoder(s_x_orig=(d_o,), d_x_encoded=d_state, lws=[7, 7], activation='relu',
                                   layer_norm=False)

        self.rssm_cell = RSSMCell(d_z=d_stoch, d_h=d_det, d_a=d_a, o_encoder=o_enc, o_decoder=o_dec, r_decoder=r_dec,
                                  term_decoder=term_dec, z_prior_lws=[9, 9], z_post_lws=[11, 11], layer_norm=True,
                                  n_hidden_layers=n_hidden_layers, rnn_type=rnn_type)

    def test_gen_seq(self):
        seq_len = 50
        d_batch = 8
        seq = self._gen_state_seq(self.rssm_cell, d_batch, seq_len)

        for i in range(seq_len):
            i_tens = torch.tensor(i, device='cpu', dtype=torch.float32)
            self.assertTrue(torch.isclose(seq['z'][i].mean(), i_tens))
            if self.rssm_cell.rnn_type == 'lstm':
                self.assertTrue(torch.isclose(seq['rnn_state'][0].mean(), i_tens))
                self.assertTrue(torch.isclose(seq['rnn_state'][1].mean(), i_tens))
            else:
                self.assertTrue(torch.isclose(seq['rnn_state'][i].mean(), i_tens))

        # plt.matshow(torch.stack(seq['z']).detach().cpu().numpy().mean(axis=-1))
        # plt.show()

    def test_states_seq_to_batch(self):
        seq_len = 25
        d_batch = 32
        seq = self._gen_seq(self.rssm_cell, d_batch, seq_len)

        batched_seq, _ = filter_mem_state_seq_to_batch(mem=seq, rssm_instance=self.rssm_cell)

        # plt.matshow(batched_seq['z'])
        # plt.tight_layout()
        # plt.show()

        for t in range(seq_len):
            t_tens = torch.tensor(t, device='cpu', dtype=torch.float32)
            i_start = t * d_batch
            i_end = (t + 1) * d_batch
            subbatch_t_current = batched_seq['z'][i_start:i_end]
            self.assertTrue(torch.isclose(subbatch_t_current.mean(), t_tens))

    def test_states_seq_to_batch_i_end(self):
        seq_len = 25
        d_batch = 32
        offset = 10
        seq = self._gen_seq(self.rssm_cell, d_batch, seq_len)

        batched_seq, _ = filter_mem_state_seq_to_batch(mem=seq, rssm_instance=self.rssm_cell, i_end=-offset)
        self.assertEqual((seq_len - offset) * d_batch, batched_seq['z'].shape[0])

        for t in range(seq_len - offset):
            t_tens = torch.tensor(t, device='cpu', dtype=torch.float32)
            i_start = t * d_batch
            i_end = (t + 1) * d_batch
            subbatch_t_current = batched_seq['z'][i_start:i_end]
            self.assertTrue(torch.isclose(subbatch_t_current.mean(), t_tens))

    def test_states_seq_to_batch_mask(self):
        seq_len = 25
        d_batch = 32
        offset = 10
        seq = self._gen_seq(self.rssm_cell, d_batch, seq_len)

        batched_seq, mask = filter_mem_state_seq_to_batch(mem=seq, rssm_instance=self.rssm_cell, i_end=-offset)


    def test_latent_overshooting(self):
        seq_len = 25
        d_batch = 32
        n_latent_steps = 10
        seq = self._gen_seq(self.rssm_cell, d_batch, seq_len)

        start_states, _ = filter_mem_state_seq_to_batch(mem=seq, rssm_instance=self.rssm_cell, i_end=-n_latent_steps)
        actions = torch.stack(seq['a'])
        terminals = torch.stack(seq['terminal'])
        zs = torch.stack(seq['z'])

        action_windows, terminal_windows, z_windows, z_post_windows = [], [], [], []

        for t in range(1, actions.shape[0] - n_latent_steps + 1):
            action_windows.append(actions[t: t + n_latent_steps])
            terminal_windows.append(terminals[t: t + n_latent_steps])
            z_windows.append(zs[t: t +n_latent_steps])
            z_post_windows.append(stack_dists(seq['z_post'][t: t + n_latent_steps]))
        actions = torch.concat(action_windows, dim=1)  # concat all windows along batch dimension
        terminals = torch.concat(terminal_windows, dim=1)
        zs = torch.concat(z_windows, dim=1)

        subsequences = torch.concat([start_states['z'].unsqueeze(0), zs], dim=0)

        #fig = plt.figure(figsize=(16, 8))
        #plt.matshow(subsequences.mean(-1).detach().cpu().numpy(), fignum=fig, aspect='auto')
        #plt.show()

    def test_avg_upwards_filter_no_padding(self):
        seq_len = 12
        batch_size = 32
        data_size = 1
        window_size = 4
        flt = AvgUpwardsFilter(window_size=window_size)

        seq = torch.randint(0, 10, (seq_len, batch_size, data_size), dtype=torch.float32)
        seq[0] = 2
        seq[-1] = 5
        mask = torch.zeros_like(seq)
        mask[-1] = 1

        seq_filtered_control = torch.zeros(seq_len // window_size, batch_size, data_size, dtype=torch.float32)
        for i_b in range(batch_size):
            for t in range(0, seq_len, window_size):
                i_chunk = t // window_size
                chunk = seq[t: t+window_size, i_b]
                m = mask[t: t+window_size, i_b]
                seq_filtered_control[i_chunk, i_b] = masked_mean(chunk, m)

        seq_filtered = flt(seq, mask=mask.to(dtype=torch.bool))
        self.assertTrue(np.isclose(torch.abs(seq_filtered - seq_filtered_control).sum().detach().cpu().numpy(), 0))


    def test_avg_upwards_filter_with_padding(self):
        seq_len = 11
        batch_size = 32
        data_size = 1
        window_size = 4
        flt = AvgUpwardsFilter(window_size=window_size)

        seq = torch.randint(0, 10, (seq_len, batch_size, data_size), dtype=torch.float32)
        seq[0] = 2
        seq[-1] = 5
        mask = torch.zeros_like(seq)
        mask[-1] = 1

        n_pad = window_size - seq_len % window_size
        seq_padded = torch.concat([seq, torch.zeros(n_pad, batch_size, data_size)])
        mask_padded = torch.concat([mask, torch.ones(n_pad, batch_size, data_size)])
        padded_seq_len = seq_padded.shape[0]
        seq_filtered_control = torch.zeros(padded_seq_len // window_size, batch_size, data_size, dtype=torch.float32)
        for i_b in range(batch_size):
            for t in range(0, padded_seq_len, window_size):
                i_chunk = t // window_size
                chunk = seq_padded[t: t+window_size, i_b]
                m = mask_padded[t: t+window_size, i_b]
                seq_filtered_control[i_chunk, i_b] = masked_mean(chunk, m)

        seq_filtered = flt(seq, mask=mask.to(dtype=torch.bool))
        self.assertTrue(np.isclose(torch.abs(seq_filtered - seq_filtered_control).sum().detach().cpu().numpy(), 0))


if __name__ == '__main__':
    unittest.main()
