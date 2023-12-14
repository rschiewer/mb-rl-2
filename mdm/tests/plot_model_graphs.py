from torchviz import make_dot

from mdm.models.building_blocks import *
from mdm.models.rssm_cell import RSSMCell


def vis_gaussian_decoder():
    d_batch = 8
    d_orig = 10
    d_enc = 32
    x_encoded = torch.rand(d_batch, d_enc)
    o_dec = GaussianDecoder(s_x_orig=(d_orig,), d_x_encoded=d_enc, lws=[16, 16], activation='relu', layer_norm=False, epsilon=0.1)
    y_dist, y_smpl = o_dec(x_encoded)
    make_dot(y_smpl.mean(), params=dict(o_dec.named_parameters())).view()


def vis_rssm():
    d_batch = 8
    d_o = 10
    d_a = 2

    d_det = 15
    d_stoch = 17
    d_state = d_det + d_stoch

    o_enc = MLPEncoder(s_x_orig=(d_o,), d_x_encoded=d_state, lws=[2, 2], activation='relu', layer_norm=True)
    o_dec = GaussianDecoder(s_x_orig=(d_o,), d_x_encoded=d_state, lws=[3, 3], activation='relu', layer_norm=False,
                            epsilon=0.1)
    r_dec = GaussianDecoder(s_x_orig=(d_o,), d_x_encoded=d_state, lws=[5, 5], activation='relu', layer_norm=False,
                            epsilon=0.1)
    term_dec = BinomialDecoder(s_x_orig=(d_o,), d_x_encoded=d_state, lws=[7, 7], activation='relu', layer_norm=False)

    rssm_cell = RSSMCell(d_z=d_stoch, d_h=d_det, d_a=d_a, o_encoder=o_enc, o_decoder=o_dec, r_decoder=r_dec,
                         term_decoder=term_dec, z_prior_lws=[9, 9], z_post_lws=[11, 11], layer_norm=True,
                         rnn_type='gru')

    a_in = torch.rand(d_batch, d_a)
    o_in = torch.rand(d_batch, d_o)
    start_state = rssm_cell.init_state(d_batch, 'cpu')

    y, next_state = rssm_cell(a=a_in, o=o_in, last_state=start_state)
    make_dot(y['o'].mean(), dict(rssm_cell.named_parameters())).view()


if __name__ == '__main__':
    #vis_gaussian_decoder()
    vis_rssm()
