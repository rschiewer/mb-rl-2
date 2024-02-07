import math
import unittest

import matplotlib.patheffects
import torch.distributions
import matplotlib.pyplot as plt

from mdm.models.building_blocks import EMAClustering
from matplotlib.patheffects import SimpleLineShadow, Normal


class TestEMAClustering(unittest.TestCase):
    def _gen_low_level_sequences(self, chunks: torch.Tensor, n_chunks_per_seq: int, n_sequences: int):
        window_size = chunks.shape[-1]
        # reshape into time and batch dimension, i.e. (T, B, D)
        sequences_chunked = chunks.reshape(n_chunks_per_seq, n_sequences, window_size)
        # bring chunk data dimension adjacent to time dimension
        sequences_chunked = sequences_chunked.swapaxes(1, 2)
        # fold data dimension into time dimension to form low level sequences that clusterer can process
        sequences = sequences_chunked.reshape(n_chunks_per_seq * window_size, n_sequences, 1)
        return sequences

    def _gen_cluster(self, dim: int, n_samples: int, loc_low: float = -1.0, loc_high: float = 1.0,
                     scale_high: float = 1.0):
        loc = torch.FloatTensor(dim).uniform_(loc_low, loc_high)
        scale = torch.FloatTensor(dim).uniform_(0.01, scale_high)
        data = torch.distributions.Normal(loc=loc, scale=scale).sample(sample_shape=(n_samples,))
        return data

    def test_static_2d(self):
        data = torch.tensor([[1.0, 1.0],
                             [1.0, 0.0],
                             [-1.0, -1.0],
                             [1.0, 0.51]], dtype=torch.float32)
        init_centroids = torch.tensor([[1.1, 1.1],
                                       [0.9, 0.1],
                                       [-1.1, -1.1]], dtype=torch.float32)
        clustering = EMAClustering(window_size=2, s_x_orig=1, n_centroids=3,
                                   alpha=0.1, dead_zone_mode='off')
        clustering.centroids.copy_(init_centroids)

        sequences = data.swapaxes(0, 1).unsqueeze(-1)
        indices = clustering(sequences)
        indices = indices.argmax(-1)

        sequences_np = sequences.detach().cpu().numpy().squeeze()
        data_flat_np = sequences_np.swapaxes(0, 1)
        indices_np = indices.detach().cpu().numpy().squeeze()

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i_centroid, centroid in enumerate(clustering.centroids):
            color = colors[i_centroid]
            centroid_np = centroid.detach().cpu().numpy()
            members_np = data_flat_np[i_centroid == indices_np]
            plt.scatter(members_np[:, 0], members_np[:, 1], color=color)
            plt.scatter(centroid_np[0], centroid_np[1], marker='x', color=color,
                        path_effects=[SimpleLineShadow((0, 0), 'black', 0.8, linewidth=3), Normal()])
        plt.show()

    def test_2d(self):
        n_samples = 5000
        n_sequences = 500
        n_chunks_per_seq = 10
        window_size = 2
        s_x_orig = 1
        n_clusters = 4
        n_centroids = 8
        n_train_steps = 500

        # gen data such that after chunking, each chunk of the sequence clearly belongs to one of the clusters
        # this means whole chunks have to be created at once
        clusters = [self._gen_cluster(dim=2, n_samples=n_samples, loc_low=-3.0, loc_high=3.0, scale_high=0.5)
                    for _ in range(n_clusters)]
        # combine cluster data to one long tensor that contains a lot of chunks
        data = torch.concat(clusters)

        # plt.scatter(data[:, 0], data[:, 1])
        # plt.show()

        # train clustering algorithm
        clustering = EMAClustering(window_size=window_size, s_x_orig=s_x_orig, n_centroids=n_centroids,
                                   alpha=0.05, dead_zone_mode='off')
        n_rows = 4
        n_cols = 4
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 10))
        axes = axes.ravel()
        plot_every = math.ceil(n_train_steps / (n_rows * n_cols))
        for i_step in range(n_train_steps):
            # randomly pick some chunks from the data
            seq_idxs = torch.randint(data.shape[0], size=(n_sequences * n_chunks_per_seq,))
            chunks_picked = data[seq_idxs]
            sequences = self._gen_low_level_sequences(chunks_picked, n_chunks_per_seq, n_sequences)
            ret = clustering.eval_step(sequences)

            if i_step % plot_every == 0:
                # reshape data into a form that is usable by the clusterer, i.e. (T, B, D)
                # for simplicity, make sequences' length equal to one chunk and adjust batch size accordingly
                all_sequences = self._gen_low_level_sequences(data, n_chunks_per_seq=1, n_sequences=data.shape[0])
                cluster_idx = clustering(all_sequences)

                # for easier plotting, bring batch dimension to front and remove dedundant data dimension of 1
                all_sequence_chunks = all_sequences.swapaxes(0, 1).squeeze(-1)
                chunk_memberships = cluster_idx.swapaxes(0, 1).squeeze(1).argmax(-1)

                all_sequence_chunks_np = all_sequence_chunks.detach().cpu().numpy()
                chunk_memberships_np = chunk_memberships.detach().cpu().numpy()
                centroids_np = clustering.centroids.detach().cpu().numpy()

                for i_centroid, centroid in enumerate(centroids_np):
                    i_subplot = int(i_step / plot_every)
                    members = all_sequence_chunks_np[chunk_memberships_np == i_centroid]
                    scatter_plot = axes[i_subplot].scatter(members[:, 0], members[:, 1], marker='.')
                    color = scatter_plot.get_facecolor()[0]
                    axes[i_subplot].scatter(centroid[0], centroid[1], marker='x', color=color,
                                            path_effects=[SimpleLineShadow((0, 0), 'black', 0.8, linewidth=3),
                                                          Normal()])
        plt.tight_layout()
        plt.show()

    def test_2d_dead_zone(self):
        n_samples = 5000
        n_sequences = 500
        n_chunks_per_seq = 10
        window_size = 2
        s_x_orig = 1
        n_clusters = 4
        n_centroids = 8
        n_train_steps = 500

        # gen data such that after chunking, each chunk of the sequence clearly belongs to one of the clusters
        # this means whole chunks have to be created at once
        clusters = [self._gen_cluster(dim=2, n_samples=n_samples, loc_low=-3.0, loc_high=3.0, scale_high=0.5)
                    for _ in range(n_clusters)]
        # combine cluster data to one long tensor that contains a lot of chunks
        data = torch.concat(clusters)

        # plt.scatter(data[:, 0], data[:, 1])
        # plt.show()

        # train clustering algorithm
        clustering = EMAClustering(window_size=window_size, s_x_orig=s_x_orig, n_centroids=n_centroids,
                                   alpha=0.05, dead_zone_mode='relative', dead_zone_size=1.0)
        n_rows = 4
        n_cols = 4
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 10))
        axes = axes.ravel()
        plot_every = math.ceil(n_train_steps / (n_rows * n_cols))
        for i_step in range(n_train_steps):
            # randomly pick some chunks from the data
            seq_idxs = torch.randint(data.shape[0], size=(n_sequences * n_chunks_per_seq,))
            chunks_picked = data[seq_idxs]
            sequences = self._gen_low_level_sequences(chunks_picked, n_chunks_per_seq, n_sequences)
            ret = clustering.eval_step(sequences)

            if i_step % plot_every == 0:
                # reshape data into a form that is usable by the clusterer, i.e. (T, B, D)
                # for simplicity, make sequences' length equal to one chunk and adjust batch size accordingly
                all_sequences = self._gen_low_level_sequences(data, n_chunks_per_seq=1, n_sequences=data.shape[0])
                cluster_idx = clustering(all_sequences)

                # for easier plotting, bring batch dimension to front and remove dedundant data dimension of 1
                all_sequence_chunks = all_sequences.swapaxes(0, 1).squeeze(-1)
                chunk_memberships = cluster_idx.swapaxes(0, 1).squeeze(1).argmax(-1)

                all_sequence_chunks_np = all_sequence_chunks.detach().cpu().numpy()
                chunk_memberships_np = chunk_memberships.detach().cpu().numpy()
                centroids_np = clustering.centroids.detach().cpu().numpy()

                i_subplot = int(i_step / plot_every)

                axes[i_subplot].scatter(all_sequence_chunks_np[:, 0], all_sequence_chunks_np[:, 1], color='black',
                                        alpha=0.5, marker='.')

                for i_centroid, centroid in enumerate(centroids_np):
                    members = all_sequence_chunks_np[chunk_memberships_np == i_centroid]
                    scatter_plot = axes[i_subplot].scatter(members[:, 0], members[:, 1], marker='.')
                    color = scatter_plot.get_facecolor()[0]
                    axes[i_subplot].scatter(centroid[0], centroid[1], marker='x', color=color,
                                            path_effects=[SimpleLineShadow((0, 0), 'black', 0.8, linewidth=3),
                                                          Normal()])
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    unittest.main()
