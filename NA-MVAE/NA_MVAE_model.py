import torch.nn as nn
import torch.multiprocessing
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from openTSNE import affinity

import numpy as np
import torch


def block(in_c, out_c):
    layers = [nn.Linear(in_c, out_c), nn.SiLU()]
    return layers


class MLP(nn.Module):  # spike-and-slab GP, D*L+1 parameters
    def __init__(self, input_dim=784, inter_dims=[500, 500, 300], output_dim=200):
        super(MLP, self).__init__()
        self.output_dim = output_dim
        model_list = [*block(input_dim, inter_dims[0])]
        for _ in range(0, len(inter_dims) - 1):
            model_list += [*block(inter_dims[_], inter_dims[_ + 1])]
        model_list += [nn.Linear(inter_dims[-1], output_dim)]
        self.encoder = nn.Sequential(*model_list)

    def forward(self, x):
        # handle nan if necessary
        x = torch.nan_to_num(x, 0.)
        e = self.encoder(x)
        return e

class NA_MVAE(nn.Module):
    def __init__(self, data_dict, emb_dim=128, device='cpu', alpha=5e-2):
        """
        integrating multiomics data by finding consensus of neighbourhood structures of different modalities
        :param data_dict: dictionary of multiomics data, each having the same size = total number of patients, missing values = NANs
        :param emb_dim: dimension of embedding
        :param device: cuda?
        :param alpha: regularisation strength
        """
        super(NA_MVAE, self).__init__()

        presence = [torch.tensor(~_.isna().to_numpy().all(1)).to(device) for _ in list(data_dict.values())]
        data_list = [torch.tensor(_.to_numpy(), dtype=torch.float32).to(device) for _ in list(data_dict.values())]

        self.input_dim_list = [_.shape[1] for _ in data_list]
        self.device = device
        self.P = np.mean(self.input_dim_list)
        self.N = data_list[0].shape[0]
        self.data_list = data_list
        self.presence = presence
        self.emb_dim = emb_dim
        self.alpha = alpha

        self.data_similarities = []
        for i in range(len(self.data_list)):
            nan_idx = ~self.data_list[i][:, 0].isnan()
            indices = nan_idx.nonzero(as_tuple=True)[0]
            temp = torch.zeros((self.data_list[i].shape[0], self.data_list[i].shape[0]))
            sim = torch.tensor(affinity.PerplexityBasedNN(self.data_list[i][nan_idx], perplexity=30, metric="euclidean").P.todense(), dtype=torch.float32)
            temp[indices[:, None], indices] += sim
            self.data_similarities += [temp * 1.0]

        self.data_idx_loader = None

        self.decoder_list = nn.ModuleList([MLP(input_dim=self.emb_dim,
                                               inter_dims=[2 * self.emb_dim, 4 * self.emb_dim,
                                                           max(4 * self.emb_dim, self.input_dim_list[i] // 4),
                                                           max(4 * self.emb_dim, self.input_dim_list[i] // 2)],
                                               output_dim=self.input_dim_list[i]) for i in range(len(self.data_list))])
        self.encoder_list = nn.ModuleList([MLP(input_dim=self.input_dim_list[i],
                                               inter_dims=[max(4 * self.emb_dim, self.input_dim_list[i] // 2),
                                                           max(4 * self.emb_dim, self.input_dim_list[i] // 4),
                                                           4 * self.emb_dim, 2 * self.emb_dim],
                                               output_dim=2 * self.emb_dim) for i in range(len(self.data_list))])

        self.noise_log_scales = nn.ParameterList([torch.zeros((1, _)) for _ in self.input_dim_list])

        self.register_buffer('prior_mean', torch.zeros(self.emb_dim))
        self.register_buffer('prior_std', torch.ones(self.emb_dim))

    def mc_kl_term(self, emb, post_mean, post_log_std, idx_list):
        loss_kl_1 = torch.distributions.kl.kl_divergence(
            torch.distributions.normal.Normal(loc=post_mean[:, None, :],
                                              scale=post_log_std[:, None, :].exp()),
            torch.distributions.normal.Normal(loc=self.prior_mean,
                                              scale=self.prior_std)).sum()

        affinities = 1.0 / (1.0 + torch.cdist(emb[None], emb[None])[0] ** 2)
        affinities.fill_diagonal_(0.0)

        loss_kl_2 = 0.
        for m in range(len(self.data_list)):
            available_id = self.presence[m][idx_list]
            overlap_m = idx_list[self.presence[m][idx_list]]  # subset of data id that has presence[m] True

            sub_affinity = affinities[available_id][:, available_id]
            Q = sub_affinity / (sub_affinity.sum() + 1e-8)

            sub_data_similarity = self.data_similarities[m][overlap_m][:, overlap_m]
            P = sub_data_similarity / sub_data_similarity.sum()

            loss_kl_2 += -1. * (P * torch.log(Q + 1e-8)).sum()

        return loss_kl_1 / (len(idx_list) * self.emb_dim) + loss_kl_2

    def get_embedding(self, new_data_list=None):
        if new_data_list is None:
            emb_store = torch.zeros((len(self.data_list), self.N, 2 * self.emb_dim))
            appearance = torch.zeros(self.N)
            for m in range(len(self.data_list)):
                idx_list = torch.tensor(range(self.N), device=self.device)
                available_id = self.presence[m][idx_list]
                appearance[available_id] += 1.
                overlap_m = idx_list[available_id]  # subset of data id that has presence[m] True
                emb_store[m, available_id, :] = self.encoder_list[m](self.data_list[m][overlap_m])
            merged_z = emb_store.sum(0) / appearance[:, None]
            return merged_z[:, :self.emb_dim], merged_z[:, self.emb_dim:]

        else:
            raise NotImplementedError('will do later')


    def predict(self):
        reconstructed = [torch.zeros_like(_) * float('nan') for _ in self.data_list]
        with torch.no_grad():
            mean_z, log_std_z = self.get_embedding()
            for _ in range(len(self.data_list)):
                reconstructed[_] = self.decoder_list[_](mean_z)

        return reconstructed

    def loss(self, idx_list):
        # it is actually a batched version!
        # get pairwise distance of the embeddings
        emb_store = torch.zeros((len(self.data_list), len(idx_list), 2 * self.emb_dim))
        appearance = torch.zeros(len(idx_list))
        for m in range(len(self.data_list)):
            available_id = self.presence[m][idx_list]
            appearance[available_id] += 1.
            overlap_m = idx_list[available_id]  # subset of data id that has presence[m] True
            emb_store[m, available_id, :] = self.encoder_list[m](self.data_list[m][overlap_m])
        merged_z = emb_store.sum(0) / appearance[:, None]
        mean_z, log_std_z = merged_z[:, :self.emb_dim], merged_z[:, self.emb_dim:]

        # push modality-specific embs closer
        A_ = emb_store.permute(1, 0, 2)  # N x M x P
        with torch.no_grad():
            is_zero = (A_ == 0).all(dim=2)
            B = ~(is_zero.unsqueeze(2) | is_zero.unsqueeze(1)) * 1.0
            B.diagonal(dim1=-2, dim2=-1).fill_(0.)

        dist_penalty = ((torch.cdist(A_, A_) * B) ** 2).sum() / B.sum()

        sample_z = mean_z + torch.randn_like(log_std_z) * log_std_z.exp()

        loss_kl = self.mc_kl_term(sample_z, mean_z, log_std_z, idx_list)

        loss_recon = 0.
        for m in range(len(self.data_list)):
            available_id = self.presence[m][idx_list]
            m_recon = self.decoder_list[m](sample_z[available_id])
            overlap_m = idx_list[self.presence[m][idx_list]]  # subset of data id that has presence[m] True
            nan_mask = ~torch.isnan(self.data_list[m][overlap_m])
            neg_gaussian_lkd = self.noise_log_scales[m] + 0.5 * ((self.data_list[m][overlap_m] - m_recon) / self.noise_log_scales[m].exp()) ** 2
            loss_recon += neg_gaussian_lkd[nan_mask].mean() * len(idx_list) + 1e-1 * (self.noise_log_scales[m] ** 2).mean()

        return loss_kl + loss_recon / len(idx_list) + self.alpha * dist_penalty / self.emb_dim

    def my_train(self, n_epoch=2000, lr=1e-3, batch_size=128):
        batch_size = self.N
        self.data_idx_loader = torch.utils.data.DataLoader(torch.tensor(range(self.N), device=self.device),
                                                           batch_size=batch_size,
                                                           shuffle=True)
        optimizer = torch.optim.Adam(lr=lr, params=self.parameters())
        for ep in range(n_epoch):
            if ep % 1000 == 0:
                print('Epoch={}'.format(ep))
            self.train(True)
            running_loss = 0.
            for batch_id in self.data_idx_loader:
                optimizer.zero_grad()
                batch_loss = self.loss(batch_id)
                batch_loss.backward()
                optimizer.step()
                running_loss += batch_loss.detach().cpu().item()
