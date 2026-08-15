import torch
import torch.nn as nn

from .vq import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):
    """Stack residual vector quantizers from coarse to fine."""

    def __init__(
        self,
        n_e_list,
        e_dim,
        sk_epsilons,
        beta=0.25,
        kmeans_init=False,
        kmeans_iters=100,
        sk_iters=100,
        enable_balance_loss=False,
        balance_temperature=1.0,
        usage_ema_decay=0.99,
        dead_code_threshold=0.0,
        dead_code_patience=0,
        dead_code_jitter=1e-4,
    ):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.vq_layers = nn.ModuleList(
            [
                VectorQuantizer(
                    n_e,
                    e_dim,
                    beta=self.beta,
                    kmeans_init=self.kmeans_init,
                    kmeans_iters=self.kmeans_iters,
                    sk_epsilon=sk_epsilon,
                    sk_iters=sk_iters,
                    enable_balance_loss=enable_balance_loss,
                    balance_temperature=balance_temperature,
                    usage_ema_decay=usage_ema_decay,
                    dead_code_threshold=dead_code_threshold,
                    dead_code_patience=dead_code_patience,
                    dead_code_jitter=dead_code_jitter,
                )
                for n_e, sk_epsilon in zip(n_e_list, sk_epsilons)
            ]
        )
        self.last_balance_loss = torch.tensor(0.0)

    def get_codebook(self):
        return torch.stack(
            [quantizer.get_codebook() for quantizer in self.vq_layers]
        )

    def get_balance_loss(self):
        return self.last_balance_loss

    def get_usage_metrics(self):
        return [quantizer.get_usage_metrics() for quantizer in self.vq_layers]

    def forward(self, x, use_sk=True):
        all_losses = []
        all_balance_losses = []
        all_indices = []
        x_q = torch.zeros_like(x)
        residual = x
        for quantizer in self.vq_layers:
            x_res, loss, indices = quantizer(residual, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res
            all_losses.append(loss)
            all_balance_losses.append(quantizer.balance_loss)
            all_indices.append(indices)

        mean_losses = torch.stack(all_losses).mean()
        self.last_balance_loss = torch.stack(all_balance_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)
        return x_q, mean_losses, all_indices

