import torch
from torch import nn
from torch.nn import functional as F

from .layers import MLPLayers
from .rq import ResidualVectorQuantizer


class RQVAE(nn.Module):
    def __init__(
        self,
        in_dim=768,
        num_emb_list=None,
        e_dim=64,
        layers=None,
        dropout_prob=0.0,
        bn=False,
        loss_type="mse",
        quant_loss_weight=1.0,
        beta=0.25,
        kmeans_init=False,
        kmeans_iters=100,
        sk_epsilons=None,
        sk_iters=100,
        balance_loss_weight=0.0,
        balance_temperature=1.0,
        usage_ema_decay=0.99,
        dead_code_threshold=0.0,
        dead_code_patience=0,
        dead_code_jitter=1e-4,
    ):
        super().__init__()
        if num_emb_list is None:
            raise ValueError("num_emb_list must be provided")
        if layers is None:
            raise ValueError("layers must be provided")
        if sk_epsilons is None:
            sk_epsilons = [0.0] * len(num_emb_list)
        if len(sk_epsilons) != len(num_emb_list):
            raise ValueError("sk_epsilons and num_emb_list must have equal length")

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.balance_loss_weight = balance_loss_weight
        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(
            layers=self.encode_layer_dims,
            dropout=self.dropout_prob,
            bn=self.bn,
        )
        self.rq = ResidualVectorQuantizer(
            num_emb_list,
            e_dim,
            beta=beta,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            sk_epsilons=sk_epsilons,
            sk_iters=sk_iters,
            enable_balance_loss=balance_loss_weight > 0,
            balance_temperature=balance_temperature,
            usage_ema_decay=usage_ema_decay,
            dead_code_threshold=dead_code_threshold,
            dead_code_patience=dead_code_patience,
            dead_code_jitter=dead_code_jitter,
        )
        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(
            layers=self.decode_layer_dims,
            dropout=self.dropout_prob,
            bn=self.bn,
        )

    def forward(self, x, use_sk=True):
        encoded = self.encoder(x)
        quantized, rq_loss, indices = self.rq(encoded, use_sk=use_sk)
        output = self.decoder(quantized)
        return output, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        encoded = self.encoder(xs)
        _, _, indices = self.rq(encoded, use_sk=use_sk)
        return indices

    def get_codebook_metrics(self):
        return self.rq.get_usage_metrics()

    def compute_loss(self, out, quant_loss, xs=None):
        if self.loss_type == "mse":
            loss_recon = F.mse_loss(out, xs, reduction="mean")
        elif self.loss_type == "l1":
            loss_recon = F.l1_loss(out, xs, reduction="mean")
        else:
            raise ValueError("incompatible loss type")
        loss_total = (
            loss_recon
            + self.quant_loss_weight * quant_loss
            + self.balance_loss_weight * self.rq.get_balance_loss()
        )
        return loss_total, loss_recon

