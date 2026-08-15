import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import kmeans, sinkhorn_algorithm


class VectorQuantizer(nn.Module):
    """Vector quantizer with balancing and deferred dead-code recovery.

    Usage buffers are non-persistent so checkpoints created by older versions
    of MiniOneRec remain loadable with ``strict=True``.
    """

    def __init__(
        self,
        n_e,
        e_dim,
        beta=0.25,
        kmeans_init=False,
        kmeans_iters=10,
        sk_epsilon=0.003,
        sk_iters=100,
        enable_balance_loss=False,
        balance_temperature=1.0,
        usage_ema_decay=0.99,
        dead_code_threshold=0.0,
        dead_code_patience=0,
        dead_code_jitter=1e-4,
    ):
        super().__init__()
        if balance_temperature <= 0:
            raise ValueError("balance_temperature must be positive")
        if not 0.0 <= usage_ema_decay < 1.0:
            raise ValueError("usage_ema_decay must be in [0, 1)")
        if dead_code_threshold < 0:
            raise ValueError("dead_code_threshold must be non-negative")
        if dead_code_patience < 0:
            raise ValueError("dead_code_patience must be non-negative")

        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters
        self.enable_balance_loss = enable_balance_loss
        self.balance_temperature = balance_temperature
        self.usage_ema_decay = usage_ema_decay
        self.dead_code_threshold = dead_code_threshold
        self.dead_code_patience = dead_code_patience
        self.dead_code_jitter = dead_code_jitter

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

        self.register_buffer(
            "ema_usage",
            torch.full((self.n_e,), 1.0 / self.n_e),
            persistent=False,
        )
        self.register_buffer(
            "inactive_steps",
            torch.zeros(self.n_e, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "last_batch_usage",
            torch.zeros(self.n_e),
            persistent=False,
        )
        self.balance_loss = torch.tensor(0.0)
        self.last_reset_count = 0
        self._pending_reset_indices = None
        self._pending_reset_values = None

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
        return z_q

    def init_emb(self, data):
        centers = kmeans(data, self.n_e, self.kmeans_iters)
        self.embedding.weight.data.copy_(centers)
        self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances):
        max_distance = distances.max()
        min_distance = distances.min()
        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        return (distances - middle) / amplitude

    def _compute_balance_loss(self, distances):
        if not self.enable_balance_loss:
            return distances.new_zeros(())
        assignments = F.softmax(
            -distances / self.balance_temperature,
            dim=-1,
        )
        average_probability = assignments.mean(dim=0).clamp_min(1e-12)
        # KL(p || Uniform) = sum p * log(p * K); zero at uniform use.
        return torch.sum(
            average_probability
            * torch.log(average_probability * self.n_e)
        )

    @torch.no_grad()
    def _apply_pending_resets(self):
        self.last_reset_count = 0
        if self._pending_reset_indices is None:
            return

        reset_indices = self._pending_reset_indices
        self.embedding.weight.data[reset_indices] = self._pending_reset_values
        revived_usage = max(self.dead_code_threshold, 1.0 / self.n_e)
        self.ema_usage[reset_indices] = revived_usage
        self.inactive_steps[reset_indices] = 0
        self.last_reset_count = int(reset_indices.numel())
        self._pending_reset_indices = None
        self._pending_reset_values = None

    @torch.no_grad()
    def _update_usage_and_queue_recovery(self, latent, distances, indices):
        counts = torch.bincount(indices, minlength=self.n_e)
        batch_usage = counts.to(dtype=latent.dtype) / max(indices.numel(), 1)
        self.last_batch_usage.copy_(batch_usage)
        if not self.training:
            return

        self.ema_usage.mul_(self.usage_ema_decay).add_(
            batch_usage,
            alpha=1.0 - self.usage_ema_decay,
        )

        used = counts > 0
        self.inactive_steps[used] = 0
        self.inactive_steps[~used] += 1

        reset_enabled = (
            self.dead_code_patience > 0
            and self.dead_code_threshold > 0
            and latent.shape[0] > 0
            and self._pending_reset_indices is None
        )
        if not reset_enabled:
            return

        dead_mask = (
            (self.inactive_steps >= self.dead_code_patience)
            & (self.ema_usage < self.dead_code_threshold)
        )
        dead_indices = torch.nonzero(dead_mask, as_tuple=False).flatten()
        if dead_indices.numel() == 0:
            return

        # Prefer residuals the current codebook represents poorly.
        nearest_distance = distances.min(dim=1).values
        candidate_order = torch.argsort(nearest_distance, descending=True)
        source_positions = torch.arange(
            dead_indices.numel(), device=latent.device
        ) % candidate_order.numel()
        replacements = latent[candidate_order[source_positions]].detach().clone()
        if self.dead_code_jitter > 0:
            replacements.add_(
                torch.randn_like(replacements) * self.dead_code_jitter
            )

        # Defer the in-place update until the next forward boundary. Mutating
        # weights here would invalidate the current autograd graph.
        self._pending_reset_indices = dead_indices.detach().clone()
        self._pending_reset_values = replacements

    def get_usage_metrics(self):
        usage = self.ema_usage.detach().float().clamp_min(0)
        probability = usage / usage.sum().clamp_min(1e-12)
        entropy = -(probability * probability.clamp_min(1e-12).log()).sum()
        batch_utilization = (self.last_batch_usage > 0).float().mean()
        dead_codes = (
            (self.inactive_steps >= max(self.dead_code_patience, 1))
            & (self.ema_usage < self.dead_code_threshold)
        ).sum()
        return {
            "batch_utilization": float(batch_utilization.item()),
            "ema_entropy": float(entropy.item()),
            "ema_perplexity": float(torch.exp(entropy).item()),
            "ema_max_share": float(probability.max().item()),
            "dead_codes": int(dead_codes.item()),
            "last_reset_count": self.last_reset_count,
        }

    def forward(self, x, use_sk=True):
        self._apply_pending_resets()
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        distances = (
            torch.sum(latent**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()
            - 2 * torch.matmul(latent, self.embedding.weight.t())
        )
        self.balance_loss = self._compute_balance_loss(distances)

        if not use_sk or self.sk_epsilon <= 0:
            indices = torch.argmin(distances, dim=-1)
        else:
            constrained_distances = self.center_distance_for_constraint(distances)
            Q = sinkhorn_algorithm(
                constrained_distances.double(),
                self.sk_epsilon,
                self.sk_iters,
            )
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print("Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)

        x_q = self.embedding(indices).view(x.shape)
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = codebook_loss + self.beta * commitment_loss
        x_q = x + (x_q - x).detach()
        self._update_usage_and_queue_recovery(
            latent, distances.detach(), indices
        )
        indices = indices.view(x.shape[:-1])
        return x_q, loss, indices
