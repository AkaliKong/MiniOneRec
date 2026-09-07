# Codebook balance and dead-code recovery

New RQ-VAE training runs enable two anti-collapse mechanisms:

1. A differentiable KL penalty makes each level's soft assignment distribution
   approach a uniform distribution.
2. Each quantizer tracks EMA code usage. A code that is continuously inactive
   and falls below the configured usage threshold is reinitialized from a
   high-error residual in the current batch.

Recovery is applied at the beginning of the next forward pass so codebook
weights referenced by the current autograd graph are never modified in place.

Validation reports utilization, perplexity, normalized entropy, and maximum
code share for every level. These metrics are stored in checkpoints under
`usage_metrics`.

```bash
bash rqvae.sh \
  --balance_loss_weight 0.01 \
  --balance_temperature 1.0 \
  --usage_ema_decay 0.99 \
  --dead_code_threshold 1e-4 \
  --dead_code_patience 100
```

Set `--balance_loss_weight 0 --dead_code_patience 0` to reproduce the previous
objective and disable recovery. Existing checkpoints remain compatible because
usage statistics are non-persistent buffers.
