import argparse
import logging
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.rqvae import RQVAE
from trainer import Trainer


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected a boolean value")


def parse_args():
    parser = argparse.ArgumentParser(description="Train RQ-VAE semantic IDs")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_step", type=int, default=50)
    parser.add_argument("--learner", type=str, default="AdamW")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant")
    parser.add_argument("--warmup_epochs", type=int, default=50)
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data/Games/Games.emb-llama-td.npy",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout_prob", type=float, default=0.0)
    parser.add_argument("--bn", type=str2bool, default=False)
    parser.add_argument("--loss_type", type=str, default="mse")
    parser.add_argument("--kmeans_init", type=str2bool, default=True)
    parser.add_argument("--kmeans_iters", type=int, default=100)
    parser.add_argument(
        "--sk_epsilons", type=float, nargs="+", default=[0.0, 0.0, 0.0]
    )
    parser.add_argument("--sk_iters", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--num_emb_list", type=int, nargs="+", default=[256, 256, 256]
    )
    parser.add_argument("--e_dim", type=int, default=32)
    parser.add_argument("--quant_loss_weight", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument(
        "--layers", type=int, nargs="+", default=[2048, 1024, 512, 256, 128, 64]
    )

    # Enabled for new training runs. Setting the weight or patience to zero
    # reproduces the previous behavior.
    parser.add_argument("--balance_loss_weight", type=float, default=0.01)
    parser.add_argument("--balance_temperature", type=float, default=1.0)
    parser.add_argument("--usage_ema_decay", type=float, default=0.99)
    parser.add_argument("--dead_code_threshold", type=float, default=1e-4)
    parser.add_argument("--dead_code_patience", type=int, default=100)
    parser.add_argument("--dead_code_jitter", type=float, default=1e-4)
    parser.add_argument("--save_limit", type=int, default=5)
    parser.add_argument("--ckpt_dir", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    seed = 2024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    print("=================================================")
    print(args)
    print("=================================================")
    logging.basicConfig(level=logging.DEBUG)

    data = EmbDataset(args.data_path)
    model = RQVAE(
        in_dim=data.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        beta=args.beta,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
        balance_loss_weight=args.balance_loss_weight,
        balance_temperature=args.balance_temperature,
        usage_ema_decay=args.usage_ema_decay,
        dead_code_threshold=args.dead_code_threshold,
        dead_code_patience=args.dead_code_patience,
        dead_code_jitter=args.dead_code_jitter,
    )
    print(model)
    data_loader = DataLoader(
        data,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    trainer = Trainer(args, model, len(data_loader))
    best_loss, best_collision_rate = trainer.fit(data_loader)
    print("Best Loss", best_loss)
    print("Best Collision Rate", best_collision_rate)

