import pathlib, sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # repo root, holds npu_support.py
import npu_support
import argparse
import random
import torch
import torch.nn as nn
import numpy as np
import os
import logging
from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.rqvae import RQVAE
from trainer import Trainer


class ResidualEncoderWrapper(nn.Module):

    def __init__(self, original_encoder):
        super().__init__()
        self.mlp = original_encoder

    def forward(self, x):
        return x + self.mlp(x)

def apply_rqkmeans_plus_strategy(model, codebook_path, device):
    logging.info(">>> [RQ-Kmeans+] Strategy: Applying Residual Connection & Warm-start...")

    if hasattr(model, 'encoder'):
        model.encoder = ResidualEncoderWrapper(model.encoder)
        model.encoder.to(device)
        logging.info("    [Structure] Encoder wrapped with Residual Connection (Z = X + MLP(X))")
    else:
        logging.error("    [Error] Could not find 'encoder' in model.")
        return model

    logging.info("    [Init] Applying Zero-Initialization to Encoder's last layer...")
    
    last_linear = None
    raw_mlp = model.encoder.mlp 
    
    if hasattr(raw_mlp, 'mlp_layers'):
        modules = list(raw_mlp.mlp_layers.modules())
    else:
        modules = list(raw_mlp.modules())

    for m in reversed(modules):
        if isinstance(m, nn.Linear):
            last_linear = m
            break
            
    if last_linear:
        with torch.no_grad():
            last_linear.weight.fill_(0.0)
            if last_linear.bias is not None:
                last_linear.bias.fill_(0.0)
        logging.info(f"    [Init] Zero-init applied to Linear layer: {last_linear}")
    else:
        logging.warning("    [Warning] Could not find last Linear layer to zero-init.")

    if not os.path.exists(codebook_path):
        raise FileNotFoundError(f"{codebook_path} not found")
        
    logging.info(f"    [Weights] Loading codebooks from {codebook_path}")
    npz_data = np.load(codebook_path)
    
    target_layers = None
    if hasattr(model, 'rq') and hasattr(model.rq, 'vq_layers'):
        target_layers = model.rq.vq_layers
    
    if target_layers:
        success_count = 0
        for i, layer in enumerate(target_layers):
            emb_layer = layer.embedding if hasattr(layer, 'embedding') else layer
            
            key = f'codebook_{i}'
            if key in npz_data:
                centroids = npz_data[key]
                with torch.no_grad():
                    emb_layer.weight.data.copy_(torch.from_numpy(centroids).to(device))
                success_count += 1
                logging.info(f"      -> Loaded Codebook Level {i}")
        
        if success_count == 0:
            logging.warning("      -> No codebooks loaded! Check .npz keys.")
    else:
        logging.error("    [Error] Could not locate VQ layers.")

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="RQ-KMeans+ Implementation")
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=5000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument("--data_path", type=str, default="../data/Games/Games.emb-llama-td.npy")
    parser.add_argument("--pretrained_codebook_path", type=str, required=True, 
                        help="Path to RQ-KMeans npz file")
    
    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256,256,256])
    parser.add_argument('--e_dim', type=int, default=2560, help='Must match input dim for RQ-KMeans+')
    parser.add_argument('--layers', type=int, nargs='+', default=[2048,1024,512,256,128,64], 
                        help='Hidden layers. Note: Last layer will be ignored/adjusted conceptually')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.25)
    
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--eval_step", type=int, default=50)
    parser.add_argument('--save_limit', type=int, default=5)
    parser.add_argument("--ckpt_dir", type=str, default="")
    parser.add_argument("--learner", type=str, default="AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument('--lr_scheduler_type', type=str, default="constant")
    parser.add_argument('--warmup_epochs', type=int, default=50)
    parser.add_argument("--dropout_prob", type=float, default=0.0)
    parser.add_argument("--bn", type=bool, default=False)
    parser.add_argument("--loss_type", type=str, default="mse")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0])
    parser.add_argument("--sk_iters", type=int, default=50)
    parser.add_argument("--kmeans_init", type=bool, default=False)
    parser.add_argument("--kmeans_iters", type=int, default=100)

    return parser.parse_args()

if __name__ == '__main__':
    # Seed setup
    seed = 2024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("=================================================")
    print("   RQ-KMeans+: Residual Initialization & Training")
    print("=================================================")

    data = EmbDataset(args.data_path)

    if args.e_dim != data.dim:
        logging.error(f"CRITICAL: For RQ-KMeans+ residual connection, 'e_dim' ({args.e_dim}) "
                      f"MUST equal input dimension ({data.dim}).")
        logging.error("Please set --e_dim 2560 (or your data dim).")
        exit(1)

    model = RQVAE(in_dim=data.dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers, 
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  beta=args.beta,
                  kmeans_init=False, 
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters,
                  )
    
    model = model.to(args.device)

    model = apply_rqkmeans_plus_strategy(model, args.pretrained_codebook_path, args.device)
    
    print(model) 

    data_loader = DataLoader(data, num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True,
                             pin_memory=True)
    
    logging.info("Starting RQ-KMeans+ Training...")
    trainer = Trainer(args, model, len(data_loader))
    best_loss, best_collision_rate = trainer.fit(data_loader)

    print(f"RQ-KMeans+ Final Result -> Loss: {best_loss}, Collision Rate: {best_collision_rate}")
