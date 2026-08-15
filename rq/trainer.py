import heapq
import logging
import math
import os
from time import time

import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from transformers import get_constant_schedule_with_warmup
from transformers import get_linear_schedule_with_warmup

from utils import delete_file, ensure_dir, get_local_time, set_color


class Trainer(object):
    def __init__(self, args, model, data_num):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()
        self.lr = args.lr
        self.learner = args.learner
        self.lr_scheduler_type = args.lr_scheduler_type
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.warmup_steps = args.warmup_epochs * data_num
        self.max_steps = args.epochs * data_num
        self.save_limit = args.save_limit
        self.best_save_heap = []
        self.newest_save_queue = []
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = torch.device(args.device)
        self.ckpt_dir = os.path.join(args.ckpt_dir, get_local_time())
        ensure_dir(self.ckpt_dir)

        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.latest_usage_metrics = []
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"
        self.optimizer = self._build_optimizer()
        self.scheduler = self._get_scheduler()
        self.model = self.model.to(self.device)

    def _build_optimizer(self):
        params = self.model.parameters()
        learner = self.learner.lower()
        if learner == "adam":
            return optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        if learner == "sgd":
            return optim.SGD(params, lr=self.lr, weight_decay=self.weight_decay)
        if learner == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=self.lr, weight_decay=self.weight_decay
            )
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(self.device)
            return optimizer
        if learner == "rmsprop":
            return optim.RMSprop(params, lr=self.lr, weight_decay=self.weight_decay)
        if learner == "adamw":
            return optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        self.logger.warning("Unrecognized optimizer; falling back to Adam")
        return optim.Adam(params, lr=self.lr)

    def _get_scheduler(self):
        if self.lr_scheduler_type.lower() == "linear":
            return get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.max_steps,
            )
        return get_constant_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.warmup_steps,
        )

    @staticmethod
    def _check_nan(loss):
        if not torch.isfinite(loss):
            raise ValueError("Training loss is nan or inf")

    def _train_epoch(self, train_data, epoch_idx):
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_balance_loss = 0.0
        total_resets = 0
        iter_data = tqdm(
            train_data,
            total=len(train_data),
            ncols=100,
            desc=set_color(f"Train {epoch_idx}", "pink"),
        )

        for data in iter_data:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out, rq_loss, _ = self.model(data)
            loss, loss_recon = self.model.compute_loss(out, rq_loss, xs=data)
            self._check_nan(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            total_balance_loss += self.model.rq.get_balance_loss().detach().item()
            total_resets += sum(
                metric["last_reset_count"]
                for metric in self.model.get_codebook_metrics()
            )

        denominator = max(len(train_data), 1)
        return (
            total_loss / denominator,
            total_recon_loss / denominator,
            total_balance_loss / denominator,
            total_resets,
        )

    @staticmethod
    def _usage_metrics(level_counts):
        metrics = []
        for counts in level_counts:
            total = counts.sum()
            probability = counts / max(total, 1)
            nonzero = probability[probability > 0]
            entropy = float(-(nonzero * np.log(nonzero)).sum())
            metrics.append(
                {
                    "used_codes": int(np.count_nonzero(counts)),
                    "codebook_size": int(len(counts)),
                    "utilization": float(np.count_nonzero(counts) / len(counts)),
                    "entropy": entropy,
                    "normalized_entropy": float(
                        entropy / math.log(len(counts)) if len(counts) > 1 else 1.0
                    ),
                    "perplexity": float(math.exp(entropy)),
                    "max_share": float(probability.max(initial=0.0)),
                }
            )
        return metrics

    @torch.no_grad()
    def _valid_epoch(self, valid_data):
        self.model.eval()
        indices_set = set()
        num_sample = 0
        level_counts = [
            np.zeros(size, dtype=np.int64) for size in self.model.rq.n_e_list
        ]
        iter_data = tqdm(
            valid_data,
            total=len(valid_data),
            ncols=100,
            desc=set_color("Evaluate", "pink"),
        )

        for data in iter_data:
            num_sample += len(data)
            indices = self.model.get_indices(data.to(self.device), use_sk=False)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            indices_set.update(map(tuple, indices.tolist()))
            for level, codebook_size in enumerate(self.model.rq.n_e_list):
                level_counts[level] += np.bincount(
                    indices[:, level], minlength=codebook_size
                )

        collision_rate = (num_sample - len(indices_set)) / max(num_sample, 1)
        return collision_rate, self._usage_metrics(level_counts)

    def _save_checkpoint(self, epoch, collision_rate=1, ckpt_file=None):
        ckpt_path = (
            os.path.join(self.ckpt_dir, ckpt_file)
            if ckpt_file
            else os.path.join(
                self.ckpt_dir,
                "epoch_%d_collision_%.4f_model.pth" % (epoch, collision_rate),
            )
        )
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "usage_metrics": self.latest_usage_metrics,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)
        self.logger.info(set_color("Saving current", "blue") + f": {ckpt_path}")
        return ckpt_path

    def _log_train_metrics(
        self, epoch_idx, elapsed, loss, recon_loss, balance_loss, reset_count
    ):
        self.logger.info(
            "epoch %d training [time: %.2fs, loss: %.4f, recon: %.4f, "
            "balance_kl: %.4f, revived_codes: %d]",
            epoch_idx,
            elapsed,
            loss,
            recon_loss,
            balance_loss,
            reset_count,
        )

    def _log_usage_metrics(self, epoch_idx, collision_rate, usage_metrics):
        summaries = []
        for level, metric in enumerate(usage_metrics):
            summaries.append(
                "L%d util=%.3f perplexity=%.1f norm_entropy=%.3f max_share=%.3f"
                % (
                    level + 1,
                    metric["utilization"],
                    metric["perplexity"],
                    metric["normalized_entropy"],
                    metric["max_share"],
                )
            )
        self.logger.info(
            "epoch %d evaluating [collision_rate: %.6f, %s]",
            epoch_idx,
            collision_rate,
            "; ".join(summaries),
        )

    def fit(self, data):
        for epoch_idx in range(self.epochs):
            training_start_time = time()
            train_loss, train_recon_loss, balance_loss, reset_count = (
                self._train_epoch(data, epoch_idx)
            )
            self._log_train_metrics(
                epoch_idx,
                time() - training_start_time,
                train_loss,
                train_recon_loss,
                balance_loss,
                reset_count,
            )

            if (epoch_idx + 1) % self.eval_step != 0:
                continue

            collision_rate, usage_metrics = self._valid_epoch(data)
            self.latest_usage_metrics = usage_metrics
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                self._save_checkpoint(epoch_idx, ckpt_file=self.best_loss_ckpt)
            if collision_rate < self.best_collision_rate:
                self.best_collision_rate = collision_rate
                self._save_checkpoint(
                    epoch_idx,
                    collision_rate=collision_rate,
                    ckpt_file=self.best_collision_ckpt,
                )

            self._log_usage_metrics(epoch_idx, collision_rate, usage_metrics)
            ckpt_path = self._save_checkpoint(
                epoch_idx, collision_rate=collision_rate
            )
            now_save = (-collision_rate, ckpt_path)
            if len(self.newest_save_queue) < self.save_limit:
                self.newest_save_queue.append(now_save)
                heapq.heappush(self.best_save_heap, now_save)
            else:
                old_save = self.newest_save_queue.pop(0)
                self.newest_save_queue.append(now_save)
                if collision_rate < -self.best_save_heap[0][0]:
                    bad_save = heapq.heappop(self.best_save_heap)
                    heapq.heappush(self.best_save_heap, now_save)
                    if bad_save not in self.newest_save_queue:
                        delete_file(bad_save[1])
                if old_save not in self.best_save_heap:
                    delete_file(old_save[1])

        return self.best_loss, self.best_collision_rate
