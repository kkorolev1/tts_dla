import random
from pathlib import Path
from random import shuffle

import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import os

from hw_tts.base import BaseTrainer
from hw_tts.utils import inf_loop, MetricTracker
from hw_tts.text import text_to_sequence
from waveglow import get_wav, get_waveglow


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            metrics,
            optimizer,
            config,
            device,
            dataloaders,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True
    ):
        super().__init__(model, criterion, metrics, optimizer, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.lr_scheduler = lr_scheduler
        self.log_step = 50

        self.train_metrics = MetricTracker(
            "loss", "grad norm", *[m.name for m in self.metrics], writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "loss", *[m.name for m in self.metrics], writer=self.writer
        )
        self.fine_tune = config["trainer"].get("fine_tune", False)
        self.grad_accum_iters = config["trainer"].get("grad_accum_iters", 1)
        self.eval_start_iter = config["trainer"].get("eval_start_iter", 0)
        self.scheduler_config = config["trainer"].get("scheduler", {
            "requires_loss": False,
            "epoch_based": False
        })
        self.scheduler_config["requires_loss"] = self.scheduler_config.get("requires_loss", False)
        self.scheduler_config["epoch_based"] = self.scheduler_config.get("epoch_based", False)
        self.batch_expand_size = self.config["trainer"]["batch_expand_size"]
        self.waveglow = get_waveglow(self.config["trainer"]["waveglow_path"])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["arch"] != self.config["arch"]:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is different from that "
                "of checkpoint. This may yield an exception while state_dict is being loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
                checkpoint["config"]["optimizer"] != self.config["optimizer"] or
                (self.lr_scheduler is not None and checkpoint["config"]["lr_scheduler"] != self.config["lr_scheduler"])
        ):
            self.logger.warning(
                "Warning: Optimizer or lr_scheduler given in config file is different "
                "from that of checkpoint. Optimizer parameters not being resumed."
            )
        elif not self.fine_tune:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["src_seq", "mel_target", "length_target", "energy_target",
                 "mel_pos", "src_pos", "pitch_target"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )
    
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        progress_bar = tqdm(range(self.len_epoch), desc='train')

        for list_batch_idx, list_batch in enumerate(self.train_dataloader):
            stop = False
            for batch_idx, batch in enumerate(list_batch):
                progress_bar.update(1)
                try:
                    batch = self.process_batch(
                        batch,
                        batch_idx,
                        is_train=True,
                        metrics=self.train_metrics,
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e) and self.skip_oom:
                        self.logger.warning("OOM on batch. Skipping batch.")
                        for p in self.model.parameters():
                            if p.grad is not None:
                                del p.grad  # free some memory
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
                full_batch_idx = batch_idx + list_batch_idx * self.batch_expand_size
                if full_batch_idx % self.log_step == 0:
                    self.writer.set_step((epoch - 1) * self.len_epoch + full_batch_idx)
                    self.logger.debug(
                        "Train Epoch: {} {} Loss: {:.6f}".format(
                            epoch, self._progress(full_batch_idx), batch["loss"].item()
                        )
                    )
                    if self.lr_scheduler is not None:
                        self.writer.add_scalar(
                            "learning rate", self.lr_scheduler.get_last_lr()[0]
                        )
                    self._log_scalars(self.train_metrics)
                    # we don't want to reset train metrics at the start of every epoch
                    # because we are interested in recent train metrics
                    last_train_metrics = self.train_metrics.result()
                    self.train_metrics.reset()
                if full_batch_idx >= self.len_epoch:
                    stop = True
                    break
            if stop:
                break
        log = last_train_metrics
        self._evaluation_epoch()

        if self.lr_scheduler is not None and self.scheduler_config["epoch_based"]:
            if self.scheduler_config["requires_loss"]:
                if "val_loss" in log:
                    self.lr_scheduler.step(log["val_loss"])
            else:
                self.lr_scheduler.step()

        return log


    def process_batch(self, batch, batch_idx, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        outputs = self.model(**batch)
        batch.update(outputs)
        mel_loss, duration_loss, energy_loss, pitch_loss = self.criterion(**batch)
        batch["mel_loss"] = mel_loss
        batch["duration_loss"] = duration_loss
        batch["energy_loss"] = energy_loss
        batch["pitch_loss"] = pitch_loss
        batch["loss"] = (mel_loss + duration_loss + energy_loss + pitch_loss) / self.grad_accum_iters
        if is_train:
            batch["loss"].backward()
            if (batch_idx + 1) % self.grad_accum_iters == 0 or (batch_idx + 1) == self.len_epoch:
                self._clip_grad_norm()
                self.optimizer.step()
                if self.lr_scheduler is not None and not self.scheduler_config["epoch_based"]:
                    self.lr_scheduler.step()
                self.train_metrics.update("grad norm", self.get_grad_norm())
                self.optimizer.zero_grad()

        metrics.update("mel_loss", batch["mel_loss"].item())
        metrics.update("duration_loss", batch["duration_loss"].item())
        metrics.update("energy_loss", batch["energy_loss"].item())
        metrics.update("pitch_loss", batch["pitch_loss"].item())
        metrics.update("loss", batch["loss"].item())
        for met in self.metrics:
            metrics.update(met.name, met(**batch))
        return batch

    @torch.no_grad
    def _evaluation_epoch(self):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        
        texts = [
            "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
            "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
            "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space"
        ]

        text_cleaners = ["english_cleaners"]
        tokenized_texts = [text_to_sequence(text, text_cleaners) for text in texts]

        sampling_rate = 22050

        for i, tokenized_text in enumerate(tokenized_texts):
            src_seq = torch.tensor(tokenized_text, device=self.device).unsqueeze(0)
            src_pos = torch.tensor(
                [i + 1 for i in range(len(tokenized_text))], device=self.device).unsqueeze(0)
            outputs = self.model(src_seq=src_seq, src_pos=src_pos)
            wav = get_wav(outputs["mel_output"].transpose(
                1, 2).to("cuda"), self.waveglow, sampling_rate=sampling_rate).unsqueeze(0)
            self._log_audio(wav, sampling_rate, f"test_{i + 1}")

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    @torch.no_grad()
    def _log_predictions(
            self,
            mel_output,
            examples_to_log=5,
            **kwargs
    ):
        if self.writer is None:
            return

        mel_output = mel_output[:examples_to_log].transpose(1, 2).to("cuda")
        sr = 22050
        wavs = get_wav(mel_output, self.waveglow, sampling_rate=sr)

        for i, wav in enumerate(wavs):
            self._log_audio(wav, sr, str(i + 1))

    def _log_audio(self, audio, sr, name):
        self.writer.add_audio(f"Audio_{name}", audio, sample_rate=sr)

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]

        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
