# -*- coding: utf-8 -*-

import sys
import os
import toml
import copy
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.quantization.quantize_fx as quantize_fx
from torch.fx import symbolic_trace
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter


sys.path.append("./")
from dataset.dataset import DNS_Dataset
from dataset.compute_metrics import compute_metric
from DCCRN.model import DCCRN
from audio.metrics import transform_pesq_range
from audio.utils import prepare_empty_path

plt.switch_backend("agg")


class Trainer:
    def __init__(self, model, train_iter, valid_iter, config, device):
        # set model
        self.model = model
        # set device
        self.device = device

        # get meta args
        self.use_quant = config["meta"]["use_quant"]
        self.use_prune = config["meta"]["use_prune"]
        self.iter_prune = config["meta"]["iter_prune"]
        self.use_kd = config["meta"]["use_kd"]
        self.use_amp = False if self.device == "cpu" or self.use_quant else config["meta"]["use_amp"]

        # set path
        base_path = config["path"]["base"]
        os.makedirs(base_path, exist_ok=True)
        # get checkpoints path
        self.checkpoints_path = os.path.join(base_path, "checkpoints", "normal")
        # get logs path
        self.logs_path = os.path.join(base_path, "logs", "train", "normal")

        # set quant path
        if self.use_quant:
            self.checkpoints_path = os.path.join(base_path, "checkpoints", "quant")
            self.logs_path = os.path.join(base_path, "logs", "train", "quant")

        # set prune path
        if self.use_prune:
            self.checkpoints_path = os.path.join(base_path, "checkpoints", "prune")
            self.logs_path = os.path.join(base_path, "logs", "train", "prune")

        # set kd path and model
        if self.use_kd:
            self.teacher_model = copy.deepcopy(self.model)

            self.checkpoints_path = os.path.join(base_path, "checkpoints", "kd")
            self.logs_path = os.path.join(base_path, "logs", "train", "kd")

            kd_model_config = copy.deepcopy(config["kd"]["model"])
            config["model"] = kd_model_config
            self.model = DCCRN(config, mode="train", device=device)

            # get kd args
            self.alpha = config["kd"]["train"]["alpha"]

        # get dataset args
        self.sr = config["dataset"]["sr"]
        self.audio_len = config["dataset"]["audio_len"]

        # get train args
        self.resume = config["train"]["resume"]
        self.n_folds = config["train"]["n_folds"]
        self.n_jobs = config["train"]["n_jobs"]
        self.epochs = config["train"]["epochs"]
        self.save_checkpoint_interval = config["train"]["save_checkpoint_interval"]
        self.valid_interval = config["train"]["valid_interval"]
        self.clip_grad_norm_value = config["train"]["clip_grad_norm_value"]
        self.audio_visual_samples = config["train"]["audio_visual_samples"]

        # set quant args
        self.qconfig_dict = {
            "": torch.quantization.get_default_qat_qconfig("qnnpack"),
            "object_type": [
                (torch.nn.LSTM, torch.quantization.default_dynamic_qconfig),
            ],
        }
        # set the qengine to control weight packing
        # torch.backends.quantized.engine = "qnnpack"

        # init common args
        self.start_epoch = 1
        self.best_score = 0.0

        # amp
        self.scaler = GradScaler(enabled=self.use_amp)

        # set iter
        self.train_iter = train_iter
        self.valid_iter = valid_iter

        # config optimizer
        self.optimizer = getattr(torch.optim, config["train"]["optimizer"])(
            params=model.parameters(),
            lr=config["train"]["lr"],
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=int(10 // self.valid_interval - 1),
            verbose=True,
        )

        # mkdir path
        prepare_empty_path([self.checkpoints_path, self.logs_path], self.resume)

        # quant
        if self.use_quant:
            self.prepare_qat()

        # prune
        if self.use_prune:
            self.uniform_l1_norm_prune(is_iter_prune=self.iter_prune)

        if self.use_kd:
            self.load_teacher_checkpoint()

        # resume
        if self.resume:
            self.resume_checkpoint()

        # config logs
        self.writer = SummaryWriter(
            log_dir=os.path.join(self.logs_path, f"start_epoch_{self.start_epoch}"),
            max_queue=5,
            flush_secs=60,
        )
        self.writer.add_text(
            tag="config",
            text_string=f"<pre>  \n{toml.dumps(config)}  \n</pre>",
            global_step=1,
        )

        # print params
        self.model.print_networks()

    def prepare_qat(self):
        # load best model
        model_path = os.path.join(self.checkpoints_path, "..", "normal", "best_model.tar")
        assert os.path.exists(model_path)
        checkpoint = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])
        # get graph module
        gm = symbolic_trace(self.model)
        model_to_quantize = copy.deepcopy(gm)
        self.model = quantize_fx.prepare_qat_fx(model_to_quantize, self.qconfig_dict)

    def quant_fx(self):
        best_prepare_qat_model = copy.deepcopy(self.model)
        self.quantized_model = quantize_fx.convert_fx(best_prepare_qat_model)

    def uniform_l1_norm_prune(self, is_iter_prune=False):
        # load best model
        if is_iter_prune:
            model_path = os.path.join(self.checkpoints_path, "prune_model.pth")
            best_model_path = os.path.join(self.checkpoints_path, "best_model.tar")
        else:
            model_path = os.path.join(self.checkpoints_path, "..", "normal", "best_model.tar")
        assert os.path.exists(model_path)
        checkpoint = torch.load(model_path, map_location="cpu")
        if is_iter_prune:
            self.model.load_state_dict(checkpoint)
            self.best_score = torch.load(best_model_path, map_location="cpu")["best_score"]
        else:
            self.model.load_state_dict(checkpoint["model"])

        for _, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                prune.ln_structured(module, name="weight", amount=0.2, n=1, dim=0)

    def permanent_prune(self):
        self.prune_model = copy.deepcopy(self.model)
        for _, module in self.prune_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                prune.remove(module, name="weight")

    def load_teacher_checkpoint(self):
        model_path = os.path.join(self.checkpoints_path, "..", "normal", "best_model.tar")
        assert os.path.exists(model_path)
        checkpoint = torch.load(model_path, map_location="cpu")

        self.teacher_model.load_state_dict(checkpoint["model"])

        print(f"Load teacher model done...")

    def save_checkpoint(self, epoch, is_best_epoch=False):
        print(f"Saving {epoch} epoch model checkpoint...")

        state_dict = {
            "epoch": epoch,
            "best_score": self.best_score,
            "model": self.model.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
        }

        # save latest_model.tar or latest_prepare_qat_model.tar
        torch.save(state_dict, os.path.join(self.checkpoints_path, "latest_model.tar"))

        # save best_model.tar
        if is_best_epoch:
            torch.save(state_dict, os.path.join(self.checkpoints_path, "best_model.tar"))
            torch.save(state_dict["model"], os.path.join(self.checkpoints_path, "best_model.pth"))

            if self.use_quant:
                self.quant_fx()
                torch.save(
                    self.quantized_model.state_dict(),
                    os.path.join(
                        self.checkpoints_path,
                        "quantized_model.pth",
                    ),
                )

            if self.use_prune:
                self.permanent_prune()
                torch.save(
                    self.prune_model.state_dict(),
                    os.path.join(
                        self.checkpoints_path,
                        "prune_model.pth",
                    ),
                )

    def resume_checkpoint(self):
        model_path = os.path.join(self.checkpoints_path, "latest_model.tar")

        assert os.path.exists(model_path)

        checkpoint = torch.load(model_path, map_location="cpu")

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_score = checkpoint["best_score"]
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.model.load_state_dict(checkpoint["model"])
        self.scaler.load_state_dict(checkpoint["scaler"])

        print(f"use quant {self.use_quant}")
        print(f"use prune {self.use_prune}")
        print(f"Model checkpoint loaded. Training will begin at {self.start_epoch} epoch.")

    def is_best_epoch(self, score):
        if score > self.best_score:
            self.best_score = score
            return True
        else:
            return False

    def audio_visualization(self, noisy, clean, enh, name, epoch):
        self.writer.add_audio(f"audio/{name}/noisy", noisy, epoch, sample_rate=self.sr)
        self.writer.add_audio(f"audio/{name}/clean", clean, epoch, sample_rate=self.sr)
        self.writer.add_audio(f"audio/{name}/enh", enh, epoch, sample_rate=self.sr)

        # Visualize the spectrogram of noisy speech, clean speech, and enhanced speech
        noisy_mag, _ = librosa.magphase(librosa.stft(noisy, n_fft=320, hop_length=160, win_length=320))
        clean_mag, _ = librosa.magphase(librosa.stft(clean, n_fft=320, hop_length=160, win_length=320))
        enh_mag, _ = librosa.magphase(librosa.stft(enh, n_fft=320, hop_length=160, win_length=320))
        fig, axes = plt.subplots(3, 1, figsize=(6, 6))
        for k, mag in enumerate([noisy_mag, clean_mag, enh_mag]):
            axes[k].set_title(
                f"mean: {np.mean(mag):.3f}, "
                f"std: {np.std(mag):.3f}, "
                f"max: {np.max(mag):.3f}, "
                f"min: {np.min(mag):.3f}"
            )
            librosa.display.specshow(
                librosa.amplitude_to_db(mag, ref=np.max), cmap="magma", y_axis="linear", ax=axes[k], sr=16000
            )
        plt.tight_layout()
        self.writer.add_figure(f"spec/{name}", fig, epoch)

    def metrics_visualization(self, enh_list, clean_list, epoch, n_folds=1, n_jobs=8):
        # get metrics
        metrics = {
            "STOI": [],
            "WB_PESQ": [],
        }

        # compute enh metrics
        compute_metric(
            enh_list,
            clean_list,
            metrics,
            n_folds=n_folds,
            n_jobs=n_jobs,
            pre_load=True,
        )

        self.writer.add_scalar("STOI/valid", metrics["STOI"], epoch)
        self.writer.add_scalar("WB_PESQ/valid", metrics["WB_PESQ"], epoch)

        return ((metrics["STOI"]) + transform_pesq_range(metrics["WB_PESQ"])) / 2

    def set_model_to_train_mode(self):
        self.model.train()

    def set_model_to_eval_mode(self):
        self.model.eval()

    def set_teacher_model_to_train_mode(self):
        self.teacher_model.train()

    def set_teacher_model_to_eval_mode(self):
        self.teacher_model.eval()

    def train_epoch(self, epoch):
        loss_total = 0.0
        for noisy, clean in tqdm(self.train_iter, desc="train"):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            self.optimizer.zero_grad()
            if self.use_kd:
                with torch.no_grad():
                    teacher_enh = self.teacher_model(noisy)
                with autocast(enabled=self.use_amp):
                    enh = self.model(noisy)
                    loss_hard = self.model.loss(enh, clean)
                    loss_soft = self.model.loss(enh, teacher_enh)
                    loss = self.alpha * loss_hard + (1.0 - self.alpha) * loss_soft
            else:
                with autocast(enabled=self.use_amp):
                    enh = self.model(noisy)
                    loss = self.model.loss(enh, clean)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_total += loss.item()

        # logs
        self.writer.add_scalar("loss/train", loss_total / len(self.train_iter), epoch)
        self.writer.add_scalar("lr", self.optimizer.state_dict()["param_groups"][0]["lr"], epoch)

    @torch.no_grad()
    def valid_epoch(self, epoch):
        noisy_list = []
        clean_list = []
        enh_list = []
        noisy_files = []

        loss_total = 0.0
        for noisy, clean, noisy_file in tqdm(self.valid_iter, desc="valid"):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            enh = self.model(noisy)
            loss = self.model.loss(enh, clean)

            loss_total += loss.item()

            noisy = noisy.detach().squeeze(0).cpu().numpy()
            clean = clean.detach().squeeze(0).cpu().numpy()
            enh = enh.detach().squeeze(0).cpu().numpy()
            assert len(noisy) == len(clean) == len(enh)

            noisy_list = np.concatenate([noisy_list, noisy], axis=0) if len(noisy_list) else noisy
            clean_list = np.concatenate([clean_list, clean], axis=0) if len(clean_list) else clean
            enh_list = np.concatenate([enh_list, enh], axis=0) if len(enh_list) else enh
            noisy_files = np.concatenate([noisy_files, noisy_file], axis=0) if len(noisy_files) else noisy_file

        # update learning rate
        self.scheduler.step(loss_total / len(self.valid_iter))

        # visual audio
        for i in range(self.audio_visual_samples):
            self.audio_visualization(noisy_list[i], clean_list[i], enh_list[i], os.path.basename(noisy_files[i]), epoch)

        # logs
        self.writer.add_scalar("loss/valid", loss_total / len(self.valid_iter), epoch)

        # visual metrics and get valid score
        metrics_score = self.metrics_visualization(
            enh_list, clean_list, epoch, n_folds=self.n_folds, n_jobs=self.n_jobs
        )

        return metrics_score

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            print(f"{'=' * 20} {epoch} epoch start {'=' * 20}")

            # train
            if self.use_kd:
                self.set_teacher_model_to_eval_mode()
            self.set_model_to_train_mode()
            self.train_epoch(epoch)

            if self.save_checkpoint_interval != 0 and (epoch % self.save_checkpoint_interval == 0):
                self.save_checkpoint(epoch)

            # valid
            if epoch % self.valid_interval == 0:
                print(f"Train has finished, Valid is in progress...")

                self.set_model_to_eval_mode()
                metric_score = self.valid_epoch(epoch)

                if self.is_best_epoch(metric_score):
                    self.save_checkpoint(epoch, is_best_epoch=True)

            print(f"{'=' * 20} {epoch} epoch end {'=' * 20}")


if __name__ == "__main__":
    # config device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # get config
    toml_path = os.path.join(os.path.dirname(__file__), "config.toml")
    config = toml.load(toml_path)

    # get dataset path
    dataset_path = os.path.join(os.getcwd(), "dataset_csv")

    cudnn_enabled = False if device == "cpu" else config["meta"]["cudnn_enabled"]
    torch.backends.cudnn.enabled = cudnn_enabled
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # get dataloader args
    batch_size = config["dataloader"]["batch_size"]
    num_workers = 0 if device == "cpu" else config["dataloader"]["num_workers"]
    drop_last = config["dataloader"]["drop_last"]
    pin_memory = config["dataloader"]["pin_memory"]

    # get train_iter
    train_set = DNS_Dataset(dataset_path, config, mode="train")
    train_iter = DataLoader(
        train_set,
        batch_size=batch_size[0],
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )

    # get valid_iter
    valid_set = DNS_Dataset(dataset_path, config, mode="valid")
    valid_iter = DataLoader(
        valid_set,
        batch_size=batch_size[1],
        shuffle=False,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )

    # config model
    model = DCCRN(config, mode="train", device=device)

    # trainer
    trainer = Trainer(model, train_iter, valid_iter, config, device)

    # train
    trainer.train()

    pass
