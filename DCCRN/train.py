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
        self.model = model.to(device)
        # set device
        self.device = device

        # get meta args
        self.use_quant = config["meta"]["use_quant"]
        self.use_amp = False if self.device == "cpu" or self.use_quant else config["meta"]["use_amp"]

        # set path
        base_path = config["path"]["base"]
        os.makedirs(base_path, exist_ok=True)
        # get checkpoints path
        self.checkpoints_path = os.path.join(base_path, "checkpoints")
        # get logs path
        self.logs_path = os.path.join(base_path, "logs", "qunat_train" if self.use_quant else "train")

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
        model_path = os.path.join(self.checkpoints_path, "best_model.tar")
        assert os.path.exists(model_path)
        checkpoint = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])
        # get graph module
        gm = symbolic_trace(self.model)
        model_to_quantize = copy.deepcopy(gm)
        self.prepare_model = quantize_fx.prepare_qat_fx(model_to_quantize, self.qconfig_dict)

    def quant_fx(self):
        best_prepare_qat_model = copy.deepcopy(self.prepare_model)
        self.quantized_model = quantize_fx.convert_fx(best_prepare_qat_model)

    def save_checkpoint(self, epoch, is_best_epoch=False):
        print(f"Saving {epoch} epoch model checkpoint...")

        state_dict = {
            "epoch": epoch,
            "best_score": self.best_score,
            "model": self.model.state_dict() if self.use_quant == False else self.prepare_model.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
        }

        # save latest_model.tar or latest_prepare_qat_model.tar
        checkpoint_name = "latest_prepare_qat_model.tar" if self.use_quant == True else "latest_model.tar"
        torch.save(state_dict, os.path.join(self.checkpoints_path, checkpoint_name))

        # save best_model.tar
        if is_best_epoch:
            best_checkpoint_name = "best_prepare_qat_model.tar" if self.use_quant == True else "best_model.tar"
            torch.save(state_dict, os.path.join(self.checkpoints_path, best_checkpoint_name))

            if self.use_quant:
                self.quant_fx()
                torch.save(self.quantized_model, os.path.join(self.checkpoints_path, "quantized_model.pth"))

    def resume_checkpoint(self):
        checkpoint_name = "latest_prepare_qat_model.tar" if self.use_quant == True else "latest_model.tar"
        model_path = os.path.join(self.checkpoints_path, checkpoint_name)

        assert os.path.exists(model_path)

        checkpoint = torch.load(model_path, map_location="cpu")

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_score = checkpoint["best_score"]
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.use_quant:
            self.prepare_model.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint["model"])
        self.scaler.load_state_dict(checkpoint["scaler"])

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
        if self.use_quant:
            self.prepare_model.train()
        else:
            self.model.train()

    def set_model_to_eval_mode(self):
        if self.use_quant:
            self.prepare_model.eval()
        else:
            self.model.eval()

    def train_epoch(self, epoch):
        loss_total = 0.0
        for noisy, clean in tqdm(self.train_iter, desc="train"):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            self.optimizer.zero_grad()
            with autocast(enabled=self.use_amp):
                enh = self.model(noisy) if self.use_quant == False else self.prepare_model(noisy)
                loss = self.model.loss(enh, clean)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters() if self.use_quant == False else self.prepare_model.parameters(),
                self.clip_grad_norm_value,
            )
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

            enh = self.model(noisy) if self.use_quant == False else self.prepare_model(noisy)
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
