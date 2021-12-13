# -*- coding: utf-8 -*-

import sys
import os
import toml
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import paddle
import paddle.nn as nn
from paddle.io import DataLoader
from paddle.amp import GradScaler, auto_cast
from visualdl import LogWriter


sys.path.append("./")
from dataset.dataset import DNS_Dataset
from DCCRN.model import DCCRN
from audio.metrics import STOI, WB_PESQ, transform_pesq_range
from audio.utils import prepare_empty_path

plt.switch_backend("agg")


class Trainer:
    def __init__(self, model, train_iter, valid_iter, config, device):
        # set path
        base_path = config["path"]["base"]
        os.makedirs(base_path, exist_ok=True)
        # get checkpoints path
        self.checkpoints_path = os.path.join(base_path, "checkpoints")
        # get logs path
        self.logs_path = os.path.join(base_path, "logs", "train")

        # get dataloader args
        self.batch_size = config["dataloader"]["batch_size"]

        # get dataset args
        self.sr = config["dataset"]["sr"]
        self.audio_len = config["dataset"]["audio_len"]

        # get train args
        self.use_amp = False if device == "cpu" else config["train"]["use_amp"]
        self.resume = config["train"]["resume"]
        self.n_folds = config["train"]["n_folds"]
        self.n_jobs = config["train"]["n_jobs"]
        self.epochs = config["train"]["epochs"]
        self.save_checkpoint_interval = config["train"]["save_checkpoint_interval"]
        self.valid_interval = config["train"]["valid_interval"]
        self.audio_visual_samples = config["train"]["audio_visual_samples"]

        # init common args
        self.start_epoch = 1
        self.best_score = 0.0

        # amp
        self.scaler = GradScaler(enable=self.use_amp)

        # set model
        self.model = model

        # set iter
        self.train_iter = train_iter
        self.valid_iter = valid_iter

        # config optimizer
        self.scheduler = paddle.optimizer.lr.ReduceOnPlateau(
            learning_rate=config["train"]["lr"], factor=0.5, patience=0, verbose=False
        )
        self.optimizer = getattr(paddle.optimizer, config["train"]["optimizer"])(
            parameters=model.parameters(),
            learning_rate=self.scheduler,
            grad_clip=nn.ClipGradByNorm(clip_norm=config["train"]["clip_grad_norm_value"]),
        )

        # mkdir path
        prepare_empty_path([self.checkpoints_path, self.logs_path], self.resume)

        # resume
        if self.resume:
            self.resume_checkpoint()

        # config logs
        self.writer = LogWriter(
            logdir=os.path.join(self.logs_path, f"start_epoch_{self.start_epoch}"), max_queue=5, flush_secs=60
        )
        self.writer.add_text(
            tag="config",
            text_string=f"<pre \n{toml.dumps(config)} \n</pre>",
            step=1,
        )

        # print params
        self.print_networks()

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

        # save latest_model.tar
        paddle.save(state_dict, os.path.join(self.checkpoints_path, "latest_model.tar"))

        # save best_model.tar
        if is_best_epoch:
            paddle.save(state_dict, os.path.join(self.checkpoints_path, "best_model.tar"))

    def resume_checkpoint(self):
        latest_model_path = os.path.join(self.checkpoints_path, "latest_model.tar")
        assert os.path.exists(latest_model_path)

        checkpoint = paddle.load(latest_model_path)

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_score = checkpoint["best_score"]
        self.scheduler.set_state_dict(checkpoint["scheduler"])
        self.optimizer.set_state_dict(checkpoint["optimizer"])
        self.model.set_state_dict(checkpoint["model"])
        self.scaler.load_state_dict(checkpoint["scaler"])

        print(f"Model checkpoint loaded. Training will begin at {self.start_epoch} epoch.")

    def print_networks(self):
        input_size = (self.batch_size[0], int(self.sr * self.audio_len))
        print(paddle.summary(self.model, input_size=input_size))

    def is_best_epoch(self, score):
        if score > self.best_score:
            self.best_score = score
            return True
        else:
            return False

    def audio_visualization(self, noisy, clean, enh, name, epoch):
        self.writer.add_audio(f"audio/{name}_noisy", noisy, epoch, sample_rate=self.sr)
        self.writer.add_audio(f"audio/{name}_clean", clean, epoch, sample_rate=self.sr)
        self.writer.add_audio(f"audio/{name}_enh", enh, epoch, sample_rate=self.sr)

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

    def metrics_visualization(self, noisy_list, clean_list, enh_list, epoch, n_folds=1, n_jobs=8):
        score = {
            "noisy": {
                "STOI": [],
                "WB_PESQ": [],
            },
            "enh": {
                "STOI": [],
                "WB_PESQ": [],
            },
        }

        split_num = len(noisy_list) // n_folds
        for n in range(n_folds):
            noisy_stoi_score = Parallel(n_jobs=n_jobs)(
                delayed(STOI)(noisy, clean)
                for noisy, clean in tqdm(
                    zip(
                        noisy_list[n * split_num : (n + 1) * split_num], clean_list[n * split_num : (n + 1) * split_num]
                    )
                )
            )
            enh_stoi_score = Parallel(n_jobs=n_jobs)(
                delayed(STOI)(noisy, clean)
                for noisy, clean in tqdm(
                    zip(enh_list[n * split_num : (n + 1) * split_num], clean_list[n * split_num : (n + 1) * split_num])
                )
            )
            score["noisy"]["STOI"].append(np.mean(noisy_stoi_score))
            score["enh"]["STOI"].append(np.mean(enh_stoi_score))

            noisy_wb_pesq_score = Parallel(n_jobs=n_jobs)(
                delayed(WB_PESQ)(noisy, clean)
                for noisy, clean in tqdm(
                    zip(
                        noisy_list[n * split_num : (n + 1) * split_num], clean_list[n * split_num : (n + 1) * split_num]
                    )
                )
            )
            enh_wb_pesq_score = Parallel(n_jobs=n_jobs)(
                delayed(WB_PESQ)(noisy, clean)
                for noisy, clean in tqdm(
                    zip(enh_list[n * split_num : (n + 1) * split_num], clean_list[n * split_num : (n + 1) * split_num])
                )
            )
            score["noisy"]["WB_PESQ"].append(np.mean(noisy_wb_pesq_score))
            score["enh"]["WB_PESQ"].append(np.mean(enh_wb_pesq_score))

        self.writer.add_scalar("STOI/valid/noisy", np.mean(score["noisy"]["STOI"]), epoch)
        self.writer.add_scalar("STOI/valid/enh", np.mean(score["enh"]["STOI"]), epoch)
        self.writer.add_scalar("WB_PESQ/valid/noisy", np.mean(score["noisy"]["WB_PESQ"]), epoch)
        self.writer.add_scalar("WB_PESQ/valid/enh", np.mean(score["enh"]["WB_PESQ"]), epoch)

        return (np.mean(score["enh"]["STOI"]) + transform_pesq_range(np.mean(score["enh"]["WB_PESQ"]))) / 2

    def hparams_visualization(self):
        hparams_dict = {
            "lr": self.optimizer.get_lr(),
        }
        metrics_list = ["lr", "STOI/valid/noisy", "STOI/valid/enh", "WB_PESQ/valid/noisy", "WB_PESQ/valid/enh"]
        self.writer.add_hparams(
            hparams_dict=hparams_dict,
            metrics_list=metrics_list,
        )

    def set_model_to_train_mode(self):
        self.model.train()

    def set_model_to_eval_mode(self):
        self.model.eval()

    def train_epoch(self, epoch):
        loss_total = 0.0
        for noisy, clean in tqdm(self.train_iter, desc="train"):
            self.optimizer.clear_grad()
            with auto_cast(enable=self.use_amp):
                enh = self.model(noisy)
                loss = self.model.loss(enh, clean)

            scaled = self.scaler.scale(loss)
            scaled.backward()
            self.scaler.minimize(self.optimizer, scaled)

            loss_total += loss.item()

        # logs
        self.writer.add_scalar("loss/train", loss_total / len(self.train_iter), epoch)
        self.writer.add_scalar("lr", self.optimizer.get_lr(), epoch)

    def valid_epoch(self, epoch):
        noisy_list = []
        clean_list = []
        enh_list = []
        noisy_files = []

        loss_total = 0.0
        for noisy, clean, noisy_file in tqdm(self.valid_iter, desc="valid"):
            with paddle.no_grad():
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
            noisy_list, clean_list, enh_list, epoch, n_folds=self.n_folds, n_jobs=self.n_jobs
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

            # logs hparams
            self.hparams_visualization()
            print(f"{'=' * 20} {epoch} epoch end {'=' * 20}")


if __name__ == "__main__":
    # config device
    device = paddle.get_device()
    paddle.set_device(device)
    print(f"device {device}")

    # get config
    toml_path = os.path.join(os.path.dirname(__file__), "config.toml")
    config = toml.load(toml_path)

    # get seed
    # seed = config["random"]["seed"]
    # np.random.seed(seed)

    # get dataset path
    dataset_path = os.path.join(os.getcwd(), "dataset_csv")

    # get dataloader args
    batch_size = config["dataloader"]["batch_size"]
    num_workers = 0 if device == "cpu" else config["dataloader"]["num_workers"]
    drop_last = config["dataloader"]["drop_last"]

    # get train_iter
    train_set = DNS_Dataset(dataset_path, config, mode="train")
    train_iter = DataLoader(
        train_set,
        batch_size=batch_size[0],
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
    )

    # get valid_iter
    valid_set = DNS_Dataset(dataset_path, config, mode="valid")
    valid_iter = DataLoader(
        valid_set,
        batch_size=batch_size[1],
        shuffle=False,
        num_workers=num_workers,
        drop_last=drop_last,
    )

    # config model
    model = DCCRN(config, mode="train")

    # trainer
    trainer = Trainer(model, train_iter, valid_iter, config, device)

    # train
    trainer.train()

    pass
