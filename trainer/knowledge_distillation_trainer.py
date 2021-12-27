# -*- coding: utf-8 -*-

import sys
import os
import argparse
import toml
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

sys.path.append(os.getcwd())
from trainer.base_trainer import BaseTrainer
from dataset.dataset import DNS_Dataset
from module.dc_crn import DCCRN
from module.ds_dc_crn import DSDCCRN
from audio.utils import prepare_empty_path


class KnowledgeDistillationTrainer(BaseTrainer):
    def __init__(self, config, teacher_model, model, train_iter, valid_iter, device="cpu"):
        super().__init__(config, model, train_iter, valid_iter, device=device)
        # get teacher model
        self.teacher_model = teacher_model
        self.teacher_model_path = config["path"]["teacher_model"]
        # get knowledge_distillation args
        self.kd_alpha = config["knowledge_distillation"]["alpha"]

        # reconfig path
        self.checkpoints_path = os.path.join(self.base_path, "checkpoints", "knowledge_distillation")
        self.logs_path = os.path.join(self.base_path, "logs", "train", "knowledge_distillation")
        # mkdir path
        prepare_empty_path([self.checkpoints_path, self.logs_path], self.resume)

        # to device
        self.teacher_model = self.teacher_model.to(self.device)
        # load teacher model checkpoint
        self.load_teacher_model()

        # set teacher model to eval
        self.teacher_model.eval()

    def load_teacher_model(self):
        assert os.path.exists(self.teacher_model_path)
        load_model = torch.load(self.teacher_model_path, map_location="cpu")

        self.teacher_model.load_state_dict(load_model)

        print(f"Load teacher model done...")

    def train_epoch(self, epoch):
        loss_total = 0.0
        for noisy, clean in tqdm(self.train_iter, desc="train"):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            # [B, S] -> [B, F, T, 2]
            noisy_spec = self.audio_stft(noisy)

            self.optimizer.zero_grad()
            with torch.no_grad():
                teacher_mask = self.teacher_model(noisy_spec)

            # [B, S]
            teacher_enh = self.audio_istft(teacher_mask, noisy_spec)

            with autocast(enabled=self.use_amp):
                mask = self.model(noisy_spec)

            # [B, S]
            enh = self.audio_istft(mask, noisy_spec)

            loss_hard = self.loss(enh, clean)
            loss_soft = self.loss(enh, teacher_enh)
            loss = self.kd_alpha * loss_hard + (1.0 - self.kd_alpha) * loss_soft
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_total += loss.item()

        # logs
        self.writer.add_scalar("loss/train", loss_total / len(self.train_iter), epoch)
        self.writer.add_scalar("lr", self.optimizer.state_dict()["param_groups"][0]["lr"], epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="knowledge distillation trainer")
    parser.add_argument("-TC", "--teacher_config", required=True, type=str, help="Teacher Config (*.toml).")
    parser.add_argument("-SC", "--student_config", required=True, type=str, help="Student Config (*.toml).")
    args = parser.parse_args()

    # config device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # get teacher config
    config = toml.load(args.teacher_config)

    # config teacher model
    teacher_model = globals().get(config["model"]["name"])(
        n_fft=config["dataset"]["n_fft"],
        rnn_layers=config["model"]["rnn_layers"],
        rnn_units=config["model"]["rnn_units"],
        kernel_num=config["model"]["kernel_num"],
        kernel_size=config["model"]["kernel_size"],
    )

    # get config
    config = toml.load(args.student_config)

    # get dataset path
    dataset_path = os.path.join(os.getcwd(), "dataset_csv")

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

    # config teacher model
    model = globals().get(config["model"]["name"])(
        n_fft=config["dataset"]["n_fft"],
        rnn_layers=config["model"]["rnn_layers"],
        rnn_units=config["model"]["rnn_units"],
        kernel_num=config["model"]["kernel_num"],
        kernel_size=config["model"]["kernel_size"],
    )

    # trainer
    trainer = KnowledgeDistillationTrainer(config, teacher_model, model, train_iter, valid_iter, device)

    # train
    trainer()

    pass
