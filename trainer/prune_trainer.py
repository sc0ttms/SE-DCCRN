# -*- coding: utf-8 -*-

import sys
import os
import copy
import toml
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from trainer.base_trainer import BaseTrainer
from dataset.dataset import DNS_Dataset
from module.dc_crn import DCCRN
from audio.utils import prepare_empty_path


class PruneTrainer(BaseTrainer):
    def __init__(self, config, model_path, model, train_iter, valid_iter, device="cpu"):
        super().__init__(config, model, train_iter, valid_iter, device=device)
        # get model path
        self.model_path = model_path
        # reconfig path
        self.checkpoints_path = os.path.join(self.base_path, "checkpoints", "prune")
        self.logs_path = os.path.join(self.base_path, "logs", "train", "prune")
        # mkdir path
        prepare_empty_path([self.checkpoints_path, self.logs_path], self.resume)

        # prune
        self.uniform_l1_norm_prune()

    def uniform_l1_norm_prune(self):
        laod_model = torch.load(self.model_path, map_location="cpu")
        # backup model
        torch.save(laod_model, os.path.join(self.checkpoints_path, "load_model.pth"))

        # load model state dict
        self.model.load_state_dict(laod_model)

        for _, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                prune.ln_structured(module, name="weight", amount=0.2, n=1, dim=0)

    def permanent_prune(self):
        self.prune_model = copy.deepcopy(self.model)
        for _, module in self.prune_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                prune.remove(module, name="weight")

    def save_checkpoint(self, epoch, is_best_epoch=False):
        print(f"Saving {epoch} epoch checkpoint, {is_best_epoch} best checkpoint...")

        state_dict = {
            "epoch": epoch,
            "best_score": self.best_score,
            "model": self.model.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
        }

        checkpoint_name_prefix = "best" if is_best_epoch else "latest"
        torch.save(state_dict, os.path.join(self.checkpoints_path, checkpoint_name_prefix + "_checkpoint.tar"))
        torch.save(state_dict["model"], os.path.join(self.checkpoints_path, checkpoint_name_prefix + "_model.pth"))

        if is_best_epoch:
            self.permanent_prune()
            torch.save(self.prune_model.state_dict(), os.path.join(self.checkpoints_path, "prune_model.pth"))


if __name__ == "__main__":
    # config device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # get config
    toml_path = os.path.join(os.getcwd(), "config", "base_config.toml")
    config = toml.load(toml_path)

    # get dataset path
    dataset_path = os.path.join(os.getcwd(), "dataset_csv")

    # get saved model path
    base_path = config["path"]["base"]
    model_path = os.path.join(base_path, "checkpoints", "base", "best_model.pth")

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
    model = DCCRN(
        n_fft=config["dataset"]["n_fft"],
        rnn_layers=config["model"]["rnn_layers"],
        rnn_units=config["model"]["rnn_units"],
        kernel_num=config["model"]["kernel_num"],
        kernel_size=config["model"]["kernel_size"],
    )

    # trainer
    trainer = PruneTrainer(config, model_path, model, train_iter, valid_iter, device)

    # train
    trainer()

    pass
