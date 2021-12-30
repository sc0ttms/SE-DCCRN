# -*- coding: utf-8 -*-

import sys
import os
import argparse
import copy
import toml
import torch
import torch.quantization.quantize_fx as quantize_fx
from torch.fx import symbolic_trace
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from trainer.base_trainer import BaseTrainer
from dataset.dataset import DNS_Dataset
from module.dc_crn import DCCRN
from audio.utils import prepare_empty_path


class QuantizationTrainer(BaseTrainer):
    def __init__(self, config, model, train_iter, valid_iter, device="cpu"):
        super().__init__(config, model, train_iter, valid_iter, device=device)
        # reconfig path
        self.checkpoints_path = os.path.join(self.base_path, "checkpoints", "quantization")
        self.logs_path = os.path.join(self.base_path, "logs", "train", "quantization")
        # mkdir path
        prepare_empty_path([self.checkpoints_path, self.logs_path], self.resume)

        # set quant args
        self.qconfig_dict = {
            "": torch.quantization.get_default_qat_qconfig("qnnpack"),
            "object_type": [
                (torch.nn.LSTM, torch.quantization.default_dynamic_qconfig),
            ],
        }
        # set the qengine to control weight packing
        # torch.backends.quantized.engine = "qnnpack"

    def prepare_qat(self):
        # get graph module
        gm = symbolic_trace(self.model)
        model_to_quantize = copy.deepcopy(gm)
        self.model = quantize_fx.prepare_qat_fx(model_to_quantize, self.qconfig_dict)

    def quant_fx(self):
        prepare_qat_model = copy.deepcopy(self.model)
        self.quantized_model = quantize_fx.convert_fx(prepare_qat_model)

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
            self.quant_fx()
            torch.save(self.quantized_model.state_dict(), os.path.join(self.checkpoints_path, "quantized_model.pth"))

    def __call__(self):
        # to device
        self.model.to(self.device)

        # init pre load model
        if self.pre_model_path:
            self.load_pre_model()

        # prepare_qat
        self.prepare_qat()

        # resume
        if self.resume:
            self.resume_checkpoint()

        # init logs
        self.init_logs()

        # loop train
        for epoch in range(self.start_epoch, self.epochs + 1):
            print(f"{'=' * 20} {epoch} epoch start {'=' * 20}")

            self.set_model_to_train_mode()
            self.train_epoch(epoch)

            if self.save_checkpoint_interval != 0 and (epoch % self.save_checkpoint_interval == 0):
                self.save_checkpoint(epoch)

            # valid
            if epoch % self.valid_interval == 0 and epoch >= self.valid_start_epoch:
                print(f"Train has finished, Valid is in progress...")

                self.set_model_to_eval_mode()
                metric_score = self.valid_epoch(epoch)

                if self.is_best_epoch(metric_score):
                    self.save_checkpoint(epoch, is_best_epoch=True)

            print(f"{'=' * 20} {epoch} epoch end {'=' * 20}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="knowledge distillation trainer")
    parser.add_argument("-C", "--config", required=True, type=str, help="Config (*.toml).")
    args = parser.parse_args()

    # config device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # get config
    config = toml.load(args.config)

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
    trainer = QuantizationTrainer(config, model, train_iter, valid_iter, device)

    # train
    trainer()

    pass
