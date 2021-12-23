# -*- coding: utf-8 -*-

import sys
import os
import toml
import torch
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from inferencer.base_inferencer import BaseInferencer
from module.dc_crn import DCCRN
from dataset.dataset import DNS_Dataset
from audio.utils import prepare_empty_path


class Quantizationferencer(BaseInferencer):
    def __init__(self, config, model_path, model, test_iter, device="cpu"):
        super().__init__(config, model_path, model, test_iter, test_iter, device=device)
        # init path
        self.output_path = os.path.join(self.base_path, "enhanced", "quantization")
        self.logs_path = os.path.join(self.base_path, "logs", "inference", "quantization")
        self.metrics_path = os.path.join(self.base_path, "metrics", "quantization")
        # mkdir path
        prepare_empty_path([self.output_path, self.logs_path, self.metrics_path])


if __name__ == "__main__":
    # config device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # get config
    toml_path = os.path.join(os.getcwd(), "config", "base_config.toml")
    config = toml.load(toml_path)

    # get saved model path
    base_path = config["path"]["base"]
    model_path = os.path.join(base_path, "checkpoints", "quantization", "quantized_model.pth")

    # get dataset path
    dataset_path = os.path.join(os.getcwd(), "dataset_csv")

    # get dataloader args
    batch_size = config["dataloader"]["batch_size"]
    num_workers = 0 if device == "cpu" else config["dataloader"]["num_workers"]
    drop_last = config["dataloader"]["drop_last"]
    pin_memory = config["dataloader"]["pin_memory"]

    # get test_iter
    test_set = DNS_Dataset(dataset_path, config, mode="test")
    test_iter = DataLoader(
        test_set,
        batch_size=batch_size[1],
        shuffle=False,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )

    # config model
    model = DCCRN(
        n_fft=config["dataset"]["n_fft"],
        rnn_layers=config["model"]["rnn_layers"],
        rnn_units=config["model"]["rnn_units"],
        kernel_num=config["model"]["kernel_num"],
        kernel_size=config["model"]["kernel_size"],
    )

    # inferencer
    inference = Quantizationferencer(config, model_path, model, test_iter, device)

    # inference
    inference()
    pass
