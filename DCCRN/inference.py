# -*- coding: utf-8 -*-

import sys
import os
import toml
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import paddle
from paddle.io import DataLoader
from visualdl import LogWriter

sys.path.append("./")
from DCCRN.model import DCCRN
from dataset.dataset import DNS_Dataset
from audio.feature import is_clipped
from audio.metrics import STOI, WB_PESQ
from audio.utils import prepare_empty_path

plt.switch_backend("agg")


class Inferencer:
    def __init__(self, model, test_iter, config):
        # set path
        base_path = os.path.abspath(config["path"]["base"])
        os.makedirs(base_path, exist_ok=True)
        # get checkpoints path
        self.checkpoints_path = os.path.join(base_path, "checkpoints")
        # get output path
        self.output_path = os.path.join(base_path, "enhanced")
        # get logs path
        self.logs_path = os.path.join(base_path, "logs", "inference")
        prepare_empty_path([self.output_path, self.logs_path])

        # set iter
        self.test_iter = test_iter

        # get model
        self.model = model
        self.load_checkpoint()

        # get dataset args
        self.sr = config["dataset"]["sr"]
        self.n_fft = config["dataset"]["n_fft"]
        self.win_len = config["dataset"]["win_len"]
        self.hop_len = config["dataset"]["hop_len"]
        self.window = paddle.to_tensor(np.hanning(self.win_len), dtype=paddle.float32)

        # get inference args
        self.n_folds = config["inference"]["n_folds"]
        self.n_jobs = config["inference"]["n_jobs"]
        self.audio_visual_samples = config["inference"]["audio_visual_samples"]

        # config logs
        self.writer = LogWriter(logdir=self.logs_path, max_queue=5, flush_secs=60)
        self.writer_text_enh_clipped_step = 1
        self.writer.add_text(
            tag="config",
            text_string=f"<pre \n{toml.dumps(config)} \n</pre>",
            step=1,
        )

    def load_checkpoint(self):
        best_model_path = os.path.join(self.checkpoints_path, "best_model.tar")
        assert os.path.exists(best_model_path)

        checkpoint = paddle.load(best_model_path)

        self.epoch = checkpoint["epoch"]
        self.model.set_state_dict(checkpoint["model"])

        print(f"Loading model checkpoint (epoch == {self.epoch})...")

    def check_clipped(self, enh, enh_file):
        if is_clipped(enh):
            self.writer.add_text(
                tag="enh_clipped",
                text_string=enh_file,
                step=self.writer_text_enh_clipped_step,
            )
        self.writer_text_enh_clipped_step += 1

    def audio_visualization(self, noisy, clean, enh, name, epoch):
        self.writer.add_audio(f"audio/noisy/{name}", noisy, epoch, sample_rate=self.sr)
        self.writer.add_audio(f"audio/clean/{name}", clean, epoch, sample_rate=self.sr)
        self.writer.add_audio(f"audio/enh/{name}", enh, epoch, sample_rate=self.sr)

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

        self.writer.add_scalar("STOI/test/noisy", np.mean(score["noisy"]["STOI"]), epoch)
        self.writer.add_scalar("STOI/test/enh", np.mean(score["enh"]["STOI"]), epoch)
        self.writer.add_scalar("WB_PESQ/test/noisy", np.mean(score["noisy"]["WB_PESQ"]), epoch)
        self.writer.add_scalar("WB_PESQ/test/enh", np.mean(score["enh"]["WB_PESQ"]), epoch)

    def save_audio(self, audio_list, audio_files, n_folds=1, n_jobs=8):
        split_num = len(audio_list) // n_folds
        for n in range(n_folds):
            Parallel(n_jobs=n_jobs)(
                delayed(sf.write)(audio_file, audio, samplerate=self.sr)
                for audio_file, audio in tqdm(
                    zip(
                        audio_files[n * split_num : (n + 1) * split_num],
                        audio_list[n * split_num : (n + 1) * split_num],
                    )
                )
            )

    def set_model_to_eval_mode(self):
        self.model.eval()

    def __call__(self):
        self.set_model_to_eval_mode()

        noisy_list = []
        clean_list = []
        noisy_files = []
        enh_list = []
        enh_files = []
        for noisy, clean, noisy_file in tqdm(self.test_iter, desc="test"):
            with paddle.no_grad():
                enh = self.model(noisy)

            noisy = noisy.detach().squeeze(0).cpu().numpy()
            clean = clean.detach().squeeze(0).cpu().numpy()
            enh = enh.detach().squeeze(0).cpu().numpy()
            assert len(noisy) == len(clean) == len(enh)

            for i in range(len(noisy_file)):
                enh_file = os.path.join(self.output_path, os.path.basename(noisy_file[i]).replace("noisy", "enh_noisy"))
                self.check_clipped(enh[i], enh_file)
                enh_files.append(enh_file)

            noisy_list = np.concatenate([noisy_list, noisy], axis=0) if len(noisy_list) else noisy
            clean_list = np.concatenate([clean_list, clean], axis=0) if len(clean_list) else clean
            enh_list = np.concatenate([enh_list, enh], axis=0) if len(enh_list) else enh
            noisy_files = np.concatenate([noisy_files, noisy_file], axis=0) if len(noisy_files) else noisy_file

        # visual audio
        for i in range(self.audio_visual_samples):
            self.audio_visualization(noisy_list[i], clean_list[i], enh_list[i], os.path.basename(noisy_files[i]), 1)

        # visual metrics and get valid score
        self.metrics_visualization(noisy_list, clean_list, enh_list, 1, n_folds=self.n_folds, n_jobs=self.n_jobs)
        # save audio
        self.save_audio(enh_list, enh_files, n_folds=self.n_folds, n_jobs=self.n_jobs)


if __name__ == "__main__":
    # config device
    device = paddle.get_device()
    paddle.set_device(device)
    print(f"device {device}")

    # get config
    toml_path = os.path.join(os.path.dirname(__file__), "config.toml")
    config = toml.load(toml_path)

    # get dataset path
    dataset_path = os.path.join(os.getcwd(), "dataset_csv")

    # get dataloader args
    batch_size = config["dataloader"]["batch_size"]
    num_workers = 0 if device == "cpu" else config["dataloader"]["num_workers"]
    drop_last = config["dataloader"]["drop_last"]

    # get test_iter
    test_set = DNS_Dataset(dataset_path, config, mode="test")
    test_iter = DataLoader(
        test_set,
        batch_size=batch_size[1],
        shuffle=False,
        num_workers=num_workers,
        drop_last=drop_last,
    )

    # config model
    model = DCCRN(config, mode="test")

    # inferencer
    inference = Inferencer(model, test_iter, config)

    # inference
    inference()
    pass
