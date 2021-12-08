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
        # get checkpoints path
        self.checkpoints_path = os.path.join(os.path.dirname(__file__), "checkpoints")
        # get output path
        self.output_path = os.path.join(os.path.dirname(__file__), "enhanced")
        # get logs path
        self.logs_path = os.path.join(os.path.dirname(__file__), "logs", "inference")
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

    def audio_visualization(self, noisy, clean, enh, name):
        self.writer.add_audio("audio/noisy", noisy, 1, sample_rate=self.sr)
        self.writer.add_audio("audio/clean", clean, 1, sample_rate=self.sr)
        self.writer.add_audio("audio/enh", enh, 1, sample_rate=self.sr)

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
            librosa.display.specshow(librosa.amplitude_to_db(mag), cmap="magma", y_axis="linear", ax=axes[k], sr=16000)
        plt.tight_layout()
        self.writer.add_figure(f"spec/{name}", fig, 1)

    def metrics_visualization(self, noisy_list, clean_list, enh_list, n_jobs=8):
        noisy_stoi_score = Parallel(n_jobs=n_jobs)(
            delayed(STOI)(noisy, clean) for noisy, clean in tqdm(zip(noisy_list, clean_list))
        )
        enh_stoi_score = Parallel(n_jobs=n_jobs)(
            delayed(STOI)(noisy, clean) for noisy, clean in tqdm(zip(enh_list, clean_list))
        )
        noisy_stoi_score_mean = np.mean(noisy_stoi_score)
        enh_stoi_score_mean = np.mean(enh_stoi_score)
        self.writer.add_scalar("STOI/test/noisy", noisy_stoi_score_mean, 1)
        self.writer.add_scalar("STOI/test/enh", enh_stoi_score_mean, 1)

        noisy_wb_pesq_score = Parallel(n_jobs=n_jobs)(
            delayed(WB_PESQ)(noisy, clean) for noisy, clean in tqdm(zip(noisy_list, clean_list))
        )
        enh_wb_pesq_score = Parallel(n_jobs=n_jobs)(
            delayed(WB_PESQ)(noisy, clean) for noisy, clean in tqdm(zip(enh_list, clean_list))
        )
        noisy_wb_pesq_score_mean = np.mean(noisy_wb_pesq_score)
        enh_wb_pesq_score_mean = np.mean(enh_wb_pesq_score)
        self.writer.add_scalar("WB_PESQ/test/noisy", noisy_wb_pesq_score_mean, 1)
        self.writer.add_scalar("WB_PESQ/test/enh", enh_wb_pesq_score_mean, 1)

    def save_audio(self, audio_list, audio_files, n_jobs=8):
        Parallel(n_jobs=n_jobs)(
            delayed(sf.write)(audio_file, audio, samplerate=self.sr)
            for audio_file, audio in tqdm(zip(audio_files, audio_list))
        )

    def __call__(self):
        self.model.eval()
        audio_visual_samples_num = 0
        noisy_list = []
        clean_list = []
        enh_list = []
        enh_files = []
        for noisy, clean, noisy_file in tqdm(self.test_iter, desc="test"):
            assert len(noisy_file) == 1

            with paddle.no_grad():
                enh = self.model(noisy)

            noisy = noisy.detach().squeeze(0).cpu().numpy()
            clean = clean.detach().squeeze(0).cpu().numpy()
            enh = enh.detach().squeeze(0).cpu().numpy()
            assert len(noisy) == len(clean) == len(enh)

            audio_visual_samples_num += 1
            if audio_visual_samples_num <= self.audio_visual_samples:
                self.audio_visualization(noisy, clean, enh, noisy_file)

            enh_file = os.path.join(self.output_path, os.path.basename(noisy_file[0]).replace("noisy", "enh_noisy"))

            noisy_list.append(noisy)
            clean_list.append(clean)
            enh_list.append(enh)
            enh_files.append(enh_file)

            self.check_clipped(enh, enh_file)

        # visual metrics
        self.metrics_visualization(noisy_list, clean_list, enh_list, n_jobs=8)
        # save audio
        self.save_audio(enh_list, enh_files, n_jobs=8)


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
    num_workers = 0 if device == "cpu" else config["dataloader"]["num_workers"]
    drop_last = config["dataloader"]["drop_last"]

    # get test_iter
    test_set = DNS_Dataset(dataset_path, config, mode="test")
    test_iter = DataLoader(
        test_set,
        batch_size=1,
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
