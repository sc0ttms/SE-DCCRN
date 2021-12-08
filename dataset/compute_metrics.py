# -*- coding: utf-8 -*-

import sys
import os
import toml
import librosa
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

sys.path.append(os.getcwd())
from audio.metrics import SI_SDR, STOI, WB_PESQ, NB_PESQ


def comptute_metric(noisy_file, clean_file, sr=16000, metric_type="STOI"):
    # get noisy, clean
    noisy, _ = librosa.load(noisy_file, sr=sr)
    clean, _ = librosa.load(clean_file, sr=sr)
    assert len(noisy) == len(clean)

    # get metric score
    if metric_type in ["SI_SDR"]:
        return SI_SDR(noisy, clean, sr=sr)
    elif metric_type in ["STOI"]:
        return STOI(noisy, clean, sr=sr)
    elif metric_type in ["WB_PESQ"]:
        return WB_PESQ(noisy, clean)
    elif metric_type in ["NB_PESQ"]:
        return NB_PESQ(noisy, clean)


if __name__ == "__main__":
    # get dataset path
    dataset_path = os.path.join(os.getcwd(), "dataset_csv")

    # get set path
    train_path = os.path.join(dataset_path, "train.csv")
    valid_path = os.path.join(dataset_path, "valid.csv")
    test_path = os.path.join(dataset_path, "test.csv")

    # get train files
    train_files = pd.read_csv(train_path).values
    train_noisy_files = train_files[:, 0].reshape(1, len(train_files))[0]
    train_clean_files = train_files[:, 1].reshape(1, len(train_files))[0]
    # get valid files
    valid_files = pd.read_csv(valid_path).values
    valid_noisy_files = valid_files[:, 0].reshape(1, len(valid_files))[0]
    valid_clean_files = valid_files[:, 1].reshape(1, len(valid_files))[0]
    # get test files
    test_files = pd.read_csv(test_path).values
    test_noisy_files = test_files[:, 0].reshape(1, len(test_files))[0]
    test_clean_files = test_files[:, 1].reshape(1, len(test_files))[0]

    # get compute metrics config
    toml_path = os.path.join(os.path.dirname(__file__), "compute_metrics_cfg.toml")
    config = toml.load(toml_path)
    # get n_jobs
    n_jobs = config["ppl"]["n_jobs"]

    # get metrics
    metrics = {
        "SI_SDR": [],
        "STOI": [],
        "WB_PESQ": [],
        "NB_PESQ": [],
    }

    # compute train metrics
    train_si_sdr_score = Parallel(n_jobs=n_jobs)(
        delayed(comptute_metric)(noisy_file, clean_file, metric_type="SI_SDR")
        for noisy_file, clean_file in tqdm(zip(train_noisy_files, train_clean_files))
    )
    train_stoi_score = Parallel(n_jobs=n_jobs)(
        delayed(comptute_metric)(noisy_file, clean_file, metric_type="STOI")
        for noisy_file, clean_file in tqdm(zip(train_noisy_files, train_clean_files))
    )
    train_wb_pesq_score = Parallel(n_jobs=n_jobs)(
        delayed(comptute_metric)(noisy_file, clean_file, metric_type="WB_PESQ")
        for noisy_file, clean_file in tqdm(zip(train_noisy_files, train_clean_files))
    )
    train_nb_pesq_score = Parallel(n_jobs=n_jobs)(
        delayed(comptute_metric)(noisy_file, clean_file, sr=8000, metric_type="NB_PESQ")
        for noisy_file, clean_file in tqdm(zip(train_noisy_files, train_clean_files))
    )
    metrics["SI_SDR"].append(np.mean(train_si_sdr_score))
    metrics["STOI"].append(np.mean(train_stoi_score))
    metrics["WB_PESQ"].append(np.mean(train_wb_pesq_score))
    metrics["NB_PESQ"].append(np.mean(train_nb_pesq_score))

    # compute valid metrics
    valid_si_sdr_score = Parallel(n_jobs=n_jobs)(
        delayed(comptute_metric)(noisy_file, clean_file, metric_type="SI_SDR")
        for noisy_file, clean_file in tqdm(zip(valid_noisy_files, valid_clean_files))
    )
    valid_stoi_score = Parallel(n_jobs=n_jobs)(
        delayed(comptute_metric)(noisy_file, clean_file, metric_type="STOI")
        for noisy_file, clean_file in tqdm(zip(valid_noisy_files, valid_clean_files))
    )
    valid_wb_pesq_score = Parallel(n_jobs=n_jobs)(
        delayed(comptute_metric)(noisy_file, clean_file, metric_type="WB_PESQ")
        for noisy_file, clean_file in tqdm(zip(valid_noisy_files, valid_clean_files))
    )
    valid_nb_pesq_score = Parallel(n_jobs=n_jobs)(
        delayed(comptute_metric)(noisy_file, clean_file, sr=8000, metric_type="NB_PESQ")
        for noisy_file, clean_file in tqdm(zip(valid_noisy_files, valid_clean_files))
    )
    metrics["SI_SDR"].append(np.mean(valid_si_sdr_score))
    metrics["STOI"].append(np.mean(valid_stoi_score))
    metrics["WB_PESQ"].append(np.mean(valid_wb_pesq_score))
    metrics["NB_PESQ"].append(np.mean(valid_nb_pesq_score))

    # compute test metrics
    test_si_sdr_score = Parallel(n_jobs=n_jobs)(
        delayed(comptute_metric)(noisy_file, clean_file, metric_type="SI_SDR")
        for noisy_file, clean_file in tqdm(zip(test_noisy_files, test_clean_files))
    )
    test_stoi_score = Parallel(n_jobs=n_jobs)(
        delayed(comptute_metric)(noisy_file, clean_file, metric_type="STOI")
        for noisy_file, clean_file in tqdm(zip(test_noisy_files, test_clean_files))
    )
    test_wb_pesq_score = Parallel(n_jobs=n_jobs)(
        delayed(comptute_metric)(noisy_file, clean_file, metric_type="WB_PESQ")
        for noisy_file, clean_file in tqdm(zip(test_noisy_files, test_clean_files))
    )
    test_nb_pesq_score = Parallel(n_jobs=n_jobs)(
        delayed(comptute_metric)(noisy_file, clean_file, sr=8000, metric_type="NB_PESQ")
        for noisy_file, clean_file in tqdm(zip(test_noisy_files, test_clean_files))
    )
    metrics["SI_SDR"].append(np.mean(test_si_sdr_score))
    metrics["STOI"].append(np.mean(test_stoi_score))
    metrics["WB_PESQ"].append(np.mean(test_wb_pesq_score))
    metrics["NB_PESQ"].append(np.mean(test_nb_pesq_score))

    df = pd.DataFrame(metrics, index=["train", "valid", "test"])
    df.to_csv(os.path.join(dataset_path, "metrics.csv"))
