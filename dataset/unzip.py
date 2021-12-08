# -*- coding: utf-8 -*-

import sys
import os
import toml


sys.path.append(os.getcwd())
from audio.utils import unzip


if __name__ == "__main__":
    # get config
    toml_path = os.path.join(os.path.dirname(__file__), "unzip_cfg.toml")
    config = toml.load(toml_path)

    # get datasets zip path
    zip_path = config["path"]["zip"]

    # extract datasets zip
    unzip_path = os.path.splitext(zip_path)[0]
    if not os.path.exists(unzip_path):
        unzip_path = unzip(zip_path)
    print(f"dataset path {unzip_path}")
