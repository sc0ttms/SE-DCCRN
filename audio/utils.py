# -*- coding: utf-8 -*-

import os
import zipfile
from tqdm import tqdm


def unzip(zip_path, unzip_path=None):
    """unzip

    Args:
        zip_path (str): zip path
        unzip_path (str, optional): unzip path. Defaults to None.

    Returns:
        unzip_path: unzip path
    """

    # set unzip path
    if unzip_path == None:
        unzip_path = os.path.splitext(zip_path)[0]

    # unzip
    with zipfile.ZipFile(zip_path) as zf:
        for file in tqdm(zf.infolist(), desc="unzip..."):
            try:
                zf.extract(file, unzip_path)
            except zipfile.error as e:
                print(e)

    return unzip_path


def prepare_empty_path(paths, resume=False):
    """prepare empty path

    Args:
        paths (list): path list
        resume (bool, optional): whether to resume. Defaults to False.
    """
    for path in paths:
        if resume:
            assert os.path.exists(path)
        else:
            os.makedirs(path, exist_ok=True)
