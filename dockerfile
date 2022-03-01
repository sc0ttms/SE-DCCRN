FROM continuumio/miniconda3
COPY requirements.txt /
RUN apt-get update \
    && apt-get install --yes build-essential
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ \
    && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ \
    && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ \
    && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ \
    && conda config --set show_channel_urls yes
RUN conda install --yes python=3.7.5 
RUN conda install --yes pytorch=1.10.2 torchvision torchaudio cpuonly
RUN while read requirement; do conda install --yes $requirement || pip install $requirement -i https://pypi.tuna.tsinghua.edu.cn/simple; done < requirements.txt \
    && rm -f requirements.txt