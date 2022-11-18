# enables cuda support in docker
FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

# Fix GPG error in `apt-get`
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# install python 3.6, pip and requirements for opencv-python 
# (see https://github.com/NVIDIA/nvidia-docker/issues/864)
RUN apt-get update && apt-get -y install \
    python3 \
    python3-pip \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# install python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install torch~=1.8 torchvision opencv-python-headless~=3.4 timm -i https://pypi.douban.com/simple

# copy inference code
WORKDIR /opt/MiDaS
COPY ./midas ./midas
COPY ./*.py ./

# download model weights so the docker image can be used offline
RUN mkdir weights && cd weights \
    && HTTPS_PROXY=http://172.16.101.68:7890 wget https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt \
    && HTTPS_PROXY=http://172.16.101.68:7890 wget https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21-f6b98070.pt
RUN python3 run.py --model_type dpt_hybrid; exit 0

# entrypoint (dont forget to mount input and output directories)
CMD python3 run.py --model_type midas_v21_small
