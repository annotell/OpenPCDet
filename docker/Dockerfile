FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04@sha256:217134b60289ced62b47463be32584329baf35cb55a9ccda6626b91bd59803be

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Install basics

RUN apt-get update && apt-get install -y python3-opencv ca-certificates python3.10 python3-pip git wget sudo ninja-build
RUN ln -sv /usr/bin/python3.10 /usr/bin/python

RUN apt-get update -y \
    && apt-get install -y build-essential \
    && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev bmon g++ libboost-all-dev

RUN pip install cmake onnx

# Install python packages
RUN PIP_INSTALL="python -m pip --no-cache-dir install" && \
    $PIP_INSTALL numpy==1.23.5 llvmlite numba

# Install torch and torchvision
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html

# Install python packages
RUN PIP_INSTALL="python -m pip --no-cache-dir install" && \
    $PIP_INSTALL tensorboardX easydict pyyaml scikit-image tqdm SharedArray six

WORKDIR /root

# A weired problem that hasn't been solved yet
RUN pip uninstall -y SharedArray && \
    pip install SharedArray

RUN pip install spconv-cu117 notebook pyquaternion

COPY OpenPCDet /root/OpenPCDet

WORKDIR /root/OpenPCDet

# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN python setup.py develop

ENV PYTHONPATH "${PYTHONPATH}:/root/OpenPCDet"