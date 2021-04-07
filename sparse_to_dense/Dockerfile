FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04
 
RUN apt-get update
RUN apt-get -y install wget gnupg2
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
 
RUN apt-get update
RUN apt-get -y install python3 \
 python3-pip \
 python3-matplotlib \
 python3-h5py \
 python3-imageio \
 python3-opencv \
 libhdf5-serial-dev \
 hdf5-tools
 
RUN python3 -m pip install scikit-image torch==1.7.1 torchvision==0.8.2 torchaudio scipy==1.2.2

WORKDIR /workspace