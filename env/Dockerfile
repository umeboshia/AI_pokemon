FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

ARG UID 1000
ENV CUDA cu113
ENV TORCH 1.11.0
ENV DEBIAN_FRONTEND=noninteractive

# install essential softwares
RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt update && apt install -y vim zsh git ssh sudo language-pack-en tmux python3-opencv
RUN update-locale LANG=en_US.UTF-8

# link to python
RUN ln -s /opt/conda/bin/python /usr/bin/python

# python -m pip install packages
RUN python -m pip install --upgrade pip
RUN python -m pip install numpy matplotlib pylint tqdm sentencepiece transformers scikit-learn tensorboard spacy==2.3.7 scispacy black seaborn rudalle==0.4.0 googletrans==3.1.0a0 google
RUN python -m pip install streamlit
RUN python -m pip install --upgrade jinja2
