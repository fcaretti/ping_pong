FROM ubuntu:jammy

USER root

RUN apt-get update && apt install -y python3 && apt install -y python3-pip && \
    pip3 install pandas && pip3 install gdown && pip3 install matplotlib && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install pyro-ppl && pip3 install scipy && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
