FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        autopoint \
        binutils \
        bison \
        build-essential \
        ca-certificates \
        clang-14 \
        clang-18 \
        cmake \
        curl \
        file \
        flex \
        gawk \
        g++-11 \
        g++-13 \
        gcc-11 \
        gcc-13 \
        gettext \
        git \
        gperf \
        help2man \
        libtool \
        libtool-bin \
        linux-libc-dev \
        m4 \
        meson \
        ninja-build \
        patch \
        perl \
        pkg-config \
        python3 \
        python3-pyelftools \
        python3-venv \
        rsync \
        texinfo \
        unzip \
        wget \
        xz-utils \
        zstd \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 130 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 130 \
    && update-alternatives --install /usr/bin/clang clang /usr/bin/clang-18 180 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-18 180 \
    && rm -rf /var/lib/apt/lists/*

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /workspace
