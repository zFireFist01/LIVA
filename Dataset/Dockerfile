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
        dpkg-dev \
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
        libatk1.0-dev \
        libcairo2-dev \
        libdatrie-dev \
        libdbus-1-dev \
        libei-dev \
        libffi-dev \
        libfontconfig1-dev \
        libfreetype-dev \
        libfribidi-dev \
        libgdk-pixbuf-2.0-dev \
        libglib2.0-dev \
        libgraphite2-dev \
        libharfbuzz-dev \
        libice-dev \
        libmd-dev \
        libmount-dev \
        libpango1.0-dev \
        libpcre2-dev \
        libpng-dev \
        libselinux1-dev \
        libsepol-dev \
        libsm-dev \
        libssl-dev \
        libthai-dev \
        libx11-dev \
        libxau-dev \
        libxcb1-dev \
        libxcomposite-dev \
        libxcursor-dev \
        libxdamage-dev \
        libxdmcp-dev \
        libxext-dev \
        libxfixes-dev \
        libxi-dev \
        libxinerama-dev \
        libxrandr-dev \
        libxrender-dev \
        libxt-dev \
        libxtst-dev \
        libxkbcommon-dev \
        libxml2-dev \
        linux-libc-dev \
        m4 \
        meson \
        ninja-build \
        patch \
        perl \
        pkg-config \
        python3 \
        python3-pyelftools \
        python3-xcbgen \
        python3-venv \
        rsync \
        texinfo \
        unzip \
        wget \
        xz-utils \
        x11proto-dev \
        xcb-proto \
        xtrans-dev \
        xutils-dev \
        zstd \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 130 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 130 \
    && update-alternatives --install /usr/bin/clang clang /usr/bin/clang-18 180 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-18 180 \
    && rm -rf /var/lib/apt/lists/*

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /workspace
