FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# -----------------------
# Strumenti di base
# -----------------------
RUN apt-get update && apt-get install -y \
    software-properties-common wget curl gnupg lsb-release build-essential python3 python3-pip git cmake file \
 && rm -rf /var/lib/apt/lists/*

# -----------------------
# GCC 11 e GCC 13
# -----------------------
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y && apt-get update && \
    apt-get install -y gcc-11 g++-11 gcc-13 g++-13 && \
    rm -rf /var/lib/apt/lists/*

# -----------------------
# LLVM/Clang 14 e 18
# -----------------------
RUN wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && ./llvm.sh 14 && rm -f llvm.sh
RUN wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && ./llvm.sh 18 && rm -f llvm.sh

# -----------------------
# Update alternatives
# -----------------------
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 13 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 13 && \
    update-alternatives --install /usr/bin/clang clang /usr/lib/llvm-14/bin/clang 14 && \
    update-alternatives --install /usr/bin/clang clang /usr/lib/llvm-18/bin/clang 18 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/lib/llvm-14/bin/clang++ 14 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/lib/llvm-18/bin/clang++ 18 && \
    update-alternatives --set gcc /usr/bin/gcc-13 && \
    update-alternatives --set g++ /usr/bin/g++-13 && \
    update-alternatives --set clang /usr/lib/llvm-18/bin/clang && \
    update-alternatives --set clang++ /usr/lib/llvm-18/bin/clang++

# -----------------------
# Install angr (symbolic execution)
# -----------------------
RUN pip install --no-cache-dir angr

# -----------------------
# Install Dwarf debugger
# -----------------------
RUN pip install --no-cache-dir dwarf-debugger

# -----------------------
# Copia la cartella libseeker_repo nel container
# -----------------------
COPY libseeker_repo/ /app/libseeker_repo/

COPY easy/ /app/easy/

# Imposta la directory di lavoro
WORKDIR /app

# Rendi eseguibili gli script shell
RUN chmod +x /app/libseeker_repo/build_lib/*.sh
RUN chmod +x /app/easy/*.sh

CMD ["/bin/bash"]


# Per cambiare versioni
#   update-alternatives --config gcc
#   update-alternatives --config clang
# oppure
#   gcc-11 main.c -o main_gcc11
#   clang-14 main.c -o main_clang14
