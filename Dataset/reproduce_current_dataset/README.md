# Reproduce Current Dataset

This folder contains the commands needed to rebuild the current dataset layout from scratch.

This is the legacy 90-case snapshot. To create the two separated datasets
(LibSeeker program inventory and non-overlapping unseen programs), use
`Dataset/create_datasets.sh`; see `Dataset/README.md`.

Anche questa pipeline legacy esclude `O1`; le ottimizzazioni ammesse per le
librerie sono `O0`, `O2`, `O3` e `Os`.

The current snapshot builds:

- source downloads from `Dataset/manifests/source_manifest.json`
- static libraries in `Dataset/builds/lib_builds`
- the randomized ELF cases listed in `Dataset/builds/elf_builds/randomized_matrix.json`
- ground truth under `GroundTruth`
- refreshed matrix metadata in `Dataset/builds/elf_builds/randomized_matrix.json`
- CU and library ground truth derived from the linker map of the final ELF link

`randomized_matrix.json` is the source of truth. The rebuild script does not create a new random matrix: it reads the cases already stored there, including the selected library optimizations and library source directories.

Current randomized dataset:

- 90 ELF cases
- 15 programs: `bash`, `gawk`, `gnuchess`, `grep`, `gzip`, `inetutils`, `less`, `make`, `nano`, `openssh`, `rsync`, `sed`, `socat`, `tar`, `wget2`
- each program is built with `gcc` and `clang` at `O0,O2,O3`
- `coreutils` and `grep-3.11` are intentionally excluded from the current matrix

Library sources:

- `Dataset/sources/lib_sources` currently contains 52 selected source directories.
- `Dataset/scripts/build_libraries.sh` uses this selected source set by default.
- Existing library variants are reused by default by this reproduction script; only missing variants are built.
- The `libraries` stage builds the library set twice: once with `gcc/g++` and once with `clang/clang++`.
- Use `Dataset/scripts/build_libraries.sh --curated` if you only want the smaller set of libraries required by the current ELF matrix.
- Libraries that failed with both toolchains or produced non-matching archive sets were removed from the selected set.
- `glibc-2.41` and `mpfr-4.2.1` are kept because they are required by the generated ELF cases, even though they are known clang/gcc exception points.
- Some source directories use CMake or Meson, so a full rebuild needs `cmake`, `meson` and `ninja` in addition to the usual Autotools stack.

Run everything from anywhere:

```bash
Dataset/reproduce_current_dataset/reproduce_dataset.sh
```

Preview commands without running them:

```bash
Dataset/reproduce_current_dataset/reproduce_dataset.sh --dry-run
```

Run individual stages:

```bash
Dataset/reproduce_current_dataset/reproduce_dataset.sh fetch
Dataset/reproduce_current_dataset/reproduce_dataset.sh libraries -j 16
Dataset/reproduce_current_dataset/reproduce_dataset.sh elves
Dataset/reproduce_current_dataset/reproduce_dataset.sh ground-truth
```

Useful options:

```bash
Dataset/reproduce_current_dataset/reproduce_dataset.sh --skip-existing
Dataset/reproduce_current_dataset/reproduce_dataset.sh --clean
```

For byte-identical binaries you need the same toolchain used for the current dataset: `gcc-16.1.1` and `clang-22.1.6`. With different compiler versions, the script still recreates the same dataset structure and build choices, but the produced binaries can differ.

Ground truth policy:

- New ELF builds are linked normally first, then the final target is relinked once with `-Wl,-Map=... -Wl,--cref`.
- `GroundTruth/*/*/ground_truth.json` marks a compilation unit as included only when it appears as `lib.a(member.o)` in that final linker map.
- Library presence is derived from the same data: a library is present only if at least one of its archives has at least one included compilation unit.
- The `ground-truth` action reruns only that final relink/metadata refresh step for existing build directories.
