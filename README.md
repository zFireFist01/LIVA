# LIVA

**Variability-Aware Library Identification in Static Binaries**

LIVA is a static binary analysis tool for identifying third-party **statically linked libraries in x86-64 ELF binaries** under compilation variability.

Instead of relying on exact function-to-function matching, LIVA combines:

- semantic similarity at **basic-block granularity**;
- structural reasoning at **compilation-unit (CU) granularity**;
- direct-call relationships;
- read-only data (`.rodata`) evidence;
- library-level aggregation across multiple compilation units.

The approach is designed to remain useful when the target executable and the candidate libraries have been built with different **compiler versions, compiler families, optimization levels, or source versions**.

LIVA was developed as part of the Master's Thesis:

> **LIVA: Variability-Aware Library Identification in Static Binaries**  
> Paolo Gennaro  
> MSc in Computer Science and Engineering  
> Politecnico di Milano, 2026

---

## Motivation

Identifying dynamically linked libraries is generally straightforward because dependency information remains available in the executable.

Static linking is different.

When a static archive is linked into an executable:

1. the dependency is no longer explicitly represented;
2. only the object files required by the linker may be included;
3. library and application code become intermixed;
4. compiler transformations may substantially modify the generated machine code.

The same source code may therefore produce different instruction sequences, function boundaries, control-flow structures, and data layouts depending on the build configuration.

LIVA addresses this problem by using **basic blocks to obtain fine-grained semantic correspondences**, while using **compilation units to determine whether those correspondences form a structurally plausible library match**.

---

## Approach

The LIVA pipeline consists of four main stages:

```mermaid
flowchart LR
    A[Target x86-64 ELF] --> C[Binary Loader]
    B[Candidate static libraries] --> AR[GNU ar]
    AR --> C

    C -->|Parsed code units| D[Embeddings Manager]
    D -->|Code units + block embeddings| E[Block Matcher]
    E -->|CU match results| F[Results Aggregator]
    F --> G[Library-level decisions]
```

### 1. Binary Parsing

The target executable and candidate static libraries are converted into a common representation containing:

- functions;
- basic blocks;
- assembly instructions;
- direct-call relationships;
- read-only data.

Candidate static archives are first decomposed into their object-file members using **GNU `ar`**. Each object file is treated as a candidate compilation unit.

Binary analysis is performed using **radare2** through **r2pipe**.

The target binary does **not** need debugging information or function names. Matching relies on information recoverable directly from the binary.

---

### 2. Basic-Block Embeddings

Assembly instructions are encoded using the pre-trained **PalmTree** assembly-language model.

Instruction embeddings are aggregated to obtain one vector representation for each basic block.

LIVA uses the public PalmTree `transformer.ep19` checkpoint without additional training or fine-tuning.

The resulting block representations allow semantically related code to remain comparable even when the exact machine-code representation changes across builds.

---

### 3. Compilation-Unit Block Matching

Each candidate compilation unit is searched inside localized regions of the target executable.

LIVA first computes similarities between the candidate and target basic-block embeddings and then evaluates whether the resulting correspondences form a plausible structural match.

The matcher considers evidence including:

- block similarity;
- block coverage;
- one-to-one block assignment;
- assignment quality;
- locality of the matching region;
- consistency of block-derived function mappings;
- function concentration and spread;
- direct-call preservation.

One-to-one block correspondence is computed using the **Hungarian assignment algorithm**.

The central idea is that an isolated similar block provides weak evidence, while a collection of mutually consistent block matches belonging to the same compilation unit provides substantially stronger evidence.

---

### 4. Evidence Aggregation

Compilation units that survive block matching are further evaluated using additional evidence.

#### Read-only data

LIVA compares candidate `.rodata` with the target executable using:

- weighted printable-string containment;
- byte n-gram containment.

Read-only data can reinforce a compatible CU match or reject a structurally plausible but data-incompatible candidate.

It cannot independently create a match without supporting code evidence.

#### Cross-CU relationships

Direct calls between functions belonging to different compilation units of the same library can also be checked after the corresponding CUs have been mapped into the target.

#### Library-level decision

Evidence from accepted compilation units is aggregated into a single library score.

The implementation supports several aggregation strategies, including:

- mean;
- maximum;
- top-3 mean;
- top-3 noisy-OR.

The resulting score is compared with the configured library-level threshold to determine whether the candidate library is present.

---

## Analysis Cache

Binary analysis and PalmTree inference are the most computationally expensive parts of the pipeline.

LIVA therefore includes a persistent, content-addressed analysis cache containing the parsed binary representation and basic-block embeddings.

Cache identity depends on the inputs that influence feature generation, including:

- input binary;
- assembly normalization;
- PalmTree model and vocabulary;
- PalmTree pooling configuration;
- parser/model implementation;
- radare2 version.

Matching thresholds are deliberately excluded from the cache identity.

This allows different matching configurations and hyperparameters to be evaluated without repeating radare2 analysis or PalmTree inference.

Cached numerical data are stored using NumPy arrays and can be accessed through memory mapping.

---

## Requirements

LIVA currently targets:

- **Linux**
- **x86-64 ELF binaries**
- **static `.a` archives**

The experimental environment used for the thesis was:

| Component | Version / configuration |
|---|---|
| OS | Ubuntu 24.04 |
| Python | 3.12 |
| radare2 | 5.5.0 |
| r2pipe | 1.9.8 |
| PyTorch | 2.13.0 |
| NumPy | 1.26.4 |
| SciPy | 1.15.2 |
| scikit-learn | 1.6.1 |
| NetworkX | 3.4.2 |
| Optuna | 4.9.0 |

The PalmTree checkpoint and its vocabulary must also be available, by default under:

```text
palmtree/model/transformer.ep19
```

GNU `ar`, provided by GNU Binutils, is required for static-archive extraction.

---

## Running LIVA

The main entry point is:

```bash
python main.py \
    --path_to_binary /path/to/target \
    --libraries_dir /path/to/static/libraries
```

By default, LIVA expects the PalmTree checkpoint at:

```text
palmtree/model/transformer.ep19
```

A custom model path can be specified with:

```bash
python main.py \
    --path_to_binary /path/to/target \
    --libraries_dir /path/to/static/libraries \
    --asm_model /path/to/transformer.ep19
```

The PalmTree execution device can also be selected explicitly:

```bash
--device cpu
--device cuda
--device cuda:0
```

or left to automatic selection:

```bash
--device auto
```

Run:

```bash
python main.py --help
```

for the complete list of matching and evidence parameters.

---

## Precomputing the Analysis Cache

For large experiments, binary analysis and PalmTree inference can be performed once before matching.

The cache builder can process ELF datasets and static-library archives:

```bash
python build_analysis_cache.py \
    --dataset-dir /path/to/binaries \
    --libraries-dir /path/to/libraries
```

After the cache has been created, matching-only experiments can reuse the stored representations without requiring GPU inference.

This separation is particularly useful when evaluating multiple matching thresholds or running experiments across several CPU-only machines.

---

## Repository Structure

The main implementation is organized as follows:

```text
.
├── main.py
│   └── Main LIVA command-line pipeline
│
├── asm.py
│   └── ELF/object parsing, assembly normalization and CodeUnit representation
│
├── model.py
│   └── PalmTree interface and embedding generation
│
├── match.py
│   └── Block matching, structural evidence, .rodata and library aggregation
│
├── analysis_cache.py
│   └── Persistent content-addressed analysis cache
│
├── numpy_cache.py
│   └── Memory-mappable representation of cached CodeUnits
│
├── archive_utils.py
│   └── Safe extraction of GNU ar archive members
│
├── build_analysis_cache.py
│   └── Precomputation of binary analysis and PalmTree embeddings
│
├── run_libseeker_batch.py
│   └── Batch experimental evaluation
│
├── optuna_threshold_search.py
│   └── Hyperparameter and decision-threshold optimization
│
└── palmtree/
    └── PalmTree model interface, vocabulary and checkpoint
```

Additional scripts are provided for cache validation, conversion, pruning, provenance management, and experiment reproducibility.

---

## Experimental Evaluation

LIVA was evaluated using completely separate development and held-out datasets.

### Held-out dataset

The final evaluation contains:

- **3,504 x86-64 ELF binaries**
- **219 programs**
- **17 source projects**
- **164 logical static-library families**
- **4,440 physical static-library builds**

The binaries cover:

**Compilers**

- GCC 11.5
- GCC 13.3
- Clang 14.0
- Clang 18.1

**Optimization levels**

- `-O0`
- `-O2`
- `-O3`
- `-Os`

The final test dataset was not used for matching-parameter selection.

Ground truth was reconstructed from GNU linker maps, making it possible to determine both which static archives actually contributed code and which individual archive members were extracted by the linker.

---

## Results

At **library-family level**, the complete LIVA pipeline achieves:

| Precision | Recall | F1 |
|---:|---:|---:|
| **0.636** | **0.840** | **0.724** |

Performance under different forms of binary variability was also evaluated:

| Setting | F1 |
|---|---:|
| Same build | **0.723** |
| Cross optimization | **0.622** |
| Cross compiler | **0.562** |
| Cross version | **0.670** |

Cross-version matching reaches a recall of **0.831**, while cross-compiler comparison is the most challenging evaluated form of variability.

These results show that library identification does not immediately fail when target and candidate code are compiled under different configurations.

---

## Why Structural Evidence Matters

An ablation study shows that basic-block similarity alone is too permissive.

| Configuration | Precision | Recall | F1 |
|---|---:|---:|---:|
| Block evidence only | 0.317 | 0.860 | 0.464 |
| Full LIVA | **0.636** | **0.840** | **0.724** |

The largest improvement comes from adding **intra-CU structural constraints**, which substantially reduce false positives while preserving most of the recall.

This result motivates the main design principle behind LIVA:

> **Basic blocks provide semantic correspondences; compilation units determine whether those correspondences are structurally credible.**

---

## Compilation-Unit Identification

LIVA also exposes the compilation units supporting a library prediction.

On the evaluable held-out CU-level domain:

| Precision | Recall | F1 |
|---:|---:|---:|
| **0.716** | **0.149** | **0.246** |

CU recovery is therefore intentionally conservative: LIVA identifies only a subset of the object files actually incorporated into the executable.

Importantly, exhaustive CU reconstruction is not required for successful library identification. A sufficiently informative subset of consistent compilation-unit matches can already support a correct library-level decision.

---

## Current Limitations

The current implementation has several limitations:

- only **x86-64 ELF** binaries are supported;
- the task is **closed-set**: candidate libraries must be supplied to LIVA;
- matching depends on the function and basic-block structure recovered by static analysis;
- extensive inlining, code reordering, function splitting/merging, and Link-Time Optimization may weaken locality and structural assumptions;
- very small compilation units provide limited structural evidence;
- exhaustive matching against very large candidate corpora remains computationally expensive;
- cross-CU call evidence is currently available only in a limited number of cases.

Support for additional architectures, more flexible structural mappings, improved candidate retrieval, and more efficient parallel matching are natural extensions of the current implementation.

---

## Research Context

LIVA builds on research in binary similarity and static-library identification, with particular inspiration from approaches based on compilation units and learned assembly representations.

The project uses **PalmTree** for instruction-level assembly embeddings and extends the compilation-unit perspective by combining fine-grained block matching with explicit structural and read-only-data evidence.

For a complete description of the approach, experimental methodology, hyperparameter selection, ablation study, and limitations, refer to the accompanying Master's thesis.

---

## Citation

If you use LIVA in academic work, please cite:

```bibtex
@mastersthesis{gennaro2026liva,
    author  = {Paolo Gennaro},
    title   = {LIVA: Variability-Aware Library Identification in Static Binaries},
    school  = {Politecnico di Milano},
    year    = {2026},
    type    = {Master's Thesis}
}
```

---

## Author

**Paolo Gennaro**  
Politecnico di Milano  
MSc in Computer Science and Engineering

Thesis advisor: **Prof. Stefano Zanero**  
Co-advisors: **Marco D'Amico** and **Lorenzo Binosi**

---

## License

See the repository license for terms of use.
