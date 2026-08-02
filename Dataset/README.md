# Dataset ELF statici A e B

Questa directory contiene soltanto manifest, script e istruzioni riproducibili.
Gli ELF e le ground truth non entrano nella repository: per default vengono
materializzati in `../Thesis_Binary_Analysis_artifacts/`.

## Struttura dei dataset

Il Dataset A contiene 3.504 ELF statici x86-64 e il Dataset B ne contiene
1.200. Entrambi sono compilati con glibc e con lo stesso insieme di librerie
statiche. O1 è esclusa sia per i programmi sia per le librerie.

Le combinazioni di compilatore e ottimizzazione sono identiche per A e B:

| Famiglia | Versione | Ottimizzazioni ELF | ELF A | ELF B |
|---|---:|---|---:|---:|
| GCC | 11 | O0, O2, O3, Os | 876 | 300 |
| GCC | 13 | O0, O2, O3, Os | 876 | 300 |
| Clang | 14 | O0, O2, O3, Os | 876 | 300 |
| Clang | 18 | O0, O2, O3, Os | 876 | 300 |

Nel Dataset A ogni cella contiene 219 ELF (1.752 GCC e 1.752 Clang); nel
Dataset B ogni cella ne contiene 75 (600 GCC e 600 Clang). Le versioni reali congelate nell'immagine sono
GCC 11.5.0, GCC 13.3.0, Clang 14.0.6 e Clang 18.1.3.

Per ogni coppia progetto/cella, le ottimizzazioni delle librerie linkate sono
scelte pseudo-casualmente tra O0, O2, O3 e Os con seed `20260731`. La scelta è
riproducibile e viene registrata nella ground truth. glibc costituisce l'unica
eccezione: non supporta una build realmente O0 e viene scelta tra O2, O3 e Os.

### Dataset A

Usa tutti i 219 nomi presenti nell'inventario LibSeeker. Ciascuno è compilato
in tutte le 16 combinazioni compilatore/ottimizzazione, per un totale di 3.504
ELF. L'elenco esatto e la provenienza sono in
`manifests/libseeker_inventory.json`.

### Dataset B

Usa 75 programmi GNU/Linux normali, tutti diversi per nome dai 219 programmi di
A. Non usa Toybox e non usa musl: applica la stessa pipeline statica glibc di A.
Tutti i 75 programmi vengono compilati in tutte le 16 celle, producendo 1.200
ELF. L'elenco esatto è in `manifests/unseen_glibc_inventory.json`.

Provenienza dei programmi B:

| Progetto | Versione | Programmi |
|---|---:|---:|
| GNU coreutils | 9.6 | 2 |
| GNU Awk | 5.3.2 | 2 |
| GNU gzip | 1.13 | 1 |
| GNU Inetutils | 2.6 | 13 |
| OpenSSH Portable | V_10_0_P2 | 7 |
| GNU tar | 1.35 | 1 |
| util-linux | v2.39.3 | 36 |
| GNU Wget2 (programmi example standalone) | 2.2.0 | 13 |

URL, versioni, SHA-256 e revisioni dei sorgenti sono congelati in
`manifests/source_manifest.json`. I due inventari sono validati come disgiunti
anche per nome; la validazione finale vieta inoltre hash ELF uguali tra A e B.

## Riproduzione

Dalla root della repository:

```bash
Dataset/scripts/run_reproduction_container.sh all -j 4
```

Azioni utili:

```bash
Dataset/scripts/run_reproduction_container.sh libseeker -j 4
Dataset/scripts/run_reproduction_container.sh unseen -j 4
Dataset/create_datasets.sh collect
Dataset/create_datasets.sh validate
```

`--skip-existing` è attivo per default e rende le build riprendibili. Per
cambiare la destinazione esterna usare `--artifact-root DIR`. `--clean` elimina
e ricrea le build richieste; va usato soltanto quando si desidera una
ricostruzione completa.

La raccolta ufficiale usa hard-link per ELF, linker map e metadati di link:
questi restano file regolari nel dataset, ma il pool A da 3.504 varianti non
duplica inutilmente i blocchi già presenti nella build root. Manifest e ground
truth JSON restano autonomi.

Output:

```text
../Thesis_Binary_Analysis_artifacts/
  datasets/{libseeker,unseen}/
  datasets_compat/{libseeker,unseen}/
  ground_truth/{libseeker,unseen}/
  ground_truth_compat/{libseeker,unseen}/
```

## Ground truth

Per ogni ELF vengono conservati il linker map esatto, SHA-256 di ELF e map,
compilatore/versione/flag, provenienza del sorgente, librerie e ottimizzazioni
selezionate, SHA-256 degli archivi e membri `.a` effettivamente inclusi dal
linker. `scripts/validate_datasets.py` controlla cardinalità, matrice completa,
staticità, bilanciamento GCC/Clang, assenza di O1, provenienza, hash, linker map,
cataloghi degli archivi e separazione tra i due dataset.

I file storici `scripts/build_toybox_dataset.py` e
`manifests/unseen_programs.txt` restano solo come materiale legacy e non sono
usati dal piano corrente.
