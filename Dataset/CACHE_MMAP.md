# Cache mmap di LIVA

La cache ufficiale appartiene a un solo shard ed è salvata in
`<artifact-root>/cache`. “Locale” indica che le cache dei quattro shard non
vengono unite: non significa che debbano necessariamente essere ricostruite sul
PC di matching. Nei pacchetti autonomi per PC2-PC4 la cache completa e validata
è già inclusa e viene trasferita insieme allo shard. Su PC1
`Dataset/shards/libseeker-shard-0/cache` è un collegamento a `Dataset/cache`.
Il launcher dei pacchetti usa la modalità cache-only: legge l'identità originale
dal manifest, mantiene la cache in sola lettura, non richiede il binario
`radare2` e interrompe il matching al primo miss.
Le entry correnti sono directory NumPy content-addressed e vengono lette con
mmap. I builder nuovi scrivono direttamente questo formato: la conversione da
pickle serve soltanto per importare una cache legacy.

## Perché `.npy` e non `.npz`

Gli array `.npy` non compressi possono essere aperti con
`numpy.load(..., mmap_mode="r")`: il kernel carica soltanto le pagine realmente
lette e può condividerle tra processi. Un `.npz` compresso dovrebbe invece
essere decompresso in RAM. Stringhe e `.rodata` sono in file binari con offset
NumPy e vengono anch'essi mappati.

Ogni entry conserva embedding dei blocchi, numero di istruzioni, confini e
visibilità delle funzioni, call target numerici, simboli di chiamata esterni e
contenuto/stringhe `.rodata`. Testo normalizzato delle istruzioni, archi CFG
intra-funzione e byte grezzi dei blocchi sono intermedi rigenerabili non usati
dal matching corrente; ne restano conteggi/dimensioni. Gli archivi e gli ELF
originali non vengono eliminati.

## Procedura di rigenerazione per PC1-PC4

Da `/home/tecnico/LIVA`:

```bash
Dataset/scripts/run_libseeker_shard.sh --shard-index 0 -j 16 --skip-fetch  # PC1, gcc-11
Dataset/scripts/run_libseeker_shard.sh --shard-index 1 -j 16 --skip-fetch  # PC2, gcc-13
Dataset/scripts/run_libseeker_shard.sh --shard-index 2 -j 16 --skip-fetch  # PC3, clang-14
Dataset/scripts/run_libseeker_shard.sh --shard-index 3 -j 16 --skip-fetch  # PC4, clang-18
```

Lo script canonico costruisce o riprende gli 876 ELF del compilatore selezionato,
crea la cache ELF, indicizza l'intera matrice delle librerie e valida tutto prima del
matching. Per fermarsi dopo la costruzione e validazione della cache aggiungere
`--skip-matching`. `--skip-cache` ha invece il significato opposto: salta il
prebuild e lascia che il matching riempia eventuali entry mancanti.

Il builder usa solo gli archivi con `status=selected` nella matrice corrente,
crea un indice per archivio e usa directory temporanee eliminate dopo ogni
archivio. Può essere ripreso perché entry e indici completi sono
content-addressed e non vengono ricalcolati.

## Conversione opzionale di una cache legacy

Questa procedura non fa parte della generazione normale dei quattro shard. Va
usata soltanto quando si importa una cache pickle preesistente, indicando
esplicitamente il manifest dello shard a cui appartengono gli ELF:

```bash
SHARD_ROOT=Dataset/shards/libseeker-shard-0
CACHE_DIR="$SHARD_ROOT/cache"

.venv/bin/python thesis_code/convert_analysis_cache.py \
  --cache-dir "$CACHE_DIR" --types CU --delete-pickle

.venv/bin/python thesis_code/prune_analysis_cache.py \
  --cache-dir "$CACHE_DIR" \
  --dataset-manifest "$SHARD_ROOT/datasets/libseeker/manifest.csv" \
  --apply

.venv/bin/python thesis_code/convert_analysis_cache.py \
  --cache-dir "$CACHE_DIR" --types ELF \
  --binary-sha-manifest "$SHARD_ROOT/datasets/libseeker/manifest.csv" \
  --delete-pickle
```

La cancellazione dei pickle avviene soltanto dopo la rilettura mmap e il
confronto di tutte le feature usate dal matching.

Per costruire la cache delle librerie con `N` processi paralleli, usare
`--shard-count N`, un diverso `--shard-index` tra `0` e `N-1` e un
`--provenance-path "$CACHE_DIR/library_provenance.shardN.jsonl"` distinto per
ogni processo. Per dividere o riprendere una coda si possono usare anche
`--start-position` e `--stop-position` (posizioni inclusive e basate da 1); la
provenienza può essere unita anche se proviene da intervalli diversi. Al
termine:

```bash
.venv/bin/python thesis_code/merge_library_provenance.py \
  --cache-dir "$CACHE_DIR"
.venv/bin/python thesis_code/build_experiment_manifest.py \
  --dataset-root "$SHARD_ROOT" --cache-dir "$CACHE_DIR"
.venv/bin/python thesis_code/prune_unreferenced_cache.py \
  --cache-dir "$CACHE_DIR" \
  --dataset-manifest "$SHARD_ROOT/datasets/libseeker/manifest.csv"
.venv/bin/python thesis_code/prune_unreferenced_cache.py \
  --cache-dir "$CACHE_DIR" \
  --dataset-manifest "$SHARD_ROOT/datasets/libseeker/manifest.csv" --apply
.venv/bin/python thesis_code/audit_analysis_cache.py \
  --cache-dir "$CACHE_DIR" \
  --dataset-manifest "$SHARD_ROOT/datasets/libseeker/manifest.csv"
```

La prima potatura è una dry-run. Quella con `--apply` è consentita soltanto se
la provenienza copre l'intera matrice; elimina entry non raggiungibili dal
manifest ELF finale o dagli indici degli archivi selezionati.

## Matrice LIVA completa su quattro PC

La matrice corrente richiesta è di 3.504 record ELF: 219 nomi per quattro
ottimizzazioni e quattro compilatori. Gli alias byte-identici restano record
distinti, ma condividono naturalmente la stessa entry cache content-addressed.

La suddivisione ufficiale assegna un compilatore a ciascun PC. Ogni script
genera 876 record ELF e analizza tutti gli ELF locali, ma indicizza l'intero
catalogo degli archivi selezionati. In questo modo ciascun PC può cercare i suoi
876 ELF contro tutte le librerie; complessivamente vengono eseguite le ricerche
di tutti i 3.504 ELF contro il catalogo completo:

```bash
Dataset/scripts/run_libseeker_shard.sh --shard-index 0 -j 16  # gcc-11
Dataset/scripts/run_libseeker_shard.sh --shard-index 1 -j 16  # gcc-13
Dataset/scripts/run_libseeker_shard.sh --shard-index 2 -j 16  # clang-14
Dataset/scripts/run_libseeker_shard.sh --shard-index 3 -j 16  # clang-18
```

In una rigenerazione distribuita, i quattro PC devono usare lo stesso seed,
`library_matrix.tsv` e la stessa directory di librerie preparata, e costruiscono
localmente le rispettive cache. Nella modalità operativa adottata per PC2-PC4,
invece, ogni cache già validata viene trasferita una volta nel relativo
pacchetto autonomo. In entrambi i casi le cache restano separate: non vengono
unite o deduplicate tra PC. Dopo il matching si raccolgono e si aggregano
esclusivamente i quattro log gzip dei risultati. La procedura completa e il
merge sono descritti in `Dataset/SHARDED_LIBSEEKER.md`.

## Informazioni per gli esperimenti

- `<shard-root>/datasets/libseeker/manifest.json` descrive programma, versione,
  compilatore, ottimizzazione, hash ELF e configurazione di link.
- `<shard-root>/ground_truth/libseeker` conserva linker map e ground truth per la
  detection delle compilation unit e il function mapping.
- `<shard-root>/cache/experiment_elf_provenance.jsonl` collega ogni ELF alla sua
  cache e agli hash della ground truth.
- `<shard-root>/cache/library_provenance.jsonl` collega ogni archivio a package,
  ruolo di versione, compilatore, ottimizzazione e indice dei membri.
- Gli indici sotto `<shard-root>/cache/archives` conservano nome/occorrenza del
  membro, hash del file oggetto e chiave della cache CU.
- `<shard-root>/cache/shard_inventory.json` è il controllo ufficiale di
  completezza dello shard.
- `cache_inventory.json`, `cache_history_summary.json`, `conversion.jsonl` e i
  report `pruned_*.jsonl` esistono soltanto quando sono stati eseguiti gli
  strumenti opzionali di audit, conversione o potatura.

La pipeline di matching carica ora una sola libreria alla volta: ciò evita di
mantenere contemporaneamente in RAM tutte le CU e tutti i mapping delle
varianti selezionate dalla matrice corrente.

## Misura di memoria

Il benchmark riproducibile è in `Dataset/cache/mmap_benchmark.json`. Sulla entry
ELF più grande (20.997 funzioni e 204.431 blocchi), il picco RSS è sceso da
844.692 KiB con pickle a 275.088 KiB con NumPy mmap (-67,43%, circa 3,07 volte
meno). La entry occupa il 31,31% in meno. La ricostruzione del grafo di oggetti
Python rende il caricamento freddo della singola entry più lento; il guadagno
end-to-end deriva soprattutto dal saltare estrazione, analisi `radare2` e
generazione degli embedding, oltre che dallo streaming di un archivio alla
volta.

La misura della precedente cache legacy a 1.272 ELF occupava 39.694.485.190 byte
(36,97 GiB di contenuto; `du -sh`
mostra 41G per l'allocazione del filesystem): 10,02 GiB di CU e 26,87 GiB di
ELF, oltre a indici e report. È un riferimento storico e non una previsione
della nuova matrice a 3.504 record. Le 1.272 entry ELF legacy sono state rimosse
dalla cache di lavoro perché non condividevano alcun hash con PC1; la misura
rimane qui soltanto come riferimento riproducibile.
