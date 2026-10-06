# Pacchetto esperimenti LibSeeker

Questo archivio contiene uno shard LibSeeker completo, autonomo e validato: non
è necessario clonare il repository. La directory radice indica il PC di
destinazione:

- PC2: shard 1, GCC 13;
- PC3: shard 2, Clang 14;
- PC4: shard 3, Clang 18.

Il pacchetto comprende i 876 ELF del PC, il relativo ground truth, l'intera
cache mmap ELF/CU, gli indici dei 4.472 archivi di libreria, tutti i 4.472 file
`.a` selezionati, i manifest e il codice necessario al matching. Sono esclusi
soltanto gli intermedi di compilazione, i log, i lock della cache e `.venv`.
I `build-info.json` sono comunque inclusi per conservare la provenienza delle
build.

## Estrazione e verifica

```bash
sha256sum -c libseeker-PCN-COMPILER-experiments.tar.zst.sha256
tar --use-compress-program=unzstd -xf libseeker-PCN-COMPILER-experiments.tar.zst
cd LIVA-PCN
```

Il file `Dataset/shards/libseeker-shard-N/cache/shard_inventory.json` deve
riportare `"valid": true`, 876 ELF e 4.472 archivi di libreria.

## Esecuzione del matching

Sostituire `N` con 1, 2 o 3 in base al PC. L'ambiente Python deve fornire i
pacchetti elencati in `python-environment.txt` ed è richiesto Python 3.12. Il
launcher usa obbligatoriamente la cache in modalità sola lettura: recupera
l'identità di analisi dal manifest incluso, non esegue `r2 -v` e termina al
primo cache miss. Il binario `radare2`, PalmTree in esecuzione e una GPU non
sono quindi necessari per il matching del pacchetto validato. `radare2` resta
necessario soltanto per rigenerare o estendere la cache. Il file delle
dipendenze deve rimanere insieme al pacchetto, ma non richiede un repository
Git.

Il launcher adatta il parallelismo alle risorse visibili sul PC. Usa al massimo
un processo per CPU, riserva il 20% della RAM disponibile e considera 1,5 GiB
per processo. I thread numerici vengono ripartiti tra i processi per evitare
oversubscription.

```bash
Dataset/scripts/run_packaged_matching.sh
```

L'output iniziale deve contenere
`Analysis cache: cache-only (read-only; radare2 disabled)`. Un errore di cache
non attiva mai una rigenerazione silenziosa.

Gli archivi creati prima di questo launcher non contengono il file: è
sufficiente copiarvi `Dataset/scripts/run_packaged_matching.sh` dopo
l'estrazione, senza ricreare l'archivio. Per forzare un valore o un device:

```bash
LIVA_MATCHING_JOBS=8 LIVA_MATCHING_DEVICE=cuda \
  Dataset/scripts/run_packaged_matching.sh
```

Per aggiornare in blocco un pacchetto già estratto alla modalità cache-only,
trasferire `libseeker-cache-only-runtime-update.tar.zst`, entrare nella sua
directory `LIVA-PCN` ed eseguire:

```bash
tar --use-compress-program=unzstd -xf \
  ../libseeker-cache-only-runtime-update.tar.zst
```

L'aggiornamento sostituisce anche `thesis_code/numpy_cache.py`; questa versione
mantiene mmap solo per gli embedding e non esaurisce i file descriptor durante
il caricamento di archivi grandi come `libX11.a`.

La cache del pacchetto è completa e validata, quindi normalmente PalmTree non
deve calcolare nuovi embedding e la VRAM non limita il numero di job. Se si
osservano memoria inutilizzata e storage non saturo, il valore può essere
aumentato manualmente; in caso di pressione sulla RAM va invece ridotto.
