# LibSeeker completo e cache su quattro PC

## Risultato

Il dataset finale contiene esattamente 3.504 coordinate:

```text
219 programmi × 4 compilatori × 4 ottimizzazioni = 3.504 ELF
```

Ogni shard contiene un compilatore completo, quindi esattamente 876 record ELF
e nessuna coordinata si sovrappone agli altri shard. Gli alias rimangono nel
manifest; se due file sono byte-identici possono condividere una singola entry
della cache, identificata dal contenuto.

Ogni ELF usa il normale linking statico: vengono selezionate versioni bilanciate
delle dipendenze richieste dal programma e il linker incorpora soltanto gli
object necessari a risolverne i simboli. Non vengono forzate librerie estranee.
Questo è distinto dal matching: ciascuno dei 3.504 ELF viene cercato contro
l'intero catalogo delle librerie. Ogni PC elabora 876 ELF, ma deve quindi avere
la cache completa degli archivi selezionati.

## Modalità operativa adottata

PC2, PC3 e PC4 ricevono ciascuno il proprio archivio completo sotto
`Dataset/exports`: non devono clonare il repository e non devono ricostruire la
cache. Dopo la verifica SHA-256, l'archivio viene estratto e il matching viene
eseguito dalla directory `LIVA-PCN` seguendo il `README.md` incluso. Le sezioni
seguenti documentano la rigenerazione degli input e degli shard, non la
procedura necessaria sui PC di matching.

Il launcher portabile abilita sempre `--analysis-cache-only`: ricava l'identità
di analisi dal manifest incluso, non interroga il `radare2` installato sul PC,
non scrive nella cache e interrompe l'esecuzione al primo miss. Il binario
`radare2` serve per rigenerare la cache, non per eseguire questi pacchetti.

Le librerie sono scelte con il seed del dataset. Il bilanciamento avviene prima
tra `current`, `minor-alternative` e `major-alternative`, dove disponibili, poi
tra le varianti di toolchain e ottimizzazione compatibili. I ruoli sono
assegnati esplicitamente per pacchetto in
`Dataset/manifests/library_version_roles.json`: la variante minor è la versione
precedente più vicina, preferibilmente nella stessa major line; la variante
major è una versione più distante, preferibilmente di una major line
precedente, oppure la più distante disponibile quando il corpus non contiene
un cambio di major. Il peso usato è il numero di ELF realmente emesso dal
progetto, non semplicemente il numero di build.

## 1. Rigenerazione: preparazione comune delle librerie

Eseguire una volta:

```bash
Dataset/prepare_libseeker_libraries.sh -j 16
```

Lo script costruisce le tre versioni, genera
`Dataset/manifests/library_matrix.tsv`, conserva i 164 archivi di riferimento e
i cinque archivi pubblici necessari al relink degli ELF (`libiconv.a`,
`libidn2.a`, `libpsl.a`, `libunistring.a`, `libzstd.a`).

Per una rigenerazione distribuita senza usare i pacchetti completi, questi
input devono essere identici su tutti i PC:

- `Dataset/builds/libraries`
- `Dataset/manifests/library_matrix.tsv`
- `Dataset/manifests/library_version_roles.json`
- il repository e i sorgenti sotto `Dataset/sources`

Tutti i PC devono usare lo stesso seed, predefinito a `20260731`. In questo
flusso di rigenerazione la cache viene costruita localmente. Nel flusso
operativo adottato, invece, ogni archivio trasferisce al PC di destinazione lo
shard e la cache già completi e validati. Le cache dei diversi shard non devono
mai essere unite.

## 2. Rigenerazione degli shard

Eseguire un solo comando per PC:

```bash
# PC 1: gcc-11
Dataset/scripts/run_libseeker_shard.sh --shard-index 0 -j 16 --skip-fetch

# PC 2: gcc-13
Dataset/scripts/run_libseeker_shard.sh --shard-index 1 -j 16 --skip-fetch

# PC 3: clang-14
Dataset/scripts/run_libseeker_shard.sh --shard-index 2 -j 16 --skip-fetch

# PC 4: clang-18
Dataset/scripts/run_libseeker_shard.sh --shard-index 3 -j 16 --skip-fetch
```

Ogni comando può essere ripreso: le build verificano il fingerprint completo
delle librerie e la cache è content-addressed. `--clean` forza soltanto la
ricostruzione dello shard del PC. `--device cuda` o `--device cpu` seleziona il
device per gli embedding. `--matching-jobs auto`, che è il valore predefinito,
adatta il numero di ELF analizzati in parallelo alle CPU e alla RAM visibili;
un intero positivo forza invece manualmente la concorrenza. Se configure, la
compilazione o il linker segnalano
un'incompatibilità riconducibile alla versione di una libreria, la build
riprova in modo deterministico le altre versioni disponibili, dalla versione
corrente alle alternative. Ogni nuovo tentativo conserva i cambi di versione
già riusciti: può quindi correggere cumulativamente più librerie incompatibili
nello stesso ELF. Quando un fallback riesce, la sequenza completa dei tentativi
viene registrata in `build-info.json`. Gli errori non riconducibili alla
versione di una libreria non attivano questi retry. Ogni comando configura i 17
progetti sorgente nelle
quattro ottimizzazioni (68 directory di build), dalle quali vengono emessi i
219 programmi per ottimizzazione, cioè 876 ELF. Quindi indicizza l'intera
matrice delle librerie e cerca ognuno dei suoi 876 ELF contro tutte le librerie.

L'output predefinito è `Dataset/shards/libseeker-shard-N`. Al termine,
`cache/shard_inventory.json` deve contenere `"valid": true`. Il risultato
trasferibile è composto soltanto da:

- `results/libseeker/results.jsonl.gz`;
- `results/libseeker/results.summary.json`.

Il JSONL gzip contiene il catalogo delle librerie una sola volta, una decisione
per ogni coppia ELF–libreria, i dettagli delle CU accettate e il mapping tra
funzione della CU di riferimento e funzione dell'ELF. Per ciascun mapping
conserva nomi, indici, indirizzi, dimensioni, `dominant_ratio`,
`coverage_mean` e `coverage_ratio`. Le CU scartate e tutte le finestre candidate
restano nei file locali di replay, così il file da trasferire rimane compatto.

### Log completi per l'ablation study offline

Il normale log compatto non può ricostruire una CU eliminata da Hungarian,
dalla struttura o da `.rodata`. Per produrre log replay-complete e applicare
anche la soglia percentuale sulle funzioni della CU, rieseguire il matching con:

```bash
Dataset/scripts/run_libseeker_shard.sh \
  --shard-index 0 \
  --skip-fetch --skip-build --skip-cache \
  --offline-ablation-features \
  --cu-function-coverage 0.60
```

Con `--resume`, i report compatti vengono riconosciuti come incompatibili con
la modalità replay e sono ricalcolati; una successiva ripresa salta invece i
report replay-complete già terminati. Questa modalità conserva nei file
`reports/current/*.features.jsonl.gz` il fronte di Pareto delle finestre che
superano B, la `.rodata` grezza, i mapping di funzione e gli archi cross-CU.
Per le CU che superano B conserva una volta il catalogo delle funzioni di
riferimento (nome, indirizzo, dimensione, blocchi e istruzioni) e, per ogni
mapping, gli indici e le tre misure `dominant_ratio`, `coverage_mean` e
`coverage_ratio`. Il catalogo delle funzioni ELF e il relativo call graph sono
scritti una sola volta per ELF. Questo rende possibile valutare offline anche
la qualità dei mapping di funzione senza duplicare i descrittori in ogni
finestra. Per una CU che non supera B rimane soltanto il miglior indizio di
funzione, utile per diagnosticare un falso negativo precoce.
Gli archi salvati sono il sottografo indotto dalle sole CU che superano B:
poiché B è sempre attivo, gli altri archi non possono influire su alcuna
configurazione B/H/S/X/R. Non modifica le similarità PalmTree e continua a
riusare la cache.

Terminata la run, il fattoriale fisso B/H/S/X/R si esegue senza radare2 o
PalmTree:

```bash
.venv/bin/python thesis_code/experiments/offline_pipeline_ablation.py \
  --batch-output-dir Dataset/shards/libseeker-shard-0/results/libseeker
```

Il pannello predefinito contiene una sola build `current`, GCC 13/O2 per
famiglia, come nel confronto di presenza in stile LibSeeker. Per confrontare
tutte le versioni senza contare le alternative perdenti come falsi positivi,
aggiungere `--candidate-panel all-versions-top1`: viene emessa una sola
decisione per famiglia, scelta dal candidato col punteggio più alto.

In questa notazione S comprende la struttura interna alla CU, inclusa la
percentuale di funzioni coperte; X contiene esclusivamente l'evidenza cross-CU.

Il bundle `results.jsonl.gz` rimane deliberatamente compatto; per un'ablazione
multi-PC devono essere raccolti anche i file `.features.jsonl.gz`. La procedura
di packaging e merge di questi log verrà definita separatamente.

## 3. Raccolta e merge

Copiare sul PC di raccolta soltanto i due file di risultato di ciascun PC,
mantenendoli in quattro directory shard separate, quindi eseguire:

```bash
.venv/bin/python Dataset/scripts/merge_libseeker_shards.py
```

L'output predefinito è
`Dataset/merged/libseeker-results/results.jsonl.gz`, accompagnato da
`results.summary.json`. Il merge aggrega esclusivamente i risultati: non copia
dataset, ground truth o cache e non esegue alcuna deduplicazione della cache.
Lavora in streaming e verifica:

- 4 shard distinti da 876 ELF e la matrice completa 219 × 4 × 4;
- lo stesso catalogo e la stessa configurazione di matching sui quattro PC;
- un risultato per ogni ELF, libreria e pipeline;
- integrità SHA-256 dei quattro log;
- coerenza dei record CU e dei mapping di funzione, incluso il
  `dominant_ratio`.

Se i file sono stati raccolti in percorsi diversi, passarli esplicitamente:

```bash
.venv/bin/python Dataset/scripts/merge_libseeker_shards.py \
  --shard-result pc1/results.jsonl.gz \
  --shard-result pc2/results.jsonl.gz \
  --shard-result pc3/results.jsonl.gz \
  --shard-result pc4/results.jsonl.gz \
  --output Dataset/merged/libseeker-results/results.jsonl.gz
```

`--replace` sostituisce atomicamente soltanto il precedente log aggregato e il
suo riepilogo. La validazione dataset/cache rimane locale a ogni PC prima del
matching; al centro viene validata la completezza e l'integrità dei risultati.
