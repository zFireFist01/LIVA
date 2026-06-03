# Report 

#### Matching a blocchi

La novita' piu' importante e' l'introduzione del `block_matching`.

Prima la pipeline decideva se una CU della libreria fosse presente nel binario usando principalmente il punteggio delle funzioni abbinate. Ora, invece, le CU candidate possono essere rivalutate confrontando direttamente gli embedding dei basic block.

Il confronto viene fatto costruendo una matrice di cosine_similarity tra i blocchi:

```text
blocchi della CU candidata
contro
blocchi del binario analizzato
```

Per ogni blocco viene calcolata la migliore corrispondenza possibile. Da questo vengono ricavate piu' metriche:

| Metrica | Significato |
| --- | --- |
| `coverage_mean` | Similarita' media dei migliori match tra blocchi |
| `coverage_min` | Similarita' minima osservata |
| `coverage_ratio` | Percentuale di blocchi sopra soglia |
| `assignment_mean` | Media del matching 1:1 tramite Hungarian |
| `assignment_min` | Valore minimo nel matching 1:1 |

Il risultato finale non dipende quindi solo da un singolo score medio, ma da una combinazione di copertura, qualita' dell'assegnamento e vincoli strutturali.


#### Due modalita' di block matching

Sono state introdotte due modalita':

| Modalita' | Descrizione |
| --- | --- |
| `function_candidates` | Applica il block matching solo alle CU gia' selezionate dal function matching |
| `all_cu` | Valuta tutte le CU eleggibili della libreria, anche se il function matching non le aveva selezionate |

La modalita' `all_cu` e' la piu' interessante per il recupero dei casi difficili. In questo caso la CU della libreria viene cercata dentro il binario usando una finestra locale di funzioni sorgente, invece di permettere match arbitrari su tutto il binario.

Questo evita che una piccola CU possa ottenere un punteggio alto prendendo blocchi simili sparsi in punti lontani e non collegati del programma.


#### Recovery band

Il commit introduce anche una fascia di recupero per i candidati.

Prima una CU sotto la soglia del function matching veniva scartata direttamente. Ora esistono due soglie:

| Parametro | Ruolo |
| --- | --- |
| `function_threshold` | Soglia principale del function matching |
| `candidate_low_threshold` | Soglia piu' bassa per mantenere candidati da verificare con block matching |

In questo modo una CU con score funzionale non abbastanza alto, ma comunque promettente, puo' essere rivalutata al livello dei blocchi.

```text
CU sopra function_threshold
  -> candidato primario

CU tra candidate_low_threshold e function_threshold
  -> candidato di recovery

CU sotto candidate_low_threshold
  -> scartata, tranne in modalita' all_cu
```

Questa modifica e' utile nei casi in cui l'ottimizzazione rovina il matching tra funzioni, ma lascia ancora riconoscibili molte porzioni di codice al livello dei basic block.


#### Localita' del matching

Per ridurre i falsi positivi e' stato aggiunto un controllo di localita'.

L'idea e' che, se due funzioni della CU target sono vicine o collegate da chiamate interne, anche le funzioni sorgente in cui vengono trovati i blocchi corrispondenti dovrebbero rimanere ragionevolmente vicine nel binario.

Il parametro principale e':

| Parametro | Descrizione |
| --- | --- |
| `block_locality_window_multiplier` | Moltiplica la dimensione della CU target per definire la finestra locale |
| `block_locality_window_padding` | Aggiunge un margine extra alla finestra |
| `block_min_edge_locality_ratio` | Percentuale minima di archi interni che devono restare locali |

Questo permette di distinguere meglio un vero match da una somiglianza casuale distribuita nel binario.


#### Concentrazione e spread

Un'altra innovazione importante e' il controllo su come i blocchi target vengono distribuiti sulle funzioni sorgente.

Sono state aggiunte due metriche:

| Metrica | Significato |
| --- | --- |
| `function_concentration` | Misura quanto i blocchi di una funzione target finiscono nella stessa funzione sorgente dominante |
| `function_spread` | Misura se funzioni target diverse finiscono in funzioni sorgente diverse |

Queste metriche servono a evitare due casi problematici:

```text
1) I blocchi di una funzione target vengono sparpagliati su troppe funzioni sorgente.
2) Molte funzioni target collassano tutte sulla stessa funzione sorgente.
```

Il primo caso indica un match poco coerente. Il secondo puo' generare falsi positivi quando tante piccole funzioni o blocchi comuni sembrano simili alla stessa regione del binario.


#### Miglioramento del function matching

Anche il matching tra funzioni e' stato aggiornato.

Oltre alla similarita' degli embedding e al bonus di call graph, ora viene considerato anche un punteggio di localita' tra funzioni.

La nuova similarita' combinata e':

```text
similarity =
    base_similarity
  + call_graph_bonus
  + function_locality_bonus
```

Il bonus di localita' premia gli assegnamenti in cui le chiamate interne della CU target vengono mappate su funzioni sorgente vicine tra loro. Questo rende piu' stabile la scelta della finestra candidata nel binario.



#### Limite ancora aperto

Nel file `Update.txt` e' stata annotata una criticita' importante:

```text
Il vincolo Hungarian 1:1 puo' penalizzare casi con ottimizzazioni, inlining o clonazione.
```

Il problema e' che in presenza di ottimizzazioni piu' blocchi target potrebbero corrispondere alla stessa regione sorgente, oppure lo stesso blocco logico potrebbe essere duplicato dal compilatore.

In questi casi un assegnamento strettamente 1:1 rischia di abbassare artificialmente `assignment_mean`.

Una possibile direzione futura e' valutare metriche:

| Alternativa | Idea |
| --- | --- |
| `many-to-one coverage` | Piu' blocchi possono matchare la stessa regione |
| `top-k coverage` | Ogni blocco considera i migliori k candidati |
| soglie diverse per `assignment_mean` | Mantenere Hungarian solo come controllo secondario |
