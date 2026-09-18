# Etichette per il consumo di suolo: che cosa è disponibile, e come usarlo

Ricerca sulle fonti di *ground truth* annuale per addestrare e validare
`gnn-sar` sulle tre aree pilota (Roma, Viterbo, Milano), con la traduzione
operativa in configurazione del progetto.

> **Come è stata fatta questa ricerca, e che cosa significa per l'affidabilità.**
> L'ambiente in cui è stata svolta **non può aprire i siti istituzionali**
> italiani (`isprambiente.gov.it`, `groupware.sinanet.isprambiente.it`,
> `geodati.gov.it`, `snpambiente.it`: tutti bloccati dalla policy di rete in
> uscita). Quello che segue è quindi ricostruito da **risultati di motore di
> ricerca** e da pagine di terze parti, non dalla lettura diretta delle schede
> di metadato. Ogni affermazione è marcata con il suo grado di certezza:
> **[verificato]** = coerente su più fonti indipendenti; **[da confermare]** =
> riportato da una sola fonte o da uno snippet; **[ignoto]** = non sono
> riuscito a stabilirlo. Le due cose che vanno controllate di persona prima di
> qualunque uso — la **licenza** dell'edizione scaricata e la **codifica dei
> pixel** — sono entrambe segnate sotto, e la seconda è ora protetta anche dal
> codice (§5).

---

## 1. Riepilogo operativo

| Fonte | Copertura temporale | Risoluzione | Adatta a | Priorità |
| --- | --- | --- | --- | --- |
| **Carta nazionale del consumo di suolo ISPRA/SNPA** | 2006, 2012, e **ogni anno dal 2015** [verificato] | 10 m [verificato] | supervisione e validazione: è il bersaglio dichiarato del progetto | **primaria** |
| **Copernicus HRL Impervious Built-Up** | 2018, 2021 (triennale) [verificato] | 10 m [verificato] | prova preliminare, transfer, controllo incrociato | ripiego |
| **Carta nazionale di copertura del suolo ISPRA** | 2012, 2018, 2021, 2022 [da confermare] | 10 m [verificato] | contesto (che cosa c'era prima), non transizioni | ausiliaria |
| **Cartografie regionali** (Lazio, Lombardia) | variabile | variabile | maschere ausiliarie, uso del suolo di riferimento | ausiliaria |

Per il progetto servono **almeno due anni consecutivi** della stessa fonte: uno
fa da baseline e le transizioni si ricavano dalla differenza fra stati
consecutivi (`annual_land_take_labels`). Con la serie annuale ISPRA dal 2015 le
tre aree pilota hanno quindi fino a **nove transizioni** supervisionabili
(2016→2024), che è molto più di quanto serva a un primo *feasibility gate*.

## 2. La carta nazionale del consumo di suolo (ISPRA/SNPA)

### 2.1 Che cos'è

Cartografia raster del **suolo consumato**, prodotta annualmente da ISPRA con
il Sistema Nazionale per la Protezione dell'Ambiente. È il dato su cui si basa
il rapporto annuale *«Consumo di suolo, dinamiche territoriali e servizi
ecosistemici»*.

- **risoluzione 10 m**, griglia regolare 10 × 10 m [verificato];
- **serie storica**: 2006, 2012 e tutti gli anni dal 2015 in poi [verificato];
- prodotta per **fotointerpretazione** su immagini satellitari e ortofoto
  [verificato];
- dal 2017 il monitoraggio è su base esclusivamente cartografica, grazie alla
  disponibilità crescente di immagini Copernicus [da confermare].

Un dettaglio che conta molto per questo progetto: ARPA Lombardia descrive il
flusso di lavoro dicendo che **ISPRA fornisce alle agenzie regionali le immagini
Sentinel-1 e Sentinel-2 a 10 m insieme a maschere ottenute da classificazione
automatica e semi-automatica** delle aree di probabile cambiamento, che gli
operatori regionali poi fotointerpretano [da confermare, fonte: pagina «Consumo
Suolo» di ARPA Lombardia]. È la conferma diretta della premessa della proposta
di dottorato: il SAR **non** è una novità da introdurre: è già nella catena, e
il margine del progetto è altrove (granularità temporale e riduzione del carico
di fotointerpretazione).

### 2.2 Dove si scarica

| canale | indirizzo | note |
| --- | --- | --- |
| pagina dati ISPRA | <https://www.isprambiente.gov.it/it/attivita/suolo-e-territorio/suolo/il-consumo-di-suolo/i-dati-sul-consumo-di-suolo> | punto d'ingresso ufficiale; rimanda al repository |
| SINAnet Groupware (repository) | <https://groupware.sinanet.isprambiente.it/uso-copertura-e-consumo-di-suolo/library/consumo-di-suolo> | è qui che stanno i file scaricabili [verificato come esistente, contenuto non ispezionato] |
| Geoportale ISPRA | <http://geoportale.isprambiente.it/> | consultazione e servizi OGC |
| servizio WMS | esiste un endpoint OGC:WMS per la carta del consumo di suolo (es. edizione 2024) [da confermare] | utile per controllo visivo in QGIS, **non** per addestrare: serve il raster |
| RNDT / INSPIRE | <https://geodati.gov.it>, <https://inspire-geoportal.ec.europa.eu> | schede di metadato con vincoli d'uso e link di distribuzione |
| contatto | `consumosuolo@isprambiente.it` | indicato nei metadati per richieste specifiche |

Il repository distribuisce, oltre alla cartografia, gli **indicatori** (suolo
consumato, consumo di suolo, consumo netto) per ogni anno e per ogni livello
amministrativo, in shapefile e formato tabellare [verificato]. Quelli non
servono all'addestramento, ma sono il modo più rapido per **verificare a
posteriori** che le etichette estratte sull'area pilota siano plausibili:
confronta gli ettari consumati che ottieni nel tuo ritaglio con quelli
pubblicati per gli stessi comuni.

**[ignoto]** Non sono riuscito a stabilire: se i raster siano distribuiti per
regione o solo come mosaico nazionale, in che formato esatto (GeoTIFF è
probabile ma non confermato), il sistema di riferimento nativo, il valore di
nodata e i nomi dei file. Sono tutte cose che si leggono in due minuti aprendo
il repository.

### 2.3 Licenza

Le fonti consultate riportano che i dati sul consumo di suolo e gli indicatori
derivati sono rilasciati in **formato aperto e liberamente accessibili con
licenza CC BY 3.0 IT** [da confermare, riportato da più snippet ma non letto
nella scheda di metadato].

Se confermato, significa che i dati si possono usare **e ridistribuire** con
attribuzione — il che cambia una cosa nella pianificazione del progetto: un
dataset derivato (le etichette per nodo dell'area pilota) sarebbe pubblicabile
insieme ai risultati, rendendo l'esperimento riproducibile da terzi. **Prima di
pubblicare qualsiasi cosa, leggi i vincoli d'uso nella scheda di metadato
dell'edizione che hai scaricato**: la licenza può cambiare fra edizioni, e
l'affermazione qui sopra non è stata verificata alla fonte.

Attribuzione suggerita, da adattare a quanto dice la scheda:

> Elaborazione su cartografia del consumo di suolo ISPRA/SNPA, edizione <anno>.

### 2.4 La codifica dei pixel: il punto da non sbagliare

Il sistema di classificazione ha un **livello primario a due classi**
[da confermare, ma coerente con più fonti]:

```
1 = suolo consumato
2 = suolo NON consumato
```

con livelli secondari e terziari dove possibile, che distinguono il consumo
**permanente** da quello **reversibile** (edifici e strade da una parte,
cantieri, piazzali non pavimentati, impianti fotovoltaici a terra dall'altra).

> ⚠️ **Questo rende pericoloso il default «qualsiasi valore ≠ 0 è sigillato».**
> Con la codifica ISPRA, `2` significa *non consumato*: trattarlo come non-zero
> etichetterebbe come consumato **tutto il territorio**. Con un raster ISPRA va
> quindi sempre dichiarato esplicitamente quali valori contano:
>
> ```json
> "labels": { "rasters": {...}, "sealed_values": [1] }
> ```
>
> e, se il file usa i codici di secondo livello, l'elenco completo dei codici che
> iniziano per 1 (per esempio `[11, 12]`), scelto in base a che cosa si vuole
> misurare — vedi sotto.
>
> Dalla versione corrente **l'ingestione si rifiuta di indovinare**: se il raster
> contiene valori diversi da 0/1 e `sealed_values` non è stato dichiarato,
> `gnn-sar ingest` si ferma e stampa i valori trovati (§5).

**Permanente o anche reversibile?** È una decisione di ricerca, non un
dettaglio tecnico:

- includere **solo il consumo permanente** dà un bersaglio più netto e più
  vicino all'irreversibilità che il modello assume per costruzione (il noisy-OR
  cumulato non torna indietro);
- includere **anche il reversibile** avvicina il bersaglio a ciò che il SAR
  vede davvero durante un cantiere, ma introduce transizioni che in un anno
  successivo possono tornare indietro — cioè esattamente ciò che la
  parametrizzazione del modello non può rappresentare. Se le includi,
  aspettati che i ritorni indietro compaiano come falsi positivi permanenti, e
  considera di trattarli come `-1` invece che come `0`.

Il fotovoltaico a terra, citato dalla proposta come *driver* emergente nel
Lazio e in particolare nel viterbese, ricade nel consumo reversibile: se è
quello il fenomeno che interessa, va incluso, con la cautela di cui sopra.

## 3. Il ripiego aperto: Copernicus HRL Impervious Built-Up

Quando serve qualcosa **subito**, senza attendere il download ISPRA o la
verifica della licenza:

- **Impervious Built-Up (IBU)**: raster **binario**, `1` = costruito, `0` =
  non costruito, derivato da serie temporali Sentinel-2 [verificato];
- **10 m**, anni di riferimento **2018 e 2021**, cadenza triennale
  [verificato]; esiste anche il layer di **cambiamento 2018–2021** a 20 m;
- distribuito per tessere da 100 × 100 km per i paesi EEA38 e come mosaico
  [verificato]; scaricabile dal Copernicus Land Monitoring Service
  (<https://land.copernicus.eu/en/products/high-resolution-layer-imperviousness>)
  e da WEkEO;
- esiste anche **Imperviousness Density** (grado di impermeabilizzazione 0–100),
  utile come soglia continua invece che binaria.

Limiti per questo progetto: **cadenza triennale**, quindi una sola transizione
2018→2021 e nessuna possibilità di valutare il ritardo di detection su base
annuale; ed è un prodotto *built-up*, semanticamente più stretto del «consumo di
suolo» italiano (che include piazzali, cave, cantieri). Va benissimo per far
girare la pipeline end-to-end su dati veri mentre si sistema il resto.

Essendo binario, qui il default va bene:

```json
"labels": { "rasters": {"2018": "ibu_2018.tif", "2021": "ibu_2021.tif"} }
```

ma con soli due anni distanti tre anni l'etichetta di transizione sarà
attribuita al 2021, mentre il cambiamento può essere avvenuto nel 2019 o nel
2020: l'esperimento misura poco. Usalo per il collaudo, non per le conclusioni.

## 4. Fonti regionali per le aree pilota

Il monitoraggio è a rete: ISPRA coordina, le agenzie regionali
fotointerpretano. Per le tre aree pilota:

| area | agenzia / portale | che cosa ci si trova |
| --- | --- | --- |
| Roma, Viterbo (Lazio) | **ARPA Lazio – SIRA**, <https://sira.arpalazio.it/home/suolo/uso-del-suolo>; **Geoportale Regione Lazio**, <https://geoportale.regione.lazio.it/> | dati sulle variazioni d'uso del suolo con frequenza annuale nell'ambito del monitoraggio nazionale [da confermare]; Carta dell'Uso del Suolo regionale 1:25.000 (classificazione Corine), citata dalla proposta come riferimento ausiliario |
| Milano (Lombardia) | **ARPA Lombardia**, <https://www.arpalombardia.it/temi-ambientali/earth-observation/consumo-suolo/>; **DUSAF** (ERSAF/Regione Lombardia) | monitoraggio annuale dell'intero territorio regionale secondo le direttive ISPRA [da confermare]; DUSAF è la carta di uso e copertura del suolo regionale, utile come strato ausiliario |

Le cartografie regionali di *uso del suolo* (Corine, DUSAF) **non** sostituiscono
la carta del consumo di suolo: hanno legenda, scala e cadenza diverse. Servono
semmai a due cose utili: escludere a priori classi dove il consumo è
implausibile, e dare un contesto di copertura ai nodi.

## 5. Che cosa cambia nel codice

Da questa ricerca sono derivate due modifiche concrete al progetto:

1. **L'ingestione non indovina più la codifica delle etichette.** Se un raster
   annuale contiene valori diversi da `0`/`1` (più il nodata) e `sealed_values`
   non è stato dichiarato, `gnn-sar ingest` si ferma con un messaggio che elenca
   i valori trovati e ricorda la convenzione ISPRA (`1` consumato, `2` non
   consumato). Il caso che avrebbe etichettato l'Italia intera come consumata
   non può più passare in silenzio.
2. **Le configurazioni delle aree pilota** (`docs/esempi/aree/*.json`) hanno già
   `"sealed_values": [1]` e un commento che spiega perché.

Per il resto, la conversione da **stato** a **transizione** era già quella
giusta per questi dati: `1` solo nell'anno del cambiamento, `0` prima e dopo,
`-1` dove un buco nella serie potrebbe nascondere la transizione, e suolo già
consumato all'inizio della serie escluso. Con la serie ISPRA annuale dal 2015
i buchi non ci sono, quindi quel meccanismo resterà quasi sempre inattivo: serve
soprattutto se si parte dal 2006/2012, dove il salto temporale è grande.

## 6. Da fare, in ordine

1. Aprire il repository SINAnet e annotare, per l'edizione più recente:
   formato, CRS, nodata, codifica esatta dei pixel, ritaglio disponibile
   (nazionale o regionale). Mettere le risposte in questo documento, sostituendo
   i **[da confermare]** e i **[ignoto]**.
2. Leggere i vincoli d'uso nella scheda di metadato e fissare la licenza.
3. Scaricare gli anni che coprono il periodo Sentinel scelto **più l'anno
   precedente** (la baseline).
4. Ritagliare ogni anno sulla griglia dell'area pilota (§6 della
   [guida](guida_dati_sentinel.md)) e verificare con `gdalinfo -hist` quali
   valori compaiono davvero: è il controllo che smaschera una codifica diversa
   dall'attesa.
5. Confrontare gli ettari consumati per comune con gli indicatori pubblicati,
   prima di fidarsi delle etichette.

## 7. Fonti consultate

Tutte via motore di ricerca, nessuna aperta direttamente (vedi la nota in cima).

- ISPRA, *I dati sul consumo di suolo* — <https://www.isprambiente.gov.it/it/attivita/suolo-e-territorio/suolo/il-consumo-di-suolo/i-dati-sul-consumo-di-suolo>
- ISPRA, *La «misura» del consumo di suolo* — <https://www.isprambiente.gov.it/it/attivita/suolo-e-territorio/suolo/il-consumo-di-suolo/la-misura-del-consumo-di-suolo>
- ISPRA, *Carta nazionale di copertura del suolo* — <https://www.isprambiente.gov.it/it/attivita/suolo-e-territorio/suolo/copertura-del-suolo/carta-nazionale-di-copertura-del-suolo>
- SINAnet Groupware, libreria *Uso, copertura e consumo di suolo* — <https://groupware.sinanet.isprambiente.it/uso-copertura-e-consumo-di-suolo/library/consumo-di-suolo>
- SNPA, *Linee Guida per il monitoraggio del consumo di suolo* (n. 48/2023 e n. 50/2024) — <https://www.snpambiente.it/category/pubblicazioni/linee-guida-snpa/>
- ARPA Lombardia, *Earth observation — Consumo suolo* — <https://www.arpalombardia.it/temi-ambientali/earth-observation/consumo-suolo/>
- ARPA Lazio, *Uso del suolo* — <https://sira.arpalazio.it/home/suolo/uso-del-suolo>
- Geoportale Regione Lazio — <https://geoportale.regione.lazio.it/>
- Copernicus Land Monitoring Service, *HRL Imperviousness* — <https://land.copernicus.eu/en/products/high-resolution-layer-imperviousness>
- INSPIRE Geoportal / RNDT, schede `ispra_rm:*` — <https://geodati.gov.it>
