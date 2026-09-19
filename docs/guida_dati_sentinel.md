# Guida: usare `gnn-sar` con dati Sentinel-1 e Sentinel-2 reali

Questa guida copre tre cose, nell'ordine in cui servono:

1. **come si usa il repository** — installazione, demo sintetica, API, test;
2. **dove e come si scaricano i dati Sentinel-1 e Sentinel-2** dall'ESA
   (Copernicus Data Space Ecosystem) e come si pre-processano;
3. **come si integrano quei dati nel progetto** — dal GeoTIFF al grafo, dal
   grafo alle mappe sub-annuali georeferenziate.

> **Stato del progetto.** Prima di questa guida il repository non leggeva dati
> reali: c'erano il modello, le loss, le metriche e un generatore sintetico, ma
> nessun ponte fra l'archivio Copernicus e il tensore `(T, N, D)` che il modello
> consuma. Quel ponte ora c'è (moduli `gnn_sar.data.raster`,
> `gnn_sar.data.sentinel`, `gnn_sar.data.labels`, `gnn_sar.data.study_area`,
> `gnn_sar.data.ingest`, `gnn_sar.data.cdse` e i comandi `gnn-sar search |
> ingest | train`). Quello che **resta fuori** è elencato nella sezione 13: in
> particolare il pre-processing SAR vero e proprio resta in SNAP, e nessuna
> validazione su dati reali è stata ancora eseguita.

**Documenti collegati**

- [`dati_etichette_ispra.md`](dati_etichette_ispra.md) — dove prendere le
  etichette annuali, con quale licenza e, soprattutto, come sono codificati i
  pixel (`1` = consumato, `2` = **non** consumato: sbagliarlo etichetta tutto).
- [`roadmap.md`](roadmap.md) — che cosa manca ancora, in che ordine, e quanto
  costa. Ci sta anche il piano per la coerenza InSAR.

**Indice**

1. [Che cos'è questo repository](#1-che-cosè-questo-repository)
2. [Installazione](#2-installazione)
3. [Usare il repository senza dati](#3-usare-il-repository-senza-dati)
4. [Quali dati servono](#4-quali-dati-servono)
5. [Dove scaricarli: Copernicus Data Space Ecosystem](#5-dove-scaricarli-copernicus-data-space-ecosystem)
6. [Pre-processing Sentinel-1](#6-pre-processing-sentinel-1)
7. [Pre-processing Sentinel-2](#7-pre-processing-sentinel-2)
8. [Etichette ISPRA/SNPA e corridoi infrastrutturali](#8-etichette-ispra-snpa-e-corridoi-infrastrutturali)
9. [Integrazione nel progetto: `gnn-sar ingest`](#9-integrazione-nel-progetto-gnn-sar-ingest)
10. [Addestramento e mappe: `gnn-sar train`](#10-addestramento-e-mappe-gnn-sar-train)
11. [La stessa cosa dall'API Python](#11-la-stessa-cosa-dallapi-python)
12. [Scelte metodologiche da non sbagliare](#12-scelte-metodologiche-da-non-sbagliare)
13. [Che cosa manca ancora](#13-che-cosa-manca-ancora)
14. [Errori frequenti](#14-errori-frequenti)
15. [Licenze e citazioni dei dati](#15-licenze-e-citazioni-dei-dati)

---

## 1. Che cos'è questo repository

Una *graph neural network* spazio-temporale che rileva e anticipa la
sigillazione del suolo a **risoluzione sub-annuale**, addestrata con le sole
etichette **annuali** disponibili (la cartografia ISPRA/SNPA). Il territorio è
un grafo: i nodi sono unità di suolo stabili (super-pixel o particelle
catastali), gli archi l'adiacenza spaziale e la connettività infrastrutturale, e
le feature dei nodi variano nel tempo seguendo la serie Sentinel-1.

Il contesto di ricerca è in [`proposta_dottorato_strutturata_v2.md`](proposta_dottorato_strutturata_v2.md);
le scelte di implementazione sono in [`../ARCHITECTURE.md`](../ARCHITECTURE.md).
Tre proprietà del modello contano per chi porta dati:

- **i nodi sono fissi**, le feature no. La segmentazione si calcola **una volta
  sola** su un composito multitemporale (o su un layer catastale); ricalcolarla
  a ogni acquisizione renderebbe privi di senso gli archi temporali;
- **il tempo è causale**: la previsione al composito `t` dipende solo da `t` e
  dal suo passato;
- **l'irreversibilità è strutturale**: la testa di detection predice la
  probabilità di *onset* per composito, e lo stato sigillato è il loro noisy-OR
  cumulato — non decrescente per costruzione. È anche il meccanismo che permette
  a un'etichetta annuale di supervisionare predizioni mensili.

## 2. Installazione

```bash
git clone https://github.com/matteo-campana/gnn-sar.git
cd gnn-sar
poetry install --with dev --extras all      # oppure: --extras "geo download segmentation"
```

Su una macchina senza GPU conviene installare prima la wheel CPU di PyTorch, che
è pinnato a `2.6.0`:

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

MLflow è una dipendenza **core**, non un extra: ogni run viene registrato, e il
posto dove un risultato vive è il suo run, non il nome di una cartella.

Gli *extras* sono opzionali e servono solo alla parte dati:

| extra | pacchetto | serve per |
| --- | --- | --- |
| `geo` | `rasterio` | leggere i GeoTIFF e scrivere le mappe (`gnn-sar ingest`, `gnn-sar train --output-dir`) |
| `download` | `requests` | interrogare e scaricare dal catalogo CDSE (`gnn-sar search`) |
| `segmentation` | `scikit-image` | super-pixel SLIC (`"method": "slic"`) |

Il cuore del pacchetto (grafo, modello, loss, metriche) resta NumPy + PyTorch e
non richiede nessuno dei tre. Fuori da Python servono, per il pre-processing:

- **ESA SNAP** (versione 10 o successive) con il comando `gpt` — serve a
  `gnn-sar preprocess s1` e `gnn-sar preprocess coherence`, cioè se parti dai
  GRD o dagli SLC invece che dai già pronti RTC (sezione 6). `gnn-sar` lo cerca
  su `PATH`, in `$GNN_SAR_GPT` e nelle installazioni tipiche; con `--gpt` glielo
  indichi a mano;
- **GDAL** (`gdal_rasterize`, `gdalbuildvrt`) per rasterizzare le strade e, se
  preferisci farlo a mano, per allineare. L'allineamento alla griglia dell'area
  lo fa comunque `gnn-sar preprocess`, via `rasterio`.

Verifica dell'installazione:

```bash
pytest                                  # deve passare tutto
gnn-sar demo --models mlp --max-epochs 5 --nodes-per-side 8 --n-time 8
```

## 3. Usare il repository senza dati

Prima di scaricare un terabyte, conviene sapere che cosa il progetto misura.

```bash
gnn-sar demo --models stgnn temporal mlp rf --max-epochs 120 \
    --nodes-per-side 20 --n-time 24 --contrast 0.3
```

Genera uno scenario sintetico di consumo di suolo (propagazione topologica,
firma di cantiere distribuita nel tempo, etichette annuali deboli), addestra il
modello completo e le sue ablazioni, e stampa il confronto: F1 e kappa annuali,
tasso di detection, ritardo medio e mediano in compositi. È il *feasibility
gate* dell'OS3 in miniatura. Attenzione: è un risultato **sul simulatore**, non
sull'Italia.

Le stesse parti si possono guidare a mano:

```python
from gnn_sar.data import make_synthetic_scenario
from gnn_sar.models import SpatioTemporalGNN
from gnn_sar.config import ModelConfig
from gnn_sar.training import SpatioTemporalGraphDataModule, SoilSealingModule
import pytorch_lightning as pl

scenario = make_synthetic_scenario(height=80, width=80, n_time=24)
data = SpatioTemporalGraphDataModule(scenario.graph)
data.setup()

module = SoilSealingModule(model=SpatioTemporalGNN(ModelConfig(in_channels=data.n_features)))
pl.Trainer(max_epochs=50).fit(module, datamodule=data)

maps = module.predict_maps(data.batch("test"))
maps["sealed_probability"]    # (T, N) mappa sub-annuale monotona
maps["onset"]                 # (N,)   primo composito segnalato per nodo
maps["early_warning"]         # (T, N) "si sigilla entro h compositi"
```

Ogni run è descritto da un `ExperimentConfig` serializzabile in JSON
(`gnn_sar.config`): grafo, modello, ottimizzazione e pesi della loss. Salvalo
insieme ai risultati — è l'unico modo per riprodurli.

```python
from gnn_sar.config import ExperimentConfig
ExperimentConfig().to_json("esperimenti/base.json")
```

## 4. Quali dati servono

| dato | che cosa fornisce | obbligatorio |
| --- | --- | --- |
| **Sentinel-1 IW, VV+VH** | il segnale primario: il backscatter sale in modo stabile quando il suolo viene impermeabilizzato, e il cantiere si vede prima | sì |
| **Coerenza InSAR** (da SLC) | persistenza: crolla durante il cantiere, risale altissima sul costruito | no, ma è la feature che distingue un cantiere da un campo arato |
| **Sentinel-2 L2A** | indici ottici (NDVI, NDBI, BSI): dove non c'è nuvola, separano il suolo nudo dal costruito meglio del SAR | no |
| **Cartografia ISPRA/SNPA** | le etichette annuali che supervisionano il modello | sì, per addestrare |
| **Rete stradale (OSM)** | la maschera dei corridoi, che dà senso agli archi di connettività | no, ma senza di essa la famiglia `connectivity` degrada a semplice prossimità |
| **Layer catastale** | nodi = particelle, invece dei super-pixel | no |

### 4.1 Le aree pilota, e la griglia che tiene insieme tutto

Le tre aree del progetto sono definite nel codice
(`gnn_sar.data.sites`), non nella prosa, perché il loro contenuto è la **griglia
unica** su cui tutto il resto deve atterrare:

```bash
gnn-sar sites            # tutte e tre
gnn-sar sites viterbo    # una sola, con la riga gdalwarp pronta
```

| area | che cos'è | griglia |
| --- | --- | --- |
| `roma` | frangia peri-urbana est/sud-est, a cavallo del GRA: espansione frammentata | 2730 × 2420 px, 10 m, EPSG:32633 |
| `viterbo` | Viterbo e la sua campagna: la provincia prima per consumo annuo, trainata dal fotovoltaico a terra | 2550 × 2090 px, 10 m, EPSG:32633 |
| `milano` | Milano sud e hinterland: il caso Pianura Padana, dove nuvole e foschia motivano l'impianto SAR-first | 2360 × 2240 px, 10 m, EPSG:32632 |

Sono volutamente piccole (~25 × 22 km, 5–7 milioni di pixel): un'area di studio
è **un solo grafo**, tenuto per intero in memoria. Per coprire una regione si
definiscono più aree, non se ne allarga una.

Milano sta in **UTM 32**, le due del Lazio in **UTM 33**: non vanno mai mescolate
dentro la stessa area di studio. Se ti serve il cluster fotovoltaico di
Tarquinia–Montalto di Castro, definisci una quarta area attorno a
11.45–11.80 E, 42.20–42.45 N invece di allargare Viterbo.

Aggiungerne una è una voce nel dizionario `PILOT_SITES`: bbox in gradi, CRS,
estensione proiettata (arrotondata a un numero intero di pixel) e una nota su
che cosa aspettarsi. Da lì derivano la query al catalogo, la griglia del
pre-processing e il controllo che l'ingestione fa sui raster.

**Il contratto di ingestione.** Qualunque sia la provenienza, `gnn-sar ingest`
si aspetta:

1. **un GeoTIFF per acquisizione**, con le bande nell'ordine dichiarato nella
   configurazione (per esempio VV e VH);
2. **la data di acquisizione nel nome del file**, come `YYYYMMDD` o
   `YYYYMMDDTHHMMSS` — i nomi originali dei prodotti Sentinel vanno benissimo;
3. **tutti i file sulla stessa griglia**: stesso CRS, stessa dimensione del
   pixel, stessa estensione, stesso allineamento. L'ingestione **verifica** e si
   rifiuta di procedere se qualcosa non torna, invece di ricampionare di
   nascosto (un ricampionamento silenzioso introduce nella serie un "cambiamento"
   che non è sul terreno);
4. **nodata dichiarato nel file**, così diventa `NaN` e non un livello di
   backscatter.

## 5. Dove scaricarli: Copernicus Data Space Ecosystem

Il **Copernicus Data Space Ecosystem** (CDSE) è il punto di distribuzione
ufficiale dei dati Sentinel; ha sostituito il vecchio Copernicus Open Access Hub
(SciHub), che non è più attivo. L'accesso è gratuito previa registrazione su
<https://dataspace.copernicus.eu>. Lo stesso account vale per tutte le
interfacce elencate qui sotto.

### 5.1 Quale prodotto scegliere

**Sentinel-1**

| prodotto | che cos'è | quando usarlo |
| --- | --- | --- |
| `IW_GRDH_1S` (GRD) | ampiezza rilevata, 10 m di pixel spacing, VV+VH sull'Europa. Non calibrato, non ortorettificato | il percorso classico: richiede SNAP (sezione 6) |
| **RTC** (`Sentinel-1-RTC`) | backscatter *radiometrically terrain corrected*, conforme CEOS CARD4L NRB, ortorettificato su DEM Copernicus, **senza** filtro speckle | il percorso consigliato: salta SNAP. Verifica prima nel catalogo la copertura spaziale e temporale sulla tua area |
| `SLC` | dato complesso, fase inclusa | serve **solo** per calcolare la coerenza InSAR (sezione 6.3). Pesante |
| `Sentinel-1-COG` | gli stessi GRD in Cloud Optimized GeoTIFF | comodo per la lettura parziale via S3/HTTP |

**Sentinel-2**: `S2MSI2A` (L2A), riflettanza al suolo, bande a 10/20/60 m più il
layer di classificazione della scena **SCL** che serve a mascherare le nuvole.
Non usare L1C: dovresti fare la correzione atmosferica da solo.

**Stato della costellazione (settembre 2026).** Sentinel-1A ha concluso le
operazioni il 30 giugno 2026 dopo dodici anni (Sentinel-1B era terminato nel
2022); la costellazione è ora **Sentinel-1C + Sentinel-1D**, con il ciclo di
ripetizione nominale di **6 giorni** ristabilito intorno a luglio 2026. Per una
serie storica lunga questo significa che il sensore cambia a metà serie e che la
frequenza di rivisita non è costante nel tempo: **verifica sempre quante
acquisizioni cadono in ogni mese** (`gnn-sar ingest` stampa quanti compositi
sono rimasti vuoti) e considera compositi mensili piuttosto che decadali sui
periodi più radi. Sentinel-2 opera con 2A, 2B e 2C.

### 5.2 Il modo più semplice: il browser

<https://browser.dataspace.copernicus.eu> — disegni l'area, filtri per missione,
data, tipo di prodotto e copertura nuvolosa, e scarichi. Perfetto per capire
quante scene esistono e per prendere le prime due o tre da usare come prova.
Impraticabile per tre anni di serie.

### 5.3 Il modo scriptabile: il catalogo OData (integrato nel progetto)

Il comando `gnn-sar search` interroga il catalogo OData e, se vuoi, scarica:

```bash
export CDSE_USERNAME='tuo.indirizzo@example.org'
export CDSE_PASSWORD='...'                 # mai scritte su disco dal progetto

gnn-sar search \
    --collection SENTINEL-1 \
    --site viterbo \
    --start 2021-01-01 --end 2024-01-01 \
    --product-type IW_GRDH_1S \
    --orbit 22 --orbit-direction DESCENDING \
    --limit 1000 \
    --output cataloghi/viterbo_s1.json
```

Stampa una riga per prodotto (data, dimensione, nome) e il volume totale, così
sai in che cosa ti stai imbarcando prima di scaricarlo. Aggiungendo
`--download data/raw/s1` scarica tutto in quella cartella.

`--site` riempie il *bounding box* dall'area pilota; in alternativa
`--bbox W S E N` prende quattro gradi a mano.

Per Sentinel-2:

```bash
gnn-sar search --collection SENTINEL-2 --site viterbo \
    --start 2021-01-01 --end 2024-01-01 \
    --product-type S2MSI2A --max-cloud-cover 40 --limit 1000
```

Dietro le quinte sono chiamate HTTP semplici, che puoi replicare a mano:

- catalogo: `https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=...`
- token: `https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token`
  (`grant_type=password`, `client_id=cdse-public`)
- download: `https://download.dataspace.copernicus.eu/odata/v1/Products(<Id>)/$value`
  con l'header `Authorization: Bearer <token>`

Il token dura pochi minuti: `gnn-sar search --download` ne prende uno per
sessione di scaricamento. Le connessioni parallele per account sono poche
(quattro è un budget sicuro).

**Perché `--orbit`.** Il backscatter dipende dalla geometria di vista. Mescolare
orbite relative diverse, o ascendente e discendente, inserisce nella serie
variazioni di qualche dB che non hanno nulla a che vedere con il suolo e che il
modello imparerebbe come "cambiamento". Scegli **una** orbita relativa che copra
tutta l'area e restaci per tutta la serie; se l'area richiede due orbite,
trattale come due aree di studio separate, oppure aggiungile come sorgenti
distinte con i propri canali.

### 5.4 Le altre interfacce (non integrate nel progetto)

- **STAC** — `https://stac.dataspace.copernicus.eu/v1`, collezioni
  `sentinel-1-grd`, `sentinel-1-rtc`, `sentinel-2-l2a`. Ottimo per esplorare che
  cosa esiste davvero (ed è il modo per verificare la copertura degli RTC).
- **S3 / `eodata`** — endpoint `https://eodata.dataspace.copernicus.eu`, bucket
  `Sentinel-1`, `Sentinel-1-COG`, `Sentinel-1-RTC`, `Sentinel-2`, con chiavi
  generate dall'S3 keys manager del proprio account. È l'accesso più veloce se
  lavori su una VM CloudFerro/CREODIAS: niente download, si legge direttamente.
- **openEO** — `https://openeo.dataspace.copernicus.eu`: descrivi il datacube
  (area, periodo, bande, `sar_backscatter`, mascheratura, aggregazione temporale)
  e ricevi già il composito. È il modo più economico di ottenere uno stack
  allineato senza SNAP e senza terabyte in locale; il risultato va poi salvato
  come GeoTIFF mensili e dato in pasto a `gnn-sar ingest` come qualunque altra
  cartella.
- **Sentinel Hub** — `https://sh.dataspace.copernicus.eu`: API di processing a
  quota (*processing units*), utile per anteprime e statistiche.

## 6. Pre-processing Sentinel-1

Obiettivo: da prodotto d'archivio a **σ⁰ calibrato, ortorettificato, sulla
griglia comune dell'area di studio, in potenza lineare**.

### 6.1 Percorso A — partire dagli RTC (consigliato)

I prodotti `Sentinel-1-RTC` del CDSE sono già calibrati e ortorettificati su DEM
Copernicus (nessun filtro speckle applicato: lo compensa la mediana del
composito mensile). Restano da fare solo due cose:

```bash
# ritaglio e allineamento sulla griglia esatta dell'area pilota
python - <<'PY'
from pathlib import Path
from gnn_sar.data.preprocess import align_to_site
from gnn_sar.data.sites import get_site

site = get_site("viterbo")
for source in sorted(Path("data/raw/rtc").glob("*.tif")):
    align_to_site(source, Path("viterbo/s1") / source.name, site)
PY
```

oppure, a mano, con GDAL e i numeri che `gnn-sar sites viterbo` stampa già
pronti:

```bash
gdalwarp $(gnn-sar sites viterbo | sed -n 's/.*gdalwarp *//p') \
         -r bilinear -dstnodata nan RTC_originale.tif viterbo/s1/S1C_..._20240117T051253.tif
```

I quattro numeri di `-te` (xmin ymin xmax ymax, nel CRS di destinazione) sono
quelli dell'area pilota e vanno usati **identici per Sentinel-1, Sentinel-2,
etichette e strade**: è esattamente il motivo per cui stanno in un posto solo.

### 6.2 Percorso B — partire dai GRD con SNAP

Il grafo pronto è in [`esempi/snap_s1_grd_preprocessing.xml`](esempi/snap_s1_grd_preprocessing.xml):
orbita precisa → rimozione del rumore termico → rimozione del rumore di bordo →
calibrazione σ⁰ (lineare) → filtro speckle Refined Lee → Range-Doppler Terrain
Correction su DEM Copernicus a 10 m, CRS unico.

Un comando solo fa tutto il lotto — SNAP su ogni prodotto, poi allineamento
alla griglia dell'area:

```bash
gnn-sar preprocess s1 \
    --site viterbo \
    --input data/raw/s1 --pattern "*.zip" \
    --output viterbo/s1 \
    --memory 8G --threads 4
```

Che cosa fa, e perché è fatto così:

- passa al grafo il **CRS dell'area pilota** (`${crs}`), quindi la stessa catena
  serve Roma, Viterbo e Milano senza modifiche;
- nomina l'uscita con la data di acquisizione, che è ciò che l'ingestione
  rileggerà;
- **salta** quello che è già stato fatto, così un lotto interrotto riparte da
  dove era arrivato (`--overwrite` per rifare);
- se una scena fallisce, la registra e **va avanti**: su trecento prodotti
  qualcuno fallisce sempre (swath tagliato, orbita precisa non ancora
  disponibile, download corrotto), e fermare il lotto sarebbe la reazione
  sbagliata. Alla fine stampa `written / skipped / failed` con l'errore di SNAP;
- `--jobs N` processa più prodotti insieme: SNAP però è già multi-thread, quindi
  oltre due si moltiplica soprattutto la memoria;
- `--graph tuo_grafo.xml` sostituisce la catena, `--parameter nome=valore` le
  passa altre variabili. Vengono passate **solo** le variabili che il grafo
  dichiara davvero, così un grafo personalizzato non riceve valori che non sa
  usare.

Alternativa programmabile per catene più complesse:
[`pyroSAR`](https://pyrosar.readthedocs.io), che genera ed esegue grafi SNAP da
Python.

**Due avvertenze.** L'uscita resta in **potenza lineare**: la conversione in dB e
il rapporto VV/VH li calcola `gnn-sar` in ingestione (`"linear": true`), perché
la mediana di un composito va fatta in potenza — mediare in dB pesa diversamente
le code. E le orbite precise sono disponibili circa venti giorni dopo
l'acquisizione: per le scene più recenti SNAP userà quelle ristituite, con una
georeferenziazione leggermente peggiore; se la serie deve essere omogenea,
riprocessale dopo.

### 6.3 Coerenza InSAR (opzionale, dai SLC)

La coerenza interferometrica è la feature che meglio separa un cantiere da un
campo arato: crolla mentre si costruisce, risale a valori altissimi sul
costruito finito. Il backscatter da solo confonde un piazzale in terra battuta
con un campo appena lavorato; la coerenza no.

```bash
gnn-sar preprocess coherence \
    --site viterbo \
    --input data/raw/slc --pattern "*.zip" \
    --output viterbo/coh \
    --baseline-days 12 --tolerance-days 2 \
    --parameter subswath=IW2 --parameter polarisation=VV \
    --memory 12G
```

Il comando sceglie le coppie a baseline costante, esegue
[`esempi/snap_s1_slc_coherence.xml`](esempi/snap_s1_slc_coherence.xml) su
ognuna e allinea il risultato alla griglia dell'area.

Quattro cose da sapere prima di lanciarlo:

1. **stessa orbita relativa**, obbligatorio: due SLC di orbite diverse non sono
   interferabili. Scaricali con `--orbit`;
2. **baseline costante**. La coerenza cala da sola col tempo, quindi mescolare
   coppie a 6 e a 12 giorni mette nella serie una decorrelazione che non ha
   nulla a che vedere con i cantieri. Il ciclo di rivisita è però cambiato nel
   tempo, quindi una serie lunga *non* ha una baseline unica: vedi
   [`roadmap.md`](roadmap.md), voce B1;
3. **un sotto-swath alla volta** (`--parameter subswath=IW2`): se l'area cade a
   cavallo di due, serve un `TOPSAR-Merge` che il grafo non fa;
4. **la coerenza descrive un intervallo, non un istante**: il prodotto è datato
   con la **seconda** acquisizione, l'unica scelta compatibile con la causalità
   che il modello impone.

Il risultato si dà a `gnn-sar` in due modi: come banda in più dello stesso file
Sentinel-1 (`"bands": {"VV": 1, "VH": 2, "coherence": 3}` — la coerenza passa
indenne, non essendo una potenza) oppure come sorgente separata di tipo `raw`.

Costo: gli SLC pesano circa il quadruplo dei GRD e la catena è molto più lenta.
**E il grafo non è mai stato eseguito su dati veri dagli autori del
repository**: provalo su una coppia e guarda l'uscita prima di lanciarlo su tre
anni.

## 7. Pre-processing Sentinel-2

Da L2A servono: le bande utili agli indici e il layer SCL, tutto a 10 m, sulla
stessa griglia. Qui SNAP non serve — L2A è già riflettanza al suolo — e il
comando fa tutto:

```bash
gnn-sar preprocess s2 \
    --site viterbo \
    --input data/raw/s2 --pattern "*.SAFE" \
    --output viterbo/s2
```

Trova le bande dentro ogni `.SAFE` (prendendo la risoluzione più fine in cui
ciascuna è offerta), le riporta sulla griglia dell'area e le impila in un unico
file nell'ordine `B02 B03 B04 B08 B11 B12 SCL` — cioè l'ordine che la
configurazione di ingestione dichiara. La riflettanza è ricampionata
**bilinearmente**, l'SCL a **nearest neighbour**: interpolare dei codici di
classe inventerebbe classi che non esistono, e l'SCL è ciò che decide di quali
pixel ci si fida. I prodotti vanno scompattati prima (`unzip`): le bande non
vengono lette dentro l'archivio.

Con `--bands B04 B08 B11` si accorcia l'elenco; `--no-scl` toglie la maschera, e
senza maschera la nuvola diventa «cambiamento».

A mano, con GDAL, sarebbe:

```bash
# B02 B03 B04 B08 sono a 10 m; B11 B12 e SCL a 20 m: si ricampionano a 10 m.
gdalbuildvrt -separate -resolution user -tr 10 10 tmp/s2.vrt \
    B02.jp2 B03.jp2 B04.jp2 B08.jp2 B11.jp2 B12.jp2 SCL.jp2

gdalwarp -t_srs EPSG:32633 -tr 10 10 -te $TE -r near \
    tmp/s2.vrt viterbo/s2/S2B_MSIL2A_20240117T100319_T33TTG.tif
```

Usa `-r near` per SCL (è una classificazione: interpolarla non ha senso) e, se
vuoi essere rigoroso, ricampiona separatamente le bande con `-r bilinear` e SCL
con `-r near`, poi uniscile.

In ingestione dichiari `"scl_band": 7` e `"scale": 0.0001` (gli interi L2A
diventano riflettanza). Le classi scartate per default sono `0, 1, 2, 3, 8, 9,
10, 11` — nodata, saturati, ombre, ombre di nuvola, nuvole medie e alte, cirri,
neve. Acqua (6) e non classificato (7) restano: sono stati legittimi del suolo, e
scartarli toglierebbe silenziosamente le zone umide di frangia.

**Perché indici e non bande.** Gli indici (NDVI, NDBI, BSI) sono rapporti:
sopravvivono alle differenze residue di illuminazione e atmosfera fra due date
molto meglio di una riflettanza, e tengono il contributo ottico a pochi canali
leggibili accanto al backscatter.

**Che cosa aspettarsi.** In Pianura Padana l'ottico è quasi inutile per mesi
interi: è esattamente la ragione per cui il progetto è *SAR-first*. Il composito
mensile mascherato è costruito per reggerlo — un pixel nuvoloso non partecipa, e
un nodo coperto a metà viene comunque misurato dalla metà libera — ma se un nodo
non è mai osservato in nessun composito l'ingestione si ferma e te lo dice,
invece di consegnare al modello dei `NaN`.

## 8. Etichette ISPRA/SNPA e corridoi infrastrutturali

### 8.1 La cartografia del consumo di suolo

ISPRA/SNPA pubblica annualmente la cartografia nazionale del consumo di suolo a
**10 m**, con l'intera serie storica (2006, 2012 e ogni anno dal 2015).
**Dove si scarica, con quale licenza e — soprattutto — come sono codificati i
pixel sta in [`dati_etichette_ispra.md`](dati_etichette_ispra.md)**, insieme
alle alternative aperte (Copernicus HRL Impervious Built-Up) e alle fonti
regionali per le tre aree pilota.

Una cosa da quel documento va ripetuta qui, perché è l'errore che rovina un
esperimento senza dare segnali: nella classificazione ISPRA **`1` significa
suolo consumato e `2` significa suolo NON consumato**. Il default «qualsiasi
valore diverso da zero è sigillato» etichetterebbe come consumato l'intero
territorio. Con un raster ISPRA va sempre dichiarato:

```json
"labels": { "rasters": {...}, "sealed_values": [1] }
```

Dalla versione corrente l'ingestione **si rifiuta di indovinare**: se trova
valori diversi da 0 e 1 senza che `sealed_values` sia stato dichiarato, si ferma
ed elenca i valori trovati.

Porta ogni anno sulla griglia di studio, con lo stesso `-te` di tutto il resto:

```bash
gdalwarp -t_srs EPSG:32633 -tr 10 10 -te $TE -r near \
    consumo_suolo_2023.tif labels/consumo_suolo_2023.tif
```

**Stato contro transizione.** La cartografia dice *dov'è* sigillato il suolo in
un dato anno; il modello va supervisionato su *quando* è stato consumato.
`gnn-sar` fa la conversione (`annual_land_take_labels`), e la fa in modo
conservativo:

- `1` solo nell'anno in cui il nodo **diventa** sigillato;
- `0` negli anni prima e — importante — anche in quelli dopo: un nodo già
  sigillato non viene consumato di nuovo;
- `-1` (ignoto) quando l'informazione non c'è. Un anno senza cartografia rende
  ignoti anche gli anni successivi finché una nuova osservazione "ancora non
  sigillato" non chiude il buco: se la transizione può essere avvenuta nel vuoto,
  nessuno dei due anni può essere chiamato positivo o negativo;
- il suolo **già sigillato nel primo anno** della serie è escluso
  (`"pre_existing": "ignore"`): non c'è nessun onset da trovare al suo interno, e
  etichettarlo `1` insegnerebbe al modello a scattare sul costruito stabile.
  L'alternativa `"negative"` lo tratta invece come classe negativa; usala solo se
  il primo anno è una baseline affidabile, e in quel caso alza
  `positive_weight` nella `LossConfig`, perché la classe negativa esplode.

Servono quindi **almeno due anni** di cartografia per avere una sola transizione
supervisionata, e conviene che il primo anno preceda l'inizio della serie
Sentinel: serve come baseline.

La soglia `threshold` decide quanta parte di un nodo deve essere sigillata
perché il nodo conti come sigillato. `0.5` è un voto di maggioranza; abbassala
per catturare elementi lineari stretti (una strada nuova dentro un nodo grande) e
aspettati in cambio più falsi onset.

### 8.2 Il corridoio infrastrutturale

Scarica la rete stradale da OpenStreetMap (Geofabrik, Overpass, `osmnx`),
bufferizzala di qualche decina di metri e rasterizzala sulla griglia di studio:

```bash
ogr2ogr -f GPKG strade.gpkg strade_osm.shp -dialect SQLite \
        -sql "SELECT ST_Buffer(geometry, 50) AS geometry FROM strade_osm"
gdal_rasterize -burn 1 -tr 10 10 -te $TE -a_nodata 0 -ot Byte \
        strade.gpkg osm/strade_buffer50m.tif
```

Nel grafo, un arco di *connettività* è mantenuto solo se **entrambi** i nodi
stanno sul corridoio: è questo che dà all'ipotesi «il consumo si propaga lungo le
infrastrutture» una forma verificabile. Senza la maschera, quella famiglia di
archi degrada a semplice prossimità fra centroidi e la distinzione dall'adiacenza
si assottiglia.

## 9. Integrazione nel progetto: `gnn-sar ingest`

### 9.1 Disposizione dei file

```
viterbo/
├── ingest.json                     # la configurazione (sotto)
├── s1/                             # un GeoTIFF per acquisizione, VV+VH lineari
│   ├── S1C_IW_GRDH_1SDV_20210112T051253_....tif
│   └── ...
├── s2/                             # un GeoTIFF per acquisizione, 6 bande + SCL
│   └── S2B_MSIL2A_20210117T100319_T33TTG.tif
├── labels/
│   ├── consumo_suolo_2020.tif      # stato sigillato, un raster per anno
│   └── ...
├── osm/strade_buffer50m.tif
└── cache/                          # l'output dell'ingestione
```

### 9.2 La configurazione

Le tre aree pilota hanno una configurazione pronta e commentata:
[`esempi/aree/roma.json`](esempi/aree/roma.json),
[`esempi/aree/viterbo.json`](esempi/aree/viterbo.json),
[`esempi/aree/milano.json`](esempi/aree/milano.json). Campo per campo:

| campo | significato |
| --- | --- |
| `site` | il nome dell'area pilota. Quando c'è, i raster vengono confrontati con la griglia di quell'area **prima** di qualunque altra cosa: un'estensione o un CRS sbagliati si scoprono sul primo file, non a fine ingestione |
| `sources[].kind` | `sentinel1` (dB + rapporto VV/VH), `sentinel2` (maschera SCL + indici), `raw` (bande così come sono) |
| `sources[].directory`, `pattern` | dove sono i file e quali prendere; i percorsi sono relativi alla cartella del config |
| `sources[].bands` | nome della banda → indice **1-based** nel file |
| `sources[].linear` | S1: il file è in potenza lineare (SNAP, RTC) e va convertito in dB |
| `sources[].ratio` | S1: aggiunge il canale VV−VH in dB |
| `sources[].scale` | fattore moltiplicativo, `0.0001` per gli interi L2A |
| `sources[].scl_band`, `indices` | S2: quale banda è la SCL e quali indici calcolare |
| `sources[].reducer` | come si combinano le acquisizioni dentro un composito (`median` di default: sopprime le code dello speckle) |
| `segmentation` | `grid` (tassellatura regolare, `block` pixel di lato), `slic` (super-pixel su composito multitemporale), `labels` (raster catastale) |
| `period`, `start`, `n_periods` | l'asse dei compositi: `"month"`, `"dekad"` o un numero di giorni |
| `statistics` | statistiche per nodo di ogni canale: `mean` è il livello, `std` la dispersione interna al nodo, che fa da tessitura |
| `labels` | i raster annuali di stato, `sealed_values` (obbligatorio con la codifica ISPRA) e le soglie della sezione 8.1 |
| `corridor` | la maschera stradale |
| `graph` | `n_neighbors`, `connectivity_radius`, `temporal_order` |
| `output` | dove scrivere la cache `.npz` |

Qualsiasi chiave che inizia con `//` viene ignorata: è il modo per commentare un
JSON.

### 9.3 Esecuzione

```bash
gnn-sar ingest docs/esempi/aree/viterbo.json
```

Stampa, mentre lavora, quello che serve a controllarlo:

```
source s1         324 acquisitions, 2021-01-12 … 2023-12-28
source s2          71 acquisitions, 2021-01-17 … 2023-12-22
grid             3500x3500 px, 10 x 10 units, crs=EPSG:32633, origin=(250000, 4715000)
composites       36 (2021-01 … 2023-12, month)
nodes from       SLIC over 8 composites of s1
nodes            5987
source s2          3 composites without an acquisition, carried forward from the previous one
labels           4 years [2020, 2021, 2022, 2023], 212 land-take nodes, 1841 unknown year-node pairs
corridor         1204 of 5987 nodes
```

Che cosa fa, in ordine: cataloga le acquisizioni e ne legge le date dai nomi;
fissa la griglia di riferimento sul primo file; costruisce l'asse dei compositi;
calcola **una sola volta** la partizione in nodi; poi, **un composito alla
volta**, legge le acquisizioni di quel mese, le riduce con la mediana, aggrega
sui nodi e libera la memoria — è il motivo per cui un'area di 35 × 35 km su tre
anni sta in pochi GB di RAM invece che in decine. Infine converte la cartografia
in etichette di transizione, costruisce gli archi e salva tutto in un `.npz`.

La cache contiene feature, archi, etichette, geometrie dei nodi,
georeferenziazione, nomi dei canali, l'asse temporale e la configurazione che
l'ha prodotta. L'ingestione si fa **una volta**; tutti gli esperimenti ripartono
da lì.

## 10. Addestramento e mappe: `gnn-sar train`

```bash
gnn-sar train viterbo/cache/viterbo.npz \
    --model stgnn \
    --config esperimenti/base.json \
    --max-epochs 200 \
    --output-dir risultati/viterbo_stgnn
```

**GPU.** Non serve nessun flag: il default è `--accelerator auto --devices 1`,
che prende la GPU quando c'è. Su una scheda Ampere o più recente conviene
aggiungere `--precision bf16-mixed`, che dimezza circa la memoria occupata dal
tensore delle feature — ed è quello che decide se un'intera provincia ci sta in
una volta. Un'area di studio è **un solo grafo full-batch**, quindi `--devices 2`
lo replicherebbe invece di dividerlo: la leva utile è la precisione, non il
numero di schede.

```bash
gnn-sar train viterbo/cache/viterbo.npz --output-dir risultati/viterbo_stgnn \
    --accelerator gpu --precision bf16-mixed --experiment-name viterbo
```

**MLflow.** Ogni run è registrato: i parametri sono l'intero
`ExperimentConfig` (nomi puntati, `training.loss.positive_weight`), le metriche
sono quelle che il loop Lightning logga epoca per epoca, e gli artefatti sono
`config.json`, `metrics.json`, il checkpoint e i GeoTIFF sub-annuali. Senza
configurare niente i run finiscono in `./mlruns`, che `mlflow ui` legge così
com'è:

```bash
mlflow ui        # http://127.0.0.1:5000
```

Per scrivere altrove: `--tracking-uri` (o `MLFLOW_TRACKING_URI`) accetta un
server, `sqlite:///mlflow.db`, Databricks. `--experiment-name` e `--run-name`
decidono come il run è archiviato, `--no-tracking` non registra nulla. Con
`gnn-sar demo` si apre **un run per modello confrontato**, con il nome del
modello come tag: il confronto dell'OS3 diventa un filtro nell'interfaccia
invece del diff di due log.

Scrive nella cartella indicata:

| file | contenuto |
| --- | --- |
| `metrics.json` | OA, F1, IoU, kappa **annuali** (l'unico confronto onesto con una cartografia annuale) e i termini della loss |
| `<modello>.ckpt` | i pesi |
| `sealed_probability.tif` | una banda per composito: la probabilità sub-annuale di suolo sigillato, non decrescente nel tempo |
| `onset.tif` | il primo composito in cui ogni nodo supera la soglia, `-1` se mai |
| `early_warning.tif` | una banda per composito: «si sigilla entro `prediction_horizon` compositi» |

L'ordine delle bande dei file multi-composito è stampato a fine run
(`2021-01, 2021-02, …`): è il solo modo per leggere `onset` come una data.

**Come leggere i numeri.**

- Le metriche sub-annuali e il *detection delay* compaiono **solo** se esistono
  date esatte (`node_labels`). Con le sole etichette annuali il modulo non le
  produce: una F1 sub-annuale calcolata contro un'etichetta annuale misurerebbe
  il modello contro un bersaglio che non esiste.
- Lo split è **geografico**, non casuale: i nodi sono ordinati lungo una
  coordinata e tagliati in bande (60/20/20). Campionare a caso metterebbe i
  vicini immediati di un nodo di test nel training, cioè esattamente i nodi fra
  cui il modello si scambia messaggi: la fuga di informazione passerebbe proprio
  attraverso il meccanismo che si vuole misurare. Anche lo standardizzatore è
  calcolato solo sui nodi di training.
- Le mappe in uscita sono invece prodotte su **tutti** i nodi: lo split serve a
  rendere oneste le metriche, non a cancellare due terzi della mappa.
- Confronta sempre con le ablazioni — `--model temporal` (niente grafo) e
  `--model mlp` (niente grafo, niente tempo), più `--model rf` come baseline
  classica. Il guadagno del grafo è la differenza fra `stgnn` e `temporal`, ed è
  esattamente la domanda dell'OS3. Sul simulatore, oggi, quel guadagno è
  negativo: è il motivo per cui servono i dati veri.

## 11. La stessa cosa dall'API Python

Se preferisci controllare ogni passaggio, o se i tuoi dati non stanno nel
formato «una cartella di GeoTIFF», l'API di libreria fa lo stesso lavoro:

```python
import numpy as np
from gnn_sar.data.raster import read_aligned_stack, list_rasters
from gnn_sar.data.sentinel import PeriodGrid, sensing_dates, sentinel1_channels
from gnn_sar.data.segmentation import NodeGeometries
from gnn_sar.data.labels import node_state_from_rasters, annual_land_take_labels
from gnn_sar.data.raster import read_raster
from gnn_sar.data.study_area import SourceStack, build_study_area

paths = list_rasters("viterbo/s1_rtc")
values, grid = read_aligned_stack(paths, bands=[1, 2])      # (K, 2, H, W)
channels, names = sentinel1_channels({"VV": values[:, 0], "VH": values[:, 1]})
stack = np.moveaxis(channels, 0, 1)                         # (K, C, H, W)

periods = PeriodGrid.covering(sensing_dates(paths), "month")
geometries = NodeGeometries.regular_grid(grid.height, grid.width, block=8)

yearly_masks = [                                            # stato sigillato, per anno
    read_raster(f"viterbo/labels/consumo_suolo_{year}.tif", bands=[1], masked=True)[0][0] > 0
    for year in (2022, 2023)
]
state = node_state_from_rasters(yearly_masks, geometries)   # (Y, N) in {0, 1, -1}
labels = annual_land_take_labels(state)                     # (Y, N) transizioni

area = build_study_area(
    [SourceStack(stack, sensing_dates(paths), names)],
    geometries,
    periods=periods,
    annual_labels={2022: labels[0], 2023: labels[1]},
    grid=grid,
)
area.save("viterbo/cache/viterbo.npz")
```

Pezzi utili da conoscere:

| funzione | a che serve |
| --- | --- |
| `NodeGeometries.from_label_image` | nodi = particelle catastali rasterizzate |
| `slic_geometries` | super-pixel SLIC su composito multitemporale |
| `composite_by_period(..., ignore_nan=True)` | compositi che ignorano i pixel mascherati |
| `fill_temporal_gaps` | chiude i buchi residui lungo il tempo |
| `aggregate_stack(..., nan_policy="omit")` | aggrega un nodo sui soli pixel osservati |
| `align_annual_labels` | dispone le etichette sugli anni che i compositi attraversano |
| `write_raster` | scrive una predizione rasterizzata come GeoTIFF |
| `load_study_area` | rilegge la cache |

`build_study_area` tiene tutto lo stack di pixel in memoria: va bene per aree
piccole e per i test, mentre per un'area vera conviene `gnn-sar ingest`, che
lavora un composito alla volta.

## 12. Scelte metodologiche da non sbagliare

1. **Una sola orbita relativa, una sola direzione di passaggio.** Vedi 5.3.
2. **Una sola griglia per tutto**: Sentinel-1, Sentinel-2, etichette, strade.
   Fissa `-te`, `-tr` e `-t_srs` una volta e riusali ovunque.
3. **Segmenta una volta sola.** I nodi devono essere gli stessi dal primo
   all'ultimo composito, altrimenti gli archi temporali collegano pezzi di terra
   diversi. È imposto dai tipi: le geometrie stanno fuori dall'asse temporale.
4. **Media in potenza, non in dB.** I file su disco restano lineari; la
   conversione avviene dopo la riduzione del composito.
5. **`-1` non è uno zero.** Un nodo senza etichetta non è un nodo dove non è
   successo niente: le loss saltano gli ignoti, e la conversione stato →
   transizione propaga l'ignoto in avanti finché non viene chiuso.
6. **Escludi il costruito preesistente** dalla supervisione, o accetta
   consapevolmente lo sbilanciamento di classe.
7. **Dimensione dei nodi.** A 10 m, `block: 8` dà nodi di 80 m: abbastanza
   piccoli da isolare un lotto in costruzione, abbastanza grandi perché la media
   interna abbassi lo speckle. Con SLIC, punta a nodi della stessa scala
   (`n_segments` ≈ area in pixel / 64).
8. **Lunghezza del composito.** Mensile è il compromesso ragionevole con
   rivisita a 6–12 giorni: abbastanza acquisizioni per una mediana robusta,
   abbastanza fine per dire qualcosa di sub-annuale. Il decadale ha senso solo
   dove la rivisita è densa e costante.
9. **Tutto è full-batch**: un'area di studio è un grafo solo. Un'area nazionale
   richiederà il *neighbour sampling* per cui GraphSAGE è nato — il modello è
   induttivo, quindi è un'aggiunta al data loader, non un cambio di architettura.
   Nel frattempo, spezza il territorio in aree di studio.

## 13. Che cosa manca ancora

Per non scambiare l'impalcatura per il sistema:

- **nessuna validazione su dati reali.** Tutto ciò che è descritto qui è
  testato su GeoTIFF sintetici scritti dai test; i numeri pubblicati nel README
  vengono dal simulatore;
- **le catene SNAP non sono mai state eseguite su prodotti veri** dagli autori:
  `gnn-sar preprocess` le orchestra (ricerca di `gpt`, ripresa, allineamento,
  rapporto sugli errori) e i grafi usano operatori standard, ma vanno verificati
  sulle prime due scene prima di lanciarli su una serie intera;
- **la baseline U-Net siamese** della proposta e i *foundation model* EO
  (Prithvi/Presto/Clay) non sono implementati;
- **nessun sampling di vicinato**: aree grandi vanno spezzate a mano;
- **la maschera di corridoio** va rasterizzata fuori dal progetto (GDAL/OSM);
- **niente merge di sotto-swath** per la coerenza, e nessuna normalizzazione
  della coerenza rispetto a baseline diverse (vedi [`roadmap.md`](roadmap.md));
- **la testa di early-warning addestrata su etichette annuali è grossolana**: un
  bersaglio annuale non sa localizzare la transizione, quindi resta a 1 per tutto
  l'anno. Impara «una transizione è vicina», non «fra `k` compositi». Dove
  esistono date esatte, usa `state_to_horizon_target`.

## 14. Errori frequenti

| messaggio | causa e rimedio |
| --- | --- |
| `no sensing date in '...'` | il nome del file non contiene una data: rinominalo includendo `YYYYMMDD`, o tieni il nome originale del prodotto |
| `... is not on the study grid ...` | una scena, la cartografia o le strade non stanno sulla griglia comune: rifai `gdalwarp` con gli stessi `-te -tr -t_srs` |
| `N of M nodes are never observed in at least one channel` | un nodo non ha nessun pixel valido in nessun composito (nuvola permanente, fuori swath): riduci l'area, togli la sorgente ottica o allenta la mascheratura |
| `K acquisitions fall outside the grid [...]` | hai fissato `start`/`n_periods` e alcune acquisizioni cadono fuori: allarga la finestra o filtra i file |
| `labels given for years [...], which the composites ... do not cover` | disallineamento fra anni della cartografia e periodo della serie — quasi sempre un anno di scarto |
| `annual_labels must be (Y, N) to match ...` | stai passando una matrice di etichette con un numero di anni diverso da quello che i compositi attraversano; usa la forma `{anno: vettore}` e lascia allineare `align_annual_labels` |
| `the graph must carry annual_labels and year_index` | stai addestrando su un'area ingerita senza sezione `labels` |
| `reading and writing GeoTIFFs requires rasterio` | manca l'extra: `poetry install --extras geo` |
| `talking to the CDSE catalogue requires requests` | manca l'extra: `poetry install --extras download` |
| `set CDSE_USERNAME and CDSE_PASSWORD ...` | credenziali non esportate nell'ambiente |
| `CDSE refused the credentials (401)` | password sbagliata, o account non ancora attivato via email |
| `slic_geometries requires scikit-image` | `poetry install --extras segmentation`, oppure usa `"method": "grid"` |
| `... holds the values [...] besides 0 and 1` | il raster delle etichette è classificato, non binario: dichiara `sealed_values` (con ISPRA, `[1]`) |
| `... is not on the grid of the '<area>' study area` | i raster non stanno sulla griglia dichiarata in `"site"`: rilancia `gnn-sar preprocess` per quell'area |
| `ESA SNAP's gpt was not found` | installa SNAP, mettilo su `PATH`, esporta `GNN_SAR_GPT` o passa `--gpt` |
| `no file found for ['B11', 'B12']` | il prodotto Sentinel-2 non è stato scompattato, o mancano le bande a 20 m |
| `unknown site 'firenze'` | le aree pilota sono `roma`, `viterbo`, `milano`; aggiungerne una è una voce in `PILOT_SITES` |

## 15. Licenze e citazioni dei dati

- **Sentinel-1 / Sentinel-2**: dati Copernicus, gratuiti e aperti secondo le
  condizioni ESA/UE. Cita i dati come «Contains modified Copernicus Sentinel data
  [anno]».
- **Cartografia ISPRA/SNPA del consumo di suolo**: verifica le condizioni
  d'uso dell'edizione scaricata. Non è ridistribuita da questo repository.
- **OpenStreetMap**: ODbL, con attribuzione «© OpenStreetMap contributors».
- **Codice di questo repository**: MIT, vedi [`../LICENSE.md`](../LICENSE.md).
