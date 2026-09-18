# Roadmap

Che cosa manca, in che ordine conviene farlo, e che cosa ciascuna voce
richiederebbe. Serve a due cose: non scambiare l'impalcatura per il sistema, e
rendere esplicito il costo di ogni pezzo mancante prima di iniziarlo.

Lo stato è quello del ramo corrente. Le voci **A** sono sulla via critica del
*feasibility gate* di primo anno (la GNN deve battere le baseline per-pixel su
un caso ristretto); le **B** servono al passo successivo; le **C** sono
desiderabili e non bloccanti.

---

## A1. Una corsa reale end-to-end su un'area pilota

**Stato: non fatto. È l'unica voce che conta davvero.**

Tutto il resto del repository è testato, ma contro dati sintetici o contro
GeoTIFF scritti dai test stessi. Nessun numero pubblicato viene da acquisizioni
reali.

Cosa serve, nell'ordine:

1. credenziali CDSE e cartografia ISPRA per l'area scelta
   (→ [`dati_etichette_ispra.md`](dati_etichette_ispra.md));
2. `gnn-sar search --site viterbo …` per censire i prodotti di **una** orbita
   relativa, e scaricarli;
3. `gnn-sar preprocess s1 --site viterbo …` (serve SNAP installato);
4. `gnn-sar preprocess s2 --site viterbo …`;
5. le etichette annuali riportate sulla griglia del sito;
6. `gnn-sar ingest docs/esempi/aree/viterbo.json`;
7. `gnn-sar train … --model stgnn` e le ablazioni `temporal`, `mlp`, `rf`.

Viterbo è il candidato migliore per primo: area meno costruita di Roma (quindi
più nodi utili e non esclusi come preesistenti), fenomeno netto e documentato
(fotovoltaico a terra), e Sentinel-2 utilizzabile più spesso che a Milano.
Roma serve a misurare il caso difficile (frangia frammentata, molti nodi già
consumati), Milano a verificare la tenuta senza ottico.

**Rischio principale**: che il numero di transizioni nell'area e nel periodo
scelti sia troppo basso per addestrare. Da controllare **prima** del download,
con gli indicatori comunali pubblicati da ISPRA: ettari consumati per anno nei
comuni dell'area, convertiti in numero atteso di nodi.

## A2. Verificare la catena SNAP su prodotti veri

**Stato: il grafo GRD è scritto e parametrizzato, mai eseguito qui.**

`docs/esempi/snap_s1_grd_preprocessing.xml` usa operatori standard, ma nessuno
lo ha ancora eseguito su un prodotto reale in questo repository. Prima di
lanciarlo su tre anni:

- eseguirlo su due scene e aprire l'uscita in SNAP o QGIS;
- controllare che i valori siano σ⁰ plausibili in potenza lineare (ordine
  `1e-3`–`1`, non dB, non DN);
- verificare la georeferenziazione contro un'ortofoto;
- controllare il bordo: `Remove-GRD-Border-Noise` a volte lascia frange.

## A3. Controlli di sanità sulle etichette

**Stato: parziale.** L'ingestione ora rifiuta di indovinare la codifica dei
pixel (→ `sealed_values`), il che chiude l'errore peggiore. Restano da
aggiungere:

- confronto degli ettari consumati per comune con gli indicatori ISPRA
  pubblicati, come test di accettazione dell'area;
- un rapporto per anno dopo l'ingestione: quanti nodi positivi, quanti esclusi
  come preesistenti, quanti ignoti — oggi `describe()` dà solo i totali;
- una maschera di esclusione opzionale (acque, aree già classificate come non
  suolo) per non contare come errori nodi dove il consumo è impossibile.

## B1. Coerenza InSAR da coppie SLC

**Stato: infrastruttura pronta, mai eseguita. Non validata.**

È la feature che, in letteratura e per ragioni fisiche, separa meglio un
cantiere da un campo arato: la coerenza crolla mentre si costruisce e risale a
valori altissimi sul costruito finito. Il backscatter da solo confonde un
piazzale in terra battuta con un campo appena lavorato; la coerenza no.

**Che cosa c'è già**

- `gnn_sar.data.preprocess.coherence_pairs` — selezione delle coppie a
  baseline temporale fissa (con tolleranza), testata;
- `gnn_sar.data.preprocess.preprocess_coherence` — esecuzione del grafo su
  ogni coppia, con ripresa, allineamento al sito e rapporto sugli errori;
- `docs/esempi/snap_s1_slc_coherence.xml` — split → orbite precise →
  Back-Geocoding → ESD → Coherence → deburst → Terrain Correction;
- `gnn-sar preprocess coherence --site … --baseline-days 12`;
- il lato consumo: la coerenza entra come banda in più della sorgente
  Sentinel-1 (`"bands": {"VV": 1, "VH": 2, "coherence": 3}`, passa senza
  conversione in dB perché è già un rapporto) o come sorgente `raw`.

**Che cosa manca, ed è la parte che costa**

1. **Esecuzione e verifica del grafo** su almeno una coppia reale. Nessuno lo
   ha ancora fatto: i parametri delle finestre di stima (`cohWinRg`/`cohWinAz`)
   e l'ESD vanno controllati sull'uscita, non sulla carta.
2. **Sotto-swath multipli.** Il grafo lavora su un sotto-swath alla volta. Roma
   e Milano potrebbero cadere a cavallo di due: serve un `TOPSAR-Merge`, che non
   c'è. Da verificare area per area prima di decidere se implementarlo.
3. **Baseline non costante.** Il ciclo di ripetizione è cambiato nel tempo
   (S1A+S1B a 6 giorni fino al 2021, S1A solo a 12, poi S1C+S1D di nuovo a 6 da
   metà 2026). Una serie lunga non ha una baseline unica: o si sceglie 12 giorni
   e si accetta di saltare i periodi che non la offrono, o si normalizza la
   coerenza per la baseline — e come farlo onestamente è una domanda aperta, non
   un dettaglio implementativo.
4. **Volume.** Gli SLC pesano circa quattro volte i GRD e la catena è molto più
   lenta. Per tre anni su un'area pilota si parla di centinaia di GB e di
   diverse ore-macchina.
5. **Come entra nel modello.** Oggi sarebbe un canale in più accanto a VV, VH e
   al rapporto. Un'alternativa più interessante è usarla come *edge feature*
   temporale (la coerenza è una proprietà della coppia, cioè dell'arco
   `(t-1, i) → (t, i)`, non del nodo): il modello attuale non ha archi con
   attributi, quindi sarebbe un cambiamento architetturale, non un canale.

**Ordine suggerito**: (1) su una coppia, poi decidere. Se la coerenza non separa
i cantieri nell'area pilota, le altre quattro voci non vanno affrontate.

## B2. Campionamento di vicinato

**Stato: non fatto.** Tutto è full-batch: un'area di studio è un grafo solo. Le
aree pilota sono dimensionate di conseguenza (~20–26 mila nodi). Per scalare a
una regione serve il *neighbour sampling* per cui GraphSAGE è nato. Il modello è
induttivo, quindi è un'aggiunta al data loader e non un cambio di architettura;
il lavoro vero è mantenere onesto lo split geografico quando i vicini vengono
campionati.

## B3. Le baseline mancanti dell'OS3

**Stato: `temporal`, `mlp` e `rf` ci sono; mancano le due che la proposta cita.**

- **U-Net siamese per-pixel**: lavora su raster, non sul grafo dei nodi. Serve
  un secondo percorso dati (patch raster + etichette rasterizzate) e un
  protocollo di confronto che non favorisca nessuno dei due — stesso split
  geografico, stesse metriche, stesse etichette annuali.
- **Foundation model EO** (Prithvi, Presto, Clay): messa a punto su questa
  serie. Costo principale: allineare la loro rappresentazione temporale con i
  compositi mensili.

## B4. Un rapporto di ingestione più severo

Oggi `gnn-sar ingest` stampa un riassunto. Servirebbe che scrivesse anche, in
un file accanto alla cache:

- quante acquisizioni per composito, con i vuoti evidenziati;
- copertura nuvolosa media per composito della sorgente ottica;
- distribuzione dei nodi per numero di pixel validi;
- istogramma dei valori delle etichette effettivamente trovati nei raster.

Sono i quattro grafici che si guardano quando un esperimento non funziona.

## C1. Ingestione via openEO

Alternativa all'ingestor locale: descrivere il datacube (area, periodo, bande,
`sar_backscatter`, mascheratura, aggregazione temporale) e ricevere i compositi
già pronti, senza SNAP e senza terabyte locali. Ridurrebbe drasticamente la
barriera d'ingresso; il costo è dipendere dalla disponibilità del servizio e
perdere il controllo fine sulla catena.

## C2. Geometrie catastali

`"segmentation": {"method": "labels"}` accetta già un raster di particelle, ma
rasterizzarne uno resta un lavoro esterno (GDAL + dati catastali, che non sono
aperti). Nodi = particelle è però la rappresentazione più vicina al modo in cui
il consumo di suolo avviene davvero, ed è quella che la proposta indica.

## C3. Trasferimento fra aree (OS4)

Il modello è induttivo per costruzione — nessuna identità di nodo è appresa — e
un test lo verifica su due grafi di dimensione diversa. Il passo successivo è
misurarlo davvero: addestrare su Viterbo, valutare su Milano senza
riaddestramento, e riportare quanto si perde. È la voce che dà il maggior
contenuto scientifico per unità di lavoro, ma richiede prima A1 su due aree.
