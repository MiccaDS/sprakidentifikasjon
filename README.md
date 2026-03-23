# Tidligere prosjekt i språkteknologi på Universitetet i Oslo

## Målet med oppgaven
Denne innleveringen har to hoveddeler og gir praktisk erfaring med grunnleggende maskinlæringsmetoder i språkteknologi:

1. **Del 1: Språkidentifikasjon med logistisk regresjon**  
   Bygge en enkel språkklassifiserer som tar fonetisk transkripsjon (IPA) av et ord som input og predikerer hvilket språk ordet tilhører (f.eks. "[bʊndɛsvɛɾfaszʊŋ]" → "tysk").  
   Vi bruker kun binære "bag-of-sounds"-features (tilstedeværelse av IPA-symboler) og trener en multi-klasse logistisk regresjon med scikit-learn.

2. **Del 2: Named Entity Recognition (NER) med Hidden Markov Model (HMM)**  
   Implementere en sekvensmodell som gjenkjenner navngitte entiteter (personer, organisasjoner, steder osv.) i norsk tekst ved hjelp av BIO-tagging, transition- og emission-sannsynligheter, og dekoding med beam search.

Målet er å forstå feature-ekstraksjon, trening av generative og diskriminative modeller, evaluering (accuracy, precision, recall, F1), og grunnleggende sekvensmodellering.

## Hva jeg har gjort
- Implementert hele `LanguageIdentifier`-klassen (feature-ekstraksjon, trening, prediksjon og evaluering).  
- Implementert hele `NamedEntityRecogniser`-klassen (`_add_counts`, `_fill_probs`, `_beam_search`, BIO-tagging og `label()`).  
- Trenet og testet begge modellene på de gitte datasettene.  
- Lagt inn forklaringer, begrunnelser, tabeller og refleksjoner i Jupyter notebooken som kreves.  
- Oppnådd ~93–94 % accuracy på språkidentifikasjon og fungerende NER-tagging (med typiske variasjoner fra beam search).


## Hvordan kjøre
1. Du må klone repoet, og kjøre pip install numpy pandas scikit-learn
2. Kjør filen språktek.py 

## Merknader
- Innleveringen består av den fylte notebooken med kode + tekstlige forklaringer.  
- Alt er testet og fungerer på det offisielle datasettet.

---

**Kort oppsummert:**  
Dette er en klassisk intro til språkteknologi: fra enkel logistisk regresjon på lydsymboler til en full HMM-basert NER-pipeline.
