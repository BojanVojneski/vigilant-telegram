# VerA - Ednostaven model za emocii

VerA e samostojna programa za prepoznavanje povekje emocii vo tekst.
Koristi TF-IDF i logistichka regresija; ne e del od pogolemiot H1-H5
istrazhuvachki pipeline.

## Podgotovka

Vo `data/simplified/` treba da se najdat `train.tsv`, `dev.tsv`,
`test.tsv` i `emotions.txt`. Toa se standardnite podelbi od GoEmotions.
Ne gi koristete `test.tsv` ili test-rezultatite za izbor na model ili prag.
TSV-datotekite so komentari se lokalni podatoci i ne se dodavaat vo Git.
Nabavete gi od [oficijalniot GoEmotions repo](https://github.com/google-research/google-research/tree/master/goemotions/data).

Instalirajte gi paketite:

```text
python -m pip install -r requirements.txt
```

## Obuka i procenka

```text
python simple_ml.py train
```

Programata sporeduva mal broj TF-IDF postavki na trenirachkiot skup, go
izbira pragot spored makro-F1 na razvojnite podatoci, a potoa ednash go
proveruva izbraniot model na test-skupot. Ne ja namaluva namerno
neutralnata klasa. Makro-F1 gi tretira emociite podednakvo; ocenite za
retkite emocii se ponepostojani bidejki imaat pomalku primeri.

Rezultatite se vo `results/ml/`: metriki po emocija, pregled na izborot na
postavki, test-rezime i akademski izvestaj. Grafikoni se sozdavaat za pragovite,
F1 po emocija, preciznost/odziv, ko-pojava na oznaki i kalibracija na skorovite.
`model.joblib` e lokalna datoteka i ne se dodava vo Git.

## Predviduvanje

Po obuka, vnesete tekst:

```text
python simple_ml.py predict --text "I am excited about this"
```

Se prikazhuvaat site emocii nad izbraniot prag so nivniot model-skor.
Skorot ne e psiholoshka procenka ili garancija za vistinskoto chuvstvo na
avtorot.

`simple_analysis.py` e dopolnitelen izvestaj za vekje oceneti komentari;
ne trenira model. CSV-so komentarite ne se vkluchuva vo repozitoriumot.
