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
izbira pragot spored makro-F0.5 na razvojnite podatoci, a potoa ednash go
proveruva izbraniot model na test-skupot. F0.5 dava pogolema tezhina na
preciznosta otkolku na odzivot: obichno znachi pomalku pogreshno dodadeni
emocii, no mozhe da propushti povekje vistinski emocii. Grafikonot za
preciznost i odziv go prikazhuva ovoj kompromis. Ne ja namaluva namerno
neutralnata klasa. Makro-merki gi tretiraat emociite podednakvo; ocenite za
retkite emocii se ponepostojani bidejki imaat pomalku primeri.

Rezultatite se vo `results/ml/`: metriki po emocija, pregled na izborot na
postavki, test-rezime i akademski izvestaj. Grafikoni se sozdavaat za
preciznost/odziv po prag, preciznost/odziv po emocija, F1 po emocija,
ko-pojava na oznaki i kalibracija na skorovite.
`model.joblib` e lokalna datoteka i ne se dodava vo Git.

## Predviduvanje

Po obuka, vnesete tekst:

```text
python simple_ml.py predict --text "I am excited about this"
```

Se prikazhuvaat site emocii nad izbraniot prag so nivniot model-skor.
Skorot ne e psiholoshka procenka ili garancija za vistinskoto chuvstvo na
avtorot.

## Opisni izvestai so ML

Po obuka, stavete ja anotiranata CSV-datoteka vo `input/final_scored_comments.csv`
i izvrshite:

```text
python simple_analysis.py
```

Skriptata povtorno gi ocenuva tekstovite so VerA modelot i gi koristi tie
skorovi za dominantna emocija i emocionalna grupa. Dominantna e emocijata so
najvisok skor od site 28 emocii; ova e odvoeno od pragot za povekje oznaki.
Grafikonite za dolzhina, psovki, ton, subjektivnost i Reddit-zaednici gi
prikazhuvaat opisnite podatoci od CSV po VerA ML-grupa; tie ne se metriki za
uspeh na klasifikatorot. Grafikonite gi prikazhuvaat goleminite na grupite;
Reddit-grafikonot prikazhuva procentualen sostav na dominantnite ML-grupi
vo 10te zaednici so najmalku 200 komentari, so brojot na primeri za sekoja.
Kratko tekstualno tolkuvanje se zapishuva vo `results/izvestaj_opisen.md`.

CSV-so komentarite sodrzhi tekst i ne se vkluchuva vo repozitoriumot.
