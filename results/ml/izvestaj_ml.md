# VerA: izvestaj za klasifikacija na emocii

## Metod

Zadachata e povekje-oznacna klasifikacija na tekst od GoEmotions. Modelot
koristi TF-IDF zborovni n-grami i logistichka regresija so one-vs-rest
klasifikacija. Izborot na C, n-grami i zaednichki prag e napraven samo na
trenirachkite i razvojnite podatoci. Pragot se izbira spored najvisok prosechen
F0.5 po emocija na razvojnata podelba; test-podelbata ne se koristi za izbor.

| Stavka | Vrednost |
|---|---:|
| C | 5.0 |
| N-gram opseg | [1, 1] |
| Izbran prag | 0.250 |
| Izbor na prag | Makro-F0.5 na razvojni podatoci |
| Trenirachki komentari | 43,410 |
| Razvojni komentari | 5,426 |
| Test komentari | 5,427 |
| Razvoen makro-F0.5 | 0.474 |
| Razvoena makro-preciznost | 0.559 |
| Razvoen makro-odziv | 0.373 |

## Rezultati na test-podelbata

| Merka | Vrednost |
|---|---:|
| Makro-preciznost | 0.514 |
| Makro-odziv | 0.382 |
| Makro-F1 | 0.423 |
| Makro-F0.5 | 0.467 |
| Mikro-F1 | 0.542 |
| Tochno poklopeni skupovi oznaki | 0.387 |

Makro-F1 ja presmetuva prosechnata F1 vrednost so ednakva tezhina za sekoja
emocija. Pragot se bira spored prosechniot F0.5 po emocija, koj dava pogolema
tezhina na preciznosta od odzivot; toa e nameren kompromis i obichno znachi
pomalku predvideni oznaki i pomal odziv. Mikro-F1 gi sobira odlukite za site
emocii pred presmetka.
Tochnoto poklopuvanje bara modelot da gi predvidi site oznaki na komentarot
bez greska, pa ovaa merka e postroga.

## Najvisoki i najniski F1 po emocija

| Emocija | Pozitivni oznaki | F1 |
|---|---:|---:|
| Blagodarnost | 352 | 0.915 |
| Zabava | 264 | 0.752 |
| Ljubov | 238 | 0.738 |
| Neutralno | 1787 | 0.638 |
| Strav | 78 | 0.636 |

| Emocija | Pozitivni oznaki | F1 |
|---|---:|---:|
| Nervoza | 23 | 0.125 |
| Olesnuvanje | 11 | 0.143 |
| Razocharuvanje | 151 | 0.185 |
| Osoznavanje | 145 | 0.250 |
| Neodobruvanje | 267 | 0.259 |

## Grafichki prikazi

- `preciznost_odziv_po_pragu.png`: kompromis megju preciznost, odziv i makro-F0.5 na razvojniot skup.
- `f1_po_emocija.png`: test-F1 i brojot na pozitivni primeri za sekoja emocija.
- `preciznost_i_odziv.png`: sporedba na preciznost i odziv za site 28 emocii na test-skupot.
- `ko_pojava_emocii.png`: procentot so emocija od kolonata megju komentarite so emocija od redot; gi prikazhuva 12te najchesti trenirachki emocii, bez trivijalnata dijagonala.
- `kalibracija_skorovi.png`: sporedba na model-skorot so empiriskiot del na pozitivni oznaki za pette najchesti emocii.

## Tolkuvanje i ogranicuvanja

Rezultatite ja merat soglasnosta so oznakite vo ovaa podelba na GoEmotions,
ne vistinskite chuvstva na avtorite. Oznakite se subjektivni i nekoi emocii
imaat mal test-primerok, pa nivnite F1 vrednosti se ponepostojani. Modelot
e ednostavna pochetna osnova, ne psiholoshki alat. Grafikonot za kalibracija
e dijagnostika; skorovite ne se koregirani ili potvrdeni kako apsolutni
verojatnosti.

Ne se koristeni hipotezi, p-vrednosti ili prichinski tvrdenja.
