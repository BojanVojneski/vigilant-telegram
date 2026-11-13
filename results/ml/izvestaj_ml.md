# VerA: izvestaj za klasifikacija na emocii

## Metod

Zadachata e povekje-oznacna klasifikacija na tekst od GoEmotions. Modelot
koristi TF-IDF zborovni n-grami i logistichka regresija so one-vs-rest
klasifikacija. Izborot na C, n-grami i zaednichki prag e napraven samo na
trenirachkite i razvojnite podatoci. Pragot e izbran spored najvisok makro-F1
na razvojnata podelba; test-podelbata ne e koristena za izbor na model.

| Stavka | Vrednost |
|---|---:|
| C | 5.0 |
| N-gram opseg | (1, 2) |
| Izbran prag | 0.125 |
| Trenirachki komentari | 43,410 |
| Razvojni komentari | 5,426 |
| Test komentari | 5,427 |
| Razvoen makro-F1 | 0.447 |

## Rezultati na test-podelbata

| Merka | Vrednost |
|---|---:|
| Makro-preciznost | 0.426 |
| Makro-odziv | 0.478 |
| Makro-F1 | 0.435 |
| Mikro-F1 | 0.530 |
| Tochno poklopeni skupovi oznaki | 0.264 |

Makro-F1 ja presmetuva prosechnata F1 vrednost so ednakva tezhina za sekoja
emocija. Mikro-F1 gi sobira odlukite za site emocii pred presmetka.
Tochnoto poklopuvanje bara modelot da gi predvidi site oznaki na komentarot
bez greska, pa ovaa merka e postroga.

## Najvisoki i najniski F1 po emocija

| Emocija | Pozitivni oznaki | F1 |
|---|---:|---:|
| Blagodarnost | 352 | 0.895 |
| Zabava | 264 | 0.799 |
| Ljubov | 238 | 0.765 |
| Strav | 78 | 0.676 |
| Kajanje | 56 | 0.662 |

| Emocija | Pozitivni oznaki | F1 |
|---|---:|---:|
| Zhalost | 6 | 0.000 |
| Nervoza | 23 | 0.129 |
| Olesnuvanje | 11 | 0.143 |
| Razocharuvanje | 151 | 0.223 |
| Grizhlivost | 135 | 0.256 |

## Grafichki prikazi

- `pragovi_makro_f1.png`: izborot na prag vrz osnova na razvojniot skup.
- `f1_po_emocija.png`: test-F1 i brojot na pozitivni primeri za sekoja emocija.
- `preciznost_i_odziv.png`: odnos megju preciznost i odziv za 10te emocii so najgolema poddrshka.
- `ko_pojava_emocii.png`: uslovna ko-pojava na 12te emocii so najgolema trenirachka poddrshka.
- `kalibracija_skorovi.png`: sporedba na model-skorot so empiriskiot del na pozitivni oznaki za pette najchesti emocii.

## Tolkuvanje i ogranicuvanja

Rezultatite ja merat soglasnosta so oznakite vo ovaa podelba na GoEmotions,
ne vistinskite chuvstva na avtorite. Oznakite se subjektivni i nekoi emocii
imaat mal test-primerok, pa nivnite F1 vrednosti se ponepostojani. Modelot
e ednostavna pochetna osnova, ne psiholoshki alat. Grafikonot za kalibracija
e dijagnostika; skorovite ne se koregirani ili potvrdeni kako apsolutni
verojatnosti.

Ne se koristeni hipotezi, p-vrednosti ili prichinski tvrdenja.
