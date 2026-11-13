# Simplified GoEmotions data

Place the official `train.tsv`, `dev.tsv`, `test.tsv`, and `emotions.txt` files
from the GoEmotions dataset here. The TSV files contain comment text and are
ignored by Git in this project.

Source:
<https://github.com/google-research/google-research/tree/master/goemotions/data>.

`ekman_mapping.json` and `sentiment_mapping.json` are copied from the official
GoEmotions repository:
<https://github.com/google-research/google-research/tree/master/goemotions/data>.

The official sentiment mapping has positive, negative, and ambiguous groups;
the `neutral` label is not included in that mapping and should be handled as
its own class where the project requires four groups.
