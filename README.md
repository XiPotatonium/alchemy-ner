# alchemy-ner

You can directly create a project based on this template to develop your own NER models.

To use alchemy, please initialize this submodule after cloning this repo by:

```bash
git submodule update --init --recursive
```

We implement the `src.task.ner.NerTask` based on `AlchemyTask`.
Within `NerTask`, meta information, data processing, output processing and evaluation are defined.

We also implement some baselines for the convenience of comparison.

## Implemented baselines

* Tagging model (`src.models.tagger.Tagger`): A simple bert-base sequence tagger with BIO taggin scheme.
* BERT-CRF model (`src.models.tagger.CRFTagger`): A simple bert-base CRF tagger with BIO taggin scheme.
* Biaffine (`src.models.biaffine.Biaffine`): A span-based model to detect nested named entities. [[code]](https://github.com/juntaoy/biaffine-ner) [[paper]](https://aclanthology.org/2020.acl-main.577).
* Propose-and-Refine (`src.models.pnr.PnRNet`): A two-stage set prediction model to detect nested named entities. [[code]](https://github.com/XiPotatonium/pnr) [[paper]](https://www.ijcai.org/proceedings/2022/613).

## Meta information

## Data processing

## Output processing

## Evaluation
