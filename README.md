# TSAttack: A Textual Adversarial Attack Toolkit for Tibetan Script

## Victim Models
You can find the victim models in [our Hugging Face collection](https://huggingface.co/collections/UTibetNLP/tibetan-victim-language-models-669f614ecea872c7211c121c) and the construction process in [our GitHub repository](https://github.com/metaphors/TibetanPLMsFineTuning). 

## Requirements
You can `python install -r requirements.txt`.
```requirements
OpenAttack==2.1.1
sentencepiece==0.2.0
scikit-learn==1.5.1
```

## Using Examples
```shell
python examples/OpenAttack/Tibetan-BERT+TU_SA.py
python examples/OpenAttack/Tibetan-BERT+TNCC-title.py
python examples/OpenAttack/Tibetan-BERT+TNCC-document.py
python examples/OpenAttack/CINO-small-v2+TU_SA.py
python examples/OpenAttack/CINO-small-v2+TNCC-title.py
python examples/OpenAttack/CINO-small-v2+TNCC-document.py
python examples/OpenAttack/CINO-base-v2+TU_SA.py
python examples/OpenAttack/CINO-base-v2+TNCC-title.py
python examples/OpenAttack/CINO-base-v2+TNCC-document.py
python examples/OpenAttack/CINO-large-v2+TU_SA.py
python examples/OpenAttack/CINO-large-v2+TNCC-title.py
python examples/OpenAttack/CINO-large-v2+TNCC-document.py
```

## First Adversarial Robustness Benchmark for Tibetan Script
AdvTS.zip  
password: 2024

## BTW
This [new repo](https://github.com/metaphors/TSAttack) is built upon the Python package of OpenAttack v2.1.1 (concise & elegant).  
The [old repo](https://github.com/metaphors/TibetanAdversarialAttack) is built upon the source code of OpenAttack HEAD (what a mess).