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

## Citation

If you think our work useful, please kindly cite our paper.

```
@inproceedings{cao-etal-2025-human,
    title = "Human-in-the-Loop Generation of Adversarial Texts: A Case Study on {T}ibetan Script",
    author = "Cao, Xi  and
      Sun, Yuan  and
      Li, Jiajun  and
      Gesang, Quzong  and
      Qun, Nuo  and
      Tashi, Nyima",
    editor = "Liu, Xuebo  and
      Purwarianti, Ayu",
    booktitle = "Proceedings of The 14th International Joint Conference on Natural Language Processing and The 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics: System Demonstrations",
    month = dec,
    year = "2025",
    address = "Mumbai, India",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.ijcnlp-demo.2/",
    pages = "9--16",
    ISBN = "979-8-89176-301-2"
}
```

```
@INPROCEEDINGS{10889732,
  author={Cao, Xi and Gesang, Quzong and Sun, Yuan and Qun, Nuo and Nyima, Tashi},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={TSCheater: Generating High-Quality Tibetan Adversarial Texts via Visual Similarity}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ICASSP49660.2025.10889732}}
```
