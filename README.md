# MG2P

Multilingual G2P API

## Install


```shell
conda create --name your_env_name python==3.10
conda activate your_env_name
pip install -r requirements.txt
```

## Usage

```
.
|--MG2P
|--test.py
```

The first time you use the repo, some pre-trained models are downloaded remotely (maybe slow).

```python
from MG2P.core.MG2P import MG2P

# test.py
g2p = MG2P()
lyrics1 = 'チャーシュー是一种barbecued pork'
lyrics2 = '踏碎凌霄 放肆桀骜'
lyrics3 = '今でもあなたはわたしの光'
print(g2p(lyrics1))
# t͡s\aas\uuzeit͡s\ib"Ar\b1kj%udp"Or\k
print(g2p(lyrics1, 'en'))
# t͡s\aas\uuzeit͡s\ib"Ar\b1kj%udp"Or\k
print(g2p(lyrics2))
# t_ha_1_5swei_^_1_5li_3_1Ns\jau_^_1fa_1_5Nsr\_=_1_5ts\je_3_1au_^_1_5
print(g2p(lyrics3))
# imademoanatawawatas\inohika4i
```

note: the model expects a 639-1 code of the language in the input lyrics, if this field is ignored the model
automatically determines the language

## Acknowledgements


* lingjzhu's [CharsiuG2P](https://github.com/lingjzhu/CharsiuG2P)
* DumoeDss and ChuxiJ's [G2P](https://github.com/BeatMagic/g2p)
* juntaosun's [LangSegment](https://github.com/juntaosun/LangSegment)
* himkt's [konoha](https://github.com/himkt/konoha)
* fxsjy's [jieba](https://github.com/fxsjy/jieba)
* hankcs's [HanLP](https://github.com/hankcs/HanLP)
* rkcosmos's [deepcut](https://github.com/rkcosmos/deepcut)
* jhasegaw's [phonecodes](https://github.com/jhasegaw/phonecodes)
* stefantaubert's [pinyin-to-ipa](https://github.com/stefantaubert/pinyin-to-ipa)
* ChuxiJ's [romaji2ipa](https://ec26ubh65w.feishu.cn/sheets/FD37spdiLhcGeEtEaFucOPX4nGg?sheet=543d9b)