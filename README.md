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
# ['t', 'S', '"', 'A', 'r\\', 's', '"', 'i', 'u', '"', 'I', 'z', '@', 'p', '"', 'O', 'r\\', 'k']
print(g2p(lyrics1, 'en'))
# ['t', 'S', '"', 'A', 'r\\', 's', '"', 'i', 'u', '"', 'I', 'z', '@', 'p', '"', 'O', 'r\\', 'k']
print(g2p(lyrics2))
# ['t_h', 'a_1_5', 's', 'w', 'ei_^_1_5', 'l', 'i_3_1', 'N', 's\\', 'j', 'au_^_1', 'f', 'a_1_5', 'N', 's', 'r\\_=_1_5', 'ts\\', 'j', 'e_3_1', 'au_^_1_5']
print(g2p(lyrics3))
# ['i', 'm', 'a', 'd', 'e', 'm', 'o', 'a', 'n', 'a', 't', 'a', 'w', 'a', 'w', 'a', 't', 'a', 's\\', 'i', 'n', 'o', 'h', 'i', 'k', 'a', '4', 'i']
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