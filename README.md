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
lyrics = 'チャーシュー是一种barbecued pork'
lyrics1 = '踏碎凌霄 放肆桀骜'
lyrics2 = '今でもあなたはわたしの光'
lyrics3 = 'charsiu is a pork'
lyrics4 = '사랑해요'
print(g2p(lyrics))
# ['t', '͡', 'ɕ', 'a', 'a', 'ɕ', 'u', 'u', 'z', 'e', 'i', 't', '͡', 'ɕ', 'i', 'b', 'a', 'ɾ', 'β', 'e', 'k', 'w', 'e', 'ð', 'p', 'o', 'ɾ', 'k'], 
# ['t', '͡', 's\\', 'a', 'a', 's\\', 'u', 'u', 'z', 'e', 'i', 't', '͡', 's\\', 'i', 'b', 'a', '4', 'B', 'e', 'k', 'w', 'e', 'D', 'p', 'o', '4', 'k']
print(g2p(lyrics1))
# ['tʰ', 'a', '⁴', 's', 'w', 'ei̯', '⁴', 'l', 'i', '²', 'ŋ', 'ɕ', 'j', 'au̯', '¹', 'f', 'a', '⁴', 'ŋ', 's', 'ɹ̩', '⁴', 'tɕ', 'j', 'e', '²', 'au̯', '⁴'], 
# ['t_h', 'a', '_4', 's', 'w', 'ei_^', '_4', 'l', 'i', '_2', 'N', 's\\', 'j', 'au_^', '_1', 'f', 'a', '_4', 'N', 's', 'r\\_=', '_4', 'ts\\', 'j', 'e', '_2', 'au_^', '_4']
print(g2p(lyrics2))
# ['i', 'm', 'a', 'd', 'e', 'm', 'o', 'a', 'n', 'a', 't', 'a', 'w', 'a', 'w', 'a', 't', 'a', 'ɕ', 'i', 'n', 'o', 'h', 'i', 'k', 'a', 'ɾ', 'i'], 
# ['i', 'm', 'a', 'd', 'e', 'm', 'o', 'a', 'n', 'a', 't', 'a', 'w', 'a', 'w', 'a', 't', 'a', 's\\', 'i', 'n', 'o', 'h', 'i', 'k', 'a', '4', 'i']
print(g2p(lyrics3))
# ['t', 'ʃ', 'ˈ', 'ɑ', 'ɹ', 's', 'ˈ', 'i', 'u', 'ˈ', 'ɪ', 'z', 'ə', 'p', 'ˈ', 'ɔ', 'ɹ', 'k'], 
# ['t', 'S', '"', 'A', 'r\\', 's', '"', 'i', 'u', '"', 'I', 'z', '@', 'p', '"', 'O', 'r\\', 'k']
print(g2p(lyrics4))
# ['sʰ', 'a̠', 'ɾ', 'a̠', 'ŋ', 'ɦ', 'ɛ̝', 'j', 'o'], 
# ['s_h', 'a_-', '4', 'a_-', 'N', 'h\\', 'E_r', 'j', 'o']

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
* CUNY-CL's [wikipron](https://github.com/CUNY-CL/wikipron)