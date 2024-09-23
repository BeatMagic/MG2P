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

该使用方法已弃用

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

***

新增并行处理方法, 请注意: 带有tag_list参数时, 仅支持每一句单语言!

```python
from MG2P.core.MG2P import MG2P

charsiu_path = 'your charsiu model path'
tokenizer_path = 'your google-byt5 model path'
g2p = MG2P(charsiu_path, tokenizer_path)
lyrics = ['チャーシュー是一种barbecued pork',
          'Ich liebe dich踏碎凌霄สวัสดี',
          'わたしの光넌 나의 빛',
          'charsiu is a pork',
          '사랑해요我爱你i love you']
print(g2p(lyrics))
#[(ipa_list,xsampa_list),(ipa_list,xsampa_list),...]
# [(['t', '͡', 'ɕ', 'a', 'a', 'ɕ', 'u', 'u', 'z', 'e', 'i', 't', '͡', 'ɕ', 'i', 'b', 'a', 'ɾ', 'β', 'e', 'k', 'w', 'e', 'ð', 'p', 'o', 'ɾ', 'k'], 
# ['t', '͡', 's\\', 'a', 'a', 's\\', 'u', 'u', 'z', 'e', 'i', 't', '͡', 's\\', 'i', 'b', 'a', '4', 'B', 'e', 'k', 'w', 'e', 'D', 'p', 'o', '4', 'k']), 
# (['ɪ', 'ç', 'ˈ', 'l', 'iː', 'b', 'ə', 'd', 'ɪ', 'ç', 'tʰ', 'a', '⁴', 's', 'w', 'ei̯', '⁴', 'l', 'i', '²', 'ŋ', 'ɕ', 'j', 'au̯', '¹', 's', 'u', 'a̯', '˩˩˦', 's', 'o', 't̚', '˨˩', 'h', 'a', '˧', '˥'], 
# ['I', 'C', '"', 'l', 'i:', 'b', '@', 'd', 'I', 'C', 't_h', 'a', '_4', 's', 'w', 'ei_^', '_4', 'l', 'i', '_2', 'N', 's\\', 'j', 'au_^', '_1', 's', 'u', 'a_^', '_5_5_2', 's', 'o', 't_}', '_4_5', 'h', 'a', '_3', '_1']), 
# (['w', 'a', 't', 'a', 'ɕ', 'i', 'n', 'o', 'h', 'i', 'k', 'a', 'ɾ', 'i', 'n', 'ʌ̹', 'n', 'n', 'a̠', 'ɰ', 'i', 'p', 'i', 't̚'], 
# ['w', 'a', 't', 'a', 's\\', 'i', 'n', 'o', 'h', 'i', 'k', 'a', '4', 'i', 'n', 'V_O', 'n', 'n', 'a_-', 'M\\', 'i', 'p', 'i', 't_}']), 
# (['t', 'ʃ', 'ˈ', 'ɑ', 'ɹ', 's', 'ˈ', 'i', 'u', 'ˈ', 'ɪ', 'z', 'ə', 'p', 'ˈ', 'ɔ', 'ɹ', 'k'], 
# ['t', 'S', '"', 'A', 'r\\', 's', '"', 'i', 'u', '"', 'I', 'z', '@', 'p', '"', 'O', 'r\\', 'k']), 
# (['sʰ', 'a̠', 'ɾ', 'a̠', 'ŋ', 'ɦ', 'ɛ̝', 'j', 'o', 'w', 'o', '³', 'ai̯', '⁴', 'n', 'i', '³', 'ˈ', 'a', 'ɪ', 'l', 'ˈ', 'ʌ', 'v', 'j', 'ˈ', 'u'], 
# ['s_h', 'a_-', '4', 'a_-', 'N', 'h\\', 'E_r', 'j', 'o', 'w', 'o', '_3', 'ai_^', '_4', 'n', 'i', '_3', '"', 'a', 'I', 'l', '"', 'V', 'v', 'j', '"', 'u'])]
lyrics = ['barbecued pork',
          'Ich liebe dich',
          'わたしの光',
          '我爱你',
          '사랑해요넌 나의 빛']
tags = ['en','de','ja','zh','ko']
print(g2p(lyrics,tags))
#[(ipa_list,xsampa_list),(ipa_list,xsampa_list),...]
# [(['b', 'ˈ', 'ɑ', 'ɹ', 'b', 'ɨ', 'k', 'j', 'ˌ', 'u', 'd', 'p', 'ˈ', 'ɔ', 'ɹ', 'k'], 
# ['b', '"', 'A', 'r\\', 'b', '1', 'k', 'j', '%', 'u', 'd', 'p', '"', 'O', 'r\\', 'k']), 
# (['ˈ', 'i', 'χ', 'ˈ', 'l', 'iː', 'b', 'ə', 'j', 'i', '˧˥'], 
# ['"', 'i', 'X', '"', 'l', 'i:', 'b', '@', 'j', 'i', '_3_1']), 
# (['w', 'a', 't', 'a', 'ɕ', 'i', 'n', 'o', 'h', 'i', 'k', 'a', 'ɾ', 'i'], 
# ['w', 'a', 't', 'a', 's\\', 'i', 'n', 'o', 'h', 'i', 'k', 'a', '4', 'i']), 
# (['w', 'o', '³', 'ai̯', '⁴', 'n', 'i', '³'], 
# ['w', 'o', '_3', 'ai_^', '_4', 'n', 'i', '_3']), 
# (['sʰ', 'a', '̠', 'ɾ', 'a', '̠', 'ŋ', 'ɦ', 'ɛ̝', 'j', 'o', 'n', 'ʌ', '̹', 'n', 'n', 'a', '̠', 'ɰ', 'i', 'p', 'i', 't̚'], 
# ['s_h', 'a', '_-', '4', 'a', '_-', 'N', 'h\\', 'E_r', 'j', 'o', 'n', 'V', '_O', 'n', 'n', 'a', '_-', 'M\\', 'i', 'p', 'i', 't_}'])]
```


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