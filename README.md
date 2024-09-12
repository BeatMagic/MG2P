# MG2P

Multilingual G2P API

## Install

***

```python
conda create --name your_env_name python==3.10
conda activate your_env_name
pip install -r requirements.txt
```
## Usage

```python
from MG2P.core.MG2P import MG2P
g2p = MG2P()
print(g2p('今でもあなたはわたしの光'))
#imademoanatahaM\ᵝatas\inoCika4i
```
## Acknowledgements

***

* lingjzhu's [CharsiuG2P](https://github.com/lingjzhu/CharsiuG2P)
* DumoeDss and ChuxiJ's [G2P](https://github.com/BeatMagic/g2p)
* juntaosun's [LangSegment](https://github.com/juntaosun/LangSegment)
* himkt's [konoha](https://github.com/himkt/konoha)
* fxsjy's [jieba](https://github.com/fxsjy/jieba)
* rkcosmos's [deepcut](https://github.com/rkcosmos/deepcut)
* jhasegaw's [phonecodes](https://github.com/jhasegaw/phonecodes)
* stefantaubert's [pinyin-to-ipa](https://github.com/stefantaubert/pinyin-to-ipa)