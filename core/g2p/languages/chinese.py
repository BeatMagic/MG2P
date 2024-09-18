import os
import re

import jieba_fast.posseg as psg
from pypinyin import lazy_pinyin, Style
import cn2an

from .symbol import punctuation
from .tone_sandhi import ToneSandhi
from .zh_normalization.text_normlization import TextNormalizer
import hanlp

rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": "…",
    "$": ".",
    "/": ",",
    "—": "-",
    "~": "…",
    "～": "…",
}
POS_MAP = {
    'AD': 'd',
    'AS': 'u',
    'BA': 'p',
    'CC': 'c',
    'CD': 'm',
    'CS': 'c',
    'DEC': 'uj',
    'DEG': 'uj',
    'DER': 'ud',
    'DEV': 'uv',
    'DT': 'r',
    'ETC': 'u',
    'FW': 'eng',
    'IJ': 'zg',
    'JJ': 'a',
    'LB': 'p',
    'LC': 'f',
    'M': 'm',
    'MSP': 'u',
    'NN': 'n',
    'NR': 'nr',
    'NT': 'm',
    'OD': 'm',
    'ON': 'o',
    'P': 'p',
    'PN': 'r',
    'SB': 'p',
    'SP': 'y',
    'VA': 'a',
    'VC': 'v',
    'VE': 'v',
    'VV': 'v',
}

current_file_path = os.path.dirname(__file__)
pinyin_to_symbol_map = {
    line.split("\t")[0]: line.strip().split("\t")[1]
    for line in open(os.path.join(current_file_path, "opencpop-strict.txt")).readlines()
}

tone_modifier = ToneSandhi()


def normalizer(x):
    return cn2an.transform(x, "an2cn")


def replace_punctuation(text):
    text = text.replace("嗯", "恩").replace("呣", "母")
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))

    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

    replaced_text = re.sub(
        r"[^\u4e00-\u9fa5" + "".join(punctuation) + r"]+", "", replaced_text
    )

    return replaced_text


def g2p(text):
    pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
    sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
    phones, word2ph = _g2p(sentences)
    return phones, word2ph


def _get_initials_finals(word):
    initials = []
    finals = []
    orig_initials = lazy_pinyin(
        word, neutral_tone_with_five=True, style=Style.INITIALS)
    orig_finals = lazy_pinyin(
        word, neutral_tone_with_five=True, style=Style.FINALS_TONE3
    )
    for c, v in zip(orig_initials, orig_finals):
        initials.append(c)
        finals.append(v)
    return initials, finals


def _g2p(segments):
    phones_list = []
    word2ph = []
    tok_fine = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
    pos = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL)
    for seg in segments:
        # Replace all English words in the sentence
        seg = re.sub("[a-zA-Z]+", "", seg)
        # seg_cut = psg.lcut(seg)
        token_list = tok_fine(seg)
        pos_list = pos(token_list)
        initials = []
        finals = []
        # seg_cut = tone_modifier.pre_merge_for_modify(seg_cut)
        # for word, pos in seg_cut:
        for word, pos in zip(token_list, pos_list):
            if pos == "eng":
                continue
            sub_initials, sub_finals = _get_initials_finals(word)
            sub_finals = tone_modifier.modified_tone(word, pos, sub_finals)
            initials.append(sub_initials)
            finals.append(sub_finals)
        initials = sum(initials, [])
        finals = sum(finals, [])
        #
        for c, v in zip(initials, finals):
            raw_pinyin = c + v
            # NOTE: post process for pypinyin outputs
            # we discriminate i, ii and iii
            if c == v:
                assert c in punctuation
                phone = [c]
                word2ph.append(1)
            else:
                v_without_tone = v[:-1]
                tone = v[-1]

                pinyin = c + v_without_tone
                assert tone in "12345"

                if c:
                    # 多音节
                    v_rep_map = {
                        "uei": "ui",
                        "iou": "iu",
                        "uen": "un",
                    }
                    if v_without_tone in v_rep_map.keys():
                        pinyin = c + v_rep_map[v_without_tone]
                else:
                    # 单音节
                    pinyin_rep_map = {
                        "ing": "ying",
                        "i": "yi",
                        "in": "yin",
                        "u": "wu",
                    }
                    if pinyin in pinyin_rep_map.keys():
                        pinyin = pinyin_rep_map[pinyin]
                    else:
                        single_rep_map = {
                            "v": "yu",
                            "e": "e",
                            "i": "y",
                            "u": "w",
                        }
                        if pinyin[0] in single_rep_map.keys():
                            pinyin = single_rep_map[pinyin[0]] + pinyin[1:]

                assert pinyin in pinyin_to_symbol_map.keys(), (pinyin, seg, raw_pinyin)
                new_c, new_v = pinyin_to_symbol_map[pinyin].split(" ")
                new_v = new_v + tone
                phone = [new_c, new_v]
                word2ph.append(len(phone))

            phones_list += phone
    return phones_list, word2ph


def text_normalize(text):
    # https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/zh_normalization
    tx = TextNormalizer()
    sentences = tx.normalize(text)
    dest_text = ""
    for sentence in sentences:
        dest_text += replace_punctuation(sentence)
    return dest_text


if __name__ == "__main__":
    text = "啊——但是《原神》是由,米哈\游自主，研发的一款全.新开放世界.冒险游戏"
    text = "呣呣呣～就是…大人的鼹鼠党吧？"
    text = "你好"
    text = text_normalize(text)
    print(g2p(text))
