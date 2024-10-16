import re
import numpy as np
from MG2P.core.phonecodes import ipa2xsampa, arpabet2ipa
from pinyin_to_ipa import pinyin_to_ipa
import os
import json
import LangSegment
from collections import deque
from loguru import logger


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

PITCH_CONTOUR_TO_IPA = {
    "˥": "¹",
    "˧˥": "²",
    "˧˩˧": "³",
    "˥˩": "⁴",
    "0": "º"
}

YUE_PITCH_CONTOUR_TO_IPA = {
    "˥": "¹",
    "˥˧": "¹",
    "˧˥": "²",
    "˧": "³",
    "˨˩": "⁴",
    "˩": "⁴",
    "˩˧": "⁵",
    "˨": "⁶"
}

ALL_LANGUAGES = ['af', 'am', 'an', 'ar', 'as', 'az', 'be', 'bg', 'bn', 'br', 'bs', 'ca', 'cs', 'cy', 'da', 'de', 'dz', 'el',
                 'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fo', 'fr', 'ga', 'gl', 'gu', 'he', 'hi', 'hr', 'ht', 'hu', 'hy',
                 'id', 'is', 'it', 'ja', 'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'ku', 'ky', 'la', 'lb', 'lo', 'lt', 'lv', 'mg',
                 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'nb', 'ne', 'nl', 'nn', 'no', 'oc', 'or', 'pa', 'pl', 'ps', 'pt', 'qu',
                 'ro', 'ru', 'rw', 'se', 'si', 'sk', 'sl', 'sq', 'sr', 'sv', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'ug', 'uk',
                 'ur', 'vi', 'vo', 'wa', 'xh', 'zh', 'zu']


def generate_sup_language_list() -> dict:
    """
    generate support languages list
    """
    mapping = {}
    tsv_path = "MG2P/core/639_1toPrefix.tsv"
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            language, prefix = parts[0], parts[1]
            mapping[language] = prefix
    return mapping


def clean_lyrics(lyrics: str) -> str:
    """
    Clean lyrics by rules, currently supported:
    timestamp, line break symbol, punctuation
    """
    # "\n\n\n攀登高峰望故乡\n |||黄沙万里[02:03:37]何处传来[02:03:327]驼铃声...声声敲心坎,盼望踏]]{上思念路[02:37]。飞纵千里山{]\n天边归雁披彩霞。"
    # line break symbol: \n, \n\n...
    lyrics = re.sub(r'\n\s*\n|\s*\n\s*', ' ', lyrics)
    # timestamp: [12:03],[12:03:02],[12:03:023],[12:03.02],[12:03.023]
    # lyrics = re.sub(r'\[\d+:\d+[:.]\d+]', ' ', lyrics)
    lyrics = re.sub(r'\[\d+:\d+[:.]?\d*\]', ' ', lyrics)
    # punctuation, only keep the English apostrophe: ,，.。!！?？....
    lyrics = re.sub(r'[^\w\s\']+', ' ', lyrics)
    # \u3000: full-width space
    lyrics = re.sub(r'\u3000', ' ', lyrics)
    return lyrics


def generate_prefix_code(lang_code: str) -> str:
    """
    generate Charsiu's prefix code
    """
    map = generate_sup_language_list()
    if lang_code in map:
        return '<' + map[lang_code] + '>: '
    else:
        return '<unk>: '


def jp_tokenizer(tokenizer, lyrics: str) -> list:
    result_list = tokenizer.tokenize(lyrics)
    result_list = [str(item) for item in result_list]
    if len(result_list) > 0:
        for i, _ in enumerate(result_list):
            result_list[i] = result_list[i].replace('▁', '')
            if result_list[i] == ' ' or result_list[i] == '':
                result_list.pop(i)
    return result_list


def pinyin_len_coefficient(pinyin: list) -> int:
    flag = 0
    for i in pinyin:
        if re.search(r'\d', i):
            flag += 1
    return flag


def zh_tone_backend(ipa: list):
    tone_ipa2xsampa = {"¹": "_1", "²": "_2", "³": "_3", "⁴": "_4", "0": "_0"}
    processed_ipa = []
    processed_xsampa = []
    ipa = deque(ipa)
    for tok in ipa:
        if tok in PITCH_CONTOUR_TO_IPA:
            processed_ipa.append(tok)
            # processed_ipa.append(PITCH_CONTOUR_TO_IPA[tok])
            processed_xsampa.append(tone_ipa2xsampa[PITCH_CONTOUR_TO_IPA[tok]])
        else:
            processed_ipa.append(tok)
            processed_xsampa.append(IPA2SAMPA([tok], zh=True)[0])
    return processed_xsampa, processed_ipa


def yue_tone_backend(ipa: list):
    tone_ipa2xsampa = {"¹": "_1", "²": "_2", "³": "_3", "⁴": "_4", "⁵": "_5", "⁶": "_6"}
    processed_ipa = []
    processed_xsampa = []
    ipa = deque(ipa)
    for tok in ipa:
        if tok in YUE_PITCH_CONTOUR_TO_IPA:
            processed_ipa.append(tok)
            # processed_ipa.append(YUE_PITCH_CONTOUR_TO_IPA[tok])
            processed_xsampa.append(tone_ipa2xsampa[YUE_PITCH_CONTOUR_TO_IPA[tok]])
        else:
            processed_ipa.append(tok)
            processed_xsampa.append(IPA2SAMPA([tok])[0])

    return processed_xsampa, processed_ipa


def IPA2SAMPA(ipa: list, zh=False) -> list:
    res = [ipa2xsampa(i, 'unk') for i in ipa]
    if zh:
        zh_tone_map = {"0": "_0", "_3_1": "_2", "_3_5_3": "_3", "_1_5": "_4"}
        res = [zh_tone_map.get(i, i) for i in res]
    return res


def combine_pinyin(pinyin: list) -> list:
    """
    :example: '['s','un1','w','u4','k','ong1','en5']' -> ['sun1','wu4','kong1','en5']
    """
    res = []
    total = ''
    while pinyin:
        current = pinyin.pop(0)
        total += current
        if any(char.isdigit() for char in total):
            res.append(total)
            total = ''

    return res


def arpa2ipa(en_lyrics: list) -> list:
    return [ipa for i in en_lyrics for ipa in arpabet2ipa(i, 'en')]


def pinyin2ipa(zh_lyrics: list):
    res = []
    tones = ["˧˥", "˧˩˧", "˥˩", "˥"]
    for i in combine_pinyin(zh_lyrics):
        i = i[2:] if i[0].isupper() else i
        i = i.replace('ir', 'i').replace('0', '').replace('E', 'e')
        try:
            ipa_list = list(pinyin_to_ipa(i)[0])
            new_ipa_list = []
            for e in ipa_list:
                ipa_tone = None
                for tone in tones:
                    if tone in e:
                        ipa_tone = tone
                        break
                if ipa_tone is None:
                    new_ipa_list.append(e)
                else:
                    new_ipa_list.append(e.replace(ipa_tone, ''))
                    new_ipa_list.append(ipa_tone)
            ipa_list = new_ipa_list
            if len(ipa_list) == 1 and not any(tone in ipa_list[0] for tone in tones):  # 轻声补充标志0
                ipa_list[0] = ipa_list[0] + '0'
        except Exception as err1:
            try:
                ipa_list = arpa2ipa([i])
            except Exception as err2:
                logger.error(f"Error in pinyin2ipa: {err1}, {err2}, phoneme: {i}")
                ipa_list = []
        res += ipa_list
    return zh_tone_backend(res)


def load_romaji2ipa_map() -> dict:
    json_path = 'MG2P/core/roma_ipa_mapping.json'
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        phoneme_to_ipa_map = {phoneme: ipa[0] for phoneme, ipa in data["phoneme2ipa"].items()}
    return phoneme_to_ipa_map


def romaji2ipa(ja_lyrics: list, roma2ipa: dict = None) -> list:
    res = []
    for i in ja_lyrics:
        i = i.lower()
        if i in roma2ipa:
            res.extend(roma2ipa[i])
    return res


def multi_lang_tokenizer(lyrics: str, lang=None) -> list:
    if lang is not None:
        LangSegment.setfilters([lang] + ALL_LANGUAGES)
    else:
        LangSegment.setfilters(ALL_LANGUAGES)
    langlist = LangSegment.getTexts(lyrics)
    return langlist


def charsiu_model_decode(model_output) -> list:
    model_output = model_output.cpu().numpy()
    results = []
    for idx in range(model_output.shape[0]):
        tokens = model_output[idx]
        tokens = (tokens[tokens > 1] - 3).astype(np.uint8)
        results.append(tokens.tobytes().decode('utf-8', errors='ignore'))
    return results
