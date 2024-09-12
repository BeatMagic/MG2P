import re
from konoha import WordTokenizer
import jieba
from deepcut import tokenize
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
import numpy as np
from phonecodes import ipa2xsampa, arpabet2ipa
from pinyin_to_ipa import pinyin_to_ipa
import g2p
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '2'


def generate_sup_language_list() -> dict:
    """
    generate support languages list
    """
    mapping = {}
    tsv_path = "639_1toPrefix.tsv"
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


def jp_tokenizer(lyrics: str) -> list:
    tokenizer = WordTokenizer('Sentencepiece', model_path='model.spm')
    result_list = tokenizer.tokenize(lyrics)
    result_list = [str(item) for item in result_list]
    if list:
        for i, _ in enumerate(result_list):
            result_list[i] = result_list[i].replace('▁', '')
            if result_list[i] == ' ' or result_list[i] == '':
                result_list.pop(i)
    # print(result_list)
    return result_list


def zh_tokenizer(lyrics: str) -> list:
    result_list = list(jieba.cut(lyrics, cut_all=False))
    return result_list


def th_tokenizer(lyrics: str) -> list:
    result_list = tokenize(lyrics)
    return result_list


def tokenize_lyrics(lyrics: str, tag: str) -> list:
    if tag == 'zh' or tag == 'zho-s':
        return zh_tokenizer(lyrics)
    if tag == 'ja' or tag == 'jpn':
        return jp_tokenizer(lyrics)
    if tag == 'th' or tag == 'tha':
        return th_tokenizer(lyrics)
    lyrics_list = lyrics.split()
    return lyrics_list


def decode(model_output) -> list:
    model_output = model_output.cpu().numpy()
    results = []
    for idx in range(model_output.shape[0]):
        tokens = model_output[idx]
        tokens = (tokens[tokens > 1] - 3).astype(np.uint8)
        results.append(tokens.tobytes().decode('utf-8', errors='ignore'))
    return results


def charsiu_g2p(lyrics: str, tag: str, use_32=False, use_fast=False) -> str:
    """
    Use Charsiu to perform G2P transformation for minority languages
    :param lyrics: lyrics grapheme str
    :param tag: 639_1 code
    :param use_32: whether to select fp32 as the model precision, the default is fp16
    :param use_fast: use Charsiu_tiny_16 model in exchange for speed, the default is Charsiu_small
    :return: lyrics phoneme str
    """
    prefix_map = generate_sup_language_list()
    # When a language is not supported (zero-shot), it uses <unk> as its prefix code.
    # In fact, Charsiu can also predict phonemes without it, but the quality will be reduced
    # lyrics = ['charsiu', 'is', 'a', 'Cantonese', 'style', 'of', 'barbecued', 'pork']
    # eng-us: ['ˈtʃɑɹsiu', 'ˈɪs', 'ˈeɪ', 'ˌkæntəˈniz', 'ˈstaɪɫ', 'ˈəf', 'ˈbɑɹbɪkˌjud', 'ˈpɔɹk']
    # unk: ['carsiw', 'iːs', 'a˧˧', 'kˌantonˈese', 'stˈaɪl', 'ɔv', 'bˈɑːbɪkjˌuːd', 'pɔrk']
    # '': ['xarɕu', 'ˈis', 'a', 'kantoneze', 'stˈaɪl', 'ɔf', 'bˈɑːbɪkjˌuːd', 'pɔrk']
    prefix = prefix_map[tag] if tag in prefix_map else 'unk'
    lyrics_list = tokenize_lyrics(lyrics, tag)
    prefix_lyrics_list = ['<' + prefix + '>: ' + i for i in lyrics_list]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2000
    if use_fast:
        model_path = 'charsiu/g2p_multilingual_byT5_tiny_16_layers_100'
        batch_size = 5000
    else:
        model_path = 'charsiu/g2p_multilingual_byT5_small_100'
        # model_path = 'D:/work/Charsiu evaluation/CharsiuPre/byT5_small_100'
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    if not use_32:
        model.half()

    tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')
    # tokenizer = AutoTokenizer.from_pretrained('D:/work/Charsiu evaluation/CharsiuPre/byt5-small')
    for i in range(0, len(prefix_lyrics_list), batch_size):
        batch = prefix_lyrics_list[i:i + batch_size]

        with torch.cuda.amp.autocast():
            out = tokenizer(batch, padding=True, add_special_tokens=False, return_tensors='pt').to(device)
            preds = model.generate(**out, num_beams=1, max_length=30)
            phones = decode(preds)
        prefix_lyrics_list[i:i + batch_size] = phones
    res = ''.join(prefix_lyrics_list)
    return res


def major_g2p(lyrics: str, tag: str) -> str:
    """
    Convert major languages to phonemes with our G2P.
    Currently, supports English and Chinese
    :param lyrics: lyrics only contain one language
    :param tag: 639_1/639_2 code
    :return: ipa list
    """
    if tag == 'zh':
        zh_pinyin = ' '.join(g2p.infer(lyrics)[0]['phones'])
        return pinyin2ipa(zh_pinyin)
    if tag == 'en':
        en_arpa = ' '.join(g2p.infer(lyrics)[0]['phones'])
        return arpa2ipa(en_arpa)


def IPA2SAMPA(ipa: str) -> str:
    return ipa2xsampa(ipa, 'unk')


def combine_pinyin(pinyin: str) -> list:
    """
    :example: 's un1 w u4 k ong1 en5' -> ['sun1','wu4','kong1','en5']
    """
    res = []
    syllables = pinyin.split()
    total = ''
    while syllables:
        current = syllables.pop(0)
        total += current
        if any(char.isdigit() for char in total):
            res.append(total)
            total = ''

    return res


def arpa2ipa(en_lyrics: str) -> str:
    res = arpabet2ipa(en_lyrics, 'en').replace(' ', '')
    return res


def pinyin2ipa(zh_lyrics: str) -> str:
    res = ''
    for i in combine_pinyin(zh_lyrics):
        i = i[2:] if i[0].isupper() else i
        i = i.replace('ir', 'i').replace('0', '')
        res += ''.join(pinyin_to_ipa(i)[0])
    return res


if __name__ == '__main__':
    # lyrics = "\n\n\n攀登高峰望故乡\n |||黄沙万里[02:03:37]何处传来[02:03:327]驼铃声...声声敲心坎,盼望踏]]{上思念路[02:37]。飞纵千里山{]\n天边归雁披彩霞。"
    # print(clean_lyrics(lyrics))

    # sentence = "二日前このへんで飞び降り自杀した人のニュースが流れてきた血まみれセーラー危ないですから"
    # print(jp_tokenizer(sentence))

    # sentence = "你好我来自时域科技是一个实习生我举起手表示我是一个员工"
    # print(zh_tokenizer(sentence))

    # sentence = "เพราะฉันไม่รู้เลยจะทำยังไงให้ชีวิตฉันสวยงามให้เธอภูมิใจไม่เก่งภาษาเคยลองทำหลายอย่างก็ไร้ราคา"
    # print(th_tokenizer(sentence))

    # pre_lyrics_list = ['<eng-us>: charsiu', '<zho-s>: 是', '<zho-s>: 一种', '<eng-us>: Cantonese',
    #                    '<eng-us>: style', '<eng-us>: of', '<eng-us>: barbecued', '<jpn>: 豚肉']
    # print(charsiu_g2p(pre_lyrics_list))

    # sentence = "hello i am timedomain"
    # print(tokenize_lyrics(sentence, 'en'))

    # text = ['ˈtʃɑɹsiu', 'ˈɪs', 'ˈeɪ', 'ˌkæntəˈniz', 'ˈstaɪɫ', 'ˈəf', 'ˈbɑɹbɪkˌjud', 'ˈpɔɹk']
    # print(IPA2SAMPA(text))

    # text = ['CH AA1 R', 'S IY1 UW0', 'IH1 Z', 'AH0', 'K AE2 N T AH0 N IY1 Z', 'S T AY1 L', 'AH1 V',
    #         'B AA1 R B IH0 K Y UW2 D', 'P AO1 R K']
    # print(arpa2ipa(text))

    # text = 's un1 w u4 k ong1 a1 en5'
    # print(pinyin2ipa(text))

    text1 = "charsiu is a pork"
    text2 = "孙悟空耳朵啊啊啊"
    text3 = ["<zho-s>: 踏碎", "<zho-s>: 凌霄", "<zho-s>: 放肆", "<zho-s>: 桀骜"]
    print(major_g2p(text1, 'en'), major_g2p(text2, 'zh'))
    print(charsiu_g2p(text3))
