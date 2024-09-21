import re
from konoha import WordTokenizer
import jieba
from deepcut import tokenize
import torch
import numpy as np
from MG2P.core.phonecodes import ipa2xsampa, arpabet2ipa
from pinyin_to_ipa import pinyin_to_ipa
import MG2P.core.g2p as g2p
import os
import json
from MG2P.core.ipa2list import ipalist2phoneme
import LangSegment

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

PITCH_CONTOUR_TO_IPA = {
    "˥": "¹",
    "˧˥": "²",
    "˧˩˧": "³",
    "˥˩": "⁴",
    "0": "º"
}


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
    # deprecated
    tokenizer = WordTokenizer('Sentencepiece', model_path='MG2P/core/model.spm')
    result_list = tokenizer.tokenize(lyrics)
    result_list = [str(item) for item in result_list]
    if list:
        for i, _ in enumerate(result_list):
            result_list[i] = result_list[i].replace('▁', '')
            if result_list[i] == ' ' or result_list[i] == '':
                result_list.pop(i)
    return result_list


def zh_tokenizer(lyrics: str) -> list:
    # deprecated
    result_list = list(jieba.cut(lyrics, cut_all=False))
    return result_list


def th_tokenizer(lyrics: str) -> list:
    result_list = tokenize(lyrics)
    return result_list


def ko_tokenizer(lyrics: str) -> list:
    lyrics = lyrics.replace(' ', '')
    tokens = []
    for i in range(0, len(lyrics), 3):
        token = lyrics[i:i + 3]
        tokens.append(token)
    return tokens


def tokenize_lyrics(lyrics: str, tag: str) -> list:
    if tag == 'zh' or tag == 'zho-s':  # deprecated
        return zh_tokenizer(lyrics)
    if tag == 'ja' or tag == 'jpn':  # deprecated
        return jp_tokenizer(lyrics)
    if tag == 'th' or tag == 'tha':
        return th_tokenizer(lyrics)
    if tag == 'ko' or tag == 'kor':
        return ko_tokenizer(lyrics)
    lyrics_list = lyrics.split()
    return lyrics_list


def pinyin_len_coefficient(pinyin: list) -> int:
    flag = 0
    for i in pinyin:
        if re.search(r'\d', i):
            flag += 1
    return flag


def zh_tone_backend(ipa: list) -> (list, list):
    tones = ["˧˥", "˧˩˧", "˥˩", "˥", "0"]
    processed_ipa = []
    split_ipa = []
    while ipa:
        current = ipa.pop(0)
        flag = True
        for tone in tones:
            if tone in current:
                split_ipa.append(current.replace(tone, ''))
                split_ipa.append(tone)
                processed_ipa.append(current.replace(tone, ''))
                processed_ipa.append(PITCH_CONTOUR_TO_IPA[tone])
                flag = False
                break
        if flag:
            split_ipa.append(current)
            processed_ipa.append(current)
    return split_ipa, processed_ipa


def IPA2SAMPA(ipa: list, zh=False) -> list:
    res = [ipa2xsampa(i, 'unk') for i in ipa]
    if zh:
        zh_tone_map = {"0": "_0", "_3_1": "_2", "_3_5_3": "_3", "_1_5": "_4"}
        for i, _ in enumerate(res):
            if res[i] in zh_tone_map:
                res[i] = zh_tone_map[res[i]]
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
    res = []
    for i in en_lyrics:
        res += arpabet2ipa(i, 'en')
    return res


def pinyin2ipa(zh_lyrics: list) -> (list, list):
    res = []
    tones = ["˧˥", "˧˩˧", "˥˩", "˥"]
    for i in combine_pinyin(zh_lyrics):
        i = i[2:] if i[0].isupper() else i
        i = i.replace('ir', 'i').replace('0', '').replace('E', 'e')
        ipa_list = list(pinyin_to_ipa(i)[0])
        if len(ipa_list) == 1 and not any(tone in ipa_list[0] for tone in tones):  # 轻声补充标志0
            ipa_list[0] = ipa_list[0] + '0'
        res += ipa_list
    return zh_tone_backend(res)


def load_romaji2ipa_map() -> dict:
    json_path = 'MG2P/core/roma_ipa_mapping.json'
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        phoneme_to_ipa_map = {phoneme: ipa[0] for phoneme, ipa in data["phoneme2ipa"].items()}
    return phoneme_to_ipa_map


def romaji2ipa(ja_lyrics: list) -> list:
    res = []
    roma2ipa = load_romaji2ipa_map()
    ja_roma_list = [i.lower() for i in ja_lyrics]
    for i in ja_roma_list:
        res += roma2ipa[i]
    return res


def multi_lang_tokenizer(lyrics: str) -> list:
    LangSegment.setfilters(
        ['af', 'am', 'an', 'ar', 'as', 'az', 'be', 'bg', 'bn', 'br', 'bs', 'ca', 'cs', 'cy', 'da', 'de', 'dz', 'el',
         'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fo', 'fr', 'ga', 'gl', 'gu', 'he', 'hi', 'hr', 'ht', 'hu', 'hy',
         'id', 'is', 'it', 'ja', 'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'ku', 'ky', 'la', 'lb', 'lo', 'lt', 'lv', 'mg',
         'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'nb', 'ne', 'nl', 'nn', 'no', 'oc', 'or', 'pa', 'pl', 'ps', 'pt', 'qu',
         'ro', 'ru', 'rw', 'se', 'si', 'sk', 'sl', 'sq', 'sr', 'sv', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'ug', 'uk',
         'ur', 'vi', 'vo', 'wa', 'xh', 'zh', 'zu'])
    langlist = LangSegment.getTexts(lyrics)
    return langlist


def decode(model_output) -> list:
    model_output = model_output.cpu().numpy()
    results = []
    for idx in range(model_output.shape[0]):
        tokens = model_output[idx]
        tokens = (tokens[tokens > 1] - 3).astype(np.uint8)
        results.append(tokens.tobytes().decode('utf-8', errors='ignore'))
    return results


def charsiu_g2p(lyrics: str, tag: str, model, tokenizer, use_32=False, use_fast=False) -> (list, list):
    """
    Use Charsiu to perform G2P transformation for minority languages
    :param lyrics: lyrics grapheme str
    :param tag: 639_1 code
    :param use_32: whether to select fp32 as the model precision, the default is fp16
    :param use_fast: use Charsiu_tiny_16 model in exchange for speed, the default is Charsiu_small
    :return: ipa phoneme list, xsampa phoneme list
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
    # if use_fast:
    #     model_path = 'charsiu/g2p_multilingual_byT5_tiny_16_layers_100'
    #     batch_size = 5000
    # else:
    #     model_path = 'charsiu/g2p_multilingual_byT5_small_100'
    # model_path = 'D:/work/Charsiu evaluation/CharsiuPre/byT5_small_100'
    # model = T5ForConditionalGeneration.from_pretrained(model_path)
    # model.to(device)
    if not use_32:
        model.half()

    # tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')
    # tokenizer = AutoTokenizer.from_pretrained('D:/work/Charsiu evaluation/CharsiuPre/byt5-small')
    for i in range(0, len(prefix_lyrics_list), batch_size):
        batch = prefix_lyrics_list[i:i + batch_size]

        with torch.cuda.amp.autocast():
            out = tokenizer(batch, padding=True, add_special_tokens=False, return_tensors='pt').to(device)
            preds = model.generate(**out, num_beams=1, max_length=61)
            phones = decode(preds)
        prefix_lyrics_list[i:i + batch_size] = phones
    ipa_list = ipalist2phoneme(prefix_lyrics_list, tag)
    xsampa = IPA2SAMPA(ipa_list)
    return ipa_list, xsampa


def charsiu_g2p_batch(lyrics_list: list, tag_list: list, model, tokenizer, batch_size=2000, use_32=False) -> (
        list, list):
    prefix_map = generate_sup_language_list()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    grapheme_list = []
    grapheme_split_list = [0]

    # 添加prefix code并计算语种分隔位置
    for i, lyrics in enumerate(lyrics_list):
        current_prefix = prefix_map[tag_list[i]] if tag_list[i] in prefix_map else 'unk'
        current_grapheme_list = tokenize_lyrics(lyrics, tag_list[i])
        grapheme_split_list.append(len(current_grapheme_list) + grapheme_split_list[-1])
        for grapheme in current_grapheme_list:
            grapheme_list.append('<' + current_prefix + '>: ' + grapheme)

    # print(grapheme_list) ['<spa>: barbecued', '<spa>: pork', '<ger>: Ich', '<ger>: liebe', '<ger>: dich', '<tha>: สว ', '<tha>: สด', '<tha>:  ', '<kor>: 넌나의', '<kor>: 빛', '<kor>: 사랑해', '<kor>: 요']
    # print(grapheme_split_list) [0, 2, 5, 8, 10, 12]

    if not use_32:
        model.half()

    # 批量跑训练
    for i in range(0, len(grapheme_list), batch_size):
        batch = grapheme_list[i:i + batch_size]

        with torch.cuda.amp.autocast():
            out = tokenizer(batch, padding=True, add_special_tokens=False, return_tensors='pt').to(device)
            preds = model.generate(**out, num_beams=1, max_length=61)
            phones = decode(preds)
        grapheme_list[i:i + batch_size] = phones
    # print(grapheme_list) ['baɾβekweð', 'poɾk', 'ɪç', 'ˈliːbə', 'dɪç', 'sua̯˩˩˦', 'sot̚˨˩', 'ha˧˥', 'nʌ̹nna̠ɰi', 'pit̚', 'sʰa̠ɾa̠ŋɦɛ̝', 'jo']

    # 每个语种出现一次作为一个列表
    ipa_list = []
    xsampa_list = []
    for i, tag in enumerate(tag_list):
        start, end = grapheme_split_list[i], grapheme_split_list[i + 1]
        current_grapheme_list = grapheme_list[start:end]
        current_ipa_list = ipalist2phoneme(current_grapheme_list, tag)
        ipa_list.append(current_ipa_list)
        current_xsampa_list = IPA2SAMPA(current_ipa_list)
        xsampa_list.append(current_xsampa_list)

    return ipa_list, xsampa_list


def major_g2p_batch(lyrics_list: list, tag_list: list) -> (list, list):
    res = g2p.infer(lyrics_list)

    ipa_list = []
    xsampa_list = []
    # ipa在生成前长度不可预见, 因此单独做计算
    for i, item in enumerate(res):
        phones = item['phones']
        if tag_list[i] == 'en':
            en_ipa = arpa2ipa(phones)
            ipa_list.append(en_ipa)
            xsampa_list.append(IPA2SAMPA(en_ipa))
        if tag_list[i] == 'zh':
            split_ipa, zh_ipa = pinyin2ipa(phones)
            ipa_list.append(zh_ipa)
            xsampa_list.append(IPA2SAMPA(split_ipa, True))
        if tag_list[i] == 'ja':
            ja_ipa = romaji2ipa(phones)
            ipa_list.append(ja_ipa)
            xsampa_list.append(IPA2SAMPA(ja_ipa))

    return ipa_list, xsampa_list


def major_g2p(lyrics: str, tag: str) -> (list, list):
    """
    Convert major languages to phonemes with our G2P.
    Currently, supports English and Chinese
    :param lyrics: lyrics only contain one language
    :param tag: 639_1/639_2 code
    :return: ipa phoneme list, xsampa phoneme list
    """
    if tag == 'zh':
        zh_pinyin = g2p.infer(lyrics, tag)[0]['phones']
        split_ipa, processed_ipa = pinyin2ipa(zh_pinyin)
        xsampa = IPA2SAMPA(split_ipa, True)
        return processed_ipa, xsampa
    if tag == 'en':
        en_arpa = g2p.infer(lyrics, tag)[0]['phones']
        ipa = arpa2ipa(en_arpa)
        xsampa = IPA2SAMPA(ipa)
        return ipa, xsampa
    if tag == 'ja':
        ja_roma = g2p.infer(lyrics, tag)[0]['phones']
        ipa = romaji2ipa(ja_roma)
        xsampa = IPA2SAMPA(ipa)
        return ipa, xsampa


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

    # text1 = "charsiu is a pork"
    # text2 = "孙悟空耳朵啊啊啊"
    # print(major_g2p(text1, 'en'), major_g2p(text2, 'zh'))

    zh_test = [
        '啊哎安肮奥额诶嗯儿哦欧', '把波白被保班本病帮崩比别表变斌不',
        '爬破派配跑剖盘喷旁朋平皮撇飘片拼铺', '马模么买明没猫某慢们忙梦米灭秒面民穆',
        '法佛反分放风服', '大的带到都单扽当等动地跌调丢点读段定', '他特踢图太忒推套偷铁弹吞汤疼听痛条天',
        '那呢你怒奶内闹牛捏男嫩您鸟年娘虐嚢能宁弄', '拉聊练凉乱咯乐力路绿来类老楼留列略蓝林论狼冷铃龙',
        '尬歌鼓该给贵高狗干跟滚瓜关光刚更宫', '卡克库开夸宽狂亏靠扣看啃坤康坑空',
        '哈和胡海黑号后汉很花欢黄魂行恒红', '几加叫建将窘卷句就接觉近均京', '其掐敲钱全强穷求且缺勤群清',
        '西许秀写学信训下小先想熊选星', '炸抓专装这只猪债追找周站准长征中', '差吃出差吹歘穿创超抽产陈纯长成冲',
        '杀射是书刷栓双晒谁少手闪身顺上声', '热日挼软入瑞绕肉然人润让扔荣', '杂泽钻子组在贼最早走赞怎尊脏增总',
        '擦测词粗菜催窜草凑参岑存仓层从', '萨色死算素赛岁骚搜三森孙桑僧送', '瓦我屋外为玩问王翁',
        '压哟也元一语要有月眼音云羊赢勇'
    ]
    res = [major_g2p(i, 'zh') for i in zh_test]
    print(res)
