import re
from konoha import WordTokenizer
import jieba
from deepcut import tokenize


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
    # line break symbol
    lyrics = re.sub(r'\n\s*\n|\s*\n\s*', ' ', lyrics)
    # timestamp
    # lyrics = re.sub(r'\[\d+:\d+[:.]\d+]', ' ', lyrics)
    lyrics = re.sub(r'\[\d+:\d+[:.]?\d*\]', ' ', lyrics)
    # punctuation, only keep the English apostrophe
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
        result_list[0] = result_list[0][1:]
    return result_list


def zh_tokenizer(lyrics: str) -> list:
    result_list = list(jieba.cut(lyrics, cut_all=False))
    return result_list


def th_tokenizer(lyrics: str) -> list:
    result_list = tokenize(lyrics)
    return result_list


if __name__ == '__main__':
    # lyrics = "\n\n\n攀登高峰望故乡\n |||黄沙万里[02:03:37]何处传来[02:03:327]驼铃声...声声敲心坎,盼望踏]]{上思念路[02:37]。飞纵千里山{]\n天边归雁披彩霞。"
    # print(clean_lyrics(lyrics))

    # sentence = "二日前このへんで飞び降り自杀した人のニュースが流れてきた血まみれセーラー危ないですから"
    # print(jp_tokenizer(sentence))

    # sentence = "你好我来自时域科技是一个实习生我举起手表示我是一个员工"
    # print(zh_tokenizer(sentence))

    sentence = "เพราะฉันไม่รู้เลยจะทำยังไงให้ชีวิตฉันสวยงามให้เธอภูมิใจไม่เก่งภาษาเคยลองทำหลายอย่างก็ไร้ราคา"
    print(th_tokenizer(sentence))
