import re
from konoha import WordTokenizer
import jieba


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


def jp_tokenizer(lyrics: str) -> str:
    tokenizer = WordTokenizer('Sentencepiece', model_path='model.spm')
    result_list = tokenizer.tokenize(lyrics)
    result_string = ' '.join([token.surface for token in result_list if token.surface != '▁'])[1:]
    return result_string


def zh_tokenizer(lyrics: str) -> str:
    result_list = jieba.cut(lyrics, cut_all=False)
    result_string = " ".join(result_list)
    return result_string


if __name__ == '__main__':
    # lyrics = "\n\n\n攀登高峰望故乡\n |||黄沙万里[02:03:37]何处传来[02:03:327]驼铃声...声声敲心坎,盼望踏]]{上思念路[02:37]。飞纵千里山{]\n天边归雁披彩霞。"
    # print(clean_lyrics(lyrics))

    # sentence = "二日前このへんで飞び降り自杀した人のニュースが流れてきた血まみれセーラー危ないですから"
    # print(jp_tokenizer(sentence))

    sentence = "你好我来自时域科技是一个实习生我举起手表示我是一个员工"
    print(zh_tokenizer(sentence))
