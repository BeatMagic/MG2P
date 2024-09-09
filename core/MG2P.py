import utils
import LangSegment


class MG2P:
    def __init__(self):
        pass

    def check_if_sup(self, language: str) -> bool:
        """
        check if the language can be converted
        :param language: language 639-1 code
        :return: [True]: this is the support language, the model can generate a good result
        [False]: This is the zero-shot language, the model tries to guess its phonemes
        """
        sup_language = utils.generate_sup_language_list()
        if language not in sup_language:
            return False
        return True

    def __call__(self, lyrics: str, tag=None, **kwargs) -> list:
        """

        """
        cleaned_lyrics = utils.clean_lyrics(lyrics)
        major_lang = ['zh', 'en', 'ja']

        prefix_map = utils.generate_sup_language_list()
        phoneme = []

        # When a language is not supported (zero-shot), it uses <unk> as its prefix code.
        # In fact, Charsiu can also predict phonemes without it, but the quality will be reduced
        # lyrics = ['charsiu', 'is', 'a', 'Cantonese', 'style', 'of', 'barbecued', 'pork']
        # eng-us: ['ˈtʃɑɹsiu', 'ˈɪs', 'ˈeɪ', 'ˌkæntəˈniz', 'ˈstaɪɫ', 'ˈəf', 'ˈbɑɹbɪkˌjud', 'ˈpɔɹk']
        # unk: ['carsiw', 'iːs', 'a˧˧', 'kˌantonˈese', 'stˈaɪl', 'ɔv', 'bˈɑːbɪkjˌuːd', 'pɔrk']
        # '': ['xarɕu', 'ˈis', 'a', 'kantoneze', 'stˈaɪl', 'ɔf', 'bˈɑːbɪkjˌuːd', 'pɔrk']

        if tag is None or len(tag) > 1:
            langlist = LangSegment.getTexts(lyrics)
            prefix_lyrics_list = []
            for item in langlist:
                prefix = prefix_map[item['lang']] if item['lang'] in prefix_map else 'unk'
                item_lyrics_list = utils.tokenize_lyrics(item['text'], item['lang'])
                item_lyrics_list = ['<' + prefix + '>: ' + i for i in item_lyrics_list]
                prefix_lyrics_list.extend(item_lyrics_list)
            print(prefix_lyrics_list)
            phoneme = utils.CharsiuG2P(prefix_lyrics_list, **kwargs)
            phoneme = utils.IPA2SAMPA(phoneme)
        else:
            lyrics_list = utils.tokenize_lyrics(cleaned_lyrics, tag)
            prefix = prefix_map[tag] if tag in prefix_map else 'unk'
            prefix_lyrics_list = ['<' + prefix + '>: ' + i for i in lyrics_list]
            phoneme = utils.CharsiuG2P(prefix_lyrics_list, **kwargs)
        phoneme = utils.IPA2SAMPA(phoneme)
        return phoneme


if __name__ == '__main__':
    g2p = MG2P()
    # print(g2p.check_if_sup('zh'))
    lyrics = 'チャーシュー是一种Cantonese风格of barbecued pork'
    print(g2p(lyrics))
