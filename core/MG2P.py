import utils
import LangSegment


class MG2P:
    def __init__(self):
        pass

    def check_if_sup(self, language: str) -> bool:
        """
        check if the language can be converted
        :param language: language 639-1 code
        :return:
        [True]: this is the support language, the model can generate a good result
        [False]: This is the zero-shot language, the model tries to guess its phonemes
        """
        sup_language = utils.generate_sup_language_list()
        if language not in sup_language:
            return False
        return True

    def __call__(self, lyrics: str, tag=None, **kwargs) -> str:
        """
        Convert the lyrics to the corresponding xsampa, Currently, Chinese and English use internal g2p,
        and other languages use CharsiuG2P

        :param lyric supports data cleaning of dirty lyrics
        :param tag the model expects a 639_1 list (even if it is one) of the languages in the input lyrics,
        and if this field is ignored the model automatically determines the language
        :param **kwargs you can adjust the speed and quality of CharsiuG2P with multiple input parameters
        use_32=True  use fp32 precision, the default is fp16
        use_fast=True use tiny_16 model, the default is small

        example: MG2P("踏碎凌霄 放肆桀骜",['zh']) -> t_ha_1_5swei_^_1_5li_3_1Ns\jau_^_1fa_1_5Nsr\_=_1_5ts\je_3_1au_^_1_5
        """
        cleaned_lyrics = utils.clean_lyrics(lyrics)
        major_lang = ['zh', 'en']

        prefix_map = utils.generate_sup_language_list()
        phoneme = ''

        # When a language is not supported (zero-shot), it uses <unk> as its prefix code.
        # In fact, Charsiu can also predict phonemes without it, but the quality will be reduced
        # lyrics = ['charsiu', 'is', 'a', 'Cantonese', 'style', 'of', 'barbecued', 'pork']
        # eng-us: ['ˈtʃɑɹsiu', 'ˈɪs', 'ˈeɪ', 'ˌkæntəˈniz', 'ˈstaɪɫ', 'ˈəf', 'ˈbɑɹbɪkˌjud', 'ˈpɔɹk']
        # unk: ['carsiw', 'iːs', 'a˧˧', 'kˌantonˈese', 'stˈaɪl', 'ɔv', 'bˈɑːbɪkjˌuːd', 'pɔrk']
        # '': ['xarɕu', 'ˈis', 'a', 'kantoneze', 'stˈaɪl', 'ɔf', 'bˈɑːbɪkjˌuːd', 'pɔrk']

        if tag is None or len(tag) > 1:
            langlist = LangSegment.getTexts(cleaned_lyrics)
            for item in langlist:
                if item['lang'] in major_lang:
                    phoneme += utils.major_g2p(item['text'], item['lang'])
                else:
                    prefix_lyrics_list = []
                    prefix = prefix_map[item['lang']] if item['lang'] in prefix_map else 'unk'
                    item_lyrics_list = utils.tokenize_lyrics(item['text'], item['lang'])
                    item_lyrics_list = ['<' + prefix + '>: ' + i for i in item_lyrics_list]
                    prefix_lyrics_list += item_lyrics_list
                    phoneme += utils.charsiu_g2p(prefix_lyrics_list, **kwargs)
        else:
            tag = tag[0]
            if tag in major_lang:
                phoneme = utils.major_g2p(cleaned_lyrics, tag)
            else:
                lyrics_list = utils.tokenize_lyrics(cleaned_lyrics, tag)
                prefix = prefix_map[tag] if tag in prefix_map else 'unk'
                prefix_lyrics_list = ['<' + prefix + '>: ' + i for i in lyrics_list]
                phoneme = utils.charsiu_g2p(prefix_lyrics_list, **kwargs)
        phoneme = utils.IPA2SAMPA(phoneme)
        return phoneme


if __name__ == '__main__':
    g2p = MG2P()
    # print(g2p.check_if_sup('zh'))
    lyrics = 'チャーシュー是一种Cantonese风格of barbecued pork'
    lyrics1 = '踏碎凌霄 放肆桀骜 世恶道险 终究难逃'
    print(g2p(lyrics1, ['zh']))
