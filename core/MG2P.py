import MG2P.core.utils as utils
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

    def __call__(self, lyrics: str, tag=None, **kwargs) -> list:
        """
        Convert the lyrics to the corresponding xsampa, Currently, Chinese, Japanese and English use internal g2p,
        and other languages use CharsiuG2P

        :param lyric supports data cleaning of dirty lyrics
        :param tag the model expects a 639_1 code of the languages in the input lyrics,
        and if this field is ignored the model automatically determines the language
        :param **kwargs you can adjust the speed and quality of CharsiuG2P with multiple input parameters
        use_32=True  use fp32 precision, the default is fp16
        use_fast=True use tiny_16 model, the default is small

        example: MG2P("踏碎凌霄 放肆桀骜",'zh') -> ['t_h', 'a_1_5', 's', 'w', 'ei_^_1_5', 'l', 'i_3_1', 'N', 's\\', 'j', 'au_^_1', 'f', 'a_1_5', 'N', 's', 'r\\_=_1_5', 'ts\\', 'j', 'e_3_1', 'au_^_1_5']
        """
        cleaned_lyrics = utils.clean_lyrics(lyrics)
        major_lang = ['zh', 'en', 'ja']
        phoneme = []

        if tag is None:
            langlist = LangSegment.getTexts(cleaned_lyrics)
            for item in langlist:
                if item['lang'] in major_lang:
                    phoneme += utils.major_g2p(item['text'], item['lang'])
                else:
                    phoneme += utils.charsiu_g2p(item['text'], item['lang'], **kwargs)
        else:
            if tag in major_lang:
                phoneme = utils.major_g2p(cleaned_lyrics, tag)
            else:
                phoneme = utils.charsiu_g2p(cleaned_lyrics, tag, **kwargs)
        phoneme = utils.IPA2SAMPA(phoneme)
        return phoneme


if __name__ == '__main__':
    g2p = MG2P()
    # print(g2p.check_if_sup('zh'))
    lyrics = 'チャーシュー是一种Cantonese风格of barbecued pork'
    lyrics1 = '踏碎凌霄 放肆桀骜 世恶道险 终究难逃'
    lyrics2 = '今でもあなたはわたしの光'
    print(g2p(lyrics), g2p(lyrics1), g2p(lyrics2))
