import MG2P.core.utils as utils
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch


class MG2P:
    def __init__(self, model_path='charsiu/g2p_multilingual_byT5_small_100', tokenizer_path='google/byt5-small'):
        self.charsiu_model = T5ForConditionalGeneration.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.charsiu_model.to(device)
        self.charsiu_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

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

    def __call__(self, lyrics: str, tag=None, **kwargs) -> (list, list):
        """
        Convert the lyrics to the corresponding ipa and xsampa, Currently, Chinese, Japanese and English use internal g2p,
        and other languages use CharsiuG2P

        :param lyric supports data cleaning of dirty lyrics
        :param tag the model expects a 639_1 code of the languages in the input lyrics,
        and if this field is ignored the model automatically determines the language
        :param **kwargs you can adjust the speed and quality of CharsiuG2P with multiple input parameters
        use_32=True  use fp32 precision, the default is fp16
        use_fast=True use tiny_16 model, the default is small
        :return: ipa phoneme list, xsampa phoneme list
        example: MG2P("踏碎凌霄 放肆桀骜",'zh') -> ['t_h', 'a_1_5', 's', 'w', 'ei_^_1_5', 'l', 'i_3_1', 'N', 's\\', 'j', 'au_^_1', 'f', 'a_1_5', 'N', 's', 'r\\_=_1_5', 'ts\\', 'j', 'e_3_1', 'au_^_1_5']
        """
        cleaned_lyrics = utils.clean_lyrics(lyrics)
        major_lang = ['zh', 'en', 'ja']
        ipa_list = []
        xsampa_list = []

        if tag is None:
            langlist = utils.multi_lang_tokenizer(cleaned_lyrics)
            for item in langlist:
                if item['lang'] in major_lang:
                    ipa, xsampa = utils.major_g2p(item['text'], item['lang'])
                    ipa_list.extend(ipa)
                    xsampa_list.extend(xsampa)
                else:
                    ipa, xsampa = utils.charsiu_g2p(item['text'], item['lang'], self.charsiu_model,
                                                    self.charsiu_tokenizer, **kwargs)
                    ipa_list.extend(ipa)
                    xsampa_list.extend(xsampa)
        else:
            if tag in major_lang:
                ipa_list, xsampa_list = utils.major_g2p(cleaned_lyrics, tag)
            else:
                ipa_list, xsampa_list = utils.charsiu_g2p(cleaned_lyrics, tag, self.charsiu_model,
                                                          self.charsiu_tokenizer, **kwargs)
        return ipa_list, xsampa_list

    def batch_infer(self, lyrics_list: list, tag: list = None, **kwargs):
        cleaned_lyrics_list = [utils.clean_lyrics(i) for i in lyrics_list]
        major_lang = ['zh', 'en', 'ja']

        g2p_process_list = []
        g2p_tag_list = []
        charsiu_process_list = []
        charsiu_tag_list = []
        raw_sequence_pos = []
        if tag is None:
            langlist = [utils.multi_lang_tokenizer(i) for i in cleaned_lyrics_list]
            flag1 = 0
            for every_lyrics in langlist:
                raw_sequence_pos.append([])
                for item in every_lyrics:
                    if item['lang'] in major_lang:
                        g2p_process_list.append(item['text'])
                        g2p_tag_list.append(item['lang'])
                        raw_sequence_pos[flag1].append(1)
                    else:
                        charsiu_process_list.append(item['text'])
                        charsiu_tag_list.append(item['lang'])
                        raw_sequence_pos[flag1].append(2)
                flag1 += 1

        # print(g2p_process_list) ['チャーシュー是一种', '踏碎凌霄', 'わたしの光', 'charsiu is a pork ', '我爱你', 'i love you ']
        # print(g2p_tag_list) ['ja', 'zh', 'ja', 'en', 'zh', 'en']
        # print(charsiu_process_list) ['barbecued pork', 'Ich liebe dich', 'สว สด ', '넌 나의 빛', '사랑해 요']
        # print(charsiu_tag_list) ['es', 'de', 'th', 'ko', 'ko']
        # print(raw_sequence_pos) [[1, 2], [2, 1, 2], [1, 2], [1], [2, 1, 1]]

        g2p_processed_ipa, g2p_processed_xsampa = utils.major_g2p_batch(g2p_process_list, g2p_tag_list)
        charsiu_processed_ipa, charsiu_processed_xsampa = utils.charsiu_g2p_batch(charsiu_process_list,
                                                                                  charsiu_tag_list, self.charsiu_model,
                                                                                  self.charsiu_tokenizer, **kwargs)
        res_list = []
        for item in raw_sequence_pos:
            cur_ipa, cur_xsampa = [], []
            for i in item:
                if i == 1:
                    cur_ipa.extend(g2p_processed_ipa.pop(0))
                    cur_xsampa.extend(g2p_processed_xsampa.pop(0))
                else:
                    cur_ipa.extend(charsiu_processed_ipa.pop(0))
                    cur_xsampa.extend(charsiu_processed_xsampa.pop(0))
            res_list.append((cur_ipa, cur_xsampa))
        return res_list


if __name__ == '__main__':
    g2p = MG2P()
    # print(g2p.check_if_sup('zh'))
    lyrics = 'チャーシュー是一种Cantonese风格of barbecued pork'
    lyrics1 = '踏碎凌霄 放肆桀骜 世恶道险 终究难逃'
    lyrics2 = '今でもあなたはわたしの光'
    print(g2p(lyrics), g2p(lyrics1), g2p(lyrics2))
