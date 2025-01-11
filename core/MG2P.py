from typing import List, Union, Tuple

import MG2P.core.utils as utils
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
from collections import defaultdict, deque
from MG2P.core.g2p import G2P as BaseG2P
from MG2P.core.ipa2list import ipalist2phoneme, load_ipa_dict
from konoha import WordTokenizer
from attacut import Tokenizer as ThaiTokenizer
import mecab_ko as MeCab
import redis
from loguru import logger
import ToJyutping


class MG2P:
    def __init__(
        self,
        model_path='charsiu/g2p_multilingual_byT5_small_100',
        tokenizer_path='google/byt5-small',
        major_lang=['zh', 'en', 'ja', "eng"],
        use_32=False,
        use_cache=False
    ):
        self.charsiu_model = T5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if not use_32:
            self.charsiu_model.to(device).half()
        else:
            self.charsiu_model.to(device)
        self.device = device
        self.charsiu_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.major_lang = major_lang
        self.roma2ipa_map = utils.load_romaji2ipa_map()
        self.prefix_map = utils.generate_sup_language_list()
        matchers, ipa_dictionaries = load_ipa_dict()
        self.matchers = matchers
        self.ipa_dictionaries = ipa_dictionaries
        self.base_g2p = BaseG2P()
        self.zh_tokenizer = self.base_g2p.tok_fine
        self.jp_tokenizer = WordTokenizer('Sentencepiece', model_path='MG2P/core/model.spm')
        self.thai_tokenizer = ThaiTokenizer()
        self.ko_tokenizer = MeCab.Tagger("-Owakati")
        if use_cache:
            self.redis_client = redis.Redis(
                host='192.168.101.22',      # Redis 服务器地址
                port=6379,             # Redis 端口
                password=None,         # 如果有密码则填写
                db=3,                  # 使用的数据库编号
                decode_responses=True  # 自动解码响应为字符串
            )
        self.use_cache = use_cache

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

    def check_if_cached(self, lyrics: str, tag: str, suffix: str) -> bool:
        """
        check if the lyrics is cached
        :param lyrics: the lyrics
        :param tag: the language of the lyrics
        :return:
        [True]: the lyrics is cached
        [False]: the lyrics is not cached
        """
        if not self.use_cache:
            return False
        key = f"mg2p-{tag}-{suffix}:{lyrics}"
        if self.redis_client.exists(key):
            return True
        return False

    def get_cached_result(self, lyrics: str, tag: str, suffix: str) -> List[str]:
        """
        get the cached result
        :param lyrics: the lyrics
        :param tag: the language of the lyrics
        :return: the cached result
        """
        key = f"mg2p-{tag}-{suffix}:{lyrics}"
        result = self.redis_client.get(key)
        return eval(result)

    def set_cached_result(self, lyrics: str, tag: str, result: Tuple[List[str], List[str]], suffix: str = "sentence"):
        """
        set the cached result
        :param lyrics: the lyrics
        :param tag: the language of the lyrics
        :param result: the result
        """
        if not self.use_cache:
            return
        key = f"mg2p-{tag}-{suffix}:{lyrics}"
        self.redis_client.set(key, str(result))

    def set_cached_result_batch(self, cached_dict, suffix: str = "sentence"):
        """
        set the cached result
        :param lyrics: the lyrics
        :param tag: the language of the lyrics
        :param result: the result
        """
        if not self.use_cache:
            return
        for (lyric, tag), result in cached_dict.items():
            key = f"mg2p-{tag}-{suffix}:{lyric}"
            self.redis_client.set(key, str(result))

    def split_with_cache(self, lyrics: List[str], tag: List[str], suffix: str = "sentence"):
        in_cached_result = {}
        to_infer_idx = set()
        to_infer_lyrics = []
        to_infer_tag = []
        for idx, (lyric_line, tag) in enumerate(zip(lyrics, tag)):
            if (lyric_line, tag) in in_cached_result:
                continue
            elif self.check_if_cached(lyric_line, tag, suffix):
                cached_result = self.get_cached_result(lyric_line, tag, suffix)
                in_cached_result[(lyric_line, tag)] = cached_result
            else:
                to_infer_idx.add(idx)
                to_infer_lyrics.append(lyric_line)
                to_infer_tag.append(tag)
        return in_cached_result, to_infer_idx, to_infer_lyrics, to_infer_tag

    def transformer_input(self, lyrics: Union[str, List[str]], tag: Union[str, List[str]] = None) -> List[str]:
        is_batch = True
        if isinstance(lyrics, str):
            lyrics = [lyrics]
            is_batch = False
        if tag is not None and isinstance(tag, str):
            tag = [tag] * len(lyrics)
        return lyrics, tag, is_batch

    def major_g2p_infer(self, lyrics_list: list, tag_list: list):
        # 处理缓存
        in_cached_result, to_infer_idx, to_infer_lyrics, to_infer_tag = self.split_with_cache(lyrics_list, tag_list, "sentence")
        if len(to_infer_lyrics) == 0:
            # 恢复原始顺序
            new_ipa_list = deque([])
            new_xsampa_list = deque([])
            for idx, (lyric_line, tag) in enumerate(zip(lyrics_list, tag_list)):
                if (lyric_line, tag) in in_cached_result:
                    new_ipa_list.append(in_cached_result[(lyric_line, tag)][0])
                    new_xsampa_list.append(in_cached_result[(lyric_line, tag)][1])
            return new_ipa_list, new_xsampa_list

        res = self.base_g2p(to_infer_lyrics, to_infer_tag)

        ipa_list = deque([])
        xsampa_list = deque([])
        # ipa在生成前长度不可预见, 因此单独做计算
        for i, item in enumerate(res):
            phones = item['phones']
            if tag_list[i] in ('en', "eng"):
                en_ipa = utils.arpa2ipa(phones)
                ipa_list.append(en_ipa)
                xsampa_list.append(utils.IPA2SAMPA(en_ipa))
            if tag_list[i] == 'zh':
                zh_xsampa, zh_ipa = utils.pinyin2ipa(phones)
                ipa_list.append(zh_ipa)
                xsampa_list.append(zh_xsampa)
            if tag_list[i] == 'ja':
                ja_ipa = utils.romaji2ipa(phones, self.roma2ipa_map)
                ipa_list.append(ja_ipa)
                xsampa_list.append(utils.IPA2SAMPA(ja_ipa))

        infered_result = {}
        for i, (lyric_line, tag) in enumerate(zip(to_infer_lyrics, to_infer_tag)):
            infered_result[(lyric_line, tag)] = (ipa_list[i], xsampa_list[i])

        self.set_cached_result_batch(infered_result, "sentence")

        in_cached_result.update(infered_result)
        # 恢复原始顺序
        new_ipa_list = deque([])
        new_xsampa_list = deque([])
        for idx, (lyric_line, tag) in enumerate(zip(lyrics_list, tag_list)):
            ipa, xsampa = in_cached_result[(lyric_line, tag)]
            new_ipa_list.append(ipa)
            new_xsampa_list.append(xsampa)

        return new_ipa_list, new_xsampa_list

    def yue_g2p_infer(self, lyrics_list: list):
        tag_list = ['yue'] * len(lyrics_list)
        # 处理缓存
        in_cached_result, to_infer_idx, to_infer_lyrics, to_infer_tag = self.split_with_cache(lyrics_list, tag_list, "sentence")
        if len(to_infer_lyrics) == 0:
            # 恢复原始顺序
            new_ipa_list = deque([])
            new_xsampa_list = deque([])
            for idx, (lyric_line, tag) in enumerate(zip(lyrics_list, tag_list)):
                if (lyric_line, tag) in in_cached_result:
                    new_ipa_list.append(in_cached_result[(lyric_line, tag)][0])
                    new_xsampa_list.append(in_cached_result[(lyric_line, tag)][1])
            return new_ipa_list, new_xsampa_list

        ipa_list = deque([])
        xsampa_list = deque([])
        for lyric_line in to_infer_lyrics:
            word_ipa_list = ToJyutping.get_ipa_list(lyric_line)
            line_ipa_list = [ipa for grapheme, ipa in word_ipa_list if ipa is not None]
            line_ipa_list = ipalist2phoneme(line_ipa_list, self.matchers, 'yue')
            processed_xsampa, processed_ipa = utils.yue_tone_backend(line_ipa_list)
            ipa_list.append(processed_ipa)
            xsampa_list.append(processed_xsampa)

        infered_result = {}
        for i, (lyric_line, tag) in enumerate(zip(to_infer_lyrics, to_infer_tag)):
            infered_result[(lyric_line, tag)] = (ipa_list[i], xsampa_list[i])

        self.set_cached_result_batch(infered_result, "sentence")

        in_cached_result.update(infered_result)
        # 恢复原始顺序
        new_ipa_list = deque([])
        new_xsampa_list = deque([])
        for idx, (lyric_line, tag) in enumerate(zip(lyrics_list, tag_list)):
            ipa, xsampa = in_cached_result[(lyric_line, tag)]
            new_ipa_list.append(ipa)
            new_xsampa_list.append(xsampa)

        return new_ipa_list, new_xsampa_list

    def tokenize_lyrics(self, lyrics, tag):
        if tag == "zh" or tag == "zho-s" or tag == "yue":
            return self.zh_tokenizer(lyrics)
        if tag == "ja" or tag == "jpn":
            return utils.jp_tokenizer(self.jp_tokenizer, lyrics)
        if tag == 'th' or tag == 'tha':
            return self.thai_tokenizer.tokenize(lyrics)
        if tag == 'ko' or tag == 'kor':
            return self.ko_tokenizer.parse(lyrics).split()
        lyrics_list = lyrics.split()
        return lyrics_list

    def charsiu_g2p_infer(self, lyrics_list: list, tag_list: list, batch_size=2000):
        if len(lyrics_list) == 0:
            return [], []
        grapheme_list = []
        grapheme_split_list = [0]

        # 添加prefix code并计算语种分隔位置
        word_tag_list = []
        for i, lyrics in enumerate(lyrics_list):
            current_prefix = self.prefix_map[tag_list[i]] if tag_list[i] in self.prefix_map else 'unk'
            current_grapheme_list = self.tokenize_lyrics(lyrics, tag_list[i])
            grapheme_split_list.append(len(current_grapheme_list) + grapheme_split_list[-1])
            word_tag_list.extend([tag_list[i]] * len(current_grapheme_list))
            grapheme_list.extend([f'<{current_prefix}>: {grapheme}' for grapheme in current_grapheme_list])

        # 处理缓存
        in_cached_result, to_infer_idx, to_infer_grapheme_list, to_infer_tag = self.split_with_cache(grapheme_list, word_tag_list, "word")

        # 转换为dict
        to_infer_grapheme_dict = {grapheme: t for grapheme, t in zip(to_infer_grapheme_list, to_infer_tag)}
        to_infer_grapheme_keys = sorted(to_infer_grapheme_dict.keys())
        infered_result = {}
        for i in range(0, len(to_infer_grapheme_keys), batch_size):
            to_infer_batch_keys = to_infer_grapheme_keys[i:i + batch_size]
            out = self.charsiu_tokenizer(to_infer_batch_keys, padding=True, add_special_tokens=False, return_tensors='pt')
            out = {k: v.to(self.device) for k, v in out.items()}
            preds = self.charsiu_model.generate(**out, num_beams=1, max_length=61)
            phones = utils.charsiu_model_decode(preds)
            infered_result.update({(to_infer_batch_keys[idx], to_infer_grapheme_dict[to_infer_batch_keys[idx]]): (phones[idx], None) for idx in range(len(to_infer_batch_keys))})

        self.set_cached_result_batch(infered_result, "word")
        in_cached_result.update(infered_result)

        to_infer_grapheme_list = deque(to_infer_grapheme_list)
        for idx, (grapheme, t) in enumerate(zip(grapheme_list, word_tag_list)):
            if (grapheme, t) in in_cached_result:
                grapheme_list[idx] = in_cached_result[(grapheme, t)][0]
            else:
                logger.warning(f"Failed to infer {grapheme} in {t}")

        # 每个语种出现一次作为一个列表
        ipa_list = deque([])
        xsampa_list = deque([])
        for i, tag in enumerate(tag_list):
            start, end = grapheme_split_list[i], grapheme_split_list[i + 1]
            current_grapheme_list = grapheme_list[start:end]
            current_ipa_list = ipalist2phoneme(current_grapheme_list, self.matchers, tag)
            ipa_list.append(current_ipa_list)
            current_xsampa_list = utils.IPA2SAMPA(current_ipa_list)
            xsampa_list.append(current_xsampa_list)

        return ipa_list, xsampa_list

    def __call__(self, lyrics: Union[str, List[str]], tag: Union[str, List[str]] = None, batch_size: int = 2000) -> List[List[str]]:
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
        if isinstance(lyrics, str) and utils.clean_lyrics(lyrics) == '':
            return [[], []]

        # 1. 检查输入，是否是字符串或者字符串列表
        lyrics, tag, is_batch = self.transformer_input(lyrics, tag)

        # 2. 清除歌词中的时间戳，换行符、标点符号等
        cleaned_lyrics_list = list(map(utils.clean_lyrics, lyrics))

        # 3. 将歌词分割成多种语言
        g2p_process_list = []
        g2p_tag_list = []
        charsiu_process_list = []
        charsiu_tag_list = []
        yue_process_list = []
        yue_tag_list = []
        g2p_sources = defaultdict(list)
        langlist = []
        for idx, cleaned_lyrics in enumerate(cleaned_lyrics_list):
            lang_objs = utils.multi_lang_tokenizer(cleaned_lyrics, lang=tag if tag is None else tag[idx])
            langlist.append(lang_objs)
        for idx, lang_objs in enumerate(langlist):
            for item in lang_objs:
                process_list = g2p_process_list if item['lang'] in self.major_lang else charsiu_process_list
                tag_list = g2p_tag_list if item['lang'] in self.major_lang else charsiu_tag_list
                source = "major" if item['lang'] in self.major_lang else "charsiu"
                if tag is not None and "yue" in tag and "zh" in item['lang']:
                    process_list = yue_process_list
                    tag_list = yue_tag_list
                    source = "yue"
                process_list.append(item['text'])
                tag_list.append(item['lang'])
                g2p_sources[idx].append(source)

        # 5. 调用不同的模型进行phneme转换
        g2p_processed_ipa, g2p_processed_xsampa = self.major_g2p_infer(g2p_process_list, g2p_tag_list)
        charsiu_processed_ipa, charsiu_processed_xsampa = self.charsiu_g2p_infer(charsiu_process_list, charsiu_tag_list, batch_size=batch_size)
        yue_processed_ipa, yue_processed_xsampa = self.yue_g2p_infer(yue_process_list)
        res_list = [
            (
                [
                    g2p_processed_ipa.popleft() if source == "major" else (yue_processed_ipa.popleft() if source == "yue" else charsiu_processed_ipa.popleft())
                    for source in g2p_sources[idx]
                ],
                [
                    g2p_processed_xsampa.popleft() if source == "major" else (yue_processed_xsampa.popleft() if source == "yue" else charsiu_processed_xsampa.popleft())
                    for source in g2p_sources[idx]
                ]
            )
            for idx in range(len(g2p_sources))
        ]
        return_res = []
        for ipa_list, xsampa_list in res_list:
            ipas = []
            xsampas = []
            for ipa in ipa_list:
                ipas.extend(ipa)
            for xsampa in xsampa_list:
                xsampas.extend(xsampa)
            return_res.append((ipas, xsampas))
        return return_res


if __name__ == '__main__':
    mg2p = MG2P()
    # print(g2p.check_if_sup('zh'))
    lyrics = 'チャーシュー是一种Cantonese风格of barbecued pork'
    lyrics1 = '踏碎凌霄 放肆桀骜 世恶道险 终究难逃'
    lyrics2 = '今でもあなたはわたしの光'
    print(mg2p(lyrics), mg2p(lyrics1), mg2p(lyrics2))
