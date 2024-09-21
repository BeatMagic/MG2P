import LangSegment
from .cleaner import clean_text
from .languages.symbol import symbol_to_id

from typing import List, Union
from loguru import logger
import hanlp


class G2P:
    def __init__(self):
        self.tok_fine = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
        self.pos = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL)

    def __call__(self, text: Union[str, List[str]], language: Union[str, List[str]] = None):
        if isinstance(text, str):
            text = [text]

        if isinstance(language, str) or language is None:
            language = [language] * len(text)

        assert len(text) == len(language), "text and language must have the same length"

        results = []
        for txt, lang in zip(text, language):
            try:
                result = self.infer_one(txt, lang)
            except Exception as e:
                logger.error(f"Error in infer_one: {e}")
                result = {
                    "phones": [],
                    "phone_ids": [],
                    "norm_text": "",
                    "word2ph": []
                }
            results.append(result)
        return results

    def infer_one(self, text: str, language: str = None):
        if language is None:
            LangSegment.setfilters(['zh', "ja", "en"])
        else:
            assert language in ["zh", "ja", "en"], "Only support zh, ja, en language, but got {}".format(language)
            LangSegment.setfilters([language])

        langlist = LangSegment.getTexts(text)
        all_norm_text = ""
        all_phones = []
        all_phone_ids = []
        all_word2ph = []
        for line in langlist:
            lang = line["lang"]
            text = line["text"]
            phones, word2ph, norm_text = clean_text(text, lang, self.tok_fine, self.pos)
            all_norm_text += norm_text
            all_phones += phones
            phone_ids = [symbol_to_id[symbol] for symbol in phones]
            all_phone_ids += phone_ids
            all_word2ph += word2ph
        return {
            "phones": all_phones,
            "phone_ids": all_phone_ids,
            "norm_text": all_norm_text,
            "word2ph": all_word2ph,
        }
