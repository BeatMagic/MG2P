from .languages import chinese, japanese, english
from .languages.symbol import symbols
from loguru import logger


language_module_map = {"zh": chinese, "ja": japanese, "en": english}

special = [
    ("￥", "zh", "SP2"),
    ("^", "zh", "SP3"),
]


def clean_special(text, language, special_s, target_symbol):
    """
    特殊静音段sp符号处理
    """
    text = text.replace(special_s, ",")
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones = language_module.g2p(norm_text)
    new_ph = []
    for ph in phones[0]:
        assert ph in symbols
        if ph == ",":
            new_ph.append(target_symbol)
        else:
            new_ph.append(ph)
    return new_ph, phones[1], norm_text


def clean_text(text, language):
    text = text.replace("%", "-").replace("￥", ",").replace("...", "…")
    if language not in language_module_map:
        language = "en"
        text = " "

    for special_s, special_l, target_symbol in special:
        if special_s in text and language == special_l:
            return clean_special(text, language, special_s, target_symbol)

    language_module = language_module_map[language]

    norm_text = language_module.text_normalize(text)

    if language == "zh":
        phones, word2ph = language_module.g2p(norm_text)
        assert len(phones) == sum(word2ph)
        assert len(norm_text) == len(word2ph)
    else:
        phones = language_module.g2p(norm_text)
        word2ph = [None]
    valid_phones = []
    for ph in phones:
        if ph in symbols:
            valid_phones.append(ph)
        else:
            logger.warning(f"ph: {ph} not in symbols {text}")
    return phones, word2ph, norm_text


if __name__ == "__main__":
    print(clean_text("你好%啊啊啊额、还是到付红四方。", "zh"))
