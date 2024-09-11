from .base import G2P


g2p_model = G2P()


def infer(text, lang=None):
    return g2p_model(text, lang)
