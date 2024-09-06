import utils


class MG2P:
    def __init__(self):
        pass

    def check_if_sup(self, language: str):
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


if __name__ == '__main__':
    g2p = MG2P()
    print(g2p.check_if_sup('zh'))
