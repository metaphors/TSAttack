from .tibetan_word2vec import TibetanWord2VecSubstitute


def get_default_substitute(lang):
    from ....tags import TAG_Tibetan
    if lang == TAG_Tibetan:
        return TibetanWord2VecSubstitute()
