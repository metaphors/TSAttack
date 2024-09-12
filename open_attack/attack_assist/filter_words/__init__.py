from .tibetan import TIBETAN_FILTER_WORDS


def get_default_filter_words(lang):
    from ...tags import TAG_Tibetan
    if lang == TAG_Tibetan:
        return TIBETAN_FILTER_WORDS
