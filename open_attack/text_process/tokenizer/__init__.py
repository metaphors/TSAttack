from .tibetan_syllable_tokenizer import TibetanSyllableTokenizer
from .TibetSegEYE.maintest import TibetanWordTokenizer


def get_default_tokenizer(lang):
    from ...tags import TAG_Tibetan
    if lang == TAG_Tibetan:
        return TibetanSyllableTokenizer()
