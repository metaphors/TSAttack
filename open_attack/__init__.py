import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from .data import add_data
from .tags import add_tags
from . import attackers
from .attack_eval import AttackEval

add_data()
add_tags()
