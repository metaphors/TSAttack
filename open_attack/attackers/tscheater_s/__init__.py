# Source Attribution:
# The majority of the code is derived from the following source:
# - OpenAttack GitHub Repository: https://github.com/thunlp/OpenAttack
# - Tag: v2.1.1
# - File: /OpenAttack/attackers/pwws/__init__.py
# - Reference Paper: OpenAttack: An Open-source Textual Adversarial Attack Toolkit (Zeng et al., ACL-IJCNLP 2021)
# Our modification is located between the comments "modification begin" and "modification end".

from typing import List, Optional
import numpy as np

# modification begin
from OpenAttack.attackers.classification import ClassificationAttacker, Classifier, ClassifierGoal
from OpenAttack.text_process.tokenizer import Tokenizer
from ...text_process.tokenizer import get_default_tokenizer
# from OpenAttack.attack_assist.substitute.word import WordSubstitute
# from ...attack_assist.substitute.word import get_default_substitute
from OpenAttack.utils import get_language, check_language, language_by_name
# from OpenAttack.exceptions import WordNotInDictionaryException
from OpenAttack.tags import Tag
from ...attack_assist.filter_words import get_default_filter_words
# modification end

# modification begin
import os
import json
# modification end

# modification begin
class TSCheater_s(ClassificationAttacker):
# modification end
    @property
    def TAGS(self):
        return { self.__lang_tag,  Tag("get_pred", "victim"), Tag("get_prob", "victim") }

    def __init__(self,
            tokenizer : Optional[Tokenizer] = None,
            # modification begin
            # substitute : Optional[WordSubstitute] = None,
            # modification end
            token_unk : str = "<UNK>",
            filter_words : List[str] = None,
            lang = None
        ):
        """
        Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency. Shuhuai Ren, Yihe Deng, Kun He, Wanxiang Che. ACL 2019.
        `[pdf] <https://www.aclweb.org/anthology/P19-1103.pdf>`__
        `[code] <https://github.com/JHL-HUST/PWWS/>`__

        Args:
            tokenizer: A tokenizer that will be used during the attack procedure. Must be an instance of :py:class:`.Tokenizer`
            # modification begin
            # substitute: A substitute that will be used during the attack procedure. Must be an instance of :py:class:`.WordSubstitute`
            # modification end
            token_unk: The token id or the token name for out-of-vocabulary words in victim model. **Default:** ``"<UNK>"``
            lang: The language used in attacker. If is `None` then `attacker` will intelligently select the language based on other parameters.
            filter_words: A list of words that will be preserved in the attack procesudre.

        :Classifier Capacity:
            * get_pred
            * get_prob

        
        """
       
        lst = []
        if tokenizer is not None:
            lst.append(tokenizer)
        # modification begin
        # if substitute is not None:
        #     lst.append(substitute)
        # modification end
        if len(lst) > 0:
            self.__lang_tag = get_language(lst)
        else:
            self.__lang_tag = language_by_name(lang)
            if self.__lang_tag is None:
                raise ValueError("Unknown language `%s`" % lang)

        # modification begin
        # if substitute is None:
        #     substitute = get_default_substitute(self.__lang_tag)
        # self.substitute = substitute
        # modification end

        if tokenizer is None:
            tokenizer = get_default_tokenizer(self.__lang_tag)
        self.tokenizer = tokenizer

        # modification begin
        # check_language([self.tokenizer, self.substitute], self.__lang_tag)
        check_language([self.tokenizer], self.__lang_tag)
        # modification end

        self.token_unk = token_unk
        if filter_words is None:
            filter_words = get_default_filter_words(self.__lang_tag)
        self.filter_words = set(filter_words)

        # modification begin
        with open(os.path.join(os.getcwd(), "data", "AttackAssist.TibetanSyllableVisualSimilarityDatabase", "similarity.json"), "r", encoding="utf-8") as file:
            self.similarity_json = json.load(file)
        # modification end

    def attack(self, victim: Classifier, sentence : str, goal : ClassifierGoal):
        x_orig = sentence.lower()


        x_orig = self.tokenizer.tokenize(x_orig)
        poss =  list(map(lambda x: x[1], x_orig)) 
        x_orig =  list(map(lambda x: x[0], x_orig)) 

        S = self.get_saliency(victim, x_orig, goal) # (len(sent), )
        S_softmax = np.exp(S - S.max())
        S_softmax = S_softmax / S_softmax.sum()

        w_star = [ self.get_wstar(victim, x_orig, i, poss[i], goal) for i in range(len(x_orig)) ]  # (len(sent), )
        H = [ (idx, w_star[idx][0], S_softmax[idx] * w_star[idx][1]) for idx in range(len(x_orig)) ]

        H = sorted(H, key=lambda x:-x[2])
        ret_sent = x_orig.copy()
        for i in range(len(H)):
            idx, wd, _ = H[i]
            if ret_sent[idx] in self.filter_words:
                continue
            ret_sent[idx] = wd
            
            curr_sent = self.tokenizer.detokenize(ret_sent)
            pred = victim.get_pred([curr_sent])[0]
            if goal.check(curr_sent, pred):
                return curr_sent
        return None


    
    def get_saliency(self, clsf, sent, goal : ClassifierGoal):
        x_hat_raw = []
        for i in range(len(sent)):
            left = sent[:i]
            right = sent[i + 1:]
            x_i_hat = left + [self.token_unk] + right
            x_hat_raw.append(self.tokenizer.detokenize(x_i_hat))
        x_hat_raw.append(self.tokenizer.detokenize(sent))
        res = clsf.get_prob(x_hat_raw)[:, goal.target]
        if not goal.targeted:
            res = res[-1] - res[:-1]
        else:
            res = res[:-1] - res[-1]
        return res

    def get_wstar(self, clsf, sent, idx, pos, goal : ClassifierGoal):
        word = sent[idx]
        # modification begin
        # try:
        #     rep_words = list(map(lambda x:x[0], self.substitute(word, pos)))
        # except WordNotInDictionaryException:
        #     rep_words = []
        # rep_words = list(filter(lambda x: x != word, rep_words))
        rep_words = []
        if word in self.similarity_json:
            for item in self.similarity_json[word]:
                if item[1] > 0.8:
                    rep_words.append(item[0])
        # modification end
        if len(rep_words) == 0:
            return ( word, 0 )
        sents = []
        for rw in rep_words:
            new_sent = sent[:idx] + [rw] + sent[idx + 1:]
            sents.append(self.tokenizer.detokenize(new_sent))
        sents.append(self.tokenizer.detokenize(sent))
        res = clsf.get_prob(sents)[:, goal.target]
        prob_orig = res[-1]
        res = res[:-1]
        if goal.targeted:
            return (rep_words[ res.argmax() ],  res.max() - prob_orig )
        else:
            return (rep_words[ res.argmin() ],  prob_orig - res.min() )
    








