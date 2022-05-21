def remove_general_ends(word):
    if len(word) > 4 and word[-2:] in {"ia", "ie"}:
        return word[:-2]
    if len(word) > 4 and word[-1:] in {"u", u"ą", "i", "a", u"ę", "y", u"ę", u"ł"}:
        return word[:-1]
    return word


def remove_diminutive(word):
    if len(word) > 6:
        if word[-5:] in {"eczek", "iczek", "iszek", "aszek", "uszek"}:
            return word[:-5]
        if word[-4:] in {"enek", "ejek", "erek"}:
            return word[:-2]
    if len(word) > 4:
        if word[-2:] in {"ek", "ak"}:
            return word[:-2]
    return word


def remove_verbs_ends(word):
    if len(word) > 5 and word.endswith("bym"):
        return word[:-3]
    if len(word) > 5 and word[-3:] in {"esz", "asz", "cie", u"eść", u"aść", u"łem", "amy", "emy"}:
        return word[:-3]
    if len(word) > 3 and word[-3:] in {"esz", "asz", u"eść", u"aść", u"eć", u"ać"}:
        return word[:-2]
    if len(word) > 3 and word[-3:] in {"aj"}:
        return word[:-1]
    if len(word) > 3 and word[-2:] in {u"ać", "em", "am", u"ał", u"ił", u"ić", u"ąc"}:
        return word[:-2]
    return word


def remove_nouns(word):
    if len(word) > 7 and word[-5:] in {"zacja", u"zacją", "zacji"}:
        return word[:-4]
    if len(word) > 6 and word[-4:] in {"acja", "acji", u"acją", "tach", "anie", "enie",
                                       "eniu", "aniu"}:
        return word[:-4]
    if len(word) > 6 and word.endswith("tyka"):
        return word[:-2]
    if len(word) > 5 and word[-3:] in {"ach", "ami", "nia", "niu", "cia", "ciu"}:
        return word[:-3]
    if len(word) > 5 and word[-3:] in {"cji", "cja", u"cją"}:
        return word[:-2]
    if len(word) > 5 and word[-2:] in {"ce", "ta"}:
        return word[:-2]
    return word


def remove_adjective_ends(word):
    if len(word) > 7 and word.startswith("naj") and (word.endswith("sze")
                                                     or word.endswith("szy")):
        return word[3:-3]
    if len(word) > 7 and word.startswith("naj") and word.endswith("szych"):
        return word[3:-5]
    if len(word) > 6 and word.endswith("czny"):
        return word[:-4]
    if len(word) > 5 and word[-3:] in {"owy", "owa", "owe", "ych", "ego"}:
        return word[:-3]
    if len(word) > 5 and word[-2:] in {"ej"}:
        return word[:-2]
    return word


def remove_adverbs_ends(word):
    if len(word) > 4 and word[:-3] in {"nie", "wie"}:
        return word[:-2]
    if len(word) > 4 and word.endswith("rze"):
        return word[:-2]
    return word


def remove_plural_forms(word):
    if len(word) > 4 and (word.endswith(u"ów") or word.endswith("om")):
        return word[:-2]
    if len(word) > 4 and word.endswith("ami"):
        return word[:-3]
    return word


def stemming(word):
    stem = word[:]
    stem = remove_nouns(stem)
    stem = remove_diminutive(stem)
    stem = remove_adjective_ends(stem)
    stem = remove_verbs_ends(stem)
    stem = remove_adverbs_ends(stem)
    stem = remove_plural_forms(stem)
    stem = remove_general_ends(stem)
    return stem
