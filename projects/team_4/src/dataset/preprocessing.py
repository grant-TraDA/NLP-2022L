import re
import pandas as pd
import string 

RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)

def preprocess_text(text, nlp):
    # remove whitespaces
    text = ' '.join(text.split())

    # tokenize
    doc = [tok for tok in nlp(text)]

    # remove RT @...
    if str(doc[0]) == "RT": 
        doc.pop(0)

    while str(doc[0]) == "@anonymized_account":
        doc.pop(0)
    while str(doc[-1]) == "@anonymized_account":
        doc.pop()

    # remove punctuation
    doc = [t for t in doc if t.text not in string.punctuation]
    
    # doc = [tok for tok in doc if not tok.is_stop]

    doc = [tok.lower_ for tok in doc]
    # doc = [RE_EMOJI.sub(r'', str_text) for str_text in doc]
    doc = [tok for tok in doc if len(tok) > 0]

    return doc