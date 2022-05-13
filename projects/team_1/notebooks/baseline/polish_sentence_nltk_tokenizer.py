import nltk

# interactive download
# nltk.download()
nltk.download('punkt')

extra_abbreviations = [
        "ps",
        "inc",
        "corp",
        "ltd",
        "Co",
        "pkt",
        "Dz.Ap",
        "Jr",
        "jr",
        "sp.k",
        "sp",
        # "Sp",
        "poj",
        "pseud",
        "krypt",
        "ws",
        "itd",
        "np",
        "sanskryt",
        "nr",
        "gł",
        "Takht",
        "tzw",
        "tzn",
        "t.zw",
        "ewan",
        "tyt",
        "fig",
        "oryg",
        "t.j",
        "vs",
        "l.mn",
        "l.poj",
        "ul",
        "al",
        "Al",
        "el",
        "tel",
        "wew",  # wewnętrzny
        "bud",
        "pok",
        "wł",
        "sam",  # samochód
        "sa",  # spółka sa.
        "wit",  # witaminy
        "mat",  # materiały
        "kat",  # kategorii
        "wg",  # według
        "btw",  #
        "itp",  #
        "wz",  # w związku
        "gosp",  #
        "dział",  #
        "hurt",  #
        "mech",  #
        "wyj",  # wyj
        "pt",  # pod tytułem
        "zew",  # zewnętrzny
    ]

position_abbrev = [
        "Ks",
        "Abp",
        "abp",
        "bp",
        "dr",
        "kard",
        "mgr",
        "prof",
        "zwycz",
        "hab",
        "arch",
        "arch.kraj",
        "B.Sc",
        "Ph.D",
        "lek",
        "med",
        "n.med",
        "bł",
        "św",
        "hr",
        "dziek",
    ]

roman_abbrev = (
        []
    )  # ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XII','XIV','XV','XVI', 'XVII', 'XVIII','XIX', 'XX', 'XXI' ]

quantity_abbrev = [
        "mln",
        "obr./min",
        "km/godz",
        "godz",
        "egz",
        "ha",
        "j.m",
        "cal",
        "obj",
        "alk",
        "wag",
        "obr",  # obroty
        "wk",
        "mm",
        "MB",  # mega bajty
        "Mb",  # mega bity
        "jedn",  # jednostkowe
        "op",
        "szt",  # sztuk
    ]  # not added: tys.

actions_abbrev = [
        "tłum",
        "tlum",
        "zob",
        "wym",
        "w/wym",
        "pot",
        "ww",
        "ogł",
        "wyd",
        "min",
        "m.i",
        "m.in",
        "in",
        "im",
        "muz",
        "tj",
        "dot",
        "wsp",
        "właść",
        "właśc",
        "przedr",
        "czyt",
        "proj",
        "dosł",
        "hist",
        "daw",
        "zwł",
        "zaw",
        "późn",
        "spr",
        "jw",
        "odp",  # odpowiedź
        "symb",  # symbol
        "klaw",  # klawiaturowe
    ]

place_abbrev = [
        "śl",
        "płd",
        "geogr",
        "zs",
        "pom",  # pomorskie
        "kuj-pom",  # kujawsko pomorskie
    ]

lang_abbrev = [
        "jęz",
        "fr",
        "franc",
        "ukr",
        "ang",
        "gr",
        "hebr",
        "czes",
        "pol",
        "niem",
        "arab",
        "egip",
        "hiszp",
        "jap",
        "chin",
        "kor",
        "tyb",
        "wiet",
        "sum",
        "chor",
        "słow",
        "węg",
        "ros",
        "boś",
        "szw",
    ]

administration = [
        "dz.urz",  # dziennik urzędowy
        "póź.zm",
        "rej",  # rejestr, rejestracyjny dowód
        "sygn",  # sygnatura
        "Dz.U",  # dziennik ustaw
        "woj",  # województow
        "ozn",  #
        "ust",  # ustawa
        "ref",  # ref
        "dz",
        "akt",  # akta
    ]

time = [
        "tyg",  # tygodniu
    ]

military_abbrev = [
        "kpt",
        "kpr",
        "obs",
        "pil",
        "mjr",
        "płk",
        "dypl",
        "pp",
        "gw",
        "dyw",
        "bryg",  # brygady
        "ppłk",
        "mar",
        "marsz",
        "rez",
        "ppor",
        "DPanc",
        "BPanc",
        "DKaw",
        "p.uł",
        "sierż",
        "post",
        "asp",
        "szt",  # sztabowy
        "podinsp",
        "kom",  # komendant, tel. komórka
        "nadkom"
    ]

extra_abbreviations = (
        extra_abbreviations
        + position_abbrev
        + quantity_abbrev
        + place_abbrev
        + actions_abbrev
        + lang_abbrev
        + administration
        + time
        + military_abbrev
    )
sentence_tokenizer = nltk.data.load('tokenizers/punkt/polish.pickle')
sentence_tokenizer._params.abbrev_types.update(extra_abbreviations)


# text = '.....'

# sentences = sentence_tokenizer.tokenize(text)