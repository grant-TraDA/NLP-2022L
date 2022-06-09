FROM python:3.6
WORKDIR nlp\_app
RUN git clone https://github.com/GoniaW/protagonist\_tagger/
WORKDIR protagonist\_tagger 
RUN pip install https://pypi.clarin-pl.eu/packages/poldeepner2-0.5.0-py3-none-any.whl && \
pip install -r requirements\_new.txt && \
python -m spacy download pl\_core\_news\_sm && \
python -m spacy download pl\_core\_news\_md && \
python -m spacy download pl\_core\_news\_lg && \
python -m spacy download xx\_ent\_wiki\_sm && \
python -m coreferee install pl
CMD /bin/bash
