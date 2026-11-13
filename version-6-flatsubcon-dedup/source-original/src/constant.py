import os
from datetime import datetime
from .helper import init_logger

LOGGER = init_logger(datetime.now().strftime('linear_eval_%d%b%Y_%H-%M-%S.log'))

# 11 BIO labels for Vietnamese Public Administration Procedures Named Entity Recognition (PAP_NER)
LABEL2ID_PAP_NER = [
    'O',
    'B-CQ', 'I-CQ',
    'B-ĐT', 'I-ĐT',
    'B-VBPL', 'I-VBPL',
    'B-NG', 'I-NG',
    'B-SL', 'I-SL'
]

ID2LABEL_PAP_NER = {idx: label for idx, label in enumerate(LABEL2ID_PAP_NER)}

LABEL_MAPPING = {
    'pap_ner': {
        'label2id': LABEL2ID_PAP_NER,
        'id2label': ID2LABEL_PAP_NER,
        'header': ['token', 'pos', 'chunk', 'ner', 'tmp']
    }
}

