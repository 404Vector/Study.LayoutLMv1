from ..dependency import *
from ..const import *


def label_mapping(labels: List[str]):
    idx2label = {v: k for v, k in enumerate(labels)}
    label2idx = {k: v for v, k in enumerate(labels)}
    return idx2label, label2idx
