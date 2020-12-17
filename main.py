# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import string
import unicodedata
import spacy

nlp = spacy.load("en_core_web_sm")

science_training_file = "./science/train.txt"


def unicode_to_ascii(s):
    all_letters = string.ascii_letters + " .,;'"
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def read_names(filename):
    with open(filename, encoding='utf-8') as f:
        names = f.read().strip().split('\n')
    return [name.split('\t') for name in names]


class NamedEntities:
    label = ""
    begin = 0
    end = 0

    def __init__(self, label, begin, end):
        self.label = label
        self.begin = begin
        self.end = end

    def getLabel(self):
        return self.label

    def getBegin(self):
        return self.begin

    def getEnd(self):
        return self.end


def named_entity_recognition(raw_data):
    namedEntities = {}
    doc = []

    i = 0
    y = 0
    sentence = ""
    for pair in raw_data:
        if len(pair) == 2:
            if i == 0:
                sentence = pair[0]
            else:
                sentence = sentence + " " + pair[0]

            if pair[1] != 'O':
                namedEntities.setdefault(y, []).append(NamedEntities(pair[1], len(sentence) - len(pair[0]), len(sentence)))
            i = i + 1
        else:
            i = 0
            if sentence != "":
                doc.append(sentence)
                y = y + 1
            sentence = ""
    if sentence != "":
        doc.append(sentence)
    # DISPLAY
    globalDisplay(doc, namedEntities)


def globalDisplay(doc, namedEntities):
    i = 0
    for sentence in doc:
        print(sentence)
        if i in namedEntities.keys():
            for namedEntitiesForSentence in namedEntities[i]:
                print(namedEntitiesForSentence.getLabel() + " " + str(namedEntitiesForSentence.getBegin()) + " " + str(namedEntitiesForSentence.getEnd()))
        i = i + 1


if __name__ == '__main__':
    raw_data = read_names(science_training_file)
    named_entity_recognition(raw_data)
