# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import string
import unicodedata
import spacy
import random

science_training_file = "./science/train.txt"
disease_training_file = "./disease/train.txt"


class NamedEntity:
    label = ""
    begin = 0
    end = 0

    def __init__(self, label, begin, end):
        self.label = remove_prefix(label, ["I-", "B-"])
        self.begin = begin
        self.end = end

    def getLabel(self):
        return self.label

    def getBegin(self):
        return self.begin

    def getEnd(self):
        return self.end

    def constructTriplet(self):
        return self.begin, self.end, self.label


def remove_prefix(text, prefixes):
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text.replace(prefix, "", 1)
            return text
    return text


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


def build_up_training_data(doc, namedEntities):
    train_data = []
    i = 0
    for sentence in doc:
        namedEntitiesForSentence = {}
        if i in namedEntities.keys():
            for namedEntityForSentence in namedEntities[i]:
                namedEntitiesForSentence.setdefault("entities", []).append(namedEntityForSentence.constructTriplet())
        train_data.append((sentence, namedEntitiesForSentence))
        i = i + 1
    return train_data


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
                namedEntities.setdefault(y, []).append(
                    NamedEntity(pair[1], len(sentence) - len(pair[0]), len(sentence)))
            i = i + 1
        else:
            i = 0
            if sentence != "":
                doc.append(sentence)
                y = y + 1
            sentence = ""
    if sentence != "":
        doc.append(sentence)

    return build_up_training_data(doc, namedEntities)


def train_nlp(train_data):
    nlp = spacy.blank("en")
    optimizer = nlp.begin_training()
    for i in range(20):
        random.shuffle(train_data)
        for text, annotations in train_data:
            nlp.update([text], [annotations], sgd=optimizer)
    nlp.to_disk("/model")
    return nlp


if __name__ == '__main__':
    raw_data = read_names(science_training_file)
    train_data = named_entity_recognition(raw_data)
    print(train_data)
    nlp = train_nlp(train_data)
    print("Done")