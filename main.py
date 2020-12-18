# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import string
import unicodedata

from pathlib import Path
import spacy
import random
from tqdm import tqdm

FORCE_UPDATE = 0
BONUS_DISPLAY = 0
output_dir = Path("C:\\Users\\Bastien\\Desktop\\ner")
# science_train_file = "./science/train.txt"
science_train_file = "./science/train_short.txt"
# science_test_file = "./science/test.txt"
science_test_file = "./science/test_short"
model = None
n_iter = 100


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


def train_nlp(output_dir, train_data):
    if model is not None:
        nlp = spacy.load(model)
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')
        print("Created blank 'en' model")

    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe('ner')

    for _, annotations in train_data:
        if annotations.get("entities") is not None:
            for ent in annotations.get("entities"):
                ner.add_label(ent[2])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            for text, annotations in tqdm(train_data):
                nlp.update(
                    [text],
                    [annotations],
                    drop=0.5,
                    sgd=optimizer,
                    losses=losses)
            if BONUS_DISPLAY:
                print(losses)

    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)


def analyse_doc(output_dir, test_data):
    print("Loading from", output_dir)
    nlp2 = spacy.load(output_dir)
    for text, _ in test_data:
        doc = nlp2(text)
        print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
        print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])


if __name__ == '__main__':
    raw_train_data = read_names(science_train_file)
    raw_test_data = read_names(science_test_file)
    train_data = named_entity_recognition(raw_train_data)
    test_data = named_entity_recognition(raw_test_data)

    # Training
    if not output_dir.exists() or FORCE_UPDATE:
        train_nlp(output_dir, train_data)
    # Testing
    analyse_doc(output_dir, test_data)

    if BONUS_DISPLAY:
        print(train_data)
