# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import string
import unicodedata

from pathlib import Path
import spacy
import random
from tqdm import tqdm

# Set to 1 to rebuild the models
FORCE_UPDATE = 0
# Set to 1 to display debug log
BONUS_DISPLAY = 0
# Set to 1 use shortened data files
USE_SHORT_VERSIONS = 0
SYMBOL_NOT_ENTITY = 'O'

if USE_SHORT_VERSIONS == 1:
    science_train_file = "./science/train_short.txt"
    science_test_file = "./science/test_short.txt"
    disease_train_file = "./disease/train_short.txt"
    disease_test_file = "./disease/test_short.txt"
    science_output_dir = Path("science/ner_short")
    disease_output_dir = Path("disease/ner_short")
else:
    science_train_file = "./science/train.txt"
    science_test_file = "./science/test.txt"
    disease_train_file = "./disease/train.txt"
    disease_test_file = "./disease/test.txt"
    science_output_dir = Path("science/ner")
    disease_output_dir = Path("disease/ner")

model = None
n_iter = 100


class NamedEntity:
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

            if pair[1] != SYMBOL_NOT_ENTITY:
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


def test_data_to_sentence(raw_data):
    doc = []
    doc_final = []
    skip_next = 0
    for pair in raw_data:
        if BONUS_DISPLAY:
            print(pair)
        if skip_next == 1:
            skip_next = 0
            doc_final.append(doc)
            doc = []
        else:
            if pair[0] == ".":
                skip_next = 1
            if pair[0] != '':
                doc.append((pair[0], pair[1]))
    doc_final.append(doc)
    return doc_final


# Resources:
# https://towardsdatascience.com/train-ner-with-custom-training-data-using-spacy-525ce748fab7
# https://www.machinelearningplus.com/nlp/training-custom-ner-model-in-spacy/
def train_nlp(output_dir, train_data):
    nlp = spacy.blank('en')
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner, last=True)

    for _, annotations in train_data:
        if annotations.get("entities") is not None:
            for ent in annotations.get("entities"):
                ner.add_label(ent[2])

    unused_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*unused_pipes):
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            for text, annotations in tqdm(train_data):
                nlp.update([text], [annotations], drop=0.5, sgd=optimizer, losses=losses)
            if BONUS_DISPLAY:
                print(losses)

    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)


def count_forgotten(sentence):
    forgotten = 0
    correct = 0
    for elt in sentence:
        if elt[1] != SYMBOL_NOT_ENTITY:
            forgotten = forgotten + 1
        else:
            correct = correct + 1
    return forgotten, correct


def accuracy(doc, sentence):
    wrong = 0
    true = 0
    for ent in doc:
        is_accurate = 0
        for elt in sentence:
            if elt[0] == ent.text and elt[1] == ent.label_:
                is_accurate = 1
                sentence.remove(elt)
                break
        true = true + is_accurate
        if is_accurate == 0:
            wrong = wrong + 1
    forgotten, true2 = count_forgotten(sentence)
    true = true + true2
    return true, wrong, forgotten


def analyse_doc(output_dir, test_data, test_as_sentence):
    nlp = spacy.load(output_dir)
    i = 0
    right = 0
    wrong = 0
    forgotten = 0
    for text, _ in test_data:
        doc = nlp(text)
        if BONUS_DISPLAY:
            print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
        right2, wrong2, forgotten2 = accuracy(doc.ents, test_as_sentence[i])
        right = right + right2
        wrong = wrong + wrong2
        forgotten = forgotten + forgotten2
        i = i + 1
    print("Accuracy Report :")
    print("Number of correct recognition :", right, "words", sep=" ")
    print("Number of wrong recognition :", wrong, "words", sep=" ")
    print("Number of recognition forgotten :", forgotten, "words", sep=" ")
    final_accuracy = (right / (right + wrong + forgotten)) * 100
    print("Overall accuracy : ", final_accuracy, "%", sep="")


def run_named_entity_recognition(train_file, test_file, output_dir):
    print("Output directory:", output_dir)

    # Training
    if not output_dir.exists() or FORCE_UPDATE:
        print("Training...")
        print("Train file:", train_file)
        raw_train_data = read_names(train_file)
        train_data = named_entity_recognition(raw_train_data)
        train_nlp(output_dir, train_data)
        if BONUS_DISPLAY:
            print(train_data)
    # Testing
    print("Testing...")
    print("Test file:", test_file)
    raw_test_data = read_names(test_file)
    test_as_sentence = test_data_to_sentence(raw_test_data)
    test_data = named_entity_recognition(raw_test_data)
    analyse_doc(output_dir, test_data, test_as_sentence)


if __name__ == '__main__':
    run_named_entity_recognition(science_train_file, science_test_file, science_output_dir)
    run_named_entity_recognition(disease_train_file, disease_test_file, disease_output_dir)
