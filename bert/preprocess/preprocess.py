from sklearn.model_selection import train_test_split

from .dictionary import IndexDictionary
from .utils import prepend_data_dir

from tqdm import tqdm
from gensim.corpora import WikiCorpus
from nltk.tokenize import sent_tokenize
import sentencepiece as spm

import re

NUMBERS = re.compile(r'\d+')
TOKENIZATION = re.compile(r'(\w+)')


def tokenize(text: str, lower: bool, **_):  # token_min_len: int, token_max_len: int,
    if lower:
        text = text.lower()
    return text.split()


def detect_sentences(raw_documents_path, sentences_detected_path, **_):
    with open(raw_documents_path) as raw_documents_file, open(sentences_detected_path, 'w') as sentences_detected_file:
        for line in tqdm(raw_documents_file):
            sentences = sent_tokenize(line.strip())
            tokenized_sentences = []
            for sentence in sentences:
                sentence = sentence.lower()
                sentence = NUMBERS.sub('N', sentence)
                tokens = [match.group() for match in TOKENIZATION.finditer(sentence)]
                if not tokens:
                    continue
                tokenized_sentences.append(' '.join(tokens))

            output_line = '|'.join(tokenized_sentences) + '\n'
            sentences_detected_file.write(output_line)


def split_sentences(sentences_detected_path, spm_input_path, **_):
    with open(sentences_detected_path) as sentences_detected_file, open(spm_input_path, 'w') as spm_input_file:
        for line in tqdm(sentences_detected_file):
            for sentence in line.strip().split('|'):
                words = sentence.split()
                for i in range(0, len(words), 254):
                    sentence_segment = words[i:i+254]
                    spm_input_file.write(' '.join(sentence_segment) + '\n')


def prepare_documents(spm_model_prefix, sentences_detected_path, prepared_documents_path, **_):
    spm_model = spm_model_prefix + '.model'
    sp_preprocessor = spm.SentencePieceProcessor()
    sp_preprocessor.Load(spm_model)

    with open(sentences_detected_path) as sentences_detected_file, \
            open(prepared_documents_path, 'w') as prepared_documents_file:
        for document in tqdm(sentences_detected_file):
            prepared_sentences = []
            pieces = []
            for sentence in document.strip().split('|'):
                sentence_pieces = sp_preprocessor.EncodeAsPieces(sentence)

                if len(sentence_pieces) <= 254:

                    if len(pieces) + len(sentence_pieces) >= 254:
                        prepared_sentences.append(' '.join(pieces))
                        pieces = sentence_pieces
                    else:
                        pieces.extend(sentence_pieces)
                else:
                    if len(pieces) > 0:
                        prepared_sentences.append(' '.join(pieces))
                    for i in range(0, len(sentence_pieces), 254):
                        sentence_pieces_segment = sentence_pieces[i:i+254]
                        prepared_sentences.append(' '.join(sentence_pieces_segment))
                    pieces = []
            if len(prepared_sentences) < 2:
                continue
            output_line = '|'.join(prepared_sentences) + '\n'
            prepared_documents_file.write(output_line)


def split_train_val(prepared_documents_path, train_path, val_path, **_):
    with open(prepared_documents_path) as prepared_documents_file:
        documents = prepared_documents_file.readlines()

    train_data, val_data = train_test_split(documents, test_size=10000)
    with open(train_path, 'w') as train_file:
        for line in train_data:
            train_file.write(line)
    with open(val_path, 'w') as val_file:
        for line in val_data:
            val_file.write(line)


def build_dictionary(train_path, dictionary_path, **_):

    def token_generator(data_path):
        with open(data_path) as file:
            for document in file:
                for sentence in document.strip().split('|'):
                    for token in sentence.split():
                        yield token

    dictionary = IndexDictionary()
    dictionary.build_vocabulary(token_generator(train_path))
    dictionary.save(dictionary_path)
    return dictionary
