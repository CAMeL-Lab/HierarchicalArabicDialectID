# -*- coding: utf-8 -*-

# MIT License
#
# Copyright 2018-2019 New York University Abu Dhabi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


'''
'''


import collections
import os.path

import kenlm
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import normalize
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.utils.dediac import dediac_ar
from utils import df2dialectsentence, levels_eval, file2dialectsentence
import time
import json

ADIDA_LABELS = frozenset(['ALE', 'ALG', 'ALX', 'AMM', 'ASW', 'BAG', 'BAS',
                          'BEI', 'BEN', 'CAI', 'DAM', 'DOH', 'FES', 'JED',
                          'JER', 'KHA', 'MOS', 'MSA', 'MUS', 'RAB', 'RIY',
                          'SAL', 'SAN', 'SFX', 'TRI', 'TUN'])

ADIDA_LABELS_EXTRA = frozenset(['BEI', 'CAI', 'DOH', 'MSA', 'RAB', 'TUN'])
_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
_CHAR_LM_DIR = os.path.join(_DATA_DIR, 'lm', 'char')
_WORD_LM_DIR = os.path.join(_DATA_DIR, 'lm', 'word')

_TRAIN_DATA_AGGREGATED_PATH = os.path.join(
    _DATA_DIR, 'MADAR-Corpus-26-train.lines')

_TRAIN_DATA_PATH = os.path.join(_DATA_DIR, 'MADAR-Corpus-26-train.lines')
_TRAIN_DATA_EXTRA_PATH = os.path.join(_DATA_DIR, 'MADAR-Corpus-6-train.lines')
_VAL_DATA_PATH = os.path.join(_DATA_DIR, 'MADAR-Corpus-26-dev.lines')
_TEST_DATA_PATH = os.path.join(_DATA_DIR, 'MADAR-Corpus-26-test.lines')


class DIDPred(collections.namedtuple('DIDPred', ['top', 'scores'])):
    """A named tuple containing dialect ID prediction results.
    Attributes:
        top (:obj:`str`): The dialect label with the highest score.
        scores (:obj:`dict`): A dictionary mapping each dialect label to it's
            computed score.
    """


class DialectIdError(Exception):
    """Base class for all CAMeL Dialect ID errors.
    """

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return str(self.msg)


class UntrainedModelError(DialectIdError):
    """Error thrown when attempting to use an untrained DialectIdentifier
    instance.
    """

    def __init__(self, msg):
        DialectIdError.__init__(self, msg)


class InvalidDataSetError(DialectIdError, ValueError):
    """Error thrown when an invalid data set name is given to eval.
    """

    def __init__(self, dataset):
        msg = ('Invalid data set name {}. Valid names are "TEST" and '
               '"VALIDATION"'.format(repr(dataset)))
        DialectIdError.__init__(self, msg)


def _normalize_lm_scores(scores):
    norm_scores = np.exp(scores)
    norm_scores = normalize(norm_scores)
    return norm_scores


def _word_to_char(txt):
    return ' '.join(list(txt.replace(' ', 'X')))


def _max_score(score_tups):
    max_score = -1
    max_dialect = None

    for dialect, score in score_tups:
        if score > max_score:
            max_score = score
            max_dialect = dialect

    return max_dialect


class DialectIdentifier(object):
    """A class for training, evaluating and running the dialect identification
    model described by Salameh et al. After initializing an instance, you must
    run the train method once before using it.
    Args:
        labels (set of str, optional): The set of dialect labels used in the
            training data in the main model.
            Defaults to ADIDA_LABELS.
        labels_extra (set of str, optional): The set of dialect labels used in
            the training data in the extra features model.
            Defaults to ADIDA_LABELS_EXTRA.
        char_lm_dir (str, optional): Path to the directory containing the
            character-based language models. If None, use the language models
            that come with this package.
            Defaults to None.
        word_lm_dir (str, optional): Path to the directory containing the
            word-based language models. If None, use the language models that
            come with this package.
            Defaults to None.
    """

    def __init__(self, labels=ADIDA_LABELS,
                 labels_extra=ADIDA_LABELS_EXTRA,
                 char_lm_dir=None,
                 word_lm_dir=None,
                 aggregated_layers=None,
                 result_file_name=None,
                 repeat_sentence_train=0,
                 repeat_sentence_eval=0,
                 extra_lm=False,
                 extra=True):
        self.exp_time = time.strftime('%Y%m%d-%H%M%S')

        if char_lm_dir is None:
            char_lm_dir = _CHAR_LM_DIR
        if word_lm_dir is None:
            word_lm_dir = _WORD_LM_DIR
        if result_file_name is None:
            result_file_name = f'{self.exp_time}.json'

        # aggregated layer
        self.aggregated_layers = aggregated_layers
        self.result_file_name = result_file_name

        # repeating sentence as input, i.e. 'A A B' -> 'A A B A A B' if repeat == 1
        self.repeat_sentence_train = repeat_sentence_train
        self.repeat_sentence_eval = repeat_sentence_eval

        # salameh
        self._labels = labels
        self._labels_extra = labels_extra
        self._labels_sorted = sorted(labels)
        self._labels_extra_sorted = sorted(labels_extra)

        self._char_lms = collections.defaultdict(kenlm.Model)
        self._word_lms = collections.defaultdict(kenlm.Model)
        self._load_lms(char_lm_dir, word_lm_dir)

        self.extra_lm = extra_lm
        self.extra = extra
        self._char_lms_extra = collections.defaultdict(kenlm.Model)
        self._word_lms_extra = collections.defaultdict(kenlm.Model)
        if extra:
            self._load_lms_extra(char_lm_dir, word_lm_dir)

        self._is_trained = False

    def _load_lms(self, char_lm_dir, word_lm_dir):
        config = kenlm.Config()
        config.show_progress = False

        for label in self._labels:
            char_lm_path = os.path.join(char_lm_dir, '{}.arpa'.format(label))
            word_lm_path = os.path.join(word_lm_dir, '{}.arpa'.format(label))
            self._char_lms[label] = kenlm.Model(char_lm_path, config)
            self._word_lms[label] = kenlm.Model(word_lm_path, config)

    def _load_lms_extra(self, char_lm_dir, word_lm_dir):
        config = kenlm.Config()
        config.show_progress = False

        for label in self._labels_extra:
            char_lm_path = os.path.join(char_lm_dir, '{}.arpa'.format(label))
            word_lm_path = os.path.join(word_lm_dir, '{}.arpa'.format(label))
            self._char_lms_extra[label] = kenlm.Model(char_lm_path, config)
            self._word_lms_extra[label] = kenlm.Model(word_lm_path, config)

    def _get_char_lm_scores(self, txt):
        chars = _word_to_char(txt)
        return np.array([self._char_lms[label].score(chars, bos=True, eos=True)
                         for label in self._labels_sorted])

    def _get_word_lm_scores(self, txt):
        return np.array([self._word_lms[label].score(txt, bos=True, eos=True)
                         for label in self._labels_sorted])

    def _get_lm_feats(self, txt):
        word_lm_scores = self._get_word_lm_scores(txt).reshape(1, -1)
        word_lm_scores = _normalize_lm_scores(word_lm_scores)
        #print('Word LM shape', word_lm_scores, word_lm_scores.shape)
        char_lm_scores = self._get_char_lm_scores(txt).reshape(1, -1)
        char_lm_scores = _normalize_lm_scores(char_lm_scores)
        #print('Char LM shape', char_lm_scores, char_lm_scores.shape)
        feats = np.concatenate((word_lm_scores, char_lm_scores), axis=1)
        return feats

    def _get_lm_feats_multi(self, sentences):
        feats_list = collections.deque()
        for sentence in sentences:
            feats_list.append(self._get_lm_feats(sentence))
        feats_matrix = np.array(feats_list)
        feats_matrix = feats_matrix.reshape((-1, 2*len(self._labels)))
        return feats_matrix

    def _get_char_lm_scores_extra(self, txt):
        chars = _word_to_char(txt)
        return np.array([self._char_lms_extra[label].score(chars, bos=True, eos=True)
                         for label in self._labels_extra_sorted])

    def _get_word_lm_scores_extra(self, txt):
        return np.array([self._word_lms_extra[label].score(txt, bos=True, eos=True)
                         for label in self._labels_extra_sorted])

    def _get_lm_feats_extra(self, txt):
        word_lm_scores_extra = self._get_word_lm_scores_extra(
            txt).reshape(1, -1)
        word_lm_scores_extra = _normalize_lm_scores(word_lm_scores_extra)
        char_lm_scores_extra = self._get_char_lm_scores_extra(
            txt).reshape(1, -1)
        char_lm_scores_extra = _normalize_lm_scores(char_lm_scores_extra)
        feats = np.concatenate(
            (word_lm_scores_extra, char_lm_scores_extra), axis=1)
        return feats

    def _get_lm_feats_multi_extra(self, sentences):
        feats_list = collections.deque()
        for sentence in sentences:
            feats_list.append(self._get_lm_feats_extra(sentence))
        feats_matrix = np.array(feats_list)
        feats_matrix = feats_matrix.reshape((-1, 2*len(self._labels_extra)))
        return feats_matrix

    def _prepare_sentences(self, sentences):

        # why use tokenization here, where in train we didn't use anything
        tokenized = [' '.join(simple_word_tokenize(dediac_ar(s)))
                     for s in sentences]
        sent_array = np.array(tokenized)
        x_trans = self._feat_union.transform(sent_array)
        if self.extra:
            x_trans_extra = self._feat_union_extra.transform(sent_array)
            x_final_extra = x_trans_extra
        # TODO:  Explore bug where just adding extra ngram layer improves accuracy significcantly
        #x_predict_extra = x_trans_extra
            if self.extra_lm:
                x_lm_feats = self._get_lm_feats_multi_extra(sentences)
                x_final_extra = sp.sparse.hstack(
                    (x_trans_extra, x_lm_feats))
            x_predict_extra = self._classifier_extra.predict_proba(
                x_final_extra)

        aggregated_prob_distrs = []
        aggregated_lm_feats = []
        # aggregated features
        if self.aggregated_layers:
            for i in range(len(self.aggregated_layers)):
                prob_distr, lm_feat = self.aggregated_layers[i].predict_proba_lm_feats(
                    sentences)
                aggregated_prob_distrs.append(prob_distr)
                aggregated_lm_feats.append(lm_feat)

        x_lm_feats = self._get_lm_feats_multi(sentences)
        x_final = sp.sparse.hstack(
            (x_trans, x_lm_feats))
        if self.extra:
            x_final = sp.sparse.hstack(
                (x_final, x_predict_extra))

        if self.aggregated_layers:
            for i in range(len(self.aggregated_layers)):
                if self.aggregated_layers[i].use_distr:
                    x_final = sp.sparse.hstack(
                        (x_final, aggregated_prob_distrs[i]))
                if self.aggregated_layers[i].use_lm:
                    x_final = sp.sparse.hstack(
                        (x_final, aggregated_lm_feats[i]))
        return x_final

    def train(self, data_path=None,
              data_extra_path=None,
              level=None,
              char_ngram_range=(1, 3),
              word_ngram_range=(1, 1),
              n_jobs=None):
        """Trains the model on a given data set.
        Args:
            data_path (str, optional): Path to main training data. If None, use
                the provided training data.
                Defaults to None.
            data_extra_path (str, optional): Path to extra features training
                data. If None,cuse the provided training data.
                Defaults to None.
            char_ngram_range (tuple, optional): The n-gram ranges to consider
                in the character-based language models.
                Defaults to (1, 3).
            word_ngram_range (tuple, optional): The n-gram ranges to consider
                in the word-based language models.
                Defaults to (1, 1).
            n_jobs (int, optional): The number of parallel jobs to use for
                computation. If None, then only 1 job is used. If -1 then all
                processors are used.
                Defaults to None.
        """

        if data_path is None:
            data_path = [_TRAIN_DATA_PATH]
        if data_extra_path is None and self.extra:
            data_extra_path = [_TRAIN_DATA_EXTRA_PATH]
        if level is None:
            level = 'city'
        print(data_path)
        y, x = file2dialectsentence(
            data_path, level, self.repeat_sentence_train, True)
        if self.extra:
            y_extra, x_extra = file2dialectsentence(
                data_extra_path, level, self.repeat_sentence_train, False, True)

        # Build and train extra classifier
        if self.extra:
            print('Build and train extra classifier')

            self._label_encoder_extra = LabelEncoder()
            self._label_encoder_extra.fit(y_extra)
            y_trans = self._label_encoder_extra.transform(y_extra)

            word_vectorizer = TfidfVectorizer(lowercase=False,
                                              ngram_range=word_ngram_range,
                                              analyzer='word',
                                              tokenizer=lambda x: x.split(' '))
            char_vectorizer = TfidfVectorizer(lowercase=False,
                                              ngram_range=char_ngram_range,
                                              analyzer='char',
                                              tokenizer=lambda x: x.split(' '))
            self._feat_union_extra = FeatureUnion([('wordgrams', word_vectorizer),
                                                   ('chargrams', char_vectorizer)])
            x_trans = self._feat_union_extra.fit_transform(x_extra)
            x_final = x_trans
            if self.extra_lm:
                x_lm_feats = self._get_lm_feats_multi_extra(x_extra)
                x_final = sp.sparse.hstack(
                    (x_trans, x_lm_feats))

            self._classifier_extra = OneVsRestClassifier(MultinomialNB(),
                                                         n_jobs=n_jobs)
            self._classifier_extra.fit(x_final, y_trans)

        # Build and train aggreggated classifier
        print('Build and train aggreggated classifier')
        if self.aggregated_layers:
            for i in range(len(self.aggregated_layers)):
                self.aggregated_layers[i].train(self.repeat_sentence_train)

        # Build and train main classifier
        print('Build and train main classifier')
        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(y)
        y_trans = self._label_encoder.transform(y)

        word_vectorizer = TfidfVectorizer(lowercase=False,
                                          ngram_range=word_ngram_range,
                                          analyzer='word',
                                          tokenizer=lambda x: x.split(' '))
        char_vectorizer = TfidfVectorizer(lowercase=False,
                                          ngram_range=char_ngram_range,
                                          analyzer='char',
                                          tokenizer=lambda x: x.split(' '))
        self._feat_union = FeatureUnion([('wordgrams', word_vectorizer),
                                         ('chargrams', char_vectorizer)])
        self._feat_union.fit(x)

        x_prepared = self._prepare_sentences(x)

        self._classifier = OneVsRestClassifier(MultinomialNB(), n_jobs=n_jobs)
        self._classifier.fit(x_prepared, y_trans)

        self._is_trained = True

    def eval(self, data_path=None, data_set='VALIDATION', level=None, save_labels='labels.csv'):
        """Evaluate the trained model on a given data set.
        Args:
            data_path (str, optional): Path to an evaluation data set.
                If None, use one of the provided data sets instead.
                Defaults to None.
            data_set (str, optional): Name of the provided data set to use.
                This is ignored if data_path is not None. Can be either
                'VALIDATION' or 'TEST'. Defaults to 'VALIDATION'.
        Returns:
            dict: A dictionary mapping an evaluation metric to its computed
            value. The metrics used are accuracy, f1_micro, f1_macro,
            recall_micro, recall_macro, precision_micro and precision_macro.
        """

        if not self._is_trained:
            raise UntrainedModelError(
                'Can\'t evaluate an untrained model.')

        if data_path is None:
            if data_set == 'VALIDATION':
                data_path = [_VAL_DATA_PATH]
            elif data_set == 'TEST':
                data_path = [_TEST_DATA_PATH]
            else:
                raise InvalidDataSetError(data_set)
        if level is None:
            level = 'city'
        y_true, x = file2dialectsentence(
            data_path, level, self.repeat_sentence_eval, True)

        # Generate predictions
        x_prepared = self._prepare_sentences(x)
        y_pred = self._classifier.predict(x_prepared)
        # print(self._classifier.predict_proba(x_prepared))
        y_pred = self._label_encoder.inverse_transform(y_pred)
        df = pd.DataFrame(columns=['gold', 'pred'])
        df['gold'] = y_true
        df['pred'] = y_pred
        df.to_csv(save_labels, sep='\t', header=True, index=False)
        # Get scores
        levels_scores = levels_eval(y_true, y_pred, level)

        return levels_scores

    def record_experiment(self, test_results, val_results):
        final_record = {}
        final_record['test_results'] = test_results
        final_record['val_results'] = val_results
        final_record['exp_time'] = self.exp_time
        final_record['repeat_sentence_train'] = self.repeat_sentence_train
        final_record['repeat_sentence_eval'] = self.repeat_sentence_eval
        if self.aggregated_layers:
            for i in range(len(self.aggregated_layers)):
                final_record[f'layer_{i}'] = self.aggregated_layers[i].dict_repr

        data = []
        if os.path.exists(self.result_file_name):
            # 1. Read file contents
            with open(self.result_file_name, "r") as file:
                data = json.load(file)

        data.append(final_record)

        with open(self.result_file_name, "w") as file:
            json.dump(data, file)

        print('Recorded experiment in:', self.result_file_name)

    def predict(self, sentences):
        """Predict the dialect probability scores for a given list of
        sentences.
        Args:
            sentences (list of str): The list of sentences.
        Returns:
            list of DIDPred: A list of prediction results, each corresponding
                to its respective sentence.
        """

        if not self._is_trained:
            raise UntrainedModelError(
                'Can\'t predict with an untrained model.')

        x_prepared = self._prepare_sentences(sentences)
        predicted_scores = self._classifier.predict_proba(x_prepared)
        result = collections.deque()
        for sentence, scores in zip(sentences, predicted_scores):
            score_tups = list(zip(self._labels_sorted, scores))
            predicted_dialect = _max_score(score_tups)
            dialect_scores = dict(score_tups)
            result.append(DIDPred(predicted_dialect, dialect_scores))

        return list(result)


if __name__ == '__main__':
    d = DialectIdentifier(
        result_file_name='love.json')
    d.train()
    scores = d.eval(data_set='TEST')
    dev = d.eval()
