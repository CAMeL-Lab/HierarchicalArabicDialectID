import os
import kenlm
import timeit
import numpy as np
import pandas as pd
import collections
import scipy as sp
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.metrics import precision_score


class LayerObject:

    def __init__(self, dict_repr):

        self.dict_repr = dict_repr
        self.level = dict_repr['level']
        if 'data_dir' not in self.dict_repr:
            self.data_dir = os.path.join(
                os.path.dirname(__file__), f'aggregated_{self.level}')
        else:
            self.data_dir = self.dict_repr['data_dir']
        self.kenlm_train = dict_repr['kenlm_train']
        self.kenlm_train_files = dict_repr['kenlm_train_files']
        self.exclude_list = dict_repr['exclude_list']

        # kenlm was not trained yet or we would like to train from scratch
        if not os.path.exists(self.data_dir) or self.kenlm_train:
            print('In KenLM Process')
            self.kenlm_process()
            print('Done KenLM')

        self.char_lm_dir = os.path.join(self.data_dir, 'lm', 'char')
        self.word_lm_dir = os.path.join(self.data_dir, 'lm', 'word')
        self.labels = []
        self.get_labels()
        self.dict_repr['labels'] = self.labels
        self.char_lms = collections.defaultdict(kenlm.Model)
        self.word_lms = collections.defaultdict(kenlm.Model)
        self.load_lms()

        self.train_path = dict_repr['train_path']

        self.use_lm = dict_repr['use_lm']
        self.use_distr = dict_repr['use_distr']
        self.cols_train = dict_repr['cols_train']

    def get_labels(self):
        self.labels = sorted([i[:-5]
                              for i in os.listdir(self.char_lm_dir) if i[-4:] == 'arpa' and i[:-5] not in self.exclude_list])

    def kenlm_process(self):
        start = timeit.default_timer()
        print('Creating list of dialects and sentences')
        dialect_list, sentence_list = file2dialectsentence(
            self.kenlm_train_files, f'{self.level}')
        print('Creating dialect dictionary')
        dialect_dict = split_by_dialect(dialect_list, sentence_list)
        create_directory(self.data_dir)
        print('Putting each dialect into file')
        dialect_dict2file(dialect_dict, self.data_dir)
        print('Creating KenLM for each dialect')
        dialect_dict_to_lm(dialect_dict, self.data_dir)
        end = timeit.default_timer()
        print('Finished creating KenLM models in ', end - start)

    def load_lms(self):
        config = kenlm.Config()
        config.show_progress = False

        for label in self.labels:
            char_lm_path = os.path.join(
                self.char_lm_dir, '{}.arpa'.format(label))
            word_lm_path = os.path.join(
                self.word_lm_dir, '{}.arpa'.format(label))
            self.char_lms[label] = kenlm.Model(
                char_lm_path, config)
            self.word_lms[label] = kenlm.Model(
                word_lm_path, config)

    def _get_char_lm_scores(self, txt):
        chars = word_to_char(txt)
        return np.array([self.char_lms[label].score(chars, bos=True, eos=True)
                         for label in self.labels])

    def _get_word_lm_scores(self, txt):
        return np.array([self.word_lms[label].score(txt, bos=True, eos=True)
                         for label in self.labels])

    def _get_lm_feats(self, txt):
        word_lm_scores = self._get_word_lm_scores(
            txt).reshape(1, -1)
        word_lm_scores = normalize_lm_scores(word_lm_scores)
        char_lm_scores = self._get_char_lm_scores(
            txt).reshape(1, -1)
        char_lm_scores = normalize_lm_scores(char_lm_scores)
        feats = np.concatenate((word_lm_scores, char_lm_scores), axis=1)
        return feats

    def get_lm_feats_multi(self, sentences):
        feats_list = collections.deque()
        for sentence in sentences:
            feats_list.append(self._get_lm_feats(sentence))
        feats_matrix = np.array(feats_list)
        # print(feats_matrix.shape)
        feats_matrix = feats_matrix.reshape((-1, len(self.labels)*2))
        # print(feats_matrix.shape)
        return feats_matrix

    def train(self, repeat=0):
        n_jobs = None
        char_ngram_range = (1, 3)
        word_ngram_range = (1, 1)

        y, sentences = file2dialectsentence(
            [self.train_path], self.level, repeat=repeat)

        # Build and train aggregated classifier
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(y)
        y_trans = self.label_encoder.transform(y)

        word_vectorizer = TfidfVectorizer(lowercase=False,
                                          ngram_range=word_ngram_range,
                                          analyzer='word',
                                          tokenizer=lambda x: x.split(' '))
        char_vectorizer = TfidfVectorizer(lowercase=False,
                                          ngram_range=char_ngram_range,
                                          analyzer='char',
                                          tokenizer=lambda x: x.split(' '))
        self.feat_union = FeatureUnion([('wordgrams', word_vectorizer),
                                        ('chargrams', char_vectorizer)])
        x_trans = self.feat_union.fit_transform(sentences)
        x_lm_feats = self.get_lm_feats_multi(sentences)
        x_final = sp.sparse.hstack(
            (x_trans, x_lm_feats))
        self.classifier = OneVsRestClassifier(MultinomialNB(),
                                              n_jobs=n_jobs)
        self.classifier.fit(x_final, y_trans)

    def predict_proba_lm_feats(self, sentences):
        x_trans = self.feat_union.transform(np.array(sentences))
        x_lm_feats = self.get_lm_feats_multi(sentences)
        x_final = sp.sparse.hstack(
            (x_trans, x_lm_feats))
        prob_distr = self.classifier.predict_proba(x_final)

        return prob_distr, x_lm_feats


def normalize_lm_scores(scores):
    norm_scores = np.exp(scores)
    norm_scores = normalize(norm_scores)
    return norm_scores


def split_by_dialect(dialect_list, sentence_list):
    char_list = []
    for s in sentence_list:
        s = word_to_char(s)
        char_list.append(s)
    dialect_dict = dict()
    for i in range(len(char_list)):
        if (dialect_list[i] in dialect_dict) == False:
            dialect_dict[dialect_list[i]] = dict()
            dialect_dict[dialect_list[i]]['word'] = []
            dialect_dict[dialect_list[i]]['char'] = []

        dialect_dict[dialect_list[i]]['word'].append(sentence_list[i])
        dialect_dict[dialect_list[i]]['char'].append(char_list[i])

    return dialect_dict


def create_directory(dir_name):
    if not os.path.exists(f'{dir_name}'):
        os.mkdir(f'{dir_name}')


def dialect_dict2file(dialect_dict, folder):
    create_directory(f'{folder}/word')
    create_directory(f'{folder}/char')

    for k in dialect_dict.keys():
        with open(f'{folder}/word/{k}.txt', 'w') as f:
            for item in dialect_dict[k]['word']:
                f.write(f'{item}\n')
        with open(f'{folder}/char/{k}.txt', 'w') as f:
            for item in dialect_dict[k]['char']:
                f.write(f'{item}\n')


def create_lm(in_file, out_file):
    location_hpc = '/scratch/nb2577/usr/lib/kenlm/build/bin/lmplz'
    location_pc = '~/kenlm/build/bin/lmplz'
    command = f'{location_pc} -o 5 < {in_file} > {out_file} --discount_fallback'
    os.system(command)


def dialect_dict_to_lm(dialect_dict, folder):
    create_directory(f'{folder}/lm')
    create_directory(f'{folder}/lm/word')
    create_directory(f'{folder}/lm/char')

    for k in dialect_dict.keys():
        create_lm(f'{folder}/word/{k}.txt', f'{folder}/lm/word/{k}.arpa')
        create_lm(f'{folder}/char/{k}.txt', f'{folder}/lm/char/{k}.arpa')


def word_to_char(txt):
    return ' '.join(list(txt.replace(' ', 'X')))


def file2dialectsentence(files, level, repeat=0):
    df = pd.read_csv(files[0], sep='\t', header=0)
    for i in range(1, len(files)):
        df = df.append(pd.read_csv(files[i], sep='\t', header=0))

    return df2dialectsentence(df, level, repeat)


def df2dialectsentence(df, level, repeat=0):
    """
        df: pd.DataFrame with sentences
        level: string representation of the level, whether it be 'city', 'country', or 'region'
        return y, x
    """
    sentence_list = df['original_sentence'].tolist()
    if repeat > 0:
        sentence_list = [' '.join([i]*(repeat+1)) for i in sentence_list]

    dialect_list = df2dialect(df, level)
    return dialect_list, sentence_list


def df2dialect(df, level):

    cols = []
    if level == 'city':
        cols = ['dialect_city_id', 'dialect_country_id', 'dialect_region_id']
    elif level == 'country':
        cols = ['dialect_country_id', 'dialect_region_id']
    elif level == 'region':
        cols = ['dialect_region_id']
    df['combined'] = df[cols].apply(
        lambda row: '-'.join(row.values.astype(str)), axis=1)

    dialect_list = df['combined'].tolist()

    return dialect_list


def single_level_eval(y_true, y_pred):

    scores = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_micro': f1_score(y_true, y_pred, average='micro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'recall_micro': recall_score(y_true, y_pred, average='micro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'precision_micro': precision_score(y_true, y_pred,
                                           average='micro'),
        'precision_macro': precision_score(y_true, y_pred, average='macro')
    }

    return scores


def levels_eval(y_true, y_pred, level):
    levels_scores = {}
    if level == 'city':
        levels_scores['city'] = single_level_eval(y_true, y_pred)
        y_true = ['-'.join(i.split('-')[1:]) for i in y_true]
        y_pred = ['-'.join(i.split('-')[1:]) for i in y_pred]
        level = 'country'

    if level == 'country':
        levels_scores['country'] = single_level_eval(y_true, y_pred)
        y_true = [i.split('-')[1] for i in y_true]
        y_pred = [i.split('-')[1] for i in y_pred]
        level = 'region'

    if level == 'region':
        levels_scores['region'] = single_level_eval(y_true, y_pred)

    return levels_scores


def whole_process(level, train_files):
    start = timeit.default_timer()
    print('Creating list of dialects and sentences')
    dialect_list, sentence_list = file2dialectsentence(train_files, f'{level}')
    print('Creating dialect dictionary')
    dialect_dict = split_by_dialect(dialect_list, sentence_list)
    folder = f'aggregated_{level}'
    create_directory(folder)
    print('Putting each dialect into file')
    dialect_dict2file(dialect_dict, folder)
    print('Creating KenLM for each dialect')
    dialect_dict_to_lm(dialect_dict, folder)
    end = timeit.default_timer()
    print('Finished in ', end - start)


if __name__ == '__main__':
    level = 'country'
    train_files = [f'../aggregated_data/{level}_train.tsv']
    whole_process(level, train_files)
    """
    layer = LayerObject(level, False, train_files,
                        [], 'aggregate_city/MADAR-Corpus-26-train.lines', None, None)
    """