import _csv
import os
import sklearn
import pandas as pd
import numpy as np
from standardize_label import standardize_labels


class DataProcess:

    def __init__(self, processed_directory, data_annotation_method, data_source,
                 source, dataset_name,
                 additional_features, additional_columns,
                 index_original_sentence,
                 lexicon_corpus, split_original_manual,
                 charmapper='ar2safebw'):
        """
            Initialize as follows:
            - processed_directory = directory where you are planning to put processed data frames
            - data_annotation_method = [manual_translation, mechanical_turk_annotators, user_level, document_level, word_level, manual_search=googling manually, informal_message= messages in messengers, hashtag_level]
            - data_source = [travel_domain, twitter, news_comments, web_mixed, speech_transcript, ldc_mixed, sms]
            - source = url_link to the dataset description
            - dataset_name = the name of the dataset, which also corresponds to the folder inside data_processed where it is saved
            - additional_features = dict({key - name of column : val - string value of that column})
            - additional_columns = dict({key - name of column : val - index of that column})
            - index_original_sentence = index of column where the dialectal sentence is saved
            - lexicon_corpus = write lexicon if lexicon, else write corpus
            - split_original_manual = if using original split of the authors write original, else write manual
            - charmapper = mapping used to transliterate Arabic script, default is Habash-Soudi-Buckwalter
        """
        self.processed_directory = processed_directory
        self.dataset_name = dataset_name
        self.data_annotation_method = data_annotation_method
        self.data_source = data_source
        self.source = source
        self.additional_features = additional_features
        self.additional_columns = additional_columns
        self.index_original_sentence = index_original_sentence
        self.lexicon_corpus = lexicon_corpus
        self.split_original_manual = split_original_manual
        self.charmapper = charmapper

    def preclean(self, filename, delimiter, header, excel, encoding):
        """
            Removes all empty rows and makes everything string
        """
        if excel:
            df = pd.read_excel(filename, header=header,
                               error_bad_lines=False, encoding=encoding)
        else:
            df = pd.read_csv(filename, delimiter=delimiter,
                             header=header, error_bad_lines=False, encoding=encoding)
        if type(self.index_original_sentence) == list:
            for i in self.index_original_sentence:
                df = df[df.iloc[:,
                                i].notna()]
                df.iloc[:,
                        i] = df.iloc[:,
                                     i].astype(str)
        else:
            df = df[df.iloc[:,
                            self.index_original_sentence].notna()]

            df.iloc[:,
                    self.index_original_sentence] = df.iloc[:,
                                                            self.index_original_sentence].astype(str)
        if excel:
            df.to_excel(filename, index=False,
                        header=header, encoding=encoding)
        else:
            df.to_csv(filename, sep=delimiter,
                      index=False, header=header, encoding=encoding)

    def preprocess(self, raw_filename,
                   dialect_city_id, dialect_province_id,
                   dialect_country_id, dialect_region_id,
                   delimiter='\t', header=0, excel=False, encoding='utf-8'):
        """
        This function will return preprocessed pd.DataFrame()
            raw_filename = csv or tsv
            dialect_city = if dialect_city of type int then assume it is index of columns, otherwise provide string of dialect for all columns
            dialect_province = if dialect_province of type int then assume it is index of columns, otherwise provide string of dialect for all columns
            dialect_country = if dialect_country of type int then assume it is index of columns, otherwise provide string of dialect for all columns
            dialect_region = if dialect_region of type int then assume it is index of columns, otherwise provide string of dialect for all columns
            delimiter = specify which delimenter is used to separate columns
            header = if first row is not header but data pass None, default assume first row is name row
        """
        # Step 0: Preclean
        self.preclean(raw_filename, delimiter, header, excel, encoding)
        # Step 1: Read raw data from file
        if excel:
            df = pd.read_excel(raw_filename, header=header,
                               error_bad_lines=False, encoding=encoding)
        else:
            df = pd.read_csv(raw_filename, delimiter=delimiter,
                             header=header, error_bad_lines=False, encoding=encoding)
        df_preprocessed = pd.DataFrame()
        # make everything string here for arabic_multi_dialect
        if type(self.index_original_sentence) == list:
            counter = 0
            for i in self.index_original_sentence:
                if counter == 0:
                    df_preprocessed['original_sentence'] = df.iloc[:, i]
                else:
                    df_preprocessed['original_sentence'] = df_preprocessed['original_sentence'].append(
                        df.iloc[:, i], ignore_index=True).unique()
        else:
            df_preprocessed['original_sentence'] = df.iloc[:,
                                                           self.index_original_sentence]
        if (type(dialect_city_id) == int):
            df_preprocessed['dialect_city_id'] = df.iloc[:, dialect_city_id]
        else:
            df_preprocessed['dialect_city_id'] = dialect_city_id
        if (type(dialect_province_id) == int):
            df_preprocessed['dialect_province_id'] = df.iloc[:,
                                                             dialect_province_id]
        else:
            df_preprocessed['dialect_province_id'] = dialect_province_id
        if (type(dialect_country_id) == int):
            df_preprocessed['dialect_country_id'] = df.iloc[:,
                                                            dialect_country_id]
        else:
            df_preprocessed['dialect_country_id'] = dialect_country_id
        if (type(dialect_region_id) == int):
            df_preprocessed['dialect_region_id'] = df.iloc[:,
                                                           dialect_region_id]
        else:
            df_preprocessed['dialect_region_id'] = dialect_region_id

        # Step 2: Put any additional columns
        for k in self.additional_columns.keys():
            df_preprocessed[k] = df.iloc[:, self.additional_columns[k]]

        return df_preprocessed

    def save_file(self, processed_filename, df, encoding='utf-8'):
        """
            Given processed_filename and pandas DataFrame it saves it into that file
        """
        df.to_csv(self.processed_directory + processed_filename,
                  sep='\t', encoding=encoding, index=False)

    def save_features(self, features_file):
        """
            features_file = file that stores features of each dataset
        """

        df_feature = pd.DataFrame()

        # Step 2: Get Column features that need to be sorted
        files = os.listdir(self.processed_directory)
        df = pd.DataFrame()
        for f in files:
            df = df.append(pd.read_csv(self.processed_directory+f,
                                       delimiter='\t'), ignore_index=True)
        # Step 1: Get common features
        df_feature['dataset_name'] = [self.dataset_name]
        df_feature['data_annotation_method'] = [
            str(df['data_annotation_method'].unique().tolist())]
        df_feature['data_source'] = [str(df['data_source'].unique().tolist())]
        df_feature['source'] = [self.source]
        df_feature['lexicon_corpus'] = [self.lexicon_corpus]
        df_feature['split_original_manual'] = [self.split_original_manual]
        # Works for array structure but need to resolve in the future TO DO
        for k in self.additional_features.keys():
            df_feature[k] = str(self.additional_features[k])
        # 2.1: Get all dialects into list
        list_dialect_city_id = str(df['dialect_city_id'].unique().tolist())
        list_dialect_province_id = str(
            df['dialect_province_id'].unique().tolist())
        list_dialect_country_id = str(
            df['dialect_country_id'].unique().tolist())
        list_dialect_region_id = str(df['dialect_region_id'].unique().tolist())

        df_feature['list_dialect_city_id'] = [list_dialect_city_id]
        df_feature['list_dialect_province_id'] = [list_dialect_province_id]
        df_feature['list_dialect_country_id'] = [list_dialect_country_id]
        df_feature['list_dialect_region_id'] = [list_dialect_region_id]

        # 2.2: Get Additional Column data
        for k in self.additional_columns.keys():
            df_feature[k] = [str(df[k].unique().tolist())]
        if os.stat(features_file).st_size != 0:
            df_features = pd.read_csv(
                features_file, delimiter='\t')
        else:
            df_features = pd.DataFrame()
        df_feature.to_csv(self.processed_directory +
                          'info_data.txt', sep='\t', index=False)
        df_features = df_features.append(df_feature, ignore_index=True)
        df_features.to_csv(features_file, sep='\t', index=False)

    def split(self, filename, train_proportion, test_proportion, dev_proportion, seed=999):
        """ 
            filename : string = file to split the data
            train_proportion : float =  0.0 < n < 1.0 
            test_proportion : float =  0.0 < n < 1.0 
            dev_proportion : float =  0.0 < n < 1.0 

            return : List<pd.DataFrame> = train, dev, test 
        """
        df = pd.read_csv(self.processed_directory +
                         filename, delimiter='\t', header=0)
        np.random.seed(seed)
        perm = np.random.permutation(df.index)
        m = len(df.index)
        train_end = int(train_proportion * m)
        dev_end = int(dev_proportion * m) + train_end
        train = df.iloc[perm[:train_end]]
        dev = df.iloc[perm[train_end:dev_end]]
        test = df.iloc[perm[dev_end:]]
        return train, dev, test

    def standardize_labels(self, filename, filename_out, levels):
        standardize_labels(filename, filename_out, levels)
