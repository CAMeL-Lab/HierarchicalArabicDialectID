from salameh import DialectIdentifier
from salameh import ADIDA_LABELS
from utils import LayerObject
import os


def run_experiment(aggregated_layers=None, repeat_train=0, repeat_eval=0,
                   file_name='results.json', labels_test_save='labels_test.csv', labels_dev_save='labels_dev.csv',
                   char_lm_dir=None, word_lm_dir=None, extra=True,
                   data_train_path=None, data_test_path='TEST', data_dev_path='VALIDATION', labels=ADIDA_LABELS):
    print('Running Experiment')
    d = DialectIdentifier(result_file_name=file_name,
                          aggregated_layers=aggregated_layers,
                          repeat_sentence_eval=repeat_eval,
                          repeat_sentence_train=repeat_train,
                          char_lm_dir=char_lm_dir,
                          word_lm_dir=word_lm_dir,
                          extra=extra, labels=labels)
    d.train(data_path=data_train_path)
    test_scores = d.eval(data_set=data_test_path, save_labels=labels_test_save)
    val_scores = d.eval(data_set=data_dev_path, save_labels=labels_dev_save)
    d.record_experiment(test_scores, val_scores)
    print(test_scores, val_scores)


def get_kenlm_train(level):
    return f'../aggregated_data/{level}_train.tsv'


def get_train_path(level):
    return ['aggregated_city/MADAR-Corpus-26-train.lines', f'../aggregated_data/{level}_train.tsv']


def get_cols_train(level):
    cols = ['dialect_city_id', 'dialect_country_id', 'dialect_region_id']
    if level == 'city':
        return cols
    elif level == 'country':
        return cols[1:]
    elif level == 'region':
        return cols[2:]


def get_single_layer_list(level, kenlm_train, exclude_list, use_lm, use_distr):
    layers = []
    for exclude in exclude_list[level]:
        for train_path in get_train_path(level):
            for lm in use_lm:
                for distr in use_distr:
                    if not lm and not distr:
                        continue
                    dict_repr = {}
                    dict_repr['level'] = level
                    dict_repr['kenlm_train'] = kenlm_train
                    dict_repr['kenlm_train_files'] = get_kenlm_train(
                        level)
                    dict_repr['exclude_list'] = exclude
                    dict_repr['train_path'] = train_path
                    dict_repr['use_lm'] = lm
                    dict_repr['use_distr'] = distr
                    dict_repr['cols_train'] = get_cols_train(
                        level)
                    layers.append(dict_repr)

    return layers


def subsets_util(levels, i, n, result, subset, j):
    # checking if all elements of the array are traverse or not
    if(i == n):
        # print the subset array
        idx = 0
        a = []
        while(idx < j):
            a.append(subset[idx])
            idx += 1

        result.append(a)
        return

    # for each index i, we have 2 options
    # case 1: i is not included in the subset
    # in this case simply increment i and move ahead
    subsets_util(levels, i+1, n, result, subset, j)
    # case 2: i is included in the subset
    # insert arr[i] at the end of subset
    # increment i and j
    subset[j] = i
    subsets_util(levels, i+1, n, result, subset, j+1)


def get_combo(levels):
    subset = [0]*2**len(levels)
    result = []

    subsets_util(levels, 0, len(levels), result, subset, 0)
    result = result[1:]  # exclude empty array
    return result


def get_layers_combinations(combos, single_layers):
    layers_combo = []
    single_layer = []
    for combo in combos:
        single_layer = []
        for i in range(len(combo)):
            if i == 0 and len(combo) == 1:
                for layer in single_layers[combo[i]]:
                    single_layer.append([layer])
            # else:
        if len(single_layer):
            layers_combo.append(single_layer)

    return layers_combo


def run_experiments(layers_combo, file_name='results.json'):
    # just Salameh
    for repeat_train in range(3):
        for repeat_eval in range(1):
            run_experiment(repeat_train=repeat_train,
                           repeat_eval=repeat_eval, file_name=file_name)
    # all different combos
    for combo in layers_combo:
        for layers in combo:
            for repeat_train in range(2):
                for repeat_eval in range(1):
                    aggregated_layers = []
                    for layer in layers:
                        l = LayerObject(layer)
                        aggregated_layers.append(l)
                    run_experiment(
                        aggregated_layers, repeat_train=repeat_train, repeat_eval=repeat_eval, file_name=file_name)


def layers_combo_experiment(layers_combo):

    region = layers_combo[0][2][0]
    country = layers_combo[1][2][0]
    city = layers_combo[2][2][0]

    #city + country
    city_layer = LayerObject(city)
    country_layer = LayerObject(country)
    run_experiment(aggregated_layers=[city_layer, country_layer], repeat_train=0,
                   repeat_eval=0, file_name='layers_combo.json', labels_test_save="city_country_combo_test.tsv",  labels_dev_save="city_country_combo_dev.csv")

    #city + region
    city_layer = LayerObject(city)
    region_layer = LayerObject(region)
    run_experiment(aggregated_layers=[city_layer, region_layer], repeat_train=0,
                   repeat_eval=0, file_name='layers_combo.json', labels_test_save="city_region_combo_test.tsv",  labels_dev_save="city_region_combo_dev.csv")

    #region + country
    region_layer = LayerObject(region)
    country_layer = LayerObject(country)
    run_experiment(aggregated_layers=[region_layer, country_layer], repeat_train=0,
                   repeat_eval=0, file_name='layers_combo.json', labels_test_save="region_country_combo_test.tsv",  labels_dev_save="region_country_combo_dev.csv")

    #city + country + region
    city_layer = LayerObject(city)
    country_layer = LayerObject(country)
    region_layer = LayerObject(region)
    run_experiment(aggregated_layers=[city_layer, country_layer, region_layer], repeat_train=0,
                   repeat_eval=0, file_name='layers_combo.json', labels_test_save="city_country_region_combo_test.tsv",  labels_dev_save="city_country_region_combo_dev.csv")


def run_aggregated_experiment(level):
    print(f'Aggregated experiment on {level} level')
    labels = [i[:-5]
              for i in os.listdir(f'aggregated_{level}/lm/char') if 'arpa' in i]
    run_experiment(aggregated_layers=None, repeat_train=0,
                   repeat_eval=0, file_name='aggregated_result.json',
                   labels_test_save=f'labels_agg_{level}_test.csv', labels_dev_save=f'labels_agg_{level}_dev.csv',
                   char_lm_dir=f'aggregated_{level}/lm/char', word_lm_dir=f'aggregated_{level}/lm/word',
                   extra=False, data_train_path=[f'../aggregated_data/{level}_train.tsv'],
                   data_test_path=[f'../aggregated_data/{level}_test.tsv'], data_dev_path=[f'../aggregated_data/{level}_dev.tsv'], labels=labels)


if __name__ == '__main__':
    """
    levels = ['city', 'country', 'region']
    kenlm_train = False
    exclude_list = {'city': [[], ['msa-msa-msa']],
                    'country': [[], ['msa-msa']],
                    'region': [[], ['msa']]}
    use_lm = [True, False]
    use_distr = [True, False]
    single_layers = []
    for level in levels:
        layers = get_single_layer_list(
            level, kenlm_train, exclude_list, use_lm, use_distr)
        single_layers.append(layers)

    combos = get_combo(levels)
    print(combos)
    layers_combo = get_layers_combinations(combos, single_layers)
    file_name = 'results_salameh_plus.json'
    run_experiments(layers_combo, file_name)
    """

    # run_experiment(aggregated_layers=None, repeat_train=0,
    #               repeat_eval=0, file_name='results_aaa.json')
    levels = ['city', 'country', 'region']
    kenlm_train = False
    exclude_list = {'city': [[], ['msa-msa-msa']],
                    'country': [[], ['msa-msa']],
                    'region': [[], ['msa']]}
    use_lm = [True, False]
    use_distr = [True, False]
    single_layers = []
    for level in levels:
        layers = get_single_layer_list(
            level, kenlm_train, exclude_list, use_lm, use_distr)
        single_layers.append(layers)

    combos = get_combo(levels)
    # print(combos)
    layers_combo = get_layers_combinations(combos, single_layers)
    # Aggregated layer experiment
    run_aggregated_experiment('city')
    run_aggregated_experiment('country')
    run_aggregated_experiment('region')
    """
    # Layers combo experiment
    layers_combo_experiment(layers_combo)
    
    #Usual expriments
    run_experiment(aggregated_layers=None, repeat_train=0,
                   repeat_eval=0, file_name='results_sal_agg.json',
                   labels_test_save="labels_test_salameh_agg.csv",  labels_dev_save="labels_dev_salameh_agg.csv",
                   data_train_path=['data/MADAR-Corpus-26-train.lines', 'data/MADAR-Corpus-26-dev.lines', 'data/MADAR-Corpus-26-test.lines'])

    run_experiment(aggregated_layers=None, repeat_train=0,
                   repeat_eval=0, file_name='results_sal_agg.json',
                   labels_test_save="camel.tsv",  labels_dev_save="labels_dev_salameh_agg.csv")

    l = LayerObject(layers_combo[0][2][0])
    l = LayerObject(layers_combo[0][2][0])
    run_experiment(aggregated_layers=[l], repeat_train=0,
                   repeat_eval=0, file_name='results_sal+best.json', labels_test_save="camel_region.tsv",  labels_dev_save="labels_dev_salameh.csv")
    """