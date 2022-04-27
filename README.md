# Aggregating Hierarchical Dialectal Data for Arabic Dialect  Classification

This repo contains code for the experiments presented in the paper: [Aggregating Hierarchical Dialectal Data for Arabic Dialect Classification (Baimukan et al., 2022)]().

## Requirements

This code was written for python>=3.7. You will also need few additional packages. Here's how you can set up the environment using conda (assuming you have conda and cuda installed):

```bash
git clone https://github.com/CAMeL-Lab/HierarchicalArabicDialectID.git
cd HierarchicalArabicDialectID

conda create -n HierarchicalArabicDialectID python=3.7
conda activate HierarchicalArabicDialectID

pip install -r requirements.txt


```

## HierarchicalArabicDialectID LMs

Our Arabic Dialectal Language Models can be found [here](https://drive.google.com/drive/folders/1-_uZnl8LamZO9RPYguJJOywJTvJtWUyg?usp=sharing)


## Classification

The code we use for classificatioin extends on [CAMeL Tools](https://github.com/CAMeL-Lab/camel_tools) [(Obeid et al. (2020)](https://aclanthology.org/2020.lrec-1.868v2.pdf), which rebuilt the model by [Salameh et al. (2018)](https://aclanthology.org/C18-1113/).
In order to run the classification experiments using the aggregragated LMs in this repo and Baimukan et al. (2022)'s model, use the following commands:

```bash
cd classification
python run_classifier.py
```

## Data Preprocessing

In order to extract certain data sets you can use the notebooks inside `data_preprocess` sub-folder and `data_process.py` module for data extraction in standard format and  `standardize_label.py` to standardize the hierarchical labels


## Citation:

If you find any of the this work useful, please cite [our paper]():
```bibtex
@inproceedings{baimukan-etal-2022-aggregating,
    title = "Aggregating Hierarchical Dialectal Data for Arabic Dialect  Classification",
    author = " Baimukan, Nurpeiis  and
      Habash, Nizar and 
      Bouamor, Houda",
    booktitle = "Proceedings of the Language Resources and Evaluation Conference (LREC)",
    month = june,
    year = "2022",
    address = "Marseille, France",
    publisher = "The European Language Resources Association",
    abstract = "Arabic is a collection of dialectal variants that are historically related but significantly different. These differences can be seen across regions, countries, and even cities in the same countries.  Previous work on Arabic Dialect Processing tasks has focused mainly on improving their performance at a particular dialect level (i.e. region, country, and/or city) using dialect-specific resources. In this paper, we present the first effort aiming at defining a standard unified hierarchical schema for dialectal Arabic labeling. This schema could be used to facilitate the use of these datasets in a joint manner. We explore fine-grained Classification of Arabic Dialects using the aggregated dialectal data from different levels and different sources.",
}
```
