# Aggregating Hierarchical Dialectal Data for Arabic Dialect  Classification

This repo contains code for the experiments presented in our paper: [Aggregating Hierarchical Dialectal Data for Arabic Dialect  Classification]().

## Requirements

This code was written for python>=3.7, pytorch 1.5.1, and transformers 3.1.0. You will also need few additional packages. Here's how you can set up the environment using conda (assuming you have conda and cuda installed):

```bash
git clone https://github.com/CAMeL-Lab/HierarchicalDID.git
cd HierarchicalDID

conda create -n HierarchicalDID python=3.7
conda activate HierarchicalDID

pip install -r requirements.txt


```

## HierarchicalDID LMs

Our Arabic Dialectal Language Models can be found [here](https://drive.google.com/drive/folders/1-_uZnl8LamZO9RPYguJJOywJTvJtWUyg?usp=sharing)


## Classification

In order to run classification experiments using aggregragated LMs and Salameh's model run following commands:

```bash
cd classification
python run_salameh.py
```

## Data Preprocessing

In order to extract certain data sets you can use the notebooks inside `data_preprocess` sub-folder and `data_process.py` module for data extraction in standard format and  `standardize_label.py` to standardize the hierarchical labels


## Citation:

If you find any of the this work useful, please cite [our paper]():
```bibtex
@inproceedings{baimukan-etal-2022-interplay,
    title = "Aggregating Hierarchical Dialectal Data for Arabic Dialect  Classification",
    author = " Baimukan, Nurpeiis  and
      Habash, Nizar and 
      Bouamor, Houda",
    booktitle = "Proceedings of the Sixth Arabic Natural Language Processing Workshop",
    month = june,
    year = "2022",
    address = "Marseille, France",
    publisher = "The European Language Resources Association",
    abstract = "Arabic is a collection of dialectal variants that are historically related but significantly different. These differences can be seen across regions, countries, and even cities in the same countries.  Previous work on Arabic Dialect Processing tasks has focused mainly on improving their performance at a particular dialect level (i.e. region, country, and/or city) using dialect-specific resources. In this paper, we present the first effort aiming at defining a standard unified hierarchical schema for dialectal Arabic labeling. This schema could be used to facilitate the use of these datasets in a joint manner. We explore fine-grained Classification of Arabic Dialects using the aggregated dialectal data from different levels and different sources.",
}
```