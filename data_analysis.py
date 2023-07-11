#!/usr/bin/env python3

# Written by Christopher Hempel (hempelc@uoguelph.ca) on 05 May 2022

# This script analyzes correlations between abundance or p-a data and environmental factors.

# Partially following https://towardsdatascience.com/machine-learning-with-python-classification-complete-tutorial-d2c99dc524ec

import os
import copy
import datetime
import logging
import pickle
import collections
import random
import warnings
import featurewiz
import pandas as pd  # v1.3.5
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib import pyplot
from skbio.stats.composition import multiplicative_replacement, clr  # v0.5.6
from sklearn import feature_selection, svm
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV

# Activate logging for debugging
# logging.basicConfig(level=logging.DEBUG,
#     format = '%(asctime)s - %(levelname)s - %(message)s')

# Define that warnings are not printed to console
warnings.filterwarnings("ignore")


# Parameters set manually
## Full path to directory that contains abundance/p-a data generated on the basis of multiple taxonomic ranks
workdir = (
    "/Users/christopherhempel/Google Drive/PhD UoG/ExStream project/data_files_for_ml/"
)
## Full path to out directory
outdir = workdir
## File that contains sample infos, i.e., classes
sample_info_file = "/Users/christopherhempel/Google Drive/PhD UoG/ExStream project/data_files_for_ml/sample_info.csv"
## Ranks to include into analysis (looped over) ["phylum", "class", "order", "family", "genus", "species"]
ranks = ["phylum", "class", "order", "family", "genus", "species"]
## Data types to include into analysis (looped over) ["abundance", "pa"]
data_types = ["abundance", "pa"]
## Seq types to include into analysis (looped over) ["metagenomics", "totalrnaseq", "16s-esv", "its-esv", "16s-otu", "its-otu", "16s-its-otu", "16s-its-esv"]
seq_types = [
    "metagenomics",
    "totalrnaseq",
    "16s-esv",
    "its-esv",
    "16s-otu",
    "its-otu",
    "16s-its-otu",
    "16s-its-esv",
]
## Select models you want to train
## ("xbg" for XGBoosting, "lsvc" for linear SVC, "rf" for random forest,
## "knn" for KNN, "svc" for SVC, "lor-ridge" for logistic regression with ridge,
## "lor-lasso" for logistic regression with lasso, "mlp" for multilayer perceptron)
# models = ["xgb", "lsvc", "knn", "svc", "rf", "lor-ridge", 'lor-lasso', 'mlp']
models = ["xgb", "lsvc", "knn", "svc", "rf", "lor-ridge", "lor-lasso"]
# Set random state for models and pond selection during train test split
random_state = 1
## Set dependent variable ("pestidice_treatment", "sediment_addition", "pesticide_and_sediment")
dependent_variable = "pesticide_and_sediment"
## Apply feature selection? (True/False)
f_select = False
## If feature selection, select based on which selection method?
## ("dt" for decision tree, "mi" for mutual_information,
## "rf" for random forest, "lasso" for lasso, "fw" for featurewiz, "rfe" for recursive feature elemination)
selection_method = "rfe"
## Apply dimensionality reduction via PCA? (True/False)
dim_red = False
## If dimensionality reduction, choose % of variance that should be covered by PCs
pca_perc = 0.5
## Show plots generated during data exploration and feature selection? (True/False)
plots = False
## Calculate number of combos
combo_num = len(ranks) * len(data_types) * len(seq_types) * len(models)
## How many repetitions
reps = 3


# Define functions
## Print datetime and text
def time_print(text):
    datetime_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{datetime_now}  ---  {text}")


# Learning curve plotting, taken from Dan Tulpan's course script
def plot_learning_curve2(
    estimator,
    title,
    X,
    y,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    pyplot.figure()

    pyplot.title(title)
    if ylim is not None:
        pyplot.ylim(*ylim)
        # pyplot.ylim([0,100])
    pyplot.xlabel("Training examples")
    pyplot.ylabel("MCC")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
        scoring=make_scorer(matthews_corrcoef),
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    pyplot.grid()
    pyplot.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    pyplot.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    pyplot.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    pyplot.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    pyplot.legend(loc="best")

    return pyplot


time_print("Script started.")


# This dict will contain all info
master_score_dict = {}
master_taxa_list_dic = {}

## Make output dir for learning curves
lcdir = os.path.join(outdir, "learning_curves")
if not os.path.exists(lcdir):
    os.makedirs(lcdir)

# Loop over ranks
counter = 1
for rank in ranks:
    print("--------------- Rank: {0} ----------------".format(rank))
    # Read in data
    abundances = pd.read_csv(os.path.join(workdir, "abundances_" + rank + ".csv"))
    abundances["taxon"] = abundances["taxon"].fillna("NA")
    abundances = abundances.set_index("taxon")
    abundances.index.name = None
    abundances = abundances.transpose()

    df_16s_esv = pd.read_parquet(
        os.path.join(
            workdir, "Cotton_16S_apscale_ESV_table_filtered_tax_clean.parquet.snappy"
        )
    )
    df_its_esv = pd.read_parquet(
        os.path.join(
            workdir, "Cotton_ITS_apscale_ESV_table_filtered_tax_clean.parquet.snappy"
        )
    )
    df_16s_otu = pd.read_parquet(
        os.path.join(
            workdir, "Cotton_16S_apscale_OTU_table_filtered_tax_clean.parquet.snappy"
        )
    )
    df_its_otu = pd.read_parquet(
        os.path.join(
            workdir, "Cotton_ITS_apscale_OTU_table_filtered_tax_clean.parquet.snappy"
        )
    )

    ## Filter taxa names
    df_16s_esv = (
        df_16s_esv.replace("^.__", "", regex=True)
        .replace("unidentified", "NA")
        .fillna("NA")
    )
    df_its_esv = (
        df_its_esv.replace("^.__", "", regex=True)
        .replace("unidentified", "NA")
        .fillna("NA")
    )
    df_16s_otu = (
        df_16s_otu.replace("^.__", "", regex=True)
        .replace("unidentified", "NA")
        .fillna("NA")
    )
    df_its_otu = (
        df_its_otu.replace("^.__", "", regex=True)
        .replace("unidentified", "NA")
        .fillna("NA")
    )

    ## Make proper species names
    df_16s_esv["Species"] = df_16s_esv["Genus"] + " " + df_16s_esv["Species"]
    df_its_esv["Species"] = df_its_esv["Genus"] + " " + df_its_esv["Species"]
    df_16s_otu["Species"] = df_16s_otu["Genus"] + " " + df_16s_otu["Species"]
    df_its_otu["Species"] = df_its_otu["Genus"] + " " + df_its_otu["Species"]

    df_16s_esv["Species"] = df_16s_esv["Species"].replace(".* NA", "NA", regex=True)
    df_its_esv["Species"] = df_its_esv["Species"].replace(".* NA", "NA", regex=True)
    df_16s_otu["Species"] = df_16s_otu["Species"].replace(".* NA", "NA", regex=True)
    df_its_otu["Species"] = df_its_otu["Species"].replace(".* NA", "NA", regex=True)

    ## Aggregate on rank
    df_16s_esv = (
        df_16s_esv.iloc[:, list(df_16s_esv.columns).index(rank.capitalize()) :]
        .groupby([rank.capitalize()])
        .sum()
    )
    df_its_esv = (
        df_its_esv.iloc[:, list(df_its_esv.columns).index(rank.capitalize()) :]
        .groupby([rank.capitalize()])
        .sum()
    )
    df_16s_otu = (
        df_16s_otu.iloc[:, list(df_16s_otu.columns).index(rank.capitalize()) :]
        .groupby([rank.capitalize()])
        .sum()
    )
    df_its_otu = (
        df_its_otu.iloc[:, list(df_its_otu.columns).index(rank.capitalize()) :]
        .groupby([rank.capitalize()])
        .sum()
    )

    ## Transform to relative
    df_16s_esv = df_16s_esv / df_16s_esv.sum()
    df_its_esv = df_its_esv / df_its_esv.sum()
    df_16s_otu = df_16s_otu / df_16s_otu.sum()
    df_its_otu = df_its_otu / df_its_otu.sum()

    ## Prepare for merge with sample info
    df_16s_esv.index.name = None
    df_16s_esv = (
        df_16s_esv.transpose().reset_index().rename(columns={"index": "sample"})
    )
    df_its_esv.index.name = None
    df_its_esv = (
        df_its_esv.transpose().reset_index().rename(columns={"index": "sample"})
    )
    df_16s_otu.index.name = None
    df_16s_otu = (
        df_16s_otu.transpose().reset_index().rename(columns={"index": "sample"})
    )
    df_its_otu.index.name = None
    df_its_otu = (
        df_its_otu.transpose().reset_index().rename(columns={"index": "sample"})
    )

    # Drop NAs:
    print(
        "Average NA abundance per metagenomics sample: {0}% (mean), {1}% (median)".format(
            round(
                100 * abundances[abundances.index.str.contains("DNA")]["NA"].mean(), 2
            ),
            round(
                100 * abundances[abundances.index.str.contains("DNA")]["NA"].median(), 2
            ),
        )
    )
    print(
        "Average NA abundance per total rna-seq sample: {0}% (mean), {1}% (median)".format(
            round(
                100 * abundances[abundances.index.str.contains("RNA")]["NA"].mean(), 2
            ),
            round(
                100 * abundances[abundances.index.str.contains("RNA")]["NA"].median(), 2
            ),
        )
    )
    print(
        "Average NA abundance per 16S ESV sample: {0}% (mean), {1}% (median)".format(
            round(100 * df_16s_esv["NA"].mean(), 2),
            round(100 * df_16s_esv["NA"].median(), 2),
        )
    )
    print(
        "Average NA abundance per ITS ESV sample: {0}% (mean), {1}% (median)".format(
            round(100 * df_its_esv["NA"].mean(), 2),
            round(100 * df_its_esv["NA"].median(), 2),
        )
    )
    print(
        "Average NA abundance per 16S OTU sample: {0}% (mean), {1}% (median)".format(
            round(100 * df_16s_otu["NA"].mean(), 2),
            round(100 * df_16s_otu["NA"].median(), 2),
        )
    )
    print(
        "Average NA abundance per ITS OTU sample: {0}% (mean), {1}% (median)".format(
            round(100 * df_its_otu["NA"].mean(), 2),
            round(100 * df_its_otu["NA"].median(), 2),
        )
    )
    abundances = abundances.drop(["NA"], axis=1)
    df_16s_esv = df_16s_esv.drop(["NA"], axis=1)
    df_its_esv = df_its_esv.drop(["NA"], axis=1)
    df_16s_otu = df_16s_otu.drop(["NA"], axis=1)
    df_its_otu = df_its_otu.drop(["NA"], axis=1)

    ## In the case that all taxa of a sample are NA, the sample has a sum of 0, so we
    ## have to drop all those samples
    abundances = abundances.loc[(abundances.sum(axis=1) != 0), :]
    df_16s_esv = df_16s_esv.loc[(df_16s_esv.sum(axis=1) != 0), :]
    df_its_esv = df_its_esv.loc[(df_its_esv.sum(axis=1) != 0), :]
    df_16s_otu = df_16s_otu.loc[(df_16s_otu.sum(axis=1) != 0), :]
    df_its_otu = df_its_otu.loc[(df_its_otu.sum(axis=1) != 0), :]

    ## Turn abundances back into relative so that their sum equals 1 after NAs have been dropped
    abundances = (
        (abundances.transpose() / abundances.transpose().sum())
        .transpose()
        .reset_index()
        .rename(columns={"index": "sample"})
    )
    df_16s_esv = (
        (
            df_16s_esv.set_index("sample").transpose()
            / df_16s_esv.set_index("sample").transpose().sum()
        )
        .transpose()
        .reset_index()
        .rename(columns={"index": "sample"})
    )
    df_its_esv = (
        (
            df_its_esv.set_index("sample").transpose()
            / df_its_esv.set_index("sample").transpose().sum()
        )
        .transpose()
        .reset_index()
        .rename(columns={"index": "sample"})
    )
    df_16s_otu = (
        (
            df_16s_otu.set_index("sample").transpose()
            / df_16s_otu.set_index("sample").transpose().sum()
        )
        .transpose()
        .reset_index()
        .rename(columns={"index": "sample"})
    )
    df_its_otu = (
        (
            df_its_otu.set_index("sample").transpose()
            / df_its_otu.set_index("sample").transpose().sum()
        )
        .transpose()
        .reset_index()
        .rename(columns={"index": "sample"})
    )

    # Merge sample info
    sample_info = pd.read_csv(sample_info_file)
    df_16s_esv = pd.merge(df_16s_esv, sample_info, how="inner").set_index("sample")
    df_its_esv = pd.merge(df_its_esv, sample_info, how="inner").set_index("sample")
    df_16s_otu = pd.merge(df_16s_otu, sample_info, how="inner").set_index("sample")
    df_its_otu = pd.merge(df_its_otu, sample_info, how="inner").set_index("sample")
    df_16s_esv.index.name = None
    df_its_esv.index.name = None
    df_16s_otu.index.name = None
    df_its_otu.index.name = None

    ## Edit sample column for metagenomcis total rnaseq samples
    sample_info["sample"] = sample_info["sample"].str.lstrip("0").str.upper()
    sample_info_dna = copy.deepcopy(sample_info)
    sample_info_dna["sample"] = sample_info_dna["sample"] + "_DNA"
    sample_info_rna = copy.deepcopy(sample_info)
    sample_info_rna["sample"] = sample_info_rna["sample"] + "_RNA"
    sample_info = pd.concat([sample_info_dna, sample_info_rna])
    df_dna_rna = pd.merge(abundances, sample_info, how="inner").set_index("sample")
    df_dna_rna.index.name = None

    ## Adapt sample names for metabarcoding samples
    df_16s_esv.index = df_16s_esv.index.str.replace(r"^0", "").str.upper()
    df_its_esv.index = df_its_esv.index.str.replace(r"^0", "").str.upper()
    df_16s_otu.index = df_16s_otu.index.str.replace(r"^0", "").str.upper()
    df_its_otu.index = df_its_otu.index.str.replace(r"^0", "").str.upper()
    df_16s_esv.index += "_16S-ESV"
    df_its_esv.index += "_ITS-ESV"
    df_16s_otu.index += "_16S-OTU"
    df_its_otu.index += "_ITS-OTU"

    # Some models don't work if feature names contain "[" and "]", so we have to remove the characters
    df_dna_rna.columns = [
        x.replace("[", "").replace("]", "") for x in df_dna_rna.columns
    ]
    df_16s_esv.columns = [
        x.replace("[", "").replace("]", "") for x in df_16s_esv.columns
    ]
    df_its_esv.columns = [
        x.replace("[", "").replace("]", "") for x in df_its_esv.columns
    ]
    df_16s_otu.columns = [
        x.replace("[", "").replace("]", "") for x in df_16s_otu.columns
    ]
    df_its_otu.columns = [
        x.replace("[", "").replace("]", "") for x in df_its_otu.columns
    ]

    # Add  column for sequencing type and combination of pesticide
    # treatment and sediment addition
    df_dna_rna["seq_type"] = df_dna_rna.index.str.contains("DNA")
    df_dna_rna["seq_type"] = df_dna_rna["seq_type"].map(
        {True: "metagenomics", False: "total_rnaseq"}
    )
    ## Change sediment yes to with (w) sediment and sediment no to without (wo) sediment
    df_dna_rna["sediment_addition"] = df_dna_rna["sediment_addition"].map(
        {"yes": "w_sediment", "no": "wo_sediment"}
    )
    df_16s_esv["sediment_addition"] = df_16s_esv["sediment_addition"].map(
        {"yes": "w_sediment", "no": "wo_sediment"}
    )
    df_its_esv["sediment_addition"] = df_its_esv["sediment_addition"].map(
        {"yes": "w_sediment", "no": "wo_sediment"}
    )
    df_16s_otu["sediment_addition"] = df_16s_otu["sediment_addition"].map(
        {"yes": "w_sediment", "no": "wo_sediment"}
    )
    df_its_otu["sediment_addition"] = df_its_otu["sediment_addition"].map(
        {"yes": "w_sediment", "no": "wo_sediment"}
    )
    ## Make new feature
    df_dna_rna["pesticide_and_sediment"] = (
        df_dna_rna["pestidice_treatment"] + "_" + df_dna_rna["sediment_addition"]
    )
    df_16s_esv["pesticide_and_sediment"] = (
        df_16s_esv["pestidice_treatment"] + "_" + df_16s_esv["sediment_addition"]
    )
    df_its_esv["pesticide_and_sediment"] = (
        df_its_esv["pestidice_treatment"] + "_" + df_its_esv["sediment_addition"]
    )
    df_16s_otu["pesticide_and_sediment"] = (
        df_16s_otu["pestidice_treatment"] + "_" + df_16s_otu["sediment_addition"]
    )
    df_its_otu["pesticide_and_sediment"] = (
        df_its_otu["pestidice_treatment"] + "_" + df_its_otu["sediment_addition"]
    )

    # Separation
    df_dna = df_dna_rna[df_dna_rna["seq_type"] == "metagenomics"]
    df_rna = df_dna_rna[df_dna_rna["seq_type"] == "total_rnaseq"]
    ## Cut dfs down to shared samples (some only worked in metagenomics or total rna-seq)
    ### Identify shared samples
    shared_samples = list(
        set(df_dna.index.str.replace("_DNA", ""))
        & set(df_rna.index.str.replace("_RNA", ""))
    )
    shared_samples_dna = [x + "_DNA" for x in shared_samples]
    shared_samples_rna = [x + "_RNA" for x in shared_samples]
    shared_samples_16s_esv = [x + "_16S-ESV" for x in shared_samples]
    shared_samples_its_esv = [x + "_ITS-ESV" for x in shared_samples]
    shared_samples_16s_otu = [x + "_16S-OTU" for x in shared_samples]
    shared_samples_its_otu = [x + "_ITS-OTU" for x in shared_samples]

    ### Subset
    df_dna = df_dna.loc[shared_samples_dna]
    df_rna = df_rna.loc[shared_samples_rna]
    df_dna_rna = df_dna_rna.loc[shared_samples_dna + shared_samples_rna]
    df_16s_esv = df_16s_esv.loc[shared_samples_16s_esv]
    df_its_esv = df_its_esv.loc[shared_samples_its_esv]
    df_16s_otu = df_16s_otu.loc[shared_samples_16s_otu]
    df_its_otu = df_its_otu.loc[shared_samples_its_otu]

    ### Split into taxa and vars
    df_taxa_dna_rna = df_dna_rna.drop(
        [
            "pestidice_treatment",
            "sediment_addition",
            "seq_type",
            "pesticide_and_sediment",
        ],
        axis=1,
    )
    df_taxa_dna = df_dna.drop(
        [
            "pestidice_treatment",
            "sediment_addition",
            "seq_type",
            "pesticide_and_sediment",
        ],
        axis=1,
    )
    df_taxa_rna = df_rna.drop(
        [
            "pestidice_treatment",
            "sediment_addition",
            "seq_type",
            "pesticide_and_sediment",
        ],
        axis=1,
    )
    df_taxa_16s_esv = df_16s_esv.drop(
        ["pestidice_treatment", "sediment_addition", "pesticide_and_sediment"], axis=1
    )
    df_taxa_its_esv = df_its_esv.drop(
        ["pestidice_treatment", "sediment_addition", "pesticide_and_sediment"], axis=1
    )
    df_taxa_16s_otu = df_16s_otu.drop(
        ["pestidice_treatment", "sediment_addition", "pesticide_and_sediment"], axis=1
    )
    df_taxa_its_otu = df_its_otu.drop(
        ["pestidice_treatment", "sediment_addition", "pesticide_and_sediment"], axis=1
    )

    df_vars_dna_rna = df_dna_rna[
        [
            "pestidice_treatment",
            "sediment_addition",
            "seq_type",
            "pesticide_and_sediment",
        ]
    ]
    df_vars_dna = df_dna[
        ["pestidice_treatment", "sediment_addition", "pesticide_and_sediment"]
    ]
    df_vars_rna = df_rna[
        ["pestidice_treatment", "sediment_addition", "pesticide_and_sediment"]
    ]
    df_vars_16s_esv = df_16s_esv[
        ["pestidice_treatment", "sediment_addition", "pesticide_and_sediment"]
    ]
    df_vars_its_esv = df_its_esv[
        ["pestidice_treatment", "sediment_addition", "pesticide_and_sediment"]
    ]
    df_vars_16s_otu = df_16s_otu[
        ["pestidice_treatment", "sediment_addition", "pesticide_and_sediment"]
    ]
    df_vars_its_otu = df_its_otu[
        ["pestidice_treatment", "sediment_addition", "pesticide_and_sediment"]
    ]

    ## Drop taxa in dfs with total abundance = 0
    df_taxa_dna = df_taxa_dna.loc[:, (df_taxa_dna.sum() != 0)]
    df_taxa_rna = df_taxa_rna.loc[:, (df_taxa_rna.sum() != 0)]
    df_taxa_16s_esv = df_taxa_16s_esv.loc[:, (df_taxa_16s_esv.sum() != 0)]
    df_taxa_its_esv = df_taxa_its_esv.loc[:, (df_taxa_its_esv.sum() != 0)]
    df_taxa_16s_otu = df_taxa_16s_otu.loc[:, (df_taxa_16s_otu.sum() != 0)]
    df_taxa_its_otu = df_taxa_its_otu.loc[:, (df_taxa_its_otu.sum() != 0)]

    # Data exploration
    dna_rna_taxa = list(df_taxa_dna_rna.columns)
    dna_taxa = list(df_taxa_dna.columns)
    rna_taxa = list(df_taxa_rna.columns)
    esv_16s_taxa = list(df_taxa_16s_esv.columns)
    esv_its_taxa = list(df_taxa_its_esv.columns)
    otu_16s_taxa = list(df_taxa_16s_otu.columns)
    otu_its_taxa = list(df_taxa_its_otu.columns)

    print("Number of taxa found in metagenomics: {0}".format(len(dna_taxa)))
    print("Number of taxa found in total rna-seq: {0}".format(len(rna_taxa)))
    print(
        "Number of taxa shared between metagenomics and total rna-seq: {0}".format(
            len([x for x in dna_taxa if x in rna_taxa])
        )
    )
    print(
        "Number of taxa unique to metagenomics: {0}".format(
            len([x for x in dna_taxa if x not in rna_taxa])
        )
    )
    print(
        "Number of taxa unique to total rna-seq: {0}".format(
            len([x for x in rna_taxa if x not in dna_taxa])
        )
    )
    print("Number of taxa found in 16s_esv: {0}".format(len(esv_16s_taxa)))
    print("Number of taxa found in its_esv: {0}".format(len(esv_its_taxa)))
    print("Number of taxa found in 16s_otu: {0}".format(len(otu_16s_taxa)))
    print("Number of taxa found in its_otu: {0}".format(len(otu_its_taxa)))

    ## Save in dict
    taxa_list_dic = {
        "dna_taxa": dna_taxa,
        "rna_taxa": rna_taxa,
        "esv_16s_taxa": esv_16s_taxa,
        "esv_its_taxa": esv_its_taxa,
        "otu_16s_taxa": otu_16s_taxa,
        "otu_its_taxa": otu_its_taxa,
    }

    ## Dependent variable distribution
    if plots == True:
        px.bar(
            df_vars_dna[dependent_variable].value_counts().sort_values(),
            orientation="h",
            title="class distribution",
        )

    # To later split the data into train and test data, some manual adjustments are required
    # due to the data structure. Therefore, I can't simply use the scikitlearn
    # train_test_split function later but rather have to manually select the test samples.
    # I know the test set will be 24 samples big for a 20% split, and I must avoid having samples
    # A and B from the same pond separately in the test and training set, respectively, because they are dependent.
    # So, I will randomly select 12 ponds with 2 available samples for DNA and RNA
    # samples and use them later for the test dataset. Some samples failed for DNA or RNA,
    # so I need to identify which ones worked in both and pick randomly from those.
    ## Remove suffixes
    sample_ponds = df_taxa_dna.index.str.replace("[AB]_DNA", "")
    ## Identify non-unique ponds (that worked for A and B)
    non_unique_ponds = [
        item for item, count in collections.Counter(sample_ponds).items() if count == 2
    ]

    for data_type in data_types:
        print("--------------- Data type: {0} ----------------".format(data_type))
        if data_type == "abundance":
            # Transform abundances by replacing 0s and taking the centered log ratio
            df_taxa_dna_rna = pd.DataFrame(
                clr(multiplicative_replacement(df_taxa_dna_rna)),
                index=df_taxa_dna_rna.index,
                columns=df_taxa_dna_rna.columns,
            )
            df_taxa_dna = pd.DataFrame(
                clr(multiplicative_replacement(df_taxa_dna)),
                index=df_taxa_dna.index,
                columns=df_taxa_dna.columns,
            )
            df_taxa_rna = pd.DataFrame(
                clr(multiplicative_replacement(df_taxa_rna)),
                index=df_taxa_rna.index,
                columns=df_taxa_rna.columns,
            )
            df_taxa_16s_esv = pd.DataFrame(
                clr(multiplicative_replacement(df_taxa_16s_esv)),
                index=df_taxa_16s_esv.index,
                columns=df_taxa_16s_esv.columns,
            )
            df_taxa_its_esv = pd.DataFrame(
                clr(multiplicative_replacement(df_taxa_its_esv)),
                index=df_taxa_its_esv.index,
                columns=df_taxa_its_esv.columns,
            )
            df_taxa_16s_otu = pd.DataFrame(
                clr(multiplicative_replacement(df_taxa_16s_otu)),
                index=df_taxa_16s_otu.index,
                columns=df_taxa_16s_otu.columns,
            )
            df_taxa_its_otu = pd.DataFrame(
                clr(multiplicative_replacement(df_taxa_its_otu)),
                index=df_taxa_its_otu.index,
                columns=df_taxa_its_otu.columns,
            )

            # Data exploration abundance
            ## Taxa abundance
            print("15 most abundant taxa in metagenomics:")
            print(df_taxa_dna.sum().sort_values(ascending=False).head(n=15))
            print("------------------------")
            print("15 most abundant taxa in total rna-seq:")
            print(df_taxa_rna.sum().sort_values(ascending=False).head(n=15))
            print("------------------------")
            print("15 most abundant taxa in 16s_esv:")
            print(df_taxa_16s_esv.sum().sort_values(ascending=False).head(n=15))
            print("------------------------")
            print("15 most abundant taxa in its_esv:")
            print(df_taxa_its_esv.sum().sort_values(ascending=False).head(n=15))
            print("------------------------")
            print("15 most abundant taxa in 16s_otu:")
            print(df_taxa_16s_otu.sum().sort_values(ascending=False).head(n=15))
            print("------------------------")
            print("15 most abundant taxa in its_otu:")
            print(df_taxa_its_otu.sum().sort_values(ascending=False).head(n=15))

            ## Taxa abundance distribution
            if plots == True:
                px.histogram(
                    df_taxa_dna.sum(), title="Taxa abundance distribution metagenomics"
                ).show()
                px.histogram(
                    df_taxa_rna.sum(), title="Taxa abundance distribution total rna-seq"
                ).show()
                px.histogram(
                    df_taxa_16s_esv.sum(), title="Taxa abundance distribution 16s_esv"
                ).show()
                px.histogram(
                    df_taxa_its_esv.sum(), title="Taxa abundance distribution its_esv"
                ).show()
                px.histogram(
                    df_taxa_16s_otu.sum(), title="Taxa abundance distribution 16s_otu"
                ).show()
                px.histogram(
                    df_taxa_its_otu.sum(), title="Taxa abundance distribution its_otu"
                ).show()

            # Standardize metagenomics and total RNA-Seq data
            df_taxa_dna_rna = pd.DataFrame(
                StandardScaler().fit_transform(df_taxa_dna_rna),
                index=df_taxa_dna_rna.index,
                columns=df_taxa_dna_rna.columns,
            )
            df_taxa_dna = pd.DataFrame(
                StandardScaler().fit_transform(df_taxa_dna),
                index=df_taxa_dna.index,
                columns=df_taxa_dna.columns,
            )
            df_taxa_rna = pd.DataFrame(
                StandardScaler().fit_transform(df_taxa_rna),
                index=df_taxa_rna.index,
                columns=df_taxa_rna.columns,
            )

        elif data_type == "pa":
            # Turn abundance data into presence/absence data (0=not found, 1=found():
            ## Replace all values above 0 with 1:
            df_taxa_dna_rna[df_taxa_dna_rna > 0] = 1
            df_taxa_dna[df_taxa_dna > 0] = 1
            df_taxa_rna[df_taxa_rna > 0] = 1
            df_taxa_16s_esv[df_taxa_16s_esv > 0] = 1
            df_taxa_its_esv[df_taxa_its_esv > 0] = 1
            df_taxa_16s_otu[df_taxa_16s_otu > 0] = 1
            df_taxa_its_otu[df_taxa_its_otu > 0] = 1

            # Data exploration pa
            ## Taxa pa
            print("15 most present taxa in metagenomics:")
            print(df_taxa_dna.sum().sort_values(ascending=False).head(n=15))
            print("------------------------")
            print("15 most present taxa in total rna-seq:")
            print(df_taxa_rna.sum().sort_values(ascending=False).head(n=15))
            print("------------------------")
            print("15 most present taxa in 16s_esv:")
            print(df_taxa_16s_esv.sum().sort_values(ascending=False).head(n=15))
            print("------------------------")
            print("15 most present taxa in its_esv:")
            print(df_taxa_its_esv.sum().sort_values(ascending=False).head(n=15))
            print("------------------------")
            print("15 most present taxa in 16s_otu:")
            print(df_taxa_16s_otu.sum().sort_values(ascending=False).head(n=15))
            print("------------------------")
            print("15 most present taxa in its_otu:")
            print(df_taxa_its_otu.sum().sort_values(ascending=False).head(n=15))
            ### Taxa presence distribution
            if plots == True:
                px.histogram(
                    df_taxa_dna.sum(), title="Taxa presence distribution metagenomics"
                ).show()
                px.histogram(
                    df_taxa_rna.sum(), title="Taxa presence distribution total rna-seq"
                ).show()
                px.histogram(
                    df_taxa_16s_esv.sum(), title="Taxa presence distribution 16s_esv"
                ).show()
                px.histogram(
                    df_taxa_its_esv.sum(), title="Taxa presence distribution its_esv"
                ).show()
                px.histogram(
                    df_taxa_16s_otu.sum(), title="Taxa presence distribution 16s_otu"
                ).show()
                px.histogram(
                    df_taxa_its_otu.sum(), title="Taxa presence distribution its_otu"
                ).show()

        ## If multi-marker approach tested, combine dataframes of markers
        if "16s-its-esv" in seq_types:
            df_taxa_16s_its_esv = pd.concat(
                [
                    df_taxa_16s_esv.reset_index().drop("index", axis=1),
                    df_taxa_its_esv.reset_index().drop("index", axis=1),
                ],
                axis=1,
            )
            df_taxa_16s_its_esv.index = df_taxa_16s_esv.index.str.replace(
                "16S", "16S-ITS"
            )
            df_vars_16s_its_esv = df_vars_16s_esv.reset_index().drop("index", axis=1)
            df_vars_16s_its_esv.index = df_vars_16s_esv.index.str.replace(
                "16S", "16S-ITS"
            )

        if "16s-its-otu" in seq_types:
            df_taxa_16s_its_otu = pd.concat(
                [
                    df_taxa_16s_otu.reset_index().drop("index", axis=1),
                    df_taxa_its_otu.reset_index().drop("index", axis=1),
                ],
                axis=1,
            )
            df_taxa_16s_its_otu.index = df_taxa_16s_otu.index.str.replace(
                "16S", "16S-ITS"
            )
            df_vars_16s_its_otu = df_vars_16s_otu.reset_index().drop("index", axis=1)
            df_vars_16s_its_otu.index = df_vars_16s_otu.index.str.replace(
                "16S", "16S-ITS"
            )

        master_taxa_list_dic[rank] = taxa_list_dic

        ## Standardize amplicon sequencing data if abundance (Note: done after multi-marker combination since StandardScaler has to be applied to all features collectively)
        if data_type == "abundance":
            df_taxa_16s_esv = pd.DataFrame(
                StandardScaler().fit_transform(df_taxa_16s_esv),
                index=df_taxa_16s_esv.index,
                columns=df_taxa_16s_esv.columns,
            )
            df_taxa_its_esv = pd.DataFrame(
                StandardScaler().fit_transform(df_taxa_its_esv),
                index=df_taxa_its_esv.index,
                columns=df_taxa_its_esv.columns,
            )
            df_taxa_16s_otu = pd.DataFrame(
                StandardScaler().fit_transform(df_taxa_16s_otu),
                index=df_taxa_16s_otu.index,
                columns=df_taxa_16s_otu.columns,
            )
            df_taxa_its_otu = pd.DataFrame(
                StandardScaler().fit_transform(df_taxa_its_otu),
                index=df_taxa_its_otu.index,
                columns=df_taxa_its_otu.columns,
            )
            if "16s-its-esv" in seq_types:
                df_taxa_16s_its_esv = pd.DataFrame(
                    StandardScaler().fit_transform(df_taxa_16s_its_esv),
                    index=df_taxa_16s_its_esv.index,
                    columns=df_taxa_16s_its_esv.columns,
                )
            if "16s-its-otu" in seq_types:
                df_taxa_16s_its_otu = pd.DataFrame(
                    StandardScaler().fit_transform(df_taxa_16s_its_otu),
                    index=df_taxa_16s_its_otu.index,
                    columns=df_taxa_16s_its_otu.columns,
                )

        for seq_type in seq_types:
            print(
                "--------------- Sequencing type: {0} ----------------".format(seq_type)
            )
            if seq_type == "metagenomics":
                df_taxa = df_taxa_dna
                df_vars = df_vars_dna
            elif seq_type == "totalrnaseq":
                df_taxa = df_taxa_rna
                df_vars = df_vars_rna
            elif seq_type == "16s-esv":
                df_taxa = df_taxa_16s_esv
                df_vars = df_vars_16s_esv
            elif seq_type == "its-esv":
                df_taxa = df_taxa_its_esv
                df_vars = df_vars_its_esv
            elif seq_type == "16s-otu":
                df_taxa = df_taxa_16s_otu
                df_vars = df_vars_16s_otu
            elif seq_type == "its-otu":
                df_taxa = df_taxa_its_otu
                df_vars = df_vars_its_otu
            elif seq_type == "16s-its-esv":
                df_taxa = df_taxa_16s_its_esv
                df_vars = df_vars_16s_its_esv
            elif seq_type == "16s-its-otu":
                df_taxa = df_taxa_16s_its_otu
                df_vars = df_vars_16s_its_otu

            # Define independent and dependent variables
            X = df_taxa
            y = df_vars[dependent_variable]
            feature_names = df_taxa.columns

            # Remove highly correlated features using the SULOV method from featurewiz,
            # unless we use featurewis for faeture selection (then it's already included)
            if f_select == False or (f_select == True and selection_method != "fw"):
                uncorrelated_features = (
                    featurewiz.FE_remove_variables_using_SULOV_method(
                        pd.concat([X, y], axis=1),
                        feature_names,
                        "Classification",
                        dependent_variable,
                        corr_limit=0.7,
                        verbose=0,
                        dask_xgboost_flag=False,
                    )
                )
                X = X[uncorrelated_features]

            if f_select == True:
                # Feature selection
                ## Naming for later
                f_select_naming = "w-fs"

                if selection_method == "lasso":
                    ## Via Lasso
                    selector = feature_selection.SelectFromModel(
                        estimator=LogisticRegression(
                            C=1,
                            penalty="l1",
                            solver="liblinear",
                            random_state=random_state,
                        ),
                        max_features=len(feature_names),
                    ).fit(X, y)
                    reduced_feature_names = list(feature_names[selector.get_support()])

                elif selection_method == "rf":
                    ## RF importance
                    rf_model_fs = RandomForestClassifier(random_state=random_state)
                    rf_model_fs.fit(X, y)
                    rf_importances = rf_model_fs.feature_importances_
                    ## Put in a pandas df
                    rf_importances_df = pd.DataFrame(
                        {"importance": rf_importances, "feature": feature_names}
                    ).sort_values("importance", ascending=False)
                    rf_importances_df["cumsum"] = rf_importances_df[
                        "importance"
                    ].cumsum(axis=0)
                    ### Plot
                    if plots == True:
                        px.line(
                            rf_importances_df,
                            x="feature",
                            y="cumsum",
                            title="Cumulative importance",
                        ).show()
                    reduced_feature_names = list(
                        rf_importances_df[rf_importances_df["importance"] != 0][
                            "feature"
                        ]
                    )

                elif selection_method == "mi":
                    # Mutual information
                    mutual_information = feature_selection.mutual_info_classif(X, y)
                    ## Put in a pandas dtf
                    mutual_information_ser = pd.Series(
                        mutual_information, index=feature_names
                    )
                    ## Keep anything with correlation (since most features actually have no correlation)
                    ## To Do: figure out if this is actually the way to go or if correlation cutoff needs to be made
                    reduced_feature_names = list(
                        mutual_information_ser[mutual_information_ser != 0].index
                    )

                elif selection_method == "dt":
                    ## DecisionTree importance
                    dt_model = DecisionTreeClassifier(random_state=random_state)
                    dt_model.fit(X, y)
                    dt_importances = dt_model.feature_importances_
                    ## Put in a pandas df
                    dt_importances_df = pd.DataFrame(
                        {"importance": dt_importances, "feature": feature_names}
                    ).sort_values("importance", ascending=False)
                    dt_importances_df["cumsum"] = dt_importances_df[
                        "importance"
                    ].cumsum(axis=0)
                    ### Plot
                    if plots == True:
                        px.line(
                            dt_importances_df,
                            x="feature",
                            y="cumsum",
                            title="Cumulative importance",
                        ).show()
                    reduced_feature_names = list(
                        dt_importances_df[dt_importances_df["importance"] != 0][
                            "feature"
                        ]
                    )

                elif selection_method == "fw":
                    # With automatization via featurewiz (removes correlated features based on manually defined correlation limit)
                    df_featurewiz = pd.concat([X, y], axis=1)
                    reduced_feature_names, reduced_df = featurewiz.featurewiz(
                        df_featurewiz, dependent_variable, corr_limit=0.7, verbose=1
                    )

                elif selection_method == "rfe":
                    # Recursive feature elimination following https://machinelearningmastery.com/rfe-feature-selection-in-python/
                    rfe = feature_selection.RFE(
                        estimator=DecisionTreeClassifier(), n_features_to_select=20
                    )
                    rfe.fit(X, y)
                    reduced_feature_names = X.columns[rfe.support_]

                ## Select features
                X = X[reduced_feature_names]
            else:
                ## Naming for later
                f_select_naming = "wo-fs"

            if dim_red == True:
                # Dimensionality reduction via PCA
                pca = PCA(n_components=20)
                pca_model = pca.fit(X).transform(X)
                pca_cumsum = pd.DataFrame(
                    {
                        "PC-wise explained variance": pca.explained_variance_ratio_,
                        "Cumulative explained variance": np.cumsum(
                            pca.explained_variance_ratio_
                        ),
                    }
                )
                if plots == True:
                    px.line(
                        pca_cumsum,
                        labels={"index": "PCs", "value": "Explained variance"},
                    ).show()
                # We select the n first PCs, where n=number of PCs that explain the % variance defined in "pca_perc"
                pc_num = (
                    len(
                        pca_cumsum[
                            pca_cumsum["Cumulative explained variance"] <= pca_perc
                        ]
                    )
                    + 1
                )
                X = pca_model[:, :pc_num]

            # Repeat tests "reps" times and take the average of the train and test
            # values (needed due to big discrepancy between test and train)

            xgb_best_mean_mcc = []
            lsvc_best_mean_mcc = []
            knn_best_mean_mcc = []
            svc_best_mean_mcc = []
            rf_best_mean_mcc = []
            lor_ridge_best_mean_mcc = []
            lor_lasso_best_mean_mcc = []
            mlp_best_mean_mcc = []

            xgb_test_score_mcc = []
            lsvc_test_score_mcc = []
            knn_test_score_mcc = []
            svc_test_score_mcc = []
            rf_test_score_mcc = []
            lor_ridge_test_score_mcc = []
            lor_lasso_test_score_mcc = []
            mlp_test_score_mcc = []

            for i in range(reps):
                ## Randomly pick 12 ponds, NOTE: key here is setting seed to i!
                ## By setting the seed to i, we make sure that no matter what
                ## combinatin of data type, seq data, etc. we process, the
                ## same test samples are used per repetition - otherwise results
                ## would not be comparable between combinations!
                random.seed(i)
                test_ponds = random.sample(non_unique_ponds, 12)
                print("Test ponds:", test_ponds)

                # Split into train and test using the randomly selected ponds = 80:20 train:test split
                X_test = X[
                    X.index.str.contains(
                        "|".join(["^" + x + "[AB]" for x in test_ponds])
                    )
                ]
                y_test = y[
                    y.index.str.contains(
                        "|".join(["^" + x + "[AB]" for x in test_ponds])
                    )
                ]
                X_train = X[
                    ~X.index.str.contains(
                        "|".join(["^" + x + "[AB]" for x in test_ponds])
                    )
                ]
                y_train = y[
                    ~y.index.str.contains(
                        "|".join(["^" + x + "[AB]" for x in test_ponds])
                    )
                ]

                # Run models
                for model in models:
                    combo_name = "_".join([rank, data_type, seq_type, model])
                    time_print(
                        "Processing combo {0}/{1}: {2} --- repetition {3}...".format(
                            str(counter), str(combo_num * reps), combo_name, i + 1
                        )
                    )

                    if model == "xgb":
                        # XGBoost #3 priority
                        ## Turn string classes into numerical classes
                        xgb_y_train = pd.Series(y_train).astype("category").cat.codes
                        xgb_y_test = pd.Series(y_test).astype("category").cat.codes
                        ## Define hyperparameters combinations to try
                        xgb_param_dic = {
                            "learning_rate": [0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01],
                            "n_estimators": [int(x) for x in np.linspace(50, 200, 5)],
                            "reg_lambda": list(np.linspace(1, 10, 5)),
                            "reg_alpha": list(np.linspace(0, 10, 5)),
                            "min_child_weight": list(np.linspace(1, 10, 5)),
                            "gamma": list(np.linspace(0, 20, 5)),
                            "max_depth": [3, 4, 5, 6],
                        }
                        ## Define model
                        xgb_model = xgb.XGBClassifier(
                            objective="multi:softmax", random_state=random_state
                        )
                        xgb_bayes_search = BayesSearchCV(
                            xgb_model,
                            xgb_param_dic,
                            scoring=make_scorer(matthews_corrcoef),
                            n_jobs=-1,
                            cv=10,
                        ).fit(X_train, xgb_y_train)
                        ## Save best mean MCC
                        xgb_best_mean_mcc.append(xgb_bayes_search.best_score_)
                        ## Train model with identified best hyperparameters and all train data
                        xgb_model.set_params(**xgb_bayes_search.best_params_)
                        xgb_model.fit(X_train, xgb_y_train)
                        # Predict test set and calculate MCC score
                        xgb_y_test_prediction = xgb_model.predict(X_test)
                        xgb_test_score_mcc.append(
                            matthews_corrcoef(xgb_y_test, xgb_y_test_prediction)
                        )
                        # Generate learning curve
                        lc = plot_learning_curve2(
                            xgb_model,
                            combo_name,
                            X_train,
                            xgb_y_train,
                            cv=10,
                            n_jobs=-1,
                        )
                        lc.tight_layout()
                        lc.savefig(
                            os.path.join(lcdir, combo_name)
                            + "_rep{0}.jpg".format(i + 1),
                            dpi=600,
                        )
                        lc.close()

                    if model == "rf":
                        ### RF #3 priority
                        rf_param_dic = {
                            "n_estimators": [
                                int(x) for x in np.linspace(50, 200, 5)
                            ],  # Number of trees in random forest
                            "criterion": ["gini", "entropy"],
                            "max_features": [
                                "auto",
                                "sqrt",
                            ],  # Number of features to consider at every split
                            "min_samples_leaf": [1, 2, 4],
                        }  # Minimum number of samples required at each leaf node
                        # Define model
                        rf_model = RandomForestClassifier(random_state=random_state)
                        rf_bayes_search = BayesSearchCV(
                            rf_model,
                            rf_param_dic,
                            scoring=make_scorer(matthews_corrcoef),
                            n_jobs=-1,
                            cv=10,
                        ).fit(X_train, y_train)
                        ## Save best mean MCC
                        rf_best_mean_mcc.append(rf_bayes_search.best_score_)
                        ## Train model with identified best hyperparameters and all train data
                        rf_model.set_params(**rf_bayes_search.best_params_)
                        rf_model.fit(X_train, y_train)
                        # Predict test set and calculate MCC score
                        y_test_prediction = rf_model.predict(X_test)
                        rf_test_score_mcc.append(
                            matthews_corrcoef(y_test, y_test_prediction)
                        )
                        # Generate learning curve
                        lc = plot_learning_curve2(
                            rf_model, combo_name, X_train, y_train, cv=10, n_jobs=-1
                        )
                        lc.tight_layout()
                        lc.savefig(
                            os.path.join(lcdir, combo_name)
                            + "_rep{0}.jpg".format(i + 1),
                            dpi=600,
                        )
                        lc.close()

                    if model == "lsvc":
                        ### Linear SVC #1 priority
                        lsvc_param_dic = {
                            "loss": ["hinge", "squared_hinge"],
                            "tol": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                            "C": np.linspace(0.5, 1.5, 10),
                            "intercept_scaling": np.linspace(0.5, 1.5, 10),
                        }
                        # Define model
                        lsvc_model = svm.LinearSVC(
                            random_state=random_state, penalty="l2"
                        )
                        lsvc_bayes_search = BayesSearchCV(
                            lsvc_model,
                            lsvc_param_dic,
                            scoring=make_scorer(matthews_corrcoef),
                            n_jobs=-1,
                            cv=10,
                        ).fit(X_train, y_train)
                        ## Save best mean MCC
                        lsvc_best_mean_mcc.append(lsvc_bayes_search.best_score_)
                        ## Train model with identified best hyperparameters and all train data
                        lsvc_model.set_params(**lsvc_bayes_search.best_params_)
                        lsvc_model.fit(X_train, y_train)
                        # Predict test set and calculate MCC score
                        y_test_prediction = lsvc_model.predict(X_test)
                        lsvc_test_score_mcc.append(
                            matthews_corrcoef(y_test, y_test_prediction)
                        )
                        # Generate learning curve
                        lc = plot_learning_curve2(
                            lsvc_model, combo_name, X_train, y_train, cv=10, n_jobs=-1
                        )
                        lc.tight_layout()
                        lc.savefig(
                            os.path.join(lcdir, combo_name)
                            + "_rep{0}.jpg".format(i + 1),
                            dpi=600,
                        )
                        lc.close()

                    if model == "svc":
                        ### SVC #3 priority
                        svc_param_dic = {
                            "C": np.linspace(
                                0.1, 10, 10
                            ),  # norm used in the penalization
                            "kernel": ["linear", "poly", "rbf", "sigmoid"],
                            "degree": [2, 3, 4],
                            "gamma": np.linspace(0.1, 10, 10),
                        }
                        # Define model
                        svc_model = svm.SVC(random_state=random_state)
                        svc_bayes_search = BayesSearchCV(
                            svc_model,
                            svc_param_dic,
                            scoring=make_scorer(matthews_corrcoef),
                            n_jobs=-1,
                            cv=10,
                        ).fit(X_train, y_train)
                        ## Save best mean MCC
                        svc_best_mean_mcc.append(svc_bayes_search.best_score_)
                        ## Train model with identified best hyperparameters and all train data
                        svc_model.set_params(**svc_bayes_search.best_params_)
                        svc_model.fit(X_train, y_train)
                        # Predict test set and calculate MCC score
                        y_test_prediction = svc_model.predict(X_test)
                        svc_test_score_mcc.append(
                            matthews_corrcoef(y_test, y_test_prediction)
                        )
                        # Generate learning curve
                        lc = plot_learning_curve2(
                            svc_model, combo_name, X_train, y_train, cv=10, n_jobs=-1
                        )
                        lc.tight_layout()
                        lc.savefig(
                            os.path.join(lcdir, combo_name)
                            + "_rep{0}.jpg".format(i + 1),
                            dpi=600,
                        )
                        lc.close()

                    if model == "lor-ridge":
                        ### Logistic regression with ridge #1 priority
                        lor_ridge_param_dic = {
                            "tol": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                            "C": np.linspace(0.5, 1.5, 10),
                            "intercept_scaling": np.linspace(0.5, 1.5, 10),
                            "solver": ["newton-cg", "lbfgs", "sag", "saga"],
                        }
                        # Define model
                        lor_ridge_model = LogisticRegression(
                            penalty="l2", random_state=random_state, max_iter=2147483647
                        )
                        lor_ridge_bayes_search = BayesSearchCV(
                            lor_ridge_model,
                            lor_ridge_param_dic,
                            scoring=make_scorer(matthews_corrcoef),
                            n_jobs=-1,
                            cv=10,
                        ).fit(X_train, y_train)
                        ## Save best mean MCC
                        lor_ridge_best_mean_mcc.append(
                            lor_ridge_bayes_search.best_score_
                        )
                        ## Train model with identified best hyperparameters and all train data
                        lor_ridge_model.set_params(
                            **lor_ridge_bayes_search.best_params_
                        )
                        lor_ridge_model.fit(X_train, y_train)
                        # Predict test set and calculate MCC score
                        y_test_prediction = lor_ridge_model.predict(X_test)
                        lor_ridge_test_score_mcc.append(
                            matthews_corrcoef(y_test, y_test_prediction)
                        )
                        # Generate learning curve
                        lc = plot_learning_curve2(
                            lor_ridge_model,
                            combo_name,
                            X_train,
                            y_train,
                            cv=10,
                            n_jobs=-1,
                        )
                        lc.tight_layout()
                        lc.savefig(
                            os.path.join(lcdir, combo_name)
                            + "_rep{0}.jpg".format(i + 1),
                            dpi=600,
                        )
                        lc.close()

                    if model == "lor-lasso":
                        ### Logistic regression with lasso #1 priority
                        lor_lasso_param_dic = {
                            "tol": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                            "C": np.linspace(0.5, 1.5, 10),
                            "intercept_scaling": np.linspace(0.5, 1.5, 10),
                        }
                        lor_lasso_model = LogisticRegression(
                            penalty="l1",
                            solver="liblinear",
                            random_state=random_state,
                            max_iter=2147483647,
                        )
                        lor_lasso_bayes_search = BayesSearchCV(
                            lor_lasso_model,
                            lor_lasso_param_dic,
                            scoring=make_scorer(matthews_corrcoef),
                            n_jobs=-1,
                            cv=10,
                        ).fit(X_train, y_train)
                        ## Save best mean MCC
                        lor_lasso_best_mean_mcc.append(
                            lor_lasso_bayes_search.best_score_
                        )
                        ## Train model with identified best hyperparameters and all train data
                        lor_lasso_model.set_params(
                            **lor_lasso_bayes_search.best_params_
                        )
                        lor_lasso_model.fit(X_train, y_train)
                        # Predict test set and calculate MCC score
                        y_test_prediction = lor_lasso_model.predict(X_test)
                        lor_lasso_test_score_mcc.append(
                            matthews_corrcoef(y_test, y_test_prediction)
                        )
                        # Generate learning curve
                        lc = plot_learning_curve2(
                            lor_lasso_model,
                            combo_name,
                            X_train,
                            y_train,
                            cv=10,
                            n_jobs=-1,
                        )
                        lc.tight_layout()
                        lc.savefig(
                            os.path.join(lcdir, combo_name)
                            + "_rep{0}.jpg".format(i + 1),
                            dpi=600,
                        )
                        lc.close()

                    if model == "mlp":
                        ### MLP #4 priority
                        mlp_param_dic = {
                            "hidden_layer_sizes": list(
                                set(
                                    [
                                        round(x)
                                        for x in np.linspace(
                                            len(X_train), len(X_train.columns), 10
                                        )
                                    ]
                                )
                            ),
                            "activation": ["tanh", "relu", "logistic", "identity"],
                            "solver": ["sgd", "adam", "lbfgs"],
                            "alpha": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                            "learning_rate": ["constant", "adaptive", "invscaling"],
                        }
                        # Define model
                        mlp_model = MLPClassifier(
                            random_state=random_state, max_iter=2147483647
                        )
                        mlp_bayes_search = BayesSearchCV(
                            mlp_model,
                            mlp_param_dic,
                            scoring=make_scorer(matthews_corrcoef),
                            n_jobs=-1,
                            cv=10,
                        ).fit(X_train, y_train)
                        ## Save best mean MCC
                        mlp_best_mean_mcc.append(mlp_bayes_search.best_score_)
                        ## Train model with identified best hyperparameters and all train data
                        mlp_model.set_params(**mlp_bayes_search.best_params_)
                        mlp_model.fit(X_train, y_train)
                        # Predict test set and calculate MCC score
                        y_test_prediction = mlp_model.predict(X_test)
                        mlp_test_score_mcc.append(
                            matthews_corrcoef(y_test, y_test_prediction)
                        )
                        # Generate learning curve
                        lc = plot_learning_curve2(
                            mlp_model, combo_name, X_train, y_train, cv=10, n_jobs=-1
                        )
                        lc.tight_layout()
                        lc.savefig(
                            os.path.join(lcdir, combo_name)
                            + "_rep{0}.jpg".format(i + 1),
                            dpi=600,
                        )
                        lc.close()

                    if model == "knn":
                        ### KNN #2 priority
                        knn_param_dic = {
                            "n_neighbors": list(range(3, 9)),
                            "algorithm": ["ball_tree", "kd_tree", "brute"],
                            "leaf_size": list(range(20, 41, 4)),
                            "weights": ["uniform", "distance"],
                            "p": [1, 2, 3],
                        }
                        # Define model
                        knn_model = KNeighborsClassifier()
                        knn_bayes_search = BayesSearchCV(
                            knn_model,
                            knn_param_dic,
                            scoring=make_scorer(matthews_corrcoef),
                            n_jobs=-1,
                            cv=10,
                        ).fit(X_train, y_train)
                        ## Save best mean MCC
                        knn_best_mean_mcc.append(knn_bayes_search.best_score_)
                        ## Train model with identified best hyperparameters and all train data
                        knn_model.set_params(**knn_bayes_search.best_params_)
                        knn_model.fit(X_train, y_train)
                        # Predict test set and calculate MCC score
                        y_test_prediction = knn_model.predict(X_test)
                        knn_test_score_mcc.append(
                            matthews_corrcoef(y_test, y_test_prediction)
                        )
                        # Generate learning curve
                        lc = plot_learning_curve2(
                            knn_model, combo_name, X_train, y_train, cv=10, n_jobs=-1
                        )
                        lc.tight_layout()
                        lc.savefig(
                            os.path.join(lcdir, combo_name)
                            + "_rep{0}.jpg".format(i + 1),
                            dpi=600,
                        )
                        lc.close()

                    counter += 1

            # Take mean and SD of results and add model results to master_score_dict
            r_d_s_f = "_".join([rank, data_type, seq_type, f_select_naming])

            if "xgb" in models:
                xgb_train_mcc_mean = sum(xgb_best_mean_mcc) / len(xgb_best_mean_mcc)
                xgb_train_mcc_sd = np.std(xgb_best_mean_mcc)
                xgb_test_mcc_mean = sum(xgb_test_score_mcc) / len(xgb_test_score_mcc)
                xgb_test_mcc_sd = np.std(xgb_test_score_mcc)
                master_score_dict[r_d_s_f + "_xgb"] = [
                    xgb_train_mcc_mean,
                    xgb_train_mcc_sd,
                    xgb_test_mcc_mean,
                    xgb_test_mcc_sd,
                ]

            if "lsvc" in models:
                lsvc_train_mcc_mean = sum(lsvc_best_mean_mcc) / len(lsvc_best_mean_mcc)
                lsvc_train_mcc_sd = np.std(lsvc_best_mean_mcc)
                lsvc_test_mcc_mean = sum(lsvc_test_score_mcc) / len(lsvc_test_score_mcc)
                lsvc_test_mcc_sd = np.std(lsvc_test_score_mcc)
                master_score_dict[r_d_s_f + "_lsvc"] = [
                    lsvc_train_mcc_mean,
                    lsvc_train_mcc_sd,
                    lsvc_test_mcc_mean,
                    lsvc_test_mcc_sd,
                ]

            if "knn" in models:
                knn_train_mcc_mean = sum(knn_best_mean_mcc) / len(knn_best_mean_mcc)
                knn_train_mcc_sd = np.std(knn_best_mean_mcc)
                knn_test_mcc_mean = sum(knn_test_score_mcc) / len(knn_test_score_mcc)
                knn_test_mcc_sd = np.std(knn_test_score_mcc)
                master_score_dict[r_d_s_f + "_knn"] = [
                    knn_train_mcc_mean,
                    knn_train_mcc_sd,
                    knn_test_mcc_mean,
                    knn_test_mcc_sd,
                ]

            if "svc" in models:
                svc_train_mcc_mean = sum(svc_best_mean_mcc) / len(svc_best_mean_mcc)
                svc_train_mcc_sd = np.std(svc_best_mean_mcc)
                svc_test_mcc_mean = sum(svc_test_score_mcc) / len(svc_test_score_mcc)
                svc_test_mcc_sd = np.std(svc_test_score_mcc)
                master_score_dict[r_d_s_f + "_svc"] = [
                    svc_train_mcc_mean,
                    svc_train_mcc_sd,
                    svc_test_mcc_mean,
                    svc_test_mcc_sd,
                ]

            if "rf" in models:
                rf_train_mcc_mean = sum(rf_best_mean_mcc) / len(rf_best_mean_mcc)
                rf_train_mcc_sd = np.std(rf_best_mean_mcc)
                rf_test_mcc_mean = sum(rf_test_score_mcc) / len(rf_test_score_mcc)
                rf_test_mcc_sd = np.std(rf_test_score_mcc)
                master_score_dict[r_d_s_f + "_rf"] = [
                    rf_train_mcc_mean,
                    rf_train_mcc_sd,
                    rf_test_mcc_mean,
                    rf_test_mcc_sd,
                ]

            if "lor-ridge" in models:
                lor_ridge_train_mcc_mean = sum(lor_ridge_best_mean_mcc) / len(
                    lor_ridge_best_mean_mcc
                )
                lor_ridge_train_mcc_sd = np.std(lor_ridge_best_mean_mcc)
                lor_ridge_test_mcc_mean = sum(lor_ridge_test_score_mcc) / len(
                    lor_ridge_test_score_mcc
                )
                lor_ridge_test_mcc_sd = np.std(lor_ridge_test_score_mcc)
                master_score_dict[r_d_s_f + "_lor-ridge"] = [
                    lor_ridge_train_mcc_mean,
                    lor_ridge_train_mcc_sd,
                    lor_ridge_test_mcc_mean,
                    lor_ridge_test_mcc_sd,
                ]

            if "lor-lasso" in models:
                lor_lasso_train_mcc_mean = sum(lor_lasso_best_mean_mcc) / len(
                    lor_lasso_best_mean_mcc
                )
                lor_lasso_train_mcc_sd = np.std(lor_lasso_best_mean_mcc)
                lor_lasso_test_mcc_mean = sum(lor_lasso_test_score_mcc) / len(
                    lor_lasso_test_score_mcc
                )
                lor_lasso_test_mcc_sd = np.std(lor_lasso_test_score_mcc)
                master_score_dict[r_d_s_f + "_lor-lasso"] = [
                    lor_lasso_train_mcc_mean,
                    lor_lasso_train_mcc_sd,
                    lor_lasso_test_mcc_mean,
                    lor_lasso_test_mcc_sd,
                ]

            if "mlp" in models:
                mlp_train_mcc_mean = sum(mlp_best_mean_mcc) / len(mlp_best_mean_mcc)
                mlp_train_mcc_sd = np.std(mlp_best_mean_mcc)
                mlp_test_mcc_mean = sum(mlp_test_score_mcc) / len(mlp_test_score_mcc)
                mlp_test_mcc_sd = np.std(mlp_test_score_mcc)
                master_score_dict[r_d_s_f + "_mlp"] = [
                    mlp_train_mcc_mean,
                    mlp_train_mcc_sd,
                    mlp_test_mcc_mean,
                    mlp_test_mcc_sd,
                ]


# Turn the master df into df and save
score_df = pd.DataFrame(master_score_dict).transpose()
score_df.columns = ["train_mcc_mean", "train_mcc_sd", "test_mcc_mean", "test_mcc_sd"]
score_df["rank"] = score_df.index.str.split("_").str[0]
score_df["datatype"] = score_df.index.str.split("_").str[1]
score_df["seqtype"] = score_df.index.str.split("_").str[2]
score_df["feature_selection"] = score_df.index.str.split("_").str[3]
score_df["model"] = score_df.index.str.split("_").str[4]
score_df.to_csv(os.path.join(outdir, "score_df.csv"), index=False)

# Save taxa lists
with open(os.path.join(outdir, "taxa_lists.pickle"), "wb") as handle:
    pickle.dump(master_taxa_list_dic, handle)

time_print("Script done.")
