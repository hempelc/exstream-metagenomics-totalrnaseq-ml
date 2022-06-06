#!/usr/bin/env python3

# Written by Christopher Hempel (hempelc@uoguelph.ca) on 05 May 2022

# This script analyzes correlations between abundance or p-a data and environmental factors.

# Partially following https://towardsdatascience.com/machine-learning-with-python-classification-complete-tutorial-d2c99dc524ec

import pandas as pd #v1.3.5
import os
import copy
import datetime
import logging
import pickle
import collections
import random
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib import pyplot
from skbio.stats.composition import multiplicative_replacement, clr #v0.5.6
from sklearn import feature_selection, svm
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, matthews_corrcoef
from featurewiz import featurewiz
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV

# Activate logging for debugging
# logging.basicConfig(level=logging.DEBUG,
#     format = '%(asctime)s - %(levelname)s - %(message)s')


# Parameters set manually
## Full path to directory that contains abundance/p-a data generated on the basis of multiple taxonomic ranks
workdir = "/Users/christopherhempel/Google Drive/PhD UoG/ExStream project/data_analysis/"
## Full path to out directory
outdir = workdir
## File that contains sample infos, i.e., classes
sample_info_file = "/Users/christopherhempel/Google Drive/PhD UoG/ExStream project/data_analysis/sample_info.csv"
## Ranks to include into analysis (looped over)
#ranks = ["phylum", "class", "order", "family", "genus", "species"]
ranks = ["phylum"]
## Set dependent variable ("pestidice_treatment", "sediment_addition", "pesticide_and_sediment")
dependent_variable = "pesticide_and_sediment"
## Apply feature selection? (True/False)
f_select = False
## If feature selection, select based on which selection method?
## ("dt" for decision tree, "mi" for mutual_information,
## "rf" for random forest, "lasso" for lasso, "fw" for featurewiz)
selection_method = "fw"
## Apply dimensionality reduction via PCA? (True/False)
dim_red = False
## If dimensionality reduction, choose % of variance that should be covered by PCs
pca_perc = 0.5
## Select models you want to train
## ("xbg" for XGBoosting, "lsvc" for linear SVC, "rf" for random forest,
## "knn" for KNN, "svc" for SVC, "lor-ridge" for logistic regression with ridge,
## "lor-lasso" for logistic regression with lasso, "mlp" for multilayer perceptron)
models = ["xgb", "lsvc", "knn", "svc", "rf", "lor-ridge", 'lor_lasso', 'mlp']
# Set random state for models and pond selection during train test split
random_state = 1
## Show plots generated during data exploration and feature selection? (True/False)
plots = True

# Define functions
## Print datetime and text
def time_print(text):
    datetime_now=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(datetime_now + "  ---  " + text)

# Learning curve plotting, taken from Dan Tulpan's course script
def plot_learning_curve2(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    pyplot.figure()

    pyplot.title(title)
    if ylim is not None:
        pyplot.ylim(*ylim)
        #pyplot.ylim([0,100])
    pyplot.xlabel("Training examples")
    pyplot.ylabel("MCC")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True, scoring=make_scorer(matthews_corrcoef))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    pyplot.grid()
    pyplot.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    pyplot.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    pyplot.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    pyplot.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    pyplot.legend(loc="best")

    return pyplot


# This dict will contain all info
master_dict = {}

## Make output dir for learning curves
lcdir = os.path.join(outdir, "learning_curves")
if not os.path.exists(lcdir):
    os.makedirs(lcdir)

# Loop over ranks
counter = 1
for rank in ranks:

    # Read in data
    abundances = pd.read_csv("/Users/christopherhempel/Google Drive/PhD UoG/ExStream project/data_analysis/abundances_" + rank + ".csv") # Adapt
    abundances["taxon"] = abundances["taxon"].fillna("NA")
    abundances = abundances.set_index("taxon")
    abundances.index.name = None
    abundances = abundances.transpose().reset_index().rename(columns={"index":'sample'})
    sample_info = pd.read_csv(sample_info_file)
    ## Edit sample column
    sample_info["sample"] = sample_info["sample"].str.lstrip('0').str.upper()
    sample_info_dna = copy.deepcopy(sample_info)
    sample_info_dna["sample"] = sample_info_dna["sample"] + "_DNA"
    sample_info_rna = copy.deepcopy(sample_info)
    sample_info_rna["sample"] = sample_info_rna["sample"] + "_RNA"
    sample_info = pd.concat([sample_info_dna, sample_info_rna])


    # Drop NAs:
    print("Average NA abundance per sample: {0}% (mean), {1}% (median)".format(round(100*abundances["NA"]\
        .mean(), 2), round(100*abundances["NA"].median(), 2)))
    print("Average NA abundance per metagenomics sample: {0}% (mean), {1}% (median)"\
        .format(round(100*abundances[abundances['sample'].str.contains('DNA')]["NA"].mean(), 2), \
        round(100*abundances[abundances['sample'].str.contains('DNA')]["NA"].median(), 2)))
    print("Average NA abundance per total rna-seq sample: {0}% (mean), {1}% (median)"\
        .format(round(100*abundances[abundances['sample'].str.contains('RNA')]["NA"].mean(), 2), \
        round(100*abundances[abundances['sample'].str.contains('RNA')]["NA"].median(), 2)))
    abundances = abundances.drop(["NA"], axis = 1).set_index('sample')
    ## In the case that all taxa of a sample are NA, the sample has a sum of 0, so we
    ## have to drop all those samples
    abundances = abundances.loc[(abundances.sum(axis=1) != 0), :]
    ## Turn coverages back into relative so that their sum equals 1
    abundances = (abundances.transpose()/abundances.transpose().sum()).transpose().reset_index()

    # Merge info
    df = pd.merge(abundances, sample_info, how='inner')
    df = df.set_index("sample")
    df.index.name = None

    # Add  column for sequencing type and combination of pesticide
    # treatment and sediment addition
    df["seq_type"] = df.index.str.contains("DNA")
    df['seq_type'] = df['seq_type'].map({True: 'metagenomics', False: 'total_rnaseq'})
    # Change sediment yes to with (w) sediment and sediment no to without (wo) sediment
    df['sediment_addition'] = df['sediment_addition'].map({"yes": 'w_sediment', "no": 'wo_sediment'})
    df['pesticide_and_sediment'] = df["pestidice_treatment"] + "_" + df["sediment_addition"]

    # Separation
    df_dna = df[df['seq_type']=="metagenomics"]
    df_rna = df[df['seq_type']=="total_rnaseq"]
    df_taxa_all = df.drop(["pestidice_treatment", "sediment_addition", "seq_type", "pesticide_and_sediment"], axis=1)
    df_taxa_dna = df_dna.drop(["pestidice_treatment", "sediment_addition", "seq_type", "pesticide_and_sediment"], axis=1)
    df_taxa_rna = df_rna.drop(["pestidice_treatment", "sediment_addition", "seq_type", "pesticide_and_sediment"], axis=1)
    df_vars_all = df[["pestidice_treatment", "sediment_addition", "seq_type", "pesticide_and_sediment"]]
    df_vars_dna = df_dna[["pestidice_treatment", "sediment_addition", "pesticide_and_sediment"]]
    df_vars_rna = df_rna[["pestidice_treatment", "sediment_addition", "pesticide_and_sediment"]]
    ## Drop taxa in dna and rna dfs with total abundance = 0 (some only occur in either or)
    df_taxa_dna = df_taxa_dna.loc[:, (df_taxa_dna.sum() != 0)]
    df_taxa_rna = df_taxa_rna.loc[:, (df_taxa_rna.sum() != 0)]


    # Data exploration overall
    ## Taxa overall
    all_taxa = list(df_taxa_all.columns)
    dna_taxa = list(df_taxa_dna.columns)
    rna_taxa = list(df_taxa_rna.columns)
    print("Number of taxa found overall: {0}".format(len(all_taxa)))
    print("Number of taxa found in metagenomics: {0}".format(len(dna_taxa)))
    print("Number of taxa found in total rna-seq: {0}".format(len(rna_taxa)))
    print("Number of taxa shared between metagenomics and total rna-seq: {0}".format(len([x for x in dna_taxa if x in rna_taxa])))
    print("Number of taxa unique to metagenomics: {0}".format(len([x for x in dna_taxa if x not in rna_taxa])))
    print("Number of taxa unique to total rna-seq: {0}".format(len([x for x in rna_taxa if x not in dna_taxa])))
    ## Dependent variable distribution
    px.bar(df_vars_dna[dependent_variable].value_counts().sort_values(), orientation='h', title='class distribution {0} metagenomics'.format(dependent_variable))
    px.bar(df_vars_rna[dependent_variable].value_counts().sort_values(), orientation='h', title='class distribution {0} total rna-seq'.format(dependent_variable))


    # To later split the data into train and test data, some manual adjustments are required
    # due to the data structure. Therefore, I can't simply use the scikitlearn
    # train_test_split function later but rather have to manually select the test samples.
    # I know the test set will be 12 samples big for a 10% split, and I must avoid having samples
    # A and B from the same pond separately in the test and training set, respectively, because they are dependent.
    # So, I will randomly select 6 ponds with 2 available samples for DNA and RNA
    # samples and use them later for the test dataset. Some samples failed for DNA or RNA,
    # so I need to identify which ones worked in both and pick randomly from those.
    ## Remove suffixes
    sample_ponds_dna = df_taxa_dna.index.str.replace('[AB]_DNA', '')
    sample_ponds_rna = df_taxa_rna.index.str.replace('[AB]_RNA', '')
    ## Identify non-unique ponds (that worked for A and B)
    non_unique_ponds_dna = [item for item, count in collections.Counter(sample_ponds_dna).items() if count == 2]
    non_unique_ponds_rna = [item for item, count in collections.Counter(sample_ponds_rna).items() if count == 2]
    ## Identify overlaps in RNA and DNA ponds
    ponds = list(set(non_unique_ponds_dna) & set(non_unique_ponds_rna))
    ## Randomly pick 6 ponds, NOTE: the ranodm seed line has to be run before every use
    ## of the random.sample method to generate reproducible results!
    random.seed(random_state)
    test_ponds = random.sample(ponds, 6)

    # Run everything for both abundance and p-a data:
    data_types = ["abundance", "pa"]
    for data_type in data_types:

        if data_type=="abundance":
            # Transform abundances by replacing 0s and taking the centered log ratio
            df_taxa_all = pd.DataFrame(clr(multiplicative_replacement(df_taxa_all))\
                , index=df_taxa_all.index, columns=df_taxa_all.columns)
            df_taxa_dna = pd.DataFrame(clr(multiplicative_replacement(df_taxa_dna))\
                , index=df_taxa_dna.index, columns=df_taxa_dna.columns)
            df_taxa_rna = pd.DataFrame(clr(multiplicative_replacement(df_taxa_rna))\
                , index=df_taxa_rna.index, columns=df_taxa_rna.columns)
            # Data exploration abundance
            ## Taxa abundance
            print("10 most abundant taxa overall:")
            print(df_taxa_all.sum().sort_values(ascending=False).head(n=15))
            print("10 most abundant taxa in metagenomics:")
            print(df_taxa_dna.sum().sort_values(ascending=False).head(n=15))
            print("10 most abundant taxa in total rna-seq:")
            print(df_taxa_rna.sum().sort_values(ascending=False).head(n=15))
            ## Taxa abundance distribution
            px.histogram(df_taxa_dna.sum(), title = "Taxa abundance distribution metagenomics").show()
            px.histogram(df_taxa_rna.sum(), title = "Taxa abundance distribution total rna-seq").show()

            # Standardize data
            df_taxa_all = pd.DataFrame(StandardScaler().fit_transform(df_taxa_all), index=df_taxa_all.index, columns=df_taxa_all.columns)
            df_taxa_dna = pd.DataFrame(StandardScaler().fit_transform(df_taxa_dna), index=df_taxa_dna.index, columns=df_taxa_dna.columns)
            df_taxa_rna = pd.DataFrame(StandardScaler().fit_transform(df_taxa_rna), index=df_taxa_rna.index, columns=df_taxa_rna.columns)

        elif data_type=="pa":
            # Turn abundance data into presence/absence data (0=not found, 1=found():
            ## Replace all values above 0 with 1:
            df_taxa_all[df_taxa_all > 0] = 1
            df_taxa_dna[df_taxa_dna > 0] = 1
            df_taxa_rna[df_taxa_rna > 0] = 1
            # Data exploration pa
            ## Taxa pa
            print("10 most present taxa overall:")
            print(df_taxa_all.sum().sort_values(ascending=False).head(n=15))
            print("10 most present taxa in metagenomics:")
            print(df_taxa_dna.sum().sort_values(ascending=False).head(n=15))
            print("10 most present taxa in total rna-seq:")
            print(df_taxa_rna.sum().sort_values(ascending=False).head(n=15))
            ### Taxa presence distribution
            px.histogram(df_taxa_dna.sum(), title = "Taxa presence distribution metagenomics").show()
            px.histogram(df_taxa_rna.sum(), title = "Taxa presence distribution total rna-seq").show()


        # Run everything for both metagenomics and total rna-seq:
        seq_types = ["metagenomics", "totalrnaseq"]
        for seq_type in seq_types:
            if seq_type=="metagenomics":
                df_taxa = df_taxa_dna
                df_vars = df_vars_dna
            else:
                df_taxa = df_taxa_rna
                df_vars = df_vars_rna

            # Define independent and dependent variables
            X = df_taxa.iloc[:, :5]
            y = df_vars[dependent_variable]
            feature_names = df_taxa.columns


            if f_select == True:
                # Feature selection
                if selection_method=="lasso":
                    ## Via Lasso
                    selector = feature_selection.SelectFromModel(estimator=LogisticRegression(C=1, penalty="l1",
                        solver='liblinear', random_state=random_state), max_features=len(feature_names)).fit(X, y)
                    reduced_feature_names = list(feature_names[selector.get_support()])

                elif selection_method=="rf":
                    ## RF importance
                    rf_model_fs = RandomForestClassifier(random_state=random_state)
                    rf_model_fs.fit(X, y)
                    rf_importances = rf_model_fs.feature_importances_
                    ## Put in a pandas df
                    rf_importances_df = pd.DataFrame({"importance":rf_importances,
                        "feature":feature_names}).sort_values("importance",
                        ascending=False)
                    rf_importances_df['cumsum'] = rf_importances_df['importance'].cumsum(axis=0)
                    ### Plot
                    if plots==True:
                        px.line(rf_importances_df, x="feature", y="cumsum", title='Cumulative importance').show()
                    reduced_feature_names = list(rf_importances_df[rf_importances_df["importance"] != 0]["feature"])

                elif selection_method=="mi":
                    # Mutual information
                    mutual_information = feature_selection.mutual_info_classif(X, y)
                    ## Put in a pandas dtf
                    mutual_information_ser = pd.Series(mutual_information, index = feature_names)
                    ## Keep anything with correlation (since most features actually have no correlation)
                    ## To Do: figure out if this is actually the way to go or if correlation cutoff needs to be made
                    reduced_feature_names = list(mutual_information_ser[mutual_information_ser != 0].index)

                elif selection_method=="dt":
                    ## DecisionTree importance
                    dt_model = DecisionTreeClassifier(random_state=random_state)
                    dt_model.fit(X, y)
                    dt_importances = dt_model.feature_importances_
                    ## Put in a pandas df
                    dt_importances_df = pd.DataFrame({"importance":dt_importances,
                        "feature":feature_names}).sort_values("importance",
                        ascending=False)
                    dt_importances_df['cumsum'] = dt_importances_df['importance'].cumsum(axis=0)
                    ### Plot
                    if plots==True:
                        px.line(dt_importances_df, x="feature", y="cumsum", title='Cumulative importance').show()
                    reduced_feature_names = list(dt_importances_df[dt_importances_df["importance"] != 0]["feature"])

                elif selection_method=="fw":
                    # With automatization via featurewiz (removes correlated features based on manually defined correlation limit)
                    df_featurewiz = pd.concat([df_taxa, df_vars[dependent_variable]], axis=1)
                    reduced_feature_names, reduced_df = featurewiz(df_featurewiz, dependent_variable, corr_limit=0.7, verbose=1)

                else:
                    print("selection_method not in available options.")

                ## Select features
                X = df_taxa[reduced_feature_names]


            if dim_red == True:
                # Dimensionality reduction via PCA
                pca = PCA(n_components=20)
                pca_model = pca.fit(X).transform(X)
                pca_cumsum = pd.DataFrame({"PC-wise explained variance": pca.explained_variance_ratio_,
                    "Cumulative explained variance": np.cumsum(pca.explained_variance_ratio_)})
                if plots == True:
                    px.line(pca_cumsum, labels={"index": "PCs", "value": "Explained variance"}).show()
                # We select the n first PCs, where n=number of PCs that explain the % variance defined in "pca_perc"
                pc_num = len(pca_cumsum[pca_cumsum["Cumulative explained variance"] <= pca_perc])+1
                X = pca_model[:,:pc_num]


            # Split into train and test using the previously defined ponds
            X_test = X[X.index.str.contains('|'.join(['^' + x + '[AB]_[DR]NA' for x in test_ponds]))]
            y_test = y[y.index.str.contains('|'.join(['^' + x + '[AB]_[DR]NA' for x in test_ponds]))]
            X_train = X[~X.index.str.contains('|'.join(['^' + x + '[AB]_[DR]NA' for x in test_ponds]))]
            y_train = y[~y.index.str.contains('|'.join(['^' + x + '[AB]_[DR]NA' for x in test_ponds]))]


            # Run models
            for model in models:
                combo_name = "_".join([rank, data_type, seq_type, model])
                combo_num = len(ranks)*len(data_types)*len(seq_types)*len(models)
                time_print("Processing combo {0}/{1}: {2}...".format(str(counter), str(combo_num), combo_name))

                if model == "xgb":
                    # XGBoost #3 priority
                    ## Turn string classes into numerical classes
                    xgb_y_train = pd.Series(y_train).astype('category').cat.codes
                    xgb_y_test = pd.Series(y_test).astype('category').cat.codes
                    ## Define hyperparameters combinations to try
                    xgb_param_dic = {"learning_rate": [0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01],
                        'n_estimators': [int(x) for x in np.linspace(50, 200, 5)],
                        "reg_lambda": list(np.linspace(1, 10, 5)),
                        "reg_alpha": list(np.linspace(0, 10, 5)),
                        "min_child_weight": list(np.linspace(1, 10, 5)),
                        "gamma": list(np.linspace(0, 20, 5)),
                        'max_depth': [3,4,5,6]}
                    ## Define model
                    xgb_model = xgb.XGBClassifier(objective="multi:softmax", random_state=random_state)
                    xgb_bayes_search = BayesSearchCV(xgb_model,
                           xgb_param_dic, scoring=make_scorer(matthews_corrcoef), n_jobs=-1, cv=10).fit(X_train, xgb_y_train)
                    ## Save best mean MCC
                    best_mean_mcc = xgb_bayes_search.best_score_
                    ## Train model with identified best hyperparameters and all train data
                    xgb_model.set_params(**xgb_bayes_search.best_params_)
                    xgb_model.fit(X_train, xgb_y_train)
                    ## Save all parameters for reproducibility
                    best_params = xgb_model.get_params()
                    # Predict test set and calculate MCC score
                    xgb_y_test_prediction = xgb_model.predict(X_test)
                    test_score_mcc = matthews_corrcoef(xgb_y_test, xgb_y_test_prediction)
                    # Generate learning curve
                    lc = plot_learning_curve2(xgb_model, combo_name, X_train, xgb_y_train, cv=10, n_jobs=-1)
                    lc.tight_layout()
                    lc.savefig(os.path.join(lcdir, combo_name) + ".jpg", dpi=600)
                    lc.close()

                if model == "rf":
                    ### RF #3 priority
                    rf_param_dic = {"n_estimators": [int(x) for x in np.linspace(50, 200, 5)], # Number of trees in random forest
                        "criterion": ["gini", "entropy"],
                        "max_features": ['auto', 'sqrt'], # Number of features to consider at every split
                        "min_samples_leaf": [1, 2, 4]} # Minimum number of samples required at each leaf node
                    # Define model
                    rf_model = RandomForestClassifier(random_state=random_state)
                    rf_bayes_search = BayesSearchCV(rf_model, rf_param_dic, scoring=make_scorer(matthews_corrcoef), n_jobs=-1, cv=10).fit(X_train, y_train)
                    ## Save best mean MCC
                    best_mean_mcc = rf_bayes_search.best_score_
                    ## Train model with identified best hyperparameters and all train data
                    rf_model.set_params(**rf_bayes_search.best_params_)
                    rf_model.fit(X_train, y_train)
                    ## Save all parameters for reproducibility
                    best_params = rf_model.get_params()
                    # Predict test set and calculate MCC score
                    y_test_prediction = rf_model.predict(X_test)
                    test_score_mcc = matthews_corrcoef(y_test, y_test_prediction)
                    # Generate learning curve
                    lc = plot_learning_curve2(rf_model, combo_name, X_train, y_train, cv=10, n_jobs=-1)
                    lc.tight_layout()
                    lc.savefig(os.path.join(lcdir, combo_name) + ".jpg", dpi=600)
                    lc.close()

                if model == "lsvc":
                    ### Linear SVC #1 priority
                    lsvc_param_dic = {'loss':['hinge', 'squared_hinge'],
                        'tol':[1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                        'C': np.linspace(0.5, 1.5, 10),
                        'intercept_scaling': np.linspace(0.5, 1.5, 10)}
                    # Define model
                    lsvc_model = svm.LinearSVC(random_state=random_state, penalty='l2')
                    lsvc_bayes_search = BayesSearchCV(lsvc_model,
                           lsvc_param_dic, scoring=make_scorer(matthews_corrcoef), n_jobs=-1, cv=10).fit(X_train, y_train)
                    ## Save best mean MCC
                    best_mean_mcc = lsvc_bayes_search.best_score_
                    ## Train model with identified best hyperparameters and all train data
                    lsvc_model.set_params(**lsvc_bayes_search.best_params_)
                    lsvc_model.fit(X_train, y_train)
                    ## Save all parameters for reproducibility
                    best_params = lsvc_model.get_params()
                    # Predict test set and calculate MCC score
                    y_test_prediction = lsvc_model.predict(X_test)
                    test_score_mcc = matthews_corrcoef(y_test, y_test_prediction)
                    # Generate learning curve
                    lc = plot_learning_curve2(lsvc_model, combo_name, X_train, y_train, cv=10, n_jobs=-1)
                    lc.tight_layout()
                    lc.savefig(os.path.join(lcdir, combo_name) + ".jpg", dpi=600)
                    lc.close()

                if model == "svc":
                    ### SVC #3 priority
                    svc_param_dic = {'C': np.linspace(0.1, 10, 10),  #norm used in the penalization
                        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                        'degree': [2,3,4],
                        'gamma': np.linspace(0.1, 10, 10)}
                    # Define model
                    svc_model = svm.SVC(random_state=random_state)
                    svc_bayes_search = BayesSearchCV(svc_model,
                           svc_param_dic, scoring=make_scorer(matthews_corrcoef), n_jobs=-1, cv=10).fit(X_train, y_train)
                    ## Save best mean MCC
                    best_mean_mcc = svc_bayes_search.best_score_
                    ## Train model with identified best hyperparameters and all train data
                    svc_model.set_params(**svc_bayes_search.best_params_)
                    svc_model.fit(X_train, y_train)
                    ## Save all parameters for reproducibility
                    best_params = svc_model.get_params()
                    # Predict test set and calculate MCC score
                    y_test_prediction = svc_model.predict(X_test)
                    test_score_mcc = matthews_corrcoef(y_test, y_test_prediction)
                    # Generate learning curve
                    lc = plot_learning_curve2(svc_model, combo_name, X_train, y_train, cv=10, n_jobs=-1)
                    lc.tight_layout()
                    lc.savefig(os.path.join(lcdir, combo_name) + ".jpg", dpi=600)
                    lc.close()

                if model == "lor-ridge":
                    ### Logistic regression with ridge #1 priority
                    lor_ridge_param_dic = {'tol': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                        'C': np.linspace(0.5, 1.5, 10),
                        'intercept_scaling': np.linspace(0.5, 1.5, 10),
                        'solver': ['newton-cg','lbfgs', 'sag', 'saga']}
                    # Define model
                    lor_ridge_model = LogisticRegression(penalty = "l2", random_state=random_state, max_iter=10000000000)
                    lor_ridge_bayes_search = BayesSearchCV(lor_ridge_model,
                           lor_ridge_param_dic, scoring=make_scorer(matthews_corrcoef), n_jobs=-1, cv=10).fit(X_train, y_train)
                    ## Save best mean MCC
                    best_mean_mcc = lor_ridge_bayes_search.best_score_
                    ## Train model with identified best hyperparameters and all train data
                    lor_ridge_model.set_params(**lor_ridge_bayes_search.best_params_)
                    lor_ridge_model.fit(X_train, y_train)
                    ## Save all parameters for reproducibility
                    best_params = lor_ridge_model.get_params()
                    # Predict test set and calculate MCC score
                    y_test_prediction = lor_ridge_model.predict(X_test)
                    test_score_mcc = matthews_corrcoef(y_test, y_test_prediction)
                    # Generate learning curve
                    lc = plot_learning_curve2(lor_ridge_model, combo_name, X_train, y_train, cv=10, n_jobs=-1)
                    lc.tight_layout()
                    lc.savefig(os.path.join(lcdir, combo_name) + ".jpg", dpi=600)
                    lc.close()

                if model == "lor-lasso":
                    ### Logistic regression with lasso #1 priority
                    lor_lasso_param_dic = {'tol': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                        'C': np.linspace(0.5, 1.5, 10),
                        'intercept_scaling': np.linspace(0.5, 1.5, 10)}
                    lor_lasso_model = LogisticRegression(penalty = "l1", solver = 'liblinear', random_state=random_state, max_iter=10000000000)
                    lor_lasso_bayes_search = BayesSearchCV(lor_lasso_model,
                           lor_lasso_param_dic, scoring=make_scorer(matthews_corrcoef), n_jobs=-1, cv=10).fit(X_train, y_train)
                    ## Save best mean MCC
                    best_mean_mcc = lor_lasso_bayes_search.best_score_
                    ## Train model with identified best hyperparameters and all train data
                    lor_lasso_model.set_params(**lor_lasso_bayes_search.best_params_)
                    lor_lasso_model.fit(X_train, y_train)
                    ## Save all parameters for reproducibility
                    best_params = lor_lasso_model.get_params()
                    # Predict test set and calculate MCC score
                    y_test_prediction = lor_lasso_model.predict(X_test)
                    test_score_mcc = matthews_corrcoef(y_test, y_test_prediction)
                    # Generate learning curve
                    lc = plot_learning_curve2(lor_lasso_model, combo_name, X_train, y_train, cv=10, n_jobs=-1)
                    lc.tight_layout()
                    lc.savefig(os.path.join(lcdir, combo_name) + ".jpg", dpi=600)
                    lc.close()

                if model == "mlp":
                    ### MLP #4 priority
                    mlp_param_dic = {'hidden_layer_sizes': [round(x) for x in np.linspace(100, len(X_train.columns), 10)],
                        'activation': ['tanh', 'relu', 'logistic', 'identity'],
                        'solver': ['sgd', 'adam', 'lbfgs'],
                        'alpha': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                        'learning_rate': ['constant','adaptive', 'invscaling']}
                    # Define model
                    mlp_model = MLPClassifier(random_state=random_state, max_iter=10000000000)
                    mlp_bayes_search = BayesSearchCV(mlp_model,
                           mlp_param_dic, scoring=make_scorer(matthews_corrcoef), n_jobs=-1, cv=10).fit(X_train, y_train)
                    ## Save best mean MCC
                    best_mean_mcc = mlp_bayes_search.best_score_
                    ## Train model with identified best hyperparameters and all train data
                    mlp_model.set_params(**mlp_bayes_search.best_params_)
                    mlp_model.fit(X_train, y_train)
                    ## Save all parameters for reproducibility
                    best_params = mlp_model.get_params()
                    # Predict test set and calculate MCC score
                    y_test_prediction = mlp_model.predict(X_test)
                    test_score_mcc = matthews_corrcoef(y_test, y_test_prediction)
                    # Generate learning curve
                    lc = plot_learning_curve2(mlp_model, combo_name, X_train, y_train, cv=10, n_jobs=-1)
                    lc.tight_layout()
                    lc.savefig(os.path.join(lcdir, combo_name) + ".jpg", dpi=600)
                    lc.close()

                if model == "knn":
                    ### KNN #2 priority
                    knn_param_dic = {'n_neighbors': list(range(3,9)),
                        'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                        'leaf_size': list(range(20, 41, 4)),
                        'weights': ['uniform', 'distance'],
                        'p': [1,2,3]}
                    # Define model
                    knn_model = KNeighborsClassifier()
                    knn_bayes_search = BayesSearchCV(knn_model, knn_param_dic, scoring=make_scorer(matthews_corrcoef), n_jobs=-1, cv=10).fit(X_train, y_train)
                    ## Save best mean MCC
                    best_mean_mcc = knn_bayes_search.best_score_
                    ## Train model with identified best hyperparameters and all train data
                    knn_model.set_params(**knn_bayes_search.best_params_)
                    knn_model.fit(X_train, y_train)
                    ## Save all parameters for reproducibility
                    best_params = knn_model.get_params()
                    # Predict test set and calculate MCC score
                    y_test_prediction = knn_model.predict(X_test)
                    test_score_mcc = matthews_corrcoef(y_test, y_test_prediction)
                    # Generate learning curve
                    lc = plot_learning_curve2(knn_model, combo_name, X_train, y_train, cv=10, n_jobs=-1)
                    lc.tight_layout()
                    lc.savefig(os.path.join(lcdir, combo_name) + ".jpg", dpi=600)
                    lc.close()

                # Add model results to master_dict
                master_dict[combo_name]=[best_mean_mcc, test_score_mcc, best_params]

                counter+=1


# Save the master dict as pickle object
with open(os.path.join(outdir, "model_scores_dict.pkl"), 'wb') as f:
    pickle.dump(master_dict, f)

# Turn the master df into df and save
score_df = pd.DataFrame(master_dict).transpose().drop(2, axis=1)
score_df.columns = ["train_score_mcc", "test_score_mcc"]
score_df["rank"] = score_df.index.str.split("_").str[0]
score_df["datatype"] = score_df.index.str.split("_").str[1]
score_df["seqtype"] = score_df.index.str.split("_").str[2]
score_df["model"] = score_df.index.str.split("_").str[3]
score_df.to_csv(os.path.join(outdir, "score_df.csv"), index=False)
