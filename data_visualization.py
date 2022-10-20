import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import numpy as np
import os
from scipy.stats import spearmanr #v1.7.3

# File paths
score_csv_file = "/Users/christopherhempel/Google Drive/PhD UoG/ExStream project/data_files_for_ml/score_dfs/master_score_df.csv"
outdir = "/Users/christopherhempel/Google Drive/PhD UoG/ExStream project/exstream_data_visualization_outdir"

if not os.path.exists(outdir):
    os.makedirs(outdir)

# Read in data
df = pd.read_csv(score_csv_file)
df.rename(columns = {"feature_selection": "feature-selection"}, inplace = True)

# Set variables
models = df["model"].unique()
dtypes = df["datatype"].unique()
ranks = ['phylum', 'class', 'order', 'family', 'genus', 'species']
seqtypes = df["seqtype"].unique()
fselects = df["feature-selection"].unique()
rank_sort_dic = {'phylum': 0, 'class': 1, 'order': 2, 'family': 3, 'genus': 4, 'species': 5}

# Line graphs
for model in models:
    for dtype in dtypes:
        for fselect in fselects:
            name = "{0}_{1}_{2}".format(model, dtype, fselect)
            modeldf = df[(df["model"]==model) & (df["datatype"]==dtype) & (df["feature-selection"]==fselect)]
            modeldf = modeldf.sort_values(by=['rank'], key=lambda x: x.map(rank_sort_dic))
            fig = px.line(modeldf, x="rank", y="test_mcc_mean", color='seqtype', title=name, markers=True)
            fig.update_yaxes(range=[-1, 1])
            fig.add_shape(type="line", x0=ranks[0], y0=0, x1=ranks[-1], y1=0, line_width=2, line_dash="dash")
            fig.show()
            fig.write_image(os.path.join(outdir, "linegraph_{0}.svg".format(name)))

for seqtype in seqtypes:
    for dtype in dtypes:
        for fselect in fselects:
            name = "{0}_{1}_{2}".format(seqtype, dtype, fselect)
            seqtypedf= df[(df["seqtype"]==seqtype) & (df["datatype"]==dtype) & (df["feature-selection"]==fselect)]
            seqtypedf = seqtypedf.sort_values(by=['rank'], key=lambda x: x.map(rank_sort_dic))
            fig = px.line(seqtypedf, x="rank", y="test_mcc_mean", color='model', title=name, markers=True)
            fig.update_yaxes(range=[-1, 1])
            fig.add_shape(type="line", x0=ranks[0], y0=0, x1=ranks[-1], y1=0, line_width=2, line_dash="dash")
            fig.show()
            fig.write_image(os.path.join(outdir, "linegraph_{0}.svg".format(name)))

for seqtype in seqtypes:
    for dtype in dtypes:
        for model in models:
            name = "{0}_{1}_{2}".format(seqtype, dtype, model)
            fselectdf= df[(df["seqtype"]==seqtype) & (df["datatype"]==dtype) & (df["model"]==model)]
            fselectdf = fselectdf.sort_values(by=['rank'], key=lambda x: x.map(rank_sort_dic))
            fig = px.line(fselectdf, x="rank", y="test_mcc_mean", color='feature-selection', title=name, markers=True)
            fig.update_yaxes(range=[-1, 1])
            fig.add_shape(type="line", x0=ranks[0], y0=0, x1=ranks[-1], y1=0, line_width=2, line_dash="dash")
            fig.show()
            fig.write_image(os.path.join(outdir, "linegraph_{0}.svg".format(name)))


# Heatmap
## Adapt seqtypes and ranks for heatmap organization
heatmap_df = df.replace('its-otu', 'a-its-otu').replace('its-esv', 'b-its-esv').replace('16s-otu', 'c-16s-otu').replace('16s-esv', 'd-16s-esv').replace('16s-its-otu', 'e-16s-its-otu').replace('16s-its-esv', 'f-16s-its-esv')
heatmap_df = heatmap_df.replace('phylum', 'a-phylum').replace('class', 'b-class').replace('order', 'c-order').replace('family', 'd-family').replace('genus', 'e-genus').replace('species', 'f-species')

## Combine columns for visualization
heatmap_df["rank+model"] = heatmap_df["rank"] + "_" + heatmap_df["model"]
heatmap_df["stype+fs"] = heatmap_df["seqtype"] + "_" + heatmap_df["feature-selection"]

heatmap_df["stype+rank"] = heatmap_df["seqtype"] + "_" + heatmap_df["rank"]
heatmap_df["fs+model"] = heatmap_df["feature-selection"] + "_" + heatmap_df["model"]

## Datatype has no significance effect, so we drop the rows with abundance (pa has the signel highest MCC)
heatmap_df = heatmap_df[heatmap_df['datatype']=="pa"]

## Visualization
### Category 1
df_wide = heatmap_df.pivot_table(index="rank+model", columns="stype+fs", values='test_mcc_mean').transpose()
heatmap = px.imshow(df_wide)
heatmap.show()
heatmap.update_xaxes(tickangle = 90)
heatmap.write_image(os.path.join(outdir, "heatmap_v1.svg"), height=600, width=1600)
heatmap.write_image(os.path.join(outdir, "heatmap_v1.png"), height=600, width=1600)

### Category 2
df_wide = heatmap_df.pivot_table(index="stype+rank", columns="fs+model", values='test_mcc_mean').transpose()
heatmap = px.imshow(df_wide)
heatmap.update_xaxes(tickangle = 90)
heatmap.show()
heatmap.write_image(os.path.join(outdir, "heatmap_v2.svg"), height=600, width=1600)
heatmap.write_image(os.path.join(outdir, "heatmap_v2.png"), height=600, width=1600)


# Coefficients and p-values
## Create additional column for sequencing method
df.loc[df["seqtype"].str.contains('16s') , 'seqmethod'] = "16s"
df.loc[df["seqtype"].str.contains('its') , 'seqmethod'] = "its"
df.loc[df["seqtype"].str.contains('16s-its') , 'seqmethod'] = "16s-its"
df.loc[df["seqtype"].str.contains('metagenomics') , 'seqmethod'] = "metagenomics"
df.loc[df["seqtype"].str.contains('totalrnaseq') , 'seqmethod'] = "totalrnaseq"

## Run correlation test for all seqtypes
### Separate
X = df[['seqmethod', 'rank', 'datatype', 'feature-selection', 'model']]
Y = df['test_mcc_mean'].to_numpy()

### Generate dummy variables
X_dummies = pd.get_dummies(X)
### Correlation estimation
cor_dic={}
for col in X_dummies.columns:
    X=np.array(X_dummies[col])
    cor=spearmanr(X,Y)
    cor_dic[col]=cor
cor_df_no_clust = pd.DataFrame(cor_dic , index=["coefficient", "p-value"]).transpose().reset_index()

## Create additional column for clustering type that only applies to amplicon seq data
df_ampseq = df[~df['seqtype'].isin(['metagenomics', 'totalrnaseq'])]
df_ampseq.loc[df_ampseq["seqtype"].str.contains('esv') , 'clustering'] = "esv"
df_ampseq.loc[df_ampseq["seqtype"].str.contains('otu') , 'clustering'] = "otu"

## Repeat correlation test for those data
### Separate
X = df_ampseq['clustering']
Y = df_ampseq['test_mcc_mean'].to_numpy()

### Generate dummy variables
X_dummies = pd.get_dummies(X)
X_dummies.columns
### Correlation estimation
cor_dic={}
for col in X_dummies.columns:
    X=np.array(X_dummies[col])
    cor=spearmanr(X,Y)
    cor_dic[col]=cor
cor_df_w_clust = pd.DataFrame(cor_dic , index=["coefficient", "p-value"]).transpose().reset_index()

## Merge dfs and adapt format
cor_df = pd.concat([cor_df_no_clust, cor_df_w_clust]).reset_index().drop(["level_0"], axis=1)
cor_df["index"] = cor_df["index"].str.replace("esv", "clustering_esv")
cor_df["index"] = cor_df["index"].str.replace("otu", "clustering_otu")

## Make significance categories
cor_df.loc[cor_df["p-value"] <= 0.001 , 'significance_cat'] = "***"
cor_df.loc[cor_df["p-value"] > 0.001 , 'significance_cat'] = "**"
cor_df.loc[cor_df["p-value"] > 0.01 , 'significance_cat'] = "*"
cor_df.loc[cor_df["p-value"] > 0.05, 'significance_cat'] = ""

## Organize df
cor_df[["category", "method"]] = cor_df["index"].str.split("_", n=-1, expand=True)
cor_df = cor_df.set_index("index")
cor_df = cor_df.reindex(["seqmethod_its", "seqmethod_16s", 'seqmethod_16s-its',
    'seqmethod_metagenomics', 'seqmethod_totalrnaseq', 'clustering_otu', 'clustering_esv',
    'rank_phylum', 'rank_class', 'rank_order', 'rank_family', 'rank_genus',
    'rank_species', 'datatype_abundance', 'datatype_pa', "feature-selection_w-fs", "feature-selection_wo-fs", 'model_knn',
    'model_lor-lasso', 'model_lor-ridge', 'model_lsvc', "model_mlp",
    'model_rf', 'model_svc', 'model_xgb'])

## Make figure
cols = ["#ba543d", "#ac9c3d", "#687ad2", "#b94a73", "#56ae6c", "#9550a1"]
fig = px.bar(cor_df, x='method', y='coefficient', color='category', text='significance_cat', color_discrete_sequence = cols)
fig.update_layout(xaxis_tickangle=30)
fig.update_traces(textposition='outside')
fig.show()
fig.write_image(os.path.join(outdir, "coefs_pvals_overall.svg"), width=1200, height=500)
fig.write_image(os.path.join(outdir, "coefs_pvals_overall.png"), width=1200, height=500)


## For seqtypes separately
for seqtype in seqtypes:
    df_seqtype = df[df["seqtype"]==seqtype].drop(["seqtype"], axis=1)
    X = df_seqtype[['rank','datatype', 'model', 'feature-selection']]
    Y = df_seqtype['test_mcc_mean'].to_numpy()

    ### Generate dummy variables
    X_dummies = pd.get_dummies(X)

    ### Correlation estimation
    cor_dic={}
    for col in X_dummies.columns:
        X=np.array(X_dummies[col])
        cor=spearmanr(X,Y)
        cor_dic[col]=cor
    cor_df = pd.DataFrame(cor_dic , index=["coefficient", "p-value"]).transpose().reset_index()

    ### Make significance categories
    cor_df.loc[cor_df["p-value"] <= 0.001 , 'significance_cat'] = "***"
    cor_df.loc[cor_df["p-value"] > 0.001 , 'significance_cat'] = "**"
    cor_df.loc[cor_df["p-value"] > 0.01 , 'significance_cat'] = "*"
    cor_df.loc[cor_df["p-value"] > 0.05, 'significance_cat'] = ""

    ### Organize df
    cor_df[["category", "method"]] = cor_df["index"].str.split("_", n=-1, expand=True)
    cor_df = cor_df.set_index("index")
    cor_df = cor_df.reindex(['rank_phylum', 'rank_class', 'rank_order', 'rank_family', 'rank_genus',
        'rank_species', 'datatype_abundance', 'datatype_pa', "feature-selection_w-fs", "feature-selection_wo-fs", 'model_knn',
        'model_lor-lasso', 'model_lor-ridge', 'model_lsvc', 'model_mlp',
        'model_rf', 'model_svc', 'model_xgb'])

    ### Make figure
    cols = ["#ba543d", "#ac9c3d", "#56ae6c", "#9550a1"]
    fig = px.bar(cor_df, x='method', y='coefficient', color='category', text='significance_cat', color_discrete_sequence = cols, title=seqtype)
    fig.update_layout(xaxis_tickangle=30)
    fig.update_traces(textposition='outside')
    fig.update_yaxes(range=[-0.68, 0.68], nticks = 8)
    fig.show()
    fig.write_image(os.path.join(outdir, "coefs_pvals_{0}.svg".format(seqtype)))
    fig.write_image(os.path.join(outdir, "coefs_pvals_{0}.png".format(seqtype)))
