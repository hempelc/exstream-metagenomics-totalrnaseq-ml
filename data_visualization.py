import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import numpy as np
import itertools
import os
from scipy.stats import pointbiserialr #v1.7.3

# File paths
score_csv_file = "/Users/christopherhempel/Google Drive/PhD UoG/ExStream project/exstream_data_visualization_outdir/master_score_df.csv"
outdir = "/Users/christopherhempel/Google Drive/PhD UoG/ExStream project/exstream_data_visualization_outdir"

if not os.path.exists(outdir):
    os.makedirs(outdir)

# Read in data
df = pd.read_csv(score_csv_file)
df.rename(columns = {"feature_selection": "feature-selection"}, inplace = True)

# Combine columns for visualization
df["fs+dtype+rank"] = df["feature-selection"] + "_" + df["datatype"] + "_" + df["rank"]
df["stype+model"] = df["seqtype"] + "_" + df["model"]
df["dtype+stype"] = df["datatype"] + "_" + df["seqtype"]

models = df["model"].unique()
dtypes = df["datatype"].unique()
ranks = df["rank"].unique()
seqtypes = df["seqtype"].unique()

# Line graphs
for model in models:
    for dtype in dtypes:
        name = "{0}_{1}".format(model, dtype)
        modeldf= df[(df["model"]==model) & (df["datatype"]==dtype)]
        fig = px.line(modeldf, x="rank", y="test_mcc_mean", color='seqtype', title=name, markers=True)
        fig.update_yaxes(range=[-1, 1])
        fig.add_shape(type="line", x0=ranks[0], y0=0, x1=ranks[-1], y1=0, line_width=2, line_dash="dash")
        fig.show()
        fig.write_image(os.path.join(outdir, "linegraph_{0}.svg".format(name)))

for seqtype in seqtypes:
    for dtype in dtypes:
        name = "{0}_{1}".format(seqtype, dtype)
        seqtypedf= df[(df["seqtype"]==seqtype) & (df["datatype"]==dtype)]
        fig = px.line(seqtypedf, x="rank", y="test_mcc_mean", color='model', title=name, markers=True)
        fig.update_yaxes(range=[-1, 1])
        fig.add_shape(type="line", x0=ranks[0], y0=0, x1=ranks[-1], y1=0, line_width=2, line_dash="dash")
        fig.show()
        fig.write_image(os.path.join(outdir, "linegraph_{0}.svg".format(name)))


# Heatmap
df_wide = df.pivot_table(index="fs+dtype+rank", columns="stype+model", values='test_mcc_mean')
heatmap = px.imshow(df_wide)
heatmap.show()
heatmap.write_image(os.path.join(outdir, "heatmap.svg"))



# Coefficients and p-values
X = df[['rank','datatype', 'seqtype', 'model', 'feature-selection']]
Y = df['test_mcc_mean']

# Generate dummy variables
X_dummies = pd.get_dummies(X)

# Correlation estimation
cor_dic={}
for col in X_dummies.columns:
    Y_cor=Y.to_numpy()
    X=np.array(X_dummies[col])
    cor=pointbiserialr(X,Y_cor)
    cor_dic[col]=cor
cor_df = pd.DataFrame(cor_dic , index=["coefficient", "p-value"]).transpose().reset_index()

# Make significance categories
cor_df.loc[cor_df["p-value"] <= 0.001 , 'significance_cat'] = "***"
cor_df.loc[cor_df["p-value"] > 0.001 , 'significance_cat'] = "**"
cor_df.loc[cor_df["p-value"] > 0.01 , 'significance_cat'] = "*"
cor_df.loc[cor_df["p-value"] > 0.05, 'significance_cat'] = ""

# Organize df
cor_df[["category", "method"]] = cor_df["index"].str.split("_", n=-1, expand=True)
cor_df = cor_df.set_index("index")
cor_df = cor_df.reindex(['rank_phylum', 'rank_class', 'rank_order', 'rank_family', 'rank_genus',
    'rank_species', 'datatype_abundance', 'datatype_pa', "seqtype_16s-esv", "seqtype_16s-otu", "seqtype_its-esv",
    "seqtype_its-otu", 'seqtype_16s-its-esv', 'seqtype_16s-its-otu', 'seqtype_metagenomics', 'seqtype_totalrnaseq', 'model_knn',
    'model_lor-lasso', 'model_lor-ridge', 'model_lsvc',
    'model_rf', 'model_svc', 'model_xgb', "feature-selection_wo-fs"])

#Temporary:
cor_df = cor_df.drop("feature-selection_wo-fs")

# Make figure
cols = ["#8B4A97", "#5D6EB4", "#A2903D", "#B24A39", "#4BA566"]
fig = px.bar(cor_df, x='method', y='coefficient', color='category', text='significance_cat', color_discrete_sequence = cols)
fig.update_layout(xaxis_tickangle=45)
fig.update_traces(textposition='outside')
fig.update_yaxes(range=[-0.6, 0.6], nticks = 8)
fig.show()
fig.write_image(os.path.join(outdir, "coefs_pvals_overall.svg"))

# For seqtypes separately
for seqtype in seqtypes:
    df_seqtype = df[df["seqtype"]==seqtype].drop("seqtype", axis=1)
    X = df_seqtype[['rank','datatype', 'model', 'feature-selection']]
    Y = df_seqtype['test_mcc_mean']

    # Generate dummy variables
    X_dummies = pd.get_dummies(X)

    # Correlation estimation
    cor_dic={}
    for col in X_dummies.columns:
        Y_cor=Y.to_numpy()
        X=np.array(X_dummies[col])
        cor=pointbiserialr(X,Y_cor)
        cor_dic[col]=cor
    cor_df = pd.DataFrame(cor_dic , index=["coefficient", "p-value"]).transpose().reset_index()

    # Make significance categories
    cor_df.loc[cor_df["p-value"] <= 0.001 , 'significance_cat'] = "***"
    cor_df.loc[cor_df["p-value"] > 0.001 , 'significance_cat'] = "**"
    cor_df.loc[cor_df["p-value"] > 0.01 , 'significance_cat'] = "*"
    cor_df.loc[cor_df["p-value"] > 0.05, 'significance_cat'] = ""

    # Organize df
    cor_df[["category", "method"]] = cor_df["index"].str.split("_", n=-1, expand=True)
    cor_df = cor_df.set_index("index")
    # cor_df = cor_df.reindex(['rank_phylum', 'rank_class', 'rank_order', 'rank_family', 'rank_genus',
    #     'rank_species', 'datatype_abundance', 'datatype_pa', "seqtype_16s-esv", "seqtype_16s-otu", "seqtype_its-esv",
    #     "seqtype_its-otu", 'seqtype_metagenomics', 'seqtype_totalrnaseq', 'model_knn',
    #     'model_lor-lasso', 'model_lor-ridge', 'model_lsvc', 'model_mlp',
    #     'model_rf', 'model_svc', 'model_xgb'])
    cor_df = cor_df.reindex(['rank_phylum', 'rank_class', 'rank_order', 'rank_family', 'rank_genus',
        'rank_species', 'datatype_abundance', 'datatype_pa', 'model_knn',
        'model_lor-lasso', 'model_lor-ridge', 'model_lsvc',
        'model_rf', 'model_svc', 'model_xgb', "feature-selection_wo-fs"])

    #Temporary:
    cor_df = cor_df.drop("feature-selection_wo-fs")

    # Make figure
    cols = ["#8B4A97", "#5D6EB4", "#B24A39", "#4BA566"]
    fig = px.bar(cor_df, x='method', y='coefficient', color='category', text='significance_cat', color_discrete_sequence = cols, title=seqtype)
    fig.update_layout(xaxis_tickangle=45)
    fig.update_traces(textposition='outside')
    fig.update_yaxes(range=[-0.6, 0.6], nticks = 8)
    fig.show()
    fig.write_image(os.path.join(outdir, "coefs_pvals_{0}.svg".format(seqtype)))
