import subprocess
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import numpy as np

# Set path to R script chordDiagram
rscript_chorddiagram = "./chorddiagram.R"

df = pd.read_csv("/Users/christopherhempel/Desktop/score_df.csv")

models = df["model"].unique()
dtypes = df["datatype"].unique()
ranks = df["rank"].unique()
seqtypes = df["seqtype"].unique()

df["rank+dtype"] = df["rank"] + "_" + df["datatype"]
df["rank+stype"] = df["rank"] + "_" + df["seqtype"]
df["rank+model"] = df["rank"] + "_" + df["model"]
df["dtype+stype"] = df["datatype"] + "_" + df["seqtype"]
df["dtype+model"] = df["datatype"] + "_" + df["model"]
df["stype+model"] = df["seqtype"] + "_" + df["model"]

# Line graphs
for model in models:
    for dtype in dtypes:
        name = "{0}_{1}".format(model, dtype)
        modeldf= df[(df["model"]==model) & (df["datatype"]==dtype)]
        fig = px.line(modeldf, x="rank", y="test_score_mcc", color='seqtype', title=name, markers=True)
        fig.update_yaxes(range=[-1, 1])
        fig.add_shape(type="line", x0=ranks[0], y0=0, x1=ranks[-1], y1=0, line_width=2, line_dash="dash")
        fig.show()

for model in models:
    modeldf= df[df["model"]==model]
    fig = px.line(modeldf, x="rank", y="test_score_mcc", color='dtype+stype', title=model, markers=True)
    fig.update_yaxes(range=[-1, 1])
    fig.add_shape(type="line", x0=ranks[0], y0=0, x1=ranks[-1], y1=0, line_width=2, line_dash="dash")
    fig.show()


for seqtype in seqtypes:
    for dtype in dtypes:
        name = "{0}_{1}".format(seqtype, dtype)
        seqtypedf= df[(df["seqtype"]==seqtype) & (df["datatype"]==dtype)]
        fig = px.line(seqtypedf, x="rank", y="test_score_mcc", color='model', title=name, markers=True)
        fig.update_yaxes(range=[-1, 1])
        fig.add_shape(type="line", x0=ranks[0], y0=0, x1=ranks[-1], y1=0, line_width=2, line_dash="dash")
        fig.show()


# Heatmaps
combos = [["rank+dtype", "stype+model"], ["rank+stype", "dtype+model"]]

for combo in combos:
    df_wide = df.pivot_table(index=combo[0], columns=combo[1], values='test_score_mcc')
    heatmap = px.imshow(df_wide)
    heatmap.show()
    #heatmap.write_image("/Users/christopherhempel/Desktop/{0}_{1}.png".format(i[0], i[1]))


# Chord diagram
# TO DO: read in taxa lists and make numbers of overlapping taxa per rank and seqtype
# Run R script manually


# Coefficients and p-values
X = df[['rank','datatype', 'seqtype', 'model']]
Y = df['test_score_mcc']

X_dummies = pd.get_dummies(X)
X_dummies = sm.add_constant(X_dummies) # adding a constant

lr_model = sm.OLS(Y, X_dummies).fit()

## Summarize the output and extract coefs and p vals
lr_summary = lr_model.summary2().tables[1][['Coef.', 'P>|t|']]
lr_summary = lr_summary.rename(columns={"Coef.": "Coefficient", "P>|t|": "p-value"})
lr_summary = lr_summary.drop("const")
lr_summary = lr_summary.reset_index()
lr_summary[['category', 'method']] = lr_summary['index'].str.split('_', expand=True)
lr_summary = lr_summary.set_index('index')

lr_summary.loc[lr_summary["p-value"] <= 0.001 , 'significance_cat'] = "***"
lr_summary.loc[lr_summary["p-value"] > 0.001 , 'significance_cat'] = "**"
lr_summary.loc[lr_summary["p-value"] > 0.01 , 'significance_cat'] = "*"
lr_summary.loc[lr_summary["p-value"] > 0.05, 'significance_cat'] = ""
lr_summary = lr_summary.reindex(['rank_phylum', 'rank_class', 'rank_order', 'rank_family', 'rank_genus',
    'rank_species', 'datatype_abundance', 'datatype_pa',
    'seqtype_metagenomics', 'seqtype_totalrnaseq', 'model_knn',
    'model_lor-lasso', 'model_lor-ridge', 'model_lsvc', 'model_mlp',
    'model_rf', 'model_svc', 'model_xgb'])

cols = ["#2995ff", "#d96c1e", "#4db444","#9f0281"]
fig = px.bar(lr_summary, x='method', y='Coefficient', color='category', text='significance_cat', color_discrete_sequence = cols)
fig.update_layout(xaxis_tickangle=45)
fig.update_traces(textposition='outside')
fig.update_yaxes(nticks = 9)
fig.show()
