import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import numpy as np
import itertools
# The pickle object has been genereated under python >=3.8 on a different machine,
# while the processing machine has python <3.8. To read in the pickle object generated
# on a higher pickle version, we need to import a specific pickle package
import pickle5 as pickle

# File paths
score_csv_file = "/Users/christopherhempel/Desktop/no_mlp_no_fs_no_multimarker/score_df.csv"
taxa_list_dic_file = "/Users/christopherhempel/Desktop/no_mlp_no_fs_no_multimarker/taxa_lists.pickle"

# Read in data
df = pd.read_csv(score_csv_file)
df.rename(columns = {"feature_selection": "feature-selection"}, inplace = True)
with open(taxa_list_dic_file, 'rb') as handle:
    taxa_list_dic = pickle.load(handle)

df["fs+dtype+rank"] = df["feature-selection"] + "_" + df["datatype"] + "_" + df["rank"]
df["stype+model"] = df["seqtype"] + "_" + df["model"]

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

for model in models:
    modeldf= df[df["model"]==model]
    fig = px.line(modeldf, x="rank", y="test_mcc_mean", color='dtype+stype', title=model, markers=True)
    fig.update_yaxes(range=[-1, 1])
    fig.add_shape(type="line", x0=ranks[0], y0=0, x1=ranks[-1], y1=0, line_width=2, line_dash="dash")
    fig.show()


for seqtype in seqtypes:
    for dtype in dtypes:
        name = "{0}_{1}".format(seqtype, dtype)
        seqtypedf= df[(df["seqtype"]==seqtype) & (df["datatype"]==dtype)]
        fig = px.line(seqtypedf, x="rank", y="test_mcc_mean", color='model', title=name, markers=True)
        fig.update_yaxes(range=[-1, 1])
        fig.add_shape(type="line", x0=ranks[0], y0=0, x1=ranks[-1], y1=0, line_width=2, line_dash="dash")
        fig.show()


# Heatmap
df_wide = df.pivot_table(index="fs+dtype+rank", columns="stype+model", values='test_mcc_mean')
heatmap = px.imshow(df_wide)
heatmap.show()
#heatmap.write_image("/Users/christopherhempel/Desktop/{0}_{1}.png".format(i[0], i[1]))



# Chord diagram
# TO DO: POLISH
rank="species"
seqtypes_chord = taxa_list_dic[rank].keys()
combos = list(itertools.combinations(seqtypes_chord, 2))
# TO DO: read in taxa lists and make numbers of overlapping taxa per rank and seqtype
# Run R script manually
from_lst = []
to_lst = []
overlap_lst = []
for i in range(len(combos)):
    set1 = set(taxa_list_dic[rank][combos[i][0]])
    set2 = set(taxa_list_dic[rank][combos[i][1]])
    overlap = len(set1 & set2)
    from_lst.append(combos[i][0])
    to_lst.append(combos[i][1])
    overlap_lst.append(overlap)
overlap_df = pd.DataFrame({"from": from_lst, "to": to_lst, "overlap": overlap_lst})
overlap_df["from"] = overlap_df["from"].str.replace("dna_taxa", "metagenomics").str.replace("rna_taxa", "totalrnaseq")\
    .str.replace("esv_16s_taxa", "16s-esv").str.replace("esv_its_taxa", "its-esv").str.replace("otu_16s_taxa", "16s-otu").str.replace("otu_its_taxa", "its-otu")
overlap_df["to"] = overlap_df["to"].str.replace("dna_taxa", "metagenomics").str.replace("rna_taxa", "totalrnaseq")\
    .str.replace("esv_16s_taxa", "16s-esv").str.replace("esv_its_taxa", "its-esv").str.replace("otu_16s_taxa", "16s-otu").str.replace("otu_its_taxa", "its-otu")
overlap_df.to_csv("/Users/christopherhempel/Desktop/overlap_df.csv", index=False)

seqtype_chord_lst = []
unique_lst = []
total_lst = []
for seqtype_chord in seqtypes_chord:
    seqtype_chord_lst.append(seqtype_chord)
    total_lst.append(len(taxa_list_dic[rank][seqtype_chord]))
    other_seqtypes = [x for x in seqtypes_chord if x != seqtype_chord]
    other_taxa = []
    for other_seqtype in other_seqtypes:
        other_taxa.extend(taxa_list_dic[rank][other_seqtype])
    other_taxa = set(other_taxa)
    unique_taxa = len([x for x in taxa_list_dic[rank][seqtype_chord] if x not in other_taxa])
    unique_lst.append(unique_taxa)

count_df = pd.DataFrame({"seqtype": seqtype_chord_lst, "unique_taxa": unique_lst, "total_taxa": total_lst})
count_df["seqtype"] = count_df["seqtype"].str.replace("dna_taxa", "metagenomics").str.replace("rna_taxa", "totalrnaseq")\
    .str.replace("esv_16s_taxa", "16s-esv").str.replace("esv_its_taxa", "its-esv").str.replace("otu_16s_taxa", "16s-otu").str.replace("otu_its_taxa", "its-otu")

count_df_short = pd.DataFrame({"seqtype": list(count_df["seqtype"])+list(count_df["seqtype"]),
    "taxa_cat": ["unique_taxa"]*len(count_df)+["total_taxa"]*len(count_df),
    "count": list(count_df["unique_taxa"])+list(count_df["total_taxa"])})

fig = px.scatter(count_df_short, x="taxa_cat", y="seqtype", size="count")
fig.show()
fig.write_image("/Users/christopherhempel/Desktop/bubbles.svg")



# Coefficients and p-values
X = df[['rank','datatype', 'seqtype', 'model', 'feature-selection']]
Y = df['test_mcc_mean']

X_dummies = pd.get_dummies(X)
X_dummies = sm.add_constant(X_dummies) # adding a constant

lr_model = sm.OLS(Y, X_dummies).fit()

## Summarize the output and extract coefs and p vals
lr_summary = lr_model.summary2().tables[1][['Coef.', 'P>|t|']]
lr_summary = lr_summary.rename(columns={"Coef.": "Coefficient", "P>|t|": "p-value"})
#lr_summary = lr_summary.drop("const")
lr_summary = lr_summary.reset_index()
lr_summary[['category', 'method']] = lr_summary['index'].str.split('_', expand=True)
lr_summary = lr_summary.set_index('index')

lr_summary.loc[lr_summary["p-value"] <= 0.001 , 'significance_cat'] = "***"
lr_summary.loc[lr_summary["p-value"] > 0.001 , 'significance_cat'] = "**"
lr_summary.loc[lr_summary["p-value"] > 0.01 , 'significance_cat'] = "*"
lr_summary.loc[lr_summary["p-value"] > 0.05, 'significance_cat'] = ""
# lr_summary = lr_summary.reindex(['rank_phylum', 'rank_class', 'rank_order', 'rank_family', 'rank_genus',
#     'rank_species', 'datatype_abundance', 'datatype_pa', "seqtype_16s-esv", "seqtype_16s-otu", "seqtype_its-esv",
#     "seqtype_its-otu", 'seqtype_metagenomics', 'seqtype_totalrnaseq', 'model_knn',
#     'model_lor-lasso', 'model_lor-ridge', 'model_lsvc', 'model_mlp',
#     'model_rf', 'model_svc', 'model_xgb'])
lr_summary = lr_summary.reindex(['rank_phylum', 'rank_class', 'rank_order', 'rank_family', 'rank_genus',
    'rank_species', 'datatype_abundance', 'datatype_pa', "seqtype_16s-esv", "seqtype_16s-otu", "seqtype_its-esv",
    "seqtype_its-otu", 'seqtype_metagenomics', 'seqtype_totalrnaseq', 'model_knn',
    'model_lor-lasso', 'model_lor-ridge', 'model_lsvc',
    'model_rf', 'model_svc', 'model_xgb', "feature-selection_wo-fs"])
cols = ["#8B4A97", "#5D6EB4", "#B24A39", "#A2903D", "#4BA566"]
fig = px.bar(lr_summary, x='method', y='Coefficient', color='category', text='significance_cat', color_discrete_sequence = cols)
fig.update_layout(xaxis_tickangle=45)
fig.update_traces(textposition='outside')
fig.update_yaxes(range=[-0.07, 0.11], nticks = 8)
fig.show()

# For types separately
for seqtype in seqtypes:
    df_seqtype = df[df["seqtype"]==seqtype].drop("seqtype", axis=1)
    X = df_seqtype[['rank','datatype', 'model', 'feature-selection']]
    Y = df_seqtype['test_mcc_mean']

    X_dummies = pd.get_dummies(X)
    X_dummies = sm.add_constant(X_dummies) # adding a constant

    lr_model = sm.OLS(Y, X_dummies).fit()

    ## Summarize the output and extract coefs and p vals
    lr_summary = lr_model.summary2().tables[1][['Coef.', 'P>|t|']]
    lr_summary = lr_summary.rename(columns={"Coef.": "Coefficient", "P>|t|": "p-value"})
    #lr_summary = lr_summary.drop("const")
    lr_summary = lr_summary.reset_index()
    lr_summary[['category', 'method']] = lr_summary['index'].str.split('_', expand=True)
    lr_summary = lr_summary.set_index('index')

    lr_summary.loc[lr_summary["p-value"] <= 0.001 , 'significance_cat'] = "***"
    lr_summary.loc[lr_summary["p-value"] > 0.001 , 'significance_cat'] = "**"
    lr_summary.loc[lr_summary["p-value"] > 0.01 , 'significance_cat'] = "*"
    lr_summary.loc[lr_summary["p-value"] > 0.05, 'significance_cat'] = ""
    # lr_summary = lr_summary.reindex(['rank_phylum', 'rank_class', 'rank_order', 'rank_family', 'rank_genus',
    #     'rank_species', 'datatype_abundance', 'datatype_pa', "seqtype_16s-esv", "seqtype_16s-otu", "seqtype_its-esv",
    #     "seqtype_its-otu", 'seqtype_metagenomics', 'seqtype_totalrnaseq', 'model_knn',
    #     'model_lor-lasso', 'model_lor-ridge', 'model_lsvc', 'model_mlp',
    #     'model_rf', 'model_svc', 'model_xgb'])
    lr_summary = lr_summary.reindex(['rank_phylum', 'rank_class', 'rank_order', 'rank_family', 'rank_genus',
        'rank_species', 'datatype_abundance', 'datatype_pa', 'model_knn',
        'model_lor-lasso', 'model_lor-ridge', 'model_lsvc',
        'model_rf', 'model_svc', 'model_xgb', "feature-selection_wo-fs"])

    cols = ["#8B4A97", "#5D6EB4", "#A2903D", "#4BA566"]
    fig = px.bar(lr_summary, x='method', y='Coefficient', color='category', text='significance_cat', color_discrete_sequence = cols, title=seqtype)
    fig.update_layout(xaxis_tickangle=45)
    fig.update_traces(textposition='outside')
    fig.update_yaxes(range=[-0.07, 0.11], nticks = 8)
    fig.show()
