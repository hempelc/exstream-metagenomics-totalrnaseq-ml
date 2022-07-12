import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import numpy as np
import itertools
import os
# The pickle object has been genereated under python >=3.8 on a different machine,
# while the processing machine has python <3.8. To read in the pickle object generated
# on a higher pickle version, we need to import a specific pickle package
import pickle5 as pickle

# File paths
score_csv_file = "/Users/christopherhempel/Google Drive/PhD UoG/ExStream project/data_analysis_no_mlp_no_fs_no_multimarker/score_df.csv"
taxa_list_dic_file = "/Users/christopherhempel/Google Drive/PhD UoG/ExStream project/data_analysis_new/taxa_lists.pickle"
outdir = "/Users/christopherhempel/Desktop/exstream_data_visualization_outdir"

if not os.path.exists(outdir):
    os.makedirs(outdir)

# Read in data
df = pd.read_csv(score_csv_file)
df.rename(columns = {"feature_selection": "feature-selection"}, inplace = True)
with open(taxa_list_dic_file, 'rb') as handle:
    taxa_list_dic = pickle.load(handle)

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



# Chord diagram
# Do for lowest and highest rank (phylum and species)
for rank in ["phylum", "species"]:
    # Get name of all seqtypes in taxa list (unfortunately I didn't name all seqtypes consistently)
    rank = "phylum"
    seqtypes_chord = taxa_list_dic[rank]["dna_taxa"].keys()
a = taxa_list_dic[rank]["16s-otu"]
sorted(a)
    # First, get number of overlapping taxa between seqtypes
    # Get all possible combinations of seqtypes
    combos = list(itertools.combinations(seqtypes_chord, 2))

    # Define number of overlapping taxa for each combo
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

    # Turn into df and change setype names for consistency
    overlap_df = pd.DataFrame({"from": from_lst, "to": to_lst, "overlap": overlap_lst})
    overlap_df["from"] = overlap_df["from"].str.replace("dna_taxa", "metagenomics").str.replace("rna_taxa", "totalrnaseq")\
        .str.replace("esv_16s_taxa", "16s-esv").str.replace("esv_its_taxa", "its-esv").str.replace("otu_16s_taxa", "16s-otu").str.replace("otu_its_taxa", "its-otu")
    overlap_df["to"] = overlap_df["to"].str.replace("dna_taxa", "metagenomics").str.replace("rna_taxa", "totalrnaseq")\
        .str.replace("esv_16s_taxa", "16s-esv").str.replace("esv_its_taxa", "its-esv").str.replace("otu_16s_taxa", "16s-otu").str.replace("otu_its_taxa", "its-otu")

    # Save for manual processing in R
    overlap_df.to_csv(os.path.join(outdir, "overlap_df_{0}.csv".format(rank)), index=False)


    # Second, get unique number of taxa for each seqtype as bubble plot
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

    # Turn into df and change setype names for consistency
    count_df = pd.DataFrame({"seqtype": seqtype_chord_lst, "unique_taxa": unique_lst, "total_taxa": total_lst})
    count_df["seqtype"] = count_df["seqtype"].str.replace("dna_taxa", "metagenomics").str.replace("rna_taxa", "totalrnaseq")\
        .str.replace("esv_16s_taxa", "16s-esv").str.replace("esv_its_taxa", "its-esv").str.replace("otu_16s_taxa", "16s-otu").str.replace("otu_its_taxa", "its-otu")

    # Change df format from wide to narrow
    count_df_narrow = pd.DataFrame({"seqtype": list(count_df["seqtype"])+list(count_df["seqtype"]),
        "taxa_cat": ["unique_taxa"]*len(count_df)+["total_taxa"]*len(count_df),
        "count": list(count_df["unique_taxa"])+list(count_df["total_taxa"])})

    # Set reference sizes for size legend
    diff = count_df_narrow["count"].max() - count_df_narrow["count"].min()
    first_ref = count_df_narrow["count"].min()
    second_ref = round(first_ref+diff/3)
    third_ref = round(second_ref+diff/3)
    last_ref = count_df_narrow["count"].max()
    count_df_narrow = count_df_narrow.append({'seqtype': 'ref1', 'taxa_cat': 'unique_taxa', 'count': first_ref}, ignore_index = True)
    count_df_narrow = count_df_narrow.append({'seqtype': 'ref2', 'taxa_cat': 'unique_taxa', 'count': second_ref}, ignore_index = True)
    count_df_narrow = count_df_narrow.append({'seqtype': 'ref1', 'taxa_cat': 'total_taxa', 'count': third_ref}, ignore_index = True)
    count_df_narrow = count_df_narrow.append({'seqtype': 'ref2', 'taxa_cat': 'total_taxa', 'count': last_ref}, ignore_index = True)

    # Make figure --> add bubbles to chorddiagram manually generated in R
    fig = px.scatter(count_df_narrow, x="taxa_cat", y="seqtype", size="count", text="count")
    fig.update_traces(textposition='top center')
    fig.show()
    fig.write_image(os.path.join(outdir, "bubbles_{0}.svg".format(rank)))



# Coefficients and p-values
X = df[['rank','datatype', 'seqtype', 'model', 'feature-selection']]
Y = df['test_mcc_mean']

# Generate dummy variables
X_dummies = pd.get_dummies(X)
X_dummies = sm.add_constant(X_dummies) # adding a constant

# Make model
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

# Make figure
cols = ["#8B4A97", "#5D6EB4", "#A2903D", "#B24A39", "#4BA566"]
fig = px.bar(lr_summary, x='method', y='Coefficient', color='category', text='significance_cat', color_discrete_sequence = cols)
fig.update_layout(xaxis_tickangle=45)
fig.update_traces(textposition='outside')
fig.update_yaxes(range=[-0.07, 0.11], nticks = 8)
fig.show()
fig.write_image(os.path.join(outdir, "coefs_pvals_overall.svg"))

# For types separately
for seqtype in seqtypes:
    df_seqtype = df[df["seqtype"]==seqtype].drop("seqtype", axis=1)
    X = df_seqtype[['rank','datatype', 'model', 'feature-selection']]
    Y = df_seqtype['test_mcc_mean']

    # Generate dummy variables
    X_dummies = pd.get_dummies(X)
    X_dummies = sm.add_constant(X_dummies) # adding a constant

    # Make model
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

    # Make figure
    cols = ["#8B4A97", "#5D6EB4", "#B24A39", "#4BA566"]
    fig = px.bar(lr_summary, x='method', y='Coefficient', color='category', text='significance_cat', color_discrete_sequence = cols, title=seqtype)
    fig.update_layout(xaxis_tickangle=45)
    fig.update_traces(textposition='outside')
    fig.update_yaxes(range=[-0.07, 0.11], nticks = 8)
    fig.show()
    fig.write_image(os.path.join(outdir, "coefs_pvals_{0}.svg".format(seqtype)))
