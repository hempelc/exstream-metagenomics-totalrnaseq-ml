#!/usr/bin/env python3

# Written by Christopher Hempel (hempelc@uoguelph.ca) on 13 May 2022

# This script analyzes DNA and RNA read numbers across samples

import pandas as pd
import plotly.express as px
import os

readnumfile = "/Users/christopherhempel/Google Drive/PhD UoG/ExStream project/sequencing_stuff/read_nums_exstream_samples.tsv"
outdir = "/Users/christopherhempel/Google Drive/PhD UoG/ExStream project/exstream_data_visualization_outdir"

df = pd.read_table(readnumfile, header=None, names=["sample", "reads"])
df_noneg = df[~df["sample"].str.contains("Neg")]
df_dna = df_noneg[df_noneg["sample"].str.contains("DNA")]
df_rna = df_noneg[df_noneg["sample"].str.contains("RNA")]
dna_num_total = df_dna["reads"].sum()
rna_num_total = df_rna["reads"].sum()
print("Total number of DNA reads: {0}".format(dna_num_total))
print("Total number of RNA reads: {0}".format(rna_num_total))
df_dna["sample"] = df_dna["sample"].str.replace("_DNA", "")
df_rna["sample"] = df_rna["sample"].str.replace("_RNA", "")
dna_mean = df_dna["reads"].mean()
rna_mean = df_rna["reads"].mean()
dna_bar = px.bar(df_dna.sort_values("reads"), x='sample', y='reads', title = "DNA samples - read count distribution (total: {0} reads)".format(dna_num_total))
rna_bar = px.bar(df_rna.sort_values("reads"), x='sample', y='reads', title = "RNA samples - read count distribution (total: {0} reads)".format(rna_num_total))
dna_bar.add_hline(y=dna_mean, line_dash="dash")
rna_bar.add_hline(y=rna_mean, line_dash="dash")
dna_bar.write_image(os.path.join(outdir, "read_distribution_dna.svg"), width=2000)
rna_bar.write_image(os.path.join(outdir, "read_distribution_rna.svg"), width=2000)
