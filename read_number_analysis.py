#!/usr/bin/env python3

# Written by Christopher Hempel (hempelc@uoguelph.ca) on 13 May 2022

# This script analyzes DNA and RNA read numbers across samples

import pandas as pd
import plotly.express as px
import os

readnumfile_omics = "/Users/christopherhempel/Google Drive/PhD UoG/ExStream project/data_files_for_ml/read_nums_exstream_samples_omics.tsv"
readnumfile_16s = "/Users/christopherhempel/Google Drive/PhD UoG/ExStream project/data_files_for_ml/read_nums_exstream_samples_16s.tsv"
readnumfile_its = "/Users/christopherhempel/Google Drive/PhD UoG/ExStream project/data_files_for_ml/read_nums_exstream_samples_its.tsv"
outdir = "/Users/christopherhempel/Google Drive/PhD UoG/ExStream project/exstream_data_visualization_outdir"

# Omics
df = pd.read_table(readnumfile_omics, header=None, names=["sample", "reads"])
df_dna = df[df["sample"].str.contains("DNA")]
df_rna = df[df["sample"].str.contains("RNA")]
dna_num_total = df_dna["reads"].sum()
rna_num_total = df_rna["reads"].sum()
print("Total number of DNA reads: {0}".format(dna_num_total))
print("Total number of RNA reads: {0}".format(rna_num_total))
df_dna["sample"] = df_dna["sample"].str.replace("_DNA", "")
df_rna["sample"] = df_rna["sample"].str.replace("_RNA", "")
dna_mean = df_dna[~df_dna["sample"].str.contains("Neg")]["reads"].mean()
dna_sd = df_dna[~df_dna["sample"].str.contains("Neg")]["reads"].std()
rna_mean = df_rna[~df_rna["sample"].str.contains("Neg")]["reads"].mean()
rna_sd = df_rna[~df_rna["sample"].str.contains("Neg")]["reads"].std()
dna_bar = px.bar(df_dna.sort_values("reads"), x='sample', y='reads', title = "DNA samples - read count distribution (total: {0} reads)".format(dna_num_total))
rna_bar = px.bar(df_rna.sort_values("reads"), x='sample', y='reads', title = "RNA samples - read count distribution (total: {0} reads)".format(rna_num_total))
dna_bar.add_hline(y=dna_mean, line_dash="dash")
rna_bar.add_hline(y=rna_mean, line_dash="dash")
dna_bar.write_image(os.path.join(outdir, "read_distribution_dna.svg"), width=2300)
rna_bar.write_image(os.path.join(outdir, "read_distribution_rna.svg"), width=2300)

# 16S
df = pd.read_table(readnumfile_16s, header=None, names=["sample", "reads"])
num_total = df["reads"].sum()
print("Total number of reads: {0}".format(num_total))
mean = df[~df["sample"].str.contains("Neg")]["reads"].mean()
sd = df[~df["sample"].str.contains("Neg")]["reads"].std()
bar = px.bar(df.sort_values("reads"), x='sample', y='reads', title = "16S samples - read count distribution (total: {0} reads)".format(num_total))
bar.add_hline(y=mean, line_dash="dash")
bar.write_image(os.path.join(outdir, "read_distribution_16s.svg"), width=2300)

# ITS
df = pd.read_table(readnumfile_its, header=None, names=["sample", "reads"])
num_total = df["reads"].sum()
print("Total number of reads: {0}".format(num_total))
mean = df[~df["sample"].str.contains("Neg")]["reads"].mean()
sd = df[~df["sample"].str.contains("Neg")]["reads"].std()
bar = px.bar(df.sort_values("reads"), x='sample', y='reads', title = "ITS-2 samples - read count distribution (total: {0} reads)".format(num_total))
bar.add_hline(y=mean, line_dash="dash")
bar.write_image(os.path.join(outdir, "read_distribution_its.svg"), width=2300)
