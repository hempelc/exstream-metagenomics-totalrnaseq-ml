#!/usr/bin/env python3

# Written by Christopher Hempel (hempelc@uoguelph.ca) on 13 May 2022

# This script analyzes my DNA and RNA read numbers across samples

import pandas as pd
import plotly.express as px

df = pd.read_table("/Users/christopherhempel/Google Drive/PhD UoG/ExStream project/data_analysis/read_nums_exstream_samples.tsv", header=None, names=["sample", "reads"])
df_noneg = df[~df["sample"].str.contains("Neg")]
df_dna = df_noneg[df_noneg["sample"].str.contains("DNA")]
df_rna = df_noneg[df_noneg["sample"].str.contains("RNA")]
print("Total number of DNA reads: {0}".format(df_dna["reads"].sum()))
print("Total number of RNA reads: {0}".format(df_rna["reads"].sum()))
px.bar(df_dna.sort_values("reads"), x='sample', y='reads', title = "DNA sample read count distribution").show()
px.bar(df_rna.sort_values("reads"), x='sample', y='reads', title = "RNA sample read count distribution").show()
