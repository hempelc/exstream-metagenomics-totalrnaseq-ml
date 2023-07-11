#!/usr/bin/env python3

# Written by Christopher Hempel (hempelc@uoguelph.ca) on 05 May 2022

# This script processes HTS files containing taxonomic and abundance data.
# Every sample needs to be in the working directory as an own directory and
# contain the HTS results as a .txt file (.tar.gz compressed). The output is
# generated in the specified output directory.

import pandas as pd  # v1.3.5
import glob
import os
import copy
import logging

# Activate logging for debugging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Parameters set manually
## Full path to directory that contains samples
workdir = "/Users/christopherhempel/Google Drive/PhD UoG/ExStream project/pipeline_results_new/"
## Full path to output dorectory in which the resulting .csv file is saved
outdir = (
    "/Users/christopherhempel/Google Drive/PhD UoG/ExStream project/data_analysis_new/"
)
## File that contains readnumbers for each sample
readnums = "/Users/christopherhempel/Google Drive/PhD UoG/ExStream project/data_analysis/read_nums_exstream_samples.tsv"
## What rank should taxa be aggregated on?
# ("phylum", "class", "order", "family", "genus", or "species")
groupby_ranks = ["phylum", "class", "order", "family", "genus", "species"]

# Make output dir
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Loop over ranks
for groupby_rank in groupby_ranks:
    # 1 Read in data
    ## 1.1 Read in data for each sample as df and add all taxa from all dfs to "all_taxa" list (needed to generate master df that
    ##   contain coverage of all taxa from all samples):
    sample_dfs = {}  # Empty dic that will eventually contain all sample dfs
    all_taxa = (
        []
    )  # Empty list that will eventually contain all taxa that appear in all samples

    ## Make a list for all sample file names:
    sample_files = glob.glob(os.path.join(workdir, "*", "*.txt*"))
    ## Select ranks to later aggregate on
    all_ranks = [
        "superkingdom",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
    ]
    used_ranks = all_ranks[: all_ranks.index(groupby_rank) + 1]
    for file in sample_files:
        ### Read in file as pandas df, fill NaN with "NA", replace "Unknown" by "NA",
        ### and change column name that gets messed up during compression
        df = pd.read_table(file)
        df = (
            df.rename(columns={df.columns[0]: "sequence_name"})
            .replace("Unknown", "NA")
            .dropna(subset=["sequence_name"])
            .fillna("NA")
        )
        if not df.empty:
            ### We need the scaffold length to determine covered bases, so
            ### if that info is not available, we have to generate it from the given sequences
            if "sequence_length" not in df.columns:
                df["sequence_length"] = df["assembly_sequence"].str.len()
            ### Apply a species filter: if a species is not 2 words (contains a space),
            ### replace species value with "NA"
            #### Therefore, first get indices of species not containing a space
            idx = (
                df["species"]
                .str.contains(" ")[df["species"].str.contains(" ") == False]
                .index
            )
            #### And replace them with "NA" in the df
            df.loc[idx, "species"] = "NA"
        else:
            df["sequence_length"] = []
        ### Determine covered bases of scaffolds to aggregate information
        ### across scaffolds with similar taxonomic annotation
        df["covered_bases"] = df["sequence_length"] * df["coverage"]
        ### Cut df down to relevant columns
        df_small = df[used_ranks + ["covered_bases", "sequence_length"]]
        ### The negative controls often have no sequences = empty dfs, therefore we need to
        ### ignore them in the next step since we get errors if we use groupby on an empty df:
        if df.empty:
            df_agg = df_small
        else:
            #### Group similar taxonomy hits and sum their covered bases and sequence length:
            df_agg = (
                df_small.groupby(list(df_small.columns)[:-2])[
                    "covered_bases", "sequence_length"
                ]
                .sum()
                .reset_index()
            )
            #### Determine average per-base coverage for each taxon
            df_agg["per_base_coverage"] = (
                df_agg["covered_bases"] / df_agg["sequence_length"]
            )
            df_agg = df_agg.drop(["sequence_length", "covered_bases"], axis=1)
            #### Turn coverages into relative abundances:
            df_agg["per_base_coverage"] = (
                df_agg["per_base_coverage"] / df_agg["per_base_coverage"].sum()
            )
        ### Rename counts col
        df_agg.rename(columns={"per_base_coverage": "abundance"}, inplace=True)
        ### Add all taxa to list "all_taxa"
        all_taxa.extend(set(df_agg[groupby_rank].tolist()))
        ### Extract sample name so that we can name dfs based on their sample name
        sample_name = file.split("/")[-2]
        ### Add df_agg to the sample_dfs dic with key=sample name
        sample_dfs[sample_name] = df_agg

    # 2 Generate master df with abundance data
    ## 2.1 Drop duplicates in taxa list
    unique_taxa = list(set(all_taxa))

    ## 2.2 Generate master df with taxa from unique_taxa list as row names:
    master_df_abun = pd.DataFrame(index=pd.Index(unique_taxa))
    ### For each df in dic with key=sample:
    for sample, data in sample_dfs.items():
        #### Make a list for abundances
        abun = []
        #### For each taxon in unique taxa list:
        for taxon in unique_taxa:
            ##### If taxon is in df groupby_rank column:
            if (data[groupby_rank] == taxon).any():
                ###### Sum up the coverage of that taxon and add it to list abun:
                abun.append(data.loc[data[groupby_rank] == taxon, "abundance"].sum())
            ##### If taxon not in df groupby_rank column, add 0 to list abun:
            else:
                abun.append(0)
        #### Make a new column in master df named after the pipeline and add
        #### taxon counts for that pipeline:
        master_df_abun[sample] = abun

    # 3 Substract negative controls
    ## Generate containing number of reads per sample
    sample_reads = pd.read_table(
        readnums, header=None, index_col=0, squeeze=True
    ).to_dict()

    ## Multiply all coverages with sample read counts to substract negatives read counts
    for sample in sample_reads.keys():
        master_df_abun[sample] = master_df_abun[sample] * sample_reads[sample]

    ## Order samples by plates
    ### Plate 1 DNA samples 1A-64A & 1B-22B
    dna_samples_p1 = master_df_abun.filter(regex="A_DNA|^([1-9]|1\d|2[0-2])B_DNA")
    ### Plate 1 DNA negative controls Neg DNA 1-12
    dna_neg_p1 = master_df_abun.filter(regex="Neg([1-9]|1[0-2])_DNA")
    ### Plate 2 DNA samples 23B-64B
    dna_samples_p2 = master_df_abun.filter(regex="^(2[3-9]|[3-6]\d)B_DNA")
    ### Plate 2 DNA negative controls Neg DNA 13-17
    dna_neg_p2 = master_df_abun.filter(regex="Neg1[3-7]_DNA")
    ### Plate 3 RNA samples 1A-64A & 1B-22B
    rna_samples_p3 = master_df_abun.filter(regex="A_RNA|^([1-9]|1\d|2[0-2])B_RNA")
    ### Plate 3 RNA negative controls samples Neg RNA 1-12
    rna_neg_p3 = master_df_abun.filter(regex="Neg([1-9]|1[0-2])_RNA")
    ### Plate 4 RNA samples 23B-64B
    rna_samples_p4 = master_df_abun.filter(regex="^(2[3-9]|[3-6]\d)B_RNA")
    ### Plate 4 RNA negative controls samples Neg RNA 13-17
    rna_neg_p4 = master_df_abun.filter(regex="Neg1[3-7]_RNA")

    ## Substract and merge dfs, turn negative counts into 0, and turn back into relative abundances
    dna_p1 = (dna_samples_p1.transpose() - dna_neg_p1.transpose().sum()).transpose()
    dna_p2 = (dna_samples_p2.transpose() - dna_neg_p2.transpose().sum()).transpose()
    rna_p3 = (rna_samples_p3.transpose() - rna_neg_p3.transpose().sum()).transpose()
    rna_p4 = (rna_samples_p4.transpose() - rna_neg_p4.transpose().sum()).transpose()
    master_df_abun_sub = pd.concat([dna_p1, dna_p2, rna_p3, rna_p4], axis=1)
    master_df_abun_sub[master_df_abun_sub < 0] = 0
    master_df_abun_sub = master_df_abun_sub.div(master_df_abun_sub.sum(), axis=1)

    ## Some taxa might just occour in negative controls and equal to zero across all
    ## samples after substraction. They can be dropped.
    master_df_abun_sub = master_df_abun_sub[master_df_abun_sub.sum(axis=1) != 0]

    ## Drop columns with NaN, which means nothing was found in that sample
    master_df_abun_sub = master_df_abun_sub.dropna(axis=1)

    # 4 Save dfs
    master_df_abun_sub.to_csv(
        os.path.join(outdir, "abundances_" + groupby_rank + ".csv"), index_label="taxon"
    )
