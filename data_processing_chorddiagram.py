# The pickle object has been genereated under python >=3.8 on a different machine,
# while the processing machine has python <3.8. To read in the pickle object generated
# on a higher pickle version, we need to import a specific pickle package
import pickle5 as pickle
import pandas as pd
import os
import csv
import itertools
import plotly.express as px

# Set variables
figure_outdir = "/Users/christopherhempel/Google Drive/PhD UoG/ExStream project/exstream_data_visualization_outdir"
indir = (
    "/Users/christopherhempel/Google Drive/PhD UoG/ExStream project/data_files_for_ml"
)
taxa_list_dict_file = "taxa_lists.pickle"
NCBI_scientific_file = "NCBI_staxids_scientific.txt"
NCBI_non_scientific_file = "NCBI_staxids_non_scientific.txt"
nodes_dmp_file = "nodes.dmp"
division_dmp_file = "division.dmp"


# Define function to get dictinory keys from values
def key_from_value(dict, value):
    return list(dict.keys())[list(dict.values()).index(value)]


# Make outdir if it doesn't exist
if not os.path.exists(figure_outdir):
    os.makedirs(figure_outdir)

# Read in files
## Taxon lists
with open(os.path.join(indir, taxa_list_dict_file), "rb") as handle:
    taxa_list_dict = pickle.load(handle)
### NCBI scientific names file as dictionary
NCBI_scientific = open(os.path.join(indir, NCBI_scientific_file), "r")
#### Set all file content to lowercase, so that matching of the files later is not
#### depending on upper- or lowercase:
NCBI_scientific_lower = (line.lower() for line in NCBI_scientific)
reader1 = csv.reader(NCBI_scientific_lower, delimiter="\t")
NCBI_scientific_dict = {}
for row in reader1:
    NCBI_scientific_dict[row[0]] = row[1]
### NCBI non-scientific names file as dictionary
NCBI_non_scientific = open(os.path.join(indir, NCBI_non_scientific_file), "r")
#### Set all file content to lowercase, so that matching of the files later is not
#### depending on upper- or lowercase:
NCBI_non_scientific_lower = (line.lower() for line in NCBI_non_scientific)
reader2 = csv.reader(NCBI_non_scientific_lower, delimiter="\t")
NCBI_non_scientific_dict = {}
for row in reader2:
    NCBI_non_scientific_dict[row[0]] = row[1]

## Files for counting
nodes_df = pd.read_csv(
    os.path.join(indir, nodes_dmp_file),
    sep="\t",
    usecols=[0, 8],
    names=["tax_id", "division_id"],
    header=None,
    dtype=str,
)
division_df = pd.read_csv(
    os.path.join(indir, division_dmp_file),
    sep="\t",
    header=None,
    usecols=[0, 4],
    names=["division_id", "division"],
    dtype=str,
)
tax_id_division_df = pd.merge(left=nodes_df, right=division_df, on="division_id").drop(
    "division_id", axis=1
)


# Translate SILVA (16S) and UNITE (ITS) taxonomy to NCBI taxonomy
for rank in ["phylum", "genus", "species"]:
    ### Get name of all seqtypes in taxa list
    seqtypes_chord = taxa_list_dict[rank].keys()
    ### Check taxa names against NCBI and only keep if match found, replace original taxa by checked taxa.
    ### Also keep taxids so that domains can be summarized later
    for seqtype_chord in seqtypes_chord:
        taxon_dic = {}
        for taxon in taxa_list_dict[rank][seqtype_chord]:
            if taxon.lower() in NCBI_scientific_dict:
                taxon_dic[taxon.lower()] = NCBI_scientific_dict[taxon.lower()]
            elif taxon.lower() in NCBI_non_scientific_dict:
                #### There are two phyla that have non-scientific names without
                #### a taxid in the scientific file, so we are replacing the two
                #### phyla names manually with the scientific ones
                try:
                    translated_taxon = key_from_value(
                        NCBI_scientific_dict, NCBI_non_scientific_dict[taxon.lower()]
                    ).lower()
                except:
                    if taxon.lower() == "gemmatimonadota":
                        translated_taxon = "gemmatimonadetes"
                    elif taxon.lower() == "elusimicrobiota":
                        translated_taxon = "elusimicrobia"
                taxon_dic[translated_taxon] = NCBI_non_scientific_dict[taxon.lower()]
        #### Remove all candidate taxa for uniformity
        taxon_dict = {
            k: v
            for k, v in taxon_dic.items()
            if "candidatus" not in k
            if "candidate" not in k
            if "[candida]" not in k
        }
        taxa_list_dict[rank][seqtype_chord] = taxon_dict


# Identify taxa ids with no division
no_ids_df = pd.DataFrame(columns=["rank", "dataset", "ids"])
for rank in ["phylum", "genus", "species"]:
    for seqtype_chord in taxa_list_dict[rank].keys():
        ## Assign divisions to all taxa
        taxid_df_tmp = (
            pd.DataFrame(taxa_list_dict[rank][seqtype_chord], index=["tax_id"])
            .transpose()
            .reset_index()
            .drop("index", axis=1)
        )
        tax_id_division_df_tmp = pd.merge(
            left=taxid_df_tmp, right=tax_id_division_df, on="tax_id", how="left"
        )
        ## Save rows with NA
        no_ids_df_tmp = tax_id_division_df_tmp[
            tax_id_division_df_tmp["division"].isna()
        ]
        no_ids_df.loc["{}_{}".format(rank, seqtype_chord)] = [
            rank,
            seqtype_chord,
            list(no_ids_df_tmp["tax_id"]),
        ]


# Manual addition - taxa that are spelled differently or not present on NCBI
# were dropped from 16S and ITS, so we add them manually.
# We also manually add taxa that have no division
## Phylum
taxa_list_dict["phylum"]["otu_its_taxa"] = {
    **taxa_list_dict["phylum"]["otu_its_taxa"],
    "monoblepharidomycetes": "451454",
    "basidiobolomycetes": "1399768",
}
taxa_list_dict["phylum"]["esv_its_taxa"]["monoblepharidomycetes"] = "451454"
taxa_list_dict["phylum"]["esv_16s_taxa"] = {
    **taxa_list_dict["phylum"]["esv_16s_taxa"],
    "deltaproteobacteria": "28221",
    "thermodesulfobacteria": "200940",
}
taxa_list_dict["phylum"]["otu_16s_taxa"]["deltaproteobacteria"] = "28221"

## Genus
taxa_list_dict["genus"]["otu_16s_taxa"]["arcobacter"] = "28196"
taxa_list_dict["genus"]["esv_16s_taxa"]["arcobacter"] = "28196"
taxa_list_dict["genus"]["esv_its_taxa"] = {
    **taxa_list_dict["genus"]["esv_its_taxa"],
    "orbilia": "47022",
    "chordomyces": "1654916",
    "teunia": "2724800",
    "helicodendron": "319381",
    "hongkongmyces": "1545378",
    "articulospora": "253308",
    "monocillium": "581188",
    "neosetophoma": "798068",
    "papiliotrema": "189450",
    "aspergillus": "5052",
    "phomatospora": "189357",
    "pseudohalonectria": "40655",
    "thelephora": "56489",
}
taxa_list_dict["genus"]["otu_its_taxa"] = {
    **taxa_list_dict["genus"]["otu_its_taxa"],
    "orbilia": "47022",
    "blastobotrys": "43971",
    "chalara": "13000",
    "chordomyces": "1654916",
    "teunia": "2724800",
    "dactylaria": "47274",
    "gyrophanopsis": "2789184",
    "helicodendron": "319381",
    "hongkongmyces": "1545378",
    "monocillium": "581188",
    "alnicola": "91662",
    "neoconiothyrium": "2093505",
    "neosetophoma": "798068",
    "aspergillus": "5052",
    "phomatospora": "189357",
    "remleria": "1776023",
    "thelephora": "56489",
    "graphium": "1503932",
    "ellisembia": "375389",
}
taxa_list_dict["genus"]["dna_taxa"]["anceyoconcha"] = "2876047"
taxa_list_dict["genus"]["rna_taxa"]["anceyoconcha"] = "2876047"

## Species
taxa_list_dict["species"]["otu_16s_taxa"]["arcobacter suis"] = "1278212"
taxa_list_dict["species"]["esv_16s_taxa"]["arcobacter suis"] = "1278212"
taxa_list_dict["species"]["esv_its_taxa"] = {
    **taxa_list_dict["species"]["esv_its_taxa"],
    "orbilia oligospora": "2813651",
    "chordomyces antarcticum": "1525356",
    "teunia cuniculi": "365330",
    "helicodendron westerdijkae": "409251",
    "hongkongmyces thailandica": "2059664",
    "articulospora tetracladia": "253309",
    "monocillium sp. 'griseo-ochraceum'": "2075169",
    "neosetophoma samarorum": "798069",
    "papiliotrema sp. 'fusca'": "192649",
    "aspergillus penicillioides": "41959",
    "phomatospora dinemasporium": "2718578",
    "pseudohalonectria adversaria": "1953033",
    "thelephora alnii": "56491",
    "niesslia mucida": "3029106",
}
taxa_list_dict["species"]["otu_its_taxa"] = {
    **taxa_list_dict["species"]["otu_its_taxa"],
    "orbilia oligospora": "2813651",
    "blastobotrys niveus": "44073",
    "chalara heteroderae": "2981829",
    "chordomyces antarcticum": "1525356",
    "teunia cuniculi": "365330",
    "dactylaria mitrata": "228885",
    "gyrophanopsis polonensis": "2789189",
    "helicodendron westerdijkae": "409251",
    "hongkongmyces thailandica": "2059664",
    "monocillium sp. 'griseo-ochraceum'": "2075169",
    "alnicola alnetorum": "242214",
    "neoconiothyrium hakeae": "2093506",
    "neosetophoma samarorum": "798069",
    "aspergillus penicillioides": "41959",
    "phomatospora dinemasporium": "2718578",
    "remleria tetraspora": "1776023",
    "thelephora alnii": "56491",
    "niesslia mucida": "3029106",
}
taxa_list_dict["species"]["dna_taxa"]["streptomyces noursei"] = "1971"
taxa_list_dict["species"]["rna_taxa"]["streptomyces noursei"] = "1971"


# Count how many bacteria, archaea, and eukaryotes are in each dataset
# Therefore, use the nodes.dmp and division.dmp from NCBI taxdump
division_count_df = pd.DataFrame(columns=["rank", "dataset", "division", "count"])
for rank in ["phylum", "genus", "species"]:
    for seqtype_chord in taxa_list_dict[rank].keys():
        ## Assign divisions to all taxa
        taxid_df_tmp = (
            pd.DataFrame(taxa_list_dict[rank][seqtype_chord], index=["tax_id"])
            .transpose()
            .reset_index()
            .drop("index", axis=1)
        )
        tax_id_division_df_tmp = pd.merge(
            left=taxid_df_tmp, right=tax_id_division_df, on="tax_id", how="left"
        )
        ## Exclude rows with NA
        tax_id_division_df_tmp = tax_id_division_df_tmp[
            ~tax_id_division_df_tmp["division"].isna()
        ]
        ## Count divisions in each dataset and add info to df
        for div in division_df["division"]:
            counter = tax_id_division_df_tmp["division"].str.count(div).sum()
            division_count_df.loc["{}_{}_{}".format(rank, seqtype_chord, div)] = [
                rank,
                seqtype_chord,
                div,
                counter,
            ]
division_count_df.to_csv(
    os.path.join(indir, "division_counts.csv".format(rank)), index=False
)


# Do the following for lowest and highest rank (phylum and species/genus)
for rank in ["phylum", "genus", "species"]:
    ## First, get number of overlapping taxa between seqtypes
    ### Get all possible combinations of seqtypes
    combos = list(itertools.combinations(seqtypes_chord, 2))

    ### Define number of overlapping taxa for each combo
    from_lst = []
    to_lst = []
    overlap_lst = []
    for i in range(len(combos)):
        set1 = set(taxa_list_dict[rank][combos[i][0]])
        set2 = set(taxa_list_dict[rank][combos[i][1]])
        overlap = len(set1 & set2)
        from_lst.append(combos[i][0])
        to_lst.append(combos[i][1])
        overlap_lst.append(overlap)

    ### Turn into df and change seqtype names for consistency
    overlap_df = pd.DataFrame({"from": from_lst, "to": to_lst, "overlap": overlap_lst})
    overlap_df = overlap_df.sort_values(["from", "overlap"], ascending=False)

    ### Second, get total and unique number of taxa for each seqtype
    overlap_df_unique = pd.DataFrame(columns=["from", "to", "overlap"])
    unique_lst = []
    total_lst = []
    seqtype_chord_lst = []
    for seqtype_chord in seqtypes_chord:
        seqtype_chord_lst.append(seqtype_chord)
        total_taxa = len(taxa_list_dict[rank][seqtype_chord])
        total_lst.append(total_taxa)
        other_seqtypes = [x for x in seqtypes_chord if x != seqtype_chord]
        other_taxa = []
        for other_seqtype in other_seqtypes:
            other_taxa.extend(taxa_list_dict[rank][other_seqtype])
        other_taxa = set(other_taxa)
        unique_taxa = len(
            [x for x in taxa_list_dict[rank][seqtype_chord] if x not in other_taxa]
        )
        unique_lst.append(unique_taxa)
        nonunique_taxa = total_taxa - unique_taxa
        # Determine factor by which unique taxa have to be multiplied for size ratio
        overlap_sum = overlap_df[
            overlap_df.apply(
                lambda r: r.str.contains(seqtype_chord, case=False).any(), axis=1
            )
        ]["overlap"].sum()
        size_factor = overlap_sum / nonunique_taxa
        unique_taxa = unique_taxa * size_factor
        # Add to overlap df unique
        overlap_df_unique.loc[len(overlap_df_unique)] = [
            seqtype_chord,
            seqtype_chord,
            unique_taxa,
        ]

    ## Organize df
    overlap_df = pd.concat([overlap_df_unique, overlap_df])
    overlap_df["from"] = (
        overlap_df["from"]
        .str.replace("dna_taxa", "Metagenomics")
        .str.replace("rna_taxa", "Total RNA-Seq")
        .str.replace("esv_16s_taxa", "16S ESV")
        .str.replace("esv_its_taxa", "ITS-2 ESV")
        .str.replace("otu_16s_taxa", "16S OTU")
        .str.replace("otu_its_taxa", "ITS-2 OTU")
    )
    overlap_df["to"] = (
        overlap_df["to"]
        .str.replace("dna_taxa", "Metagenomics")
        .str.replace("rna_taxa", "Total RNA-Seq")
        .str.replace("esv_16s_taxa", "16S ESV")
        .str.replace("esv_its_taxa", "ITS-2 ESV")
        .str.replace("otu_16s_taxa", "16S OTU")
        .str.replace("otu_its_taxa", "ITS-2 OTU")
    )

    ### Save for manual processing in R
    overlap_df.to_csv(
        os.path.join(indir, "overlap_df_{0}.csv".format(rank)), index=False
    )

    ### Finally, make a bubble plot for number of taxa
    #### Turn count lists df and change setype names for consistency
    count_df = pd.DataFrame(
        {
            "seqtype": seqtype_chord_lst,
            "unique_taxa": unique_lst,
            "total_taxa": total_lst,
        }
    )
    count_df["seqtype"] = (
        count_df["seqtype"]
        .str.replace("dna_taxa", "metagenomics")
        .str.replace("rna_taxa", "totalrnaseq")
        .str.replace("esv_16s_taxa", "16s-esv")
        .str.replace("esv_its_taxa", "its-esv")
        .str.replace("otu_16s_taxa", "16s-otu")
        .str.replace("otu_its_taxa", "its-otu")
    )

    #### Change df format from wide to narrow
    count_df_narrow = pd.DataFrame(
        {
            "seqtype": list(count_df["seqtype"]) + list(count_df["seqtype"]),
            "taxa_cat": ["unique_taxa"] * len(count_df)
            + ["total_taxa"] * len(count_df),
            "count": list(count_df["unique_taxa"]) + list(count_df["total_taxa"]),
        }
    )

    #### Manually make reference bubbles for phylum and species:
    if rank == "phylum":
        count_df_narrow = count_df_narrow.append(
            {"seqtype": "ref1", "taxa_cat": "unique_taxa", "count": 10},
            ignore_index=True,
        )
        count_df_narrow = count_df_narrow.append(
            {"seqtype": "ref2", "taxa_cat": "unique_taxa", "count": 50},
            ignore_index=True,
        )
        count_df_narrow = count_df_narrow.append(
            {"seqtype": "ref1", "taxa_cat": "total_taxa", "count": 100},
            ignore_index=True,
        )
        count_df_narrow = count_df_narrow.append(
            {"seqtype": "ref2", "taxa_cat": "total_taxa", "count": 150},
            ignore_index=True,
        )

    elif rank == "species":
        count_df_narrow = count_df_narrow.append(
            {"seqtype": "ref1", "taxa_cat": "unique_taxa", "count": 100},
            ignore_index=True,
        )
        count_df_narrow = count_df_narrow.append(
            {"seqtype": "ref2", "taxa_cat": "unique_taxa", "count": 1000},
            ignore_index=True,
        )
        count_df_narrow = count_df_narrow.append(
            {"seqtype": "ref1", "taxa_cat": "total_taxa", "count": 2500},
            ignore_index=True,
        )
        count_df_narrow = count_df_narrow.append(
            {"seqtype": "ref2", "taxa_cat": "total_taxa", "count": 4000},
            ignore_index=True,
        )

    elif rank == "genus":
        count_df_narrow = count_df_narrow.append(
            {"seqtype": "ref1", "taxa_cat": "unique_taxa", "count": 100},
            ignore_index=True,
        )
        count_df_narrow = count_df_narrow.append(
            {"seqtype": "ref2", "taxa_cat": "unique_taxa", "count": 1000},
            ignore_index=True,
        )
        count_df_narrow = count_df_narrow.append(
            {"seqtype": "ref1", "taxa_cat": "total_taxa", "count": 2000},
            ignore_index=True,
        )
        count_df_narrow = count_df_narrow.append(
            {"seqtype": "ref2", "taxa_cat": "total_taxa", "count": 3000},
            ignore_index=True,
        )

    #### Make figure --> add bubbles to chorddiagram manually generated in R
    fig = px.scatter(
        count_df_narrow, x="taxa_cat", y="seqtype", size="count", text="count"
    )
    fig.update_traces(textposition="top center")
    fig.show()
    fig.write_image(os.path.join(figure_outdir, "bubbles_{0}.svg".format(rank)))
