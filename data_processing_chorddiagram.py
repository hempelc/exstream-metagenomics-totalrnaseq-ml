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
taxa_list_dic_file = "/Users/christopherhempel/Google Drive/PhD UoG/ExStream project/data_files_for_ml/taxa_lists.pickle"
NCBI_scientific_file="/Users/christopherhempel/Google Drive/PhD UoG/ExStream project/data_files_for_ml/NCBI_staxids_scientific.txt"
NCBI_non_scientific_file="/Users/christopherhempel/Google Drive/PhD UoG/ExStream project/data_files_for_ml/NCBI_staxids_non_scientific.txt"

# Define function to get dictinory keys from values
def key_from_value(dict, value):
    return list(dict.keys())[list(dict.values()).index(value)]

# Make outdir if it doesn't exist
if not os.path.exists(figure_outdir):
    os.makedirs(figure_outdir)

# Read in files
## Taxon lists
with open(taxa_list_dic_file, 'rb') as handle:
    taxa_list_dic = pickle.load(handle)
# NCBI scientific names file as dictionary
NCBI_scientific = open(NCBI_scientific_file,'r')
## Set all file content to lowercase, so that matching of the files later is not
## depending on upper- or lowercase:
NCBI_scientific_lower = (line.lower() for line in NCBI_scientific)
reader1=csv.reader(NCBI_scientific_lower, delimiter='\t')
NCBI_scientific_dict={}
for row in reader1:
	NCBI_scientific_dict[row[0]]=row[1]
# NCBI non-scientific names file as dictionary
NCBI_non_scientific = open(NCBI_non_scientific_file,'r')
## Set all file content to lowercase, so that matching of the files later is not
## depending on upper- or lowercase:
NCBI_non_scientific_lower = (line.lower() for line in NCBI_non_scientific)
reader2=csv.reader(NCBI_non_scientific_lower, delimiter='\t')
NCBI_non_scientific_dict={}
for row in reader2:
	NCBI_non_scientific_dict[row[0]]=row[1]


# Translate SILVA (16S) and UNITE (ITS) taxonomy to NCBI taxonomy
## Do for lowest and highest rank (phylum and species)
for rank in ["phylum", "species"]:
    ### Get name of all seqtypes in taxa list
    seqtypes_chord = taxa_list_dic[rank].keys()
    ### Check taxa names against NCBI and only keep if match found, replace original taxa by checked taxa
    for seqtype_chord in seqtypes_chord:
        taxon_lst = []
        for taxon in taxa_list_dic[rank][seqtype_chord]:
            if taxon.lower() in NCBI_scientific_dict:
                taxon_lst.append(taxon.lower())
            elif taxon.lower() in NCBI_non_scientific_dict:
                try:
                    translated_taxon = key_from_value(NCBI_scientific_dict, NCBI_non_scientific_dict[taxon.lower()])
                    taxon_lst.append(translated_taxon.lower())
                except:
                    taxon_lst.append(taxon.lower())
        taxa_list_dic[rank][seqtype_chord] = taxon_lst

## Manual addition - taxa that are spelled differently or not present on NCBI
## were dropped from 16S and ITS, so we add them manually
taxa_list_dic["phylum"]["otu_its_taxa"].extend(["monoblepharidomycetes", "basidiobolomycetes"])
taxa_list_dic["phylum"]["esv_its_taxa"].extend(["monoblepharidomycetes"])
taxa_list_dic["phylum"]["otu_16s_taxa"].extend(["candidatus cloacimonetes",
    "candidate division fcpu426", "candidate division lcp-89", "candidatus latescibacteria",
    "deltaproteobacteria", "candidatus eremiobacteraeota"])
taxa_list_dic["phylum"]["esv_16s_taxa"].extend(["candidate division fcpu426",
    "candidate division lcp-89", "candidatus latescibacteria", "deltaproteobacteria",
    "candidatus eremiobacteraeota"])
taxa_list_dic["species"]["otu_16s_taxa"].extend(['arcobacter suis'])
taxa_list_dic["species"]["esv_16s_taxa"].extend(['arcobacter suis'])
taxa_list_dic["species"]["esv_its_taxa"].extend(["orbilia oligospora",
    "chordomyces antarcticum", "teunia cuniculi", "helicodendron westerdijkae",
    "hongkongmyces thailandica", "articulospora tetracladia", 'monocillium sp. \'griseo-ochraceum\'',
    "neosetophoma samarorum", 'papiliotrema sp. \'fusca\'', "aspergillus penicillioides",
    "phomatospora dinemasporium", "pseudohalonectria adversaria", "thelephora alnii"])
taxa_list_dic["species"]["otu_its_taxa"].extend(["orbilia oligospora", 'blastobotrys niveus',
    '[candida] nanaspora', 'chalara heteroderae', 'chordomyces antarcticum', 'teunia cuniculi',
    'dactylaria mitrata', 'gyrophanopsis polonensis', 'helicodendron westerdijkae',
    'hongkongmyces thailandica', 'monocillium sp. \'griseo-ochraceum\''
    'alnicola alnetorum', 'neoconiothyrium hakeae', "neosetophoma samarorum",
    "aspergillus penicillioides", "phomatospora dinemasporium", 'remleria tetraspora',
    "thelephora alnii"])


# Do for lowest and highest rank (phylum and species)
for rank in ["phylum", "species"]:
    ## First, get number of overlapping taxa between seqtypes
    ### Get all possible combinations of seqtypes
    combos = list(itertools.combinations(seqtypes_chord, 2))

    ### Define number of overlapping taxa for each combo
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

    ### Turn into df and change seqtype names for consistency
    overlap_df = pd.DataFrame({"from": from_lst, "to": to_lst, "overlap": overlap_lst})
    overlap_df["from"] = overlap_df["from"].str.replace("dna_taxa", "metagenomics").str.replace("rna_taxa", "totalrnaseq")\
        .str.replace("esv_16s_taxa", "16s-esv").str.replace("esv_its_taxa", "its-esv").str.replace("otu_16s_taxa", "16s-otu").str.replace("otu_its_taxa", "its-otu")
    overlap_df["to"] = overlap_df["to"].str.replace("dna_taxa", "metagenomics").str.replace("rna_taxa", "totalrnaseq")\
        .str.replace("esv_16s_taxa", "16s-esv").str.replace("esv_its_taxa", "its-esv").str.replace("otu_16s_taxa", "16s-otu").str.replace("otu_its_taxa", "its-otu")

    ### Save for manual processing in R
    overlap_df.to_csv(os.path.join(figure_outdir, "overlap_df_{0}.csv".format(rank)), index=False)


    ### Second, get unique number of taxa for each seqtype as bubble plot
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

    ### Turn into df and change setype names for consistency
    count_df = pd.DataFrame({"seqtype": seqtype_chord_lst, "unique_taxa": unique_lst, "total_taxa": total_lst})
    count_df["seqtype"] = count_df["seqtype"].str.replace("dna_taxa", "metagenomics").str.replace("rna_taxa", "totalrnaseq")\
        .str.replace("esv_16s_taxa", "16s-esv").str.replace("esv_its_taxa", "its-esv").str.replace("otu_16s_taxa", "16s-otu").str.replace("otu_its_taxa", "its-otu")

    ### Change df format from wide to narrow
    count_df_narrow = pd.DataFrame({"seqtype": list(count_df["seqtype"])+list(count_df["seqtype"]),
        "taxa_cat": ["unique_taxa"]*len(count_df)+["total_taxa"]*len(count_df),
        "count": list(count_df["unique_taxa"])+list(count_df["total_taxa"])})

    ### Manually make reference bubbles for phylum and species:
    if rank=="phylum":
        count_df_narrow = count_df_narrow.append({'seqtype': 'ref1', 'taxa_cat': 'unique_taxa', 'count': 10}, ignore_index = True)
        count_df_narrow = count_df_narrow.append({'seqtype': 'ref2', 'taxa_cat': 'unique_taxa', 'count': 50}, ignore_index = True)
        count_df_narrow = count_df_narrow.append({'seqtype': 'ref1', 'taxa_cat': 'total_taxa', 'count': 100}, ignore_index = True)
        count_df_narrow = count_df_narrow.append({'seqtype': 'ref2', 'taxa_cat': 'total_taxa', 'count': 150}, ignore_index = True)

    elif rank=="species":
        count_df_narrow = count_df_narrow.append({'seqtype': 'ref1', 'taxa_cat': 'unique_taxa', 'count': 100}, ignore_index = True)
        count_df_narrow = count_df_narrow.append({'seqtype': 'ref2', 'taxa_cat': 'unique_taxa', 'count': 1000}, ignore_index = True)
        count_df_narrow = count_df_narrow.append({'seqtype': 'ref1', 'taxa_cat': 'total_taxa', 'count': 2500}, ignore_index = True)
        count_df_narrow = count_df_narrow.append({'seqtype': 'ref2', 'taxa_cat': 'total_taxa', 'count': 4000}, ignore_index = True)

    ### Make figure --> add bubbles to chorddiagram manually generated in R
    fig = px.scatter(count_df_narrow, x="taxa_cat", y="seqtype", size="count", text="count")
    fig.update_traces(textposition='top center')
    fig.show()
    fig.write_image(os.path.join(figure_outdir, "bubbles_{0}.svg".format(rank)))
