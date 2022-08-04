library('circlize')
library('glue')

# Phylum or species
rank="species"

# Set in and out file names
dffile = glue("/Users/christopherhempel/Google Drive/PhD UoG/ExStream project/data_files_for_ml/overlap_df_{rank}.csv")
outfile = glue("/Users/christopherhempel/Google Drive/PhD UoG/ExStream project/exstream_data_visualization_outdir/chorddiagram_{rank}.svg")

# Read in file
df = read.csv(dffile)

# Set colors
grid.col <- c("its-esv" = "#aaa1ff", "its-otu" = "#2e7500", "16s-esv" = "#002a90",
              "16s-otu" = "#88e8a6", "metagenomics" = "#a9006b", "totalrnaseq" = "#d4631c")

# Set Chord diagram gap options
circos.par(gap.after = c(rep(10, 6)))

# Open the graphics device
svg(outfile, width=3, height=3)

# Make Chord diagram
chordDiagram(df, grid.col = grid.col, annotationTrack = c("name","grid"), 
             order = c("metagenomics", "totalrnaseq", "its-otu", "its-esv", "16s-esv", "16s-otu"))

# Close the graphics device
dev.off()

