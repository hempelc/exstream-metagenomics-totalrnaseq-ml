library('circlize')
library('glue')

rank="phylum"

dffile = glue("/Users/christopherhempel/Desktop/exstream_data_visualization_outdir/overlap_df_{rank}.csv")
outfile = glue("/Users/christopherhempel/Desktop/exstream_data_visualization_outdir/chorddiagram_{rank}.svg")

df = read.csv(dffile)

grid.col <- c("its-esv" = "#aaa1ff", "its-otu" = "#2e7500", "16s-esv" = "#002a90",
              "16s-otu" = "#88e8a6", "metagenomics" = "#a9006b", "totalrnaseq" = "#d4631c")

circos.par(gap.after = c(rep(10, 6)))

svg(outfile, width=3, height=3)

chordDiagram(df, grid.col = grid.col, annotationTrack = c("name","grid"), 
             order = c("metagenomics", "totalrnaseq", "its-otu", "its-esv", "16s-esv", "16s-otu"))

# Close the graphics device
dev.off()

