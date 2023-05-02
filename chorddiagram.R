library('circlize')
library('glue')

# Phylum or species/genus
rank="phylum"

# Set in and out file names
dffile = glue("/Users/christopherhempel/Google Drive/PhD UoG/ExStream project/data_files_for_ml/overlap_df_{rank}.csv")
outfile = glue("/Users/christopherhempel/Google Drive/PhD UoG/ExStream project/exstream_data_visualization_outdir/chorddiagram_{rank}.svg")

# Read in file
## Note: sort df manually before loading it in so that self-connections always come first
df = read.csv(dffile)

# Set colors
grid.col <- c("ITS-2 ESV" = "#aaa1ff", "ITS-2 OTU" = "#2e7500", "16S ESV" = "#002a90",
              "16S OTU" = "#88e8a6", "Metagenomics" = "#a9006b", "Total RNA-Seq" = "#d4631c")

# Set Chord diagram gap options
circos.par(gap.after = c(rep(10, 6)))

# Open the graphics device
svg(outfile, width=3, height=3)

# Make Chord diagram
chordDiagram(df, grid.col = grid.col, self.link = 1, link.sort = "asis",   
             order = c("Metagenomics", "Total RNA-Seq", "ITS-2 OTU", "ITS-2 ESV", "16S ESV", "16S OTU"), annotationTrack = c("grid"), annotationTrackHeight = mm_h(5))

for(si in get.all.sector.index()) {
  xlim = get.cell.meta.data("xlim", sector.index = si, track.index = 1)
  ylim = get.cell.meta.data("ylim", sector.index = si, track.index = 1)
  circos.text(mean(xlim), mean(ylim), si, sector.index = si, track.index = 1, 
              facing = "bending.inside", niceFacing = TRUE, col = "white")
}

# Close the graphics device
dev.off()

