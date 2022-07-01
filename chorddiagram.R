library('circlize')

dffile = "filepath"
outfile = "outpath"

df = data.frame(from = c("its-esv", "its-otu", "16s-esv", "16s-otu", "metagenomics", "totalrnaseq", "its-otu", "16s-esv", "16s-otu", "metagenomics", "totalrnaseq", "16s-esv", "16s-otu", "metagenomics", "totalrnaseq", "16s-otu", "metagenomics", "totalrnaseq", "metagenomics", "totalrnaseq", "totalrnaseq"),
                to = c(rep("its-esv", 6), rep("its-otu", 5), rep("16s-esv", 4), rep("16s-otu", 3), rep("metagenomics", 2), rep("totalrnaseq", 1)),
                value = c(20, 2, 3, 4, 5, 6, 20, 2, 3, 4,5,20,2,3,4,20,2,3,20,2,20))

grid.col <- c("its-esv" = "#aaa1ff", "its-otu" = "#2e7500", "16s-esv" = "#002a90",
              "16s-otu" = "#88e8a6", "metagenomics" = "#a9006b", "totalrnaseq" = "#d4631c")

circos.par(gap.after = c(rep(10, 6)))

svg(outfile)

chordDiagram(df, grid.col = grid.col, annotationTrack = c("name","grid"))

# Close the graphics device
dev.off()
