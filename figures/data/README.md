# TCGA pathway inputs

`tcga_pancan_gene_ids.txt` is the ordered identifier column recovered from
`EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena`, downloaded from the
TCGA PanCanAtlas Xena hub on 2026-10-26.  It has 20,531 identifiers after the
header and is used only to annotate the unchanged columns in the UCI-derived
expression matrix.

`reactome_pathways_2026-06-21.gmt` is Reactome's downloaded pathway GMT file.
The analysis tests every human Reactome pathway with 15--100 uniquely matched
genes.  This size rule is fixed before examining BRCA-versus-LUAD results.

The analysis output records SHA-256 checksums for both files.
