# this is an experiment with the maftools

require(maftools)

#laml.maf = system.file('extdata', 'tcga_laml.maf.gz', package = 'maftools') #path to TCGA LAML MAF file
#laml.clin = system.file('extdata', 'tcga_laml_annot.tsv', package = 'maftools') # clinical information containing survival information and histology. This is optional

laml = read.maf(maf = laml.maf, clinicalData = laml.clin)