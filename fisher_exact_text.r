dataset <- read.delim("~/Desktop/MSc Bioinformatics & Systems Biology/B4FTM/CATS/dataset.txt") #read file
subgroups <- dataset$Subgroup #factor of 3 levels, 1 per subtype
table1<-table(subgroups, dataset$X1) #example of contingency table
dataset_size<-dim(dataset)
pvalue<-c() #initialize vector

#loop to generate contingency table per feature and save pvalues in vector
for (i in 2:(length(dataset)-1)) {
  factor_chrom <- as.factor(dataset[,i])
  contingency_table<-table(subgroups, factor_chrom)
  fisher<- fisher.test(contingency_table)
  p_value <- fisher$p.value
  pvalue[i]<- p_value
}

#generating the ranking
columns<-colnames(dataset[,-length(dataset)]) #vector of column names (features)
chrom_pvalue<- data.frame(columns, pvalue) #dataframe of pvalue next to feature
ranking<-chrom_pvalue[order(pvalue), ] #ascendent order of pvalues
write.csv(ranking, file="./ranking.csv")
