library(dplyr)
#read file
data = "transposed_dataset.txt"
dataset = read.delim(data)

s_file = "shuffled_rows_df.txt"
shuffles = read.delim(s_file)
shuffles = shuffles[,2:51]

for (i in 1:50){
  # set ranges
  range_1 = 1:33
  range_2 = 34:66
  range_3 = 67:100
  
  start = c(1, 34, 67)
  end = c(33, 66, 100)
  
  folds = list(range_1, range_2, range_3)
  
  #reorder dataset 
  shuf = shuffles[,i]
  dataset = dataset[match(shuf, dataset$X),]
  
  for (range in 1:length(folds)){
    print("entered range loop")
    #rows = training_sets[range]}
    print("made rows var")
    subset = dataset[start[range]:end[range],]
    print("made subset")
    pvalues = c()
    subgroups = subset$Subgroup
    trash = c()
    
    for (feature in 2:(length(subset)-1)){
      print("entered feature loop")
      #print(subset)
      #print(feature)
      calls = as.factor(subset[,feature])
      contingency_table = table(subgroups, calls)
      if(dim(contingency_table)[2] == 1){
        print("get rekt lmao")
        trash = c(trash,feature)
        next
      }
     
      fisher = fisher.test(contingency_table)
      p = fisher$p.value
      pvalues = c(pvalues, p)
    }
    print("end feature loop")
    all_features = colnames(subset[,2:2835])
    if(length(trash)>0){all_features = all_features[-trash]}
    features_pvalues = data.frame(all_features, pvalues)
    colnames(features_pvalues) = c("feature", "pvalue")
    ranking = features_pvalues[order(features_pvalues$pvalue),]
    #savedir = dir.create("Kfold")
    outpath = paste("Kfold", "/", i, "_ranking_", range, ".csv", sep = "")
    print(outpath)
    write.csv(ranking, file = outpath, row.names = FALSE)
  }
  print("end range loop")
}