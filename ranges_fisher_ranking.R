#read file
f = "transposed_dataset.txt"
dataset = read.delim(f)

# set ranges
range_1 = 1:20
range_2 = 21:40
range_3 = 41:60
range_4 = 61:80
range_5 = 81:100

start = c(1, 21, 41, 60, 80)
end = c(20, 40, 60, 80, 100)

training_sets = list(range_1, range_2, range_3, range_4, range_5)

for (range in 1:length(training_sets)){
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
  outpath = paste("Kfold", "/ranking_", range, ".csv", sep = "")
  print(outpath)
  write.csv(ranking, file = outpath, row.names = FALSE)
}
print("end range loop")
