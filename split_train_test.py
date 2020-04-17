import numpy

# iteration times can be changed, I just set it 3
ITERATION_TIME = 3

file = open('dataset.txt')
input = file.readlines()

######### use the first column to choose the train and test group
# put the first column in a list
Array_list=[]
for line in input:
    line = line.split('\t')
    Array_list.append(line[0])
Array_list.remove('')
# change list to array
Array_list= numpy.array(Array_list)

######### begin iteration to get several train and test group
for i in range(1,ITERATION_TIME+1):
    train_i_str = 'Train' + str(i)
    train_filename = train_i_str + '.txt'
    f_train = open(train_filename, 'w')
    test_i_str = 'Test' + str(i)
    test_filename = test_i_str + '.txt'
    f_test = open(test_filename,'w')

    # have a randomly permute index, x.shape[0] means the column numbers
    indixs = numpy.random.permutation(Array_list.shape[0])
    # choose the 80% train 20% test by random to have the index
    train_idx, test_idx = indixs[:80], indixs[80:]
    # take the train and test group of the chosen index
    train, test = Array_list[train_idx,], Array_list[test_idx,]

    # change array to list
    train = train.tolist()
    test = test.tolist()

    # write the first line to the train.txt and test.txt
    f_test.write(input[0])
    f_train.write(input[0])

    # put the train and test group to txt files
    for line in input:
        line2 = line.split('\t')
        if line2[0] in train:
            f_train.write(line)

        elif line2[0] in test:
            f_test.write(line)

    f_train.close()
    f_test.close()










