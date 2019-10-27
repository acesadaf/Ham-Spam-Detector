import glob
import os
import re 
import math
import numpy as np


def word_list_maker(address):
    file_list = glob.glob(os.path.join(os.getcwd(), address, "*.txt"))
    AllFiles = []

    for file_path in file_list:
        with open(file_path, errors = 'ignore') as f_input:
            AllFiles.append(f_input.read())

    split_version = []

    for file in AllFiles:
        split_version.append(re.split('[ \n]', file))

    unique_words = []
    for current_file in split_version:
        for word in current_file:
            if(word not in unique_words):
                unique_words.append(word)

    return unique_words

def generator_for_training(address):
    file_list = glob.glob(os.path.join(os.getcwd(), address, "*.txt"))
    AllFiles = []

    for file_path in file_list:
        with open(file_path, errors = 'ignore') as f_input:
            AllFiles.append(f_input.read())
 

    split_version = []

    for file in AllFiles:
        split_version.append(re.split('[ \n]', file))
    
    return split_version

def make_input_vectors(unique_words, class1_train_files, class0_train_files):
    number_of_features = len(unique_words)
    number_of_weights = number_of_features + 1
    

    
    x_l1 = np.zeros((len(class1_train_files), number_of_weights))
    for i,file in enumerate(class1_train_files):
        x_l1[i,0] = 1
        for j, word in enumerate(unique_words):
            for fileword in file:
                if fileword == word:
                    x_l1[i, j+1] += 1
    
    x_l0 = np.zeros((len(class0_train_files), number_of_weights)) 
    for i,file in enumerate(class0_train_files):
        x_l0[i,0] = 1
        for j, word in enumerate(unique_words):
            for fileword in file:
                if fileword == word:
                    x_l0[i, j+1] += 1

    return x_l1, x_l0


def train_parameters(class1_inputs, class0_inputs, n, lam, max_iter):
    
    number_of_weights = class1_inputs.shape[1]
    num_of_class_1_inputs = class1_inputs.shape[0]
    num_of_class_0_inputs = class0_inputs.shape[0]
    w = np.zeros(number_of_weights)

    for iterator in range(0,max_iter):
        xerror = np.zeros(number_of_weights)
        for i in range(0, num_of_class_1_inputs):
            z = np.dot(w, class1_inputs[i,:])*(-1)

            P_Class1 = (1)/(1+np.exp(z))
            error = 1 - P_Class1

            xerror = xerror + (class1_inputs[i,:]*error)

        for i in range(0, num_of_class_0_inputs):
            z = np.dot(w, class0_inputs[i,:])*(-1)

            P_Class1 = (1)/(1+np.exp(z))
            error = 0 - P_Class1

            xerror = xerror + (class0_inputs[i,:]*error)

        for i in range(0, number_of_weights):
            w[i] = w[i] + n*xerror[i] - n*lam*w[i]


    return w



def test(w, unique_words, test_files, testclass):
    total = 0
    correct = 0
    for file in test_files:
        x = np.zeros(len(w))
        x[0] = 1
        for idx,word in enumerate(unique_words):
            for fileword in file:
                if fileword == word:
                    x[idx+1] += 1

        z = (np.dot(w, x))*(-1)

        if(testclass == 1):
            if(z<0):
                correct += 1
        else:
            if(z>0):
                correct += 1
    
        total += 1

    return (correct/total)


print("Logistic Regression initiated. Results will appear one by one for different parameter settings. It might take some time, please wait.")
print("-------------------------------------------------------------------------")

import warnings
warnings.filterwarnings("ignore")

    
ham_uniq_words = word_list_maker("train\ham")
spam_uniq_words = word_list_maker("train\spam")

all_unique_words = ham_uniq_words.copy()
for word in spam_uniq_words:
    if word not in all_unique_words:
        all_unique_words.append(word)


class1_train_files = generator_for_training("train\ham")
class0_train_files = generator_for_training("train\spam")
test_files1 = generator_for_training("test\ham")
test_files0 = generator_for_training("test\spam")

num_test_files1 = len(test_files1)
num_test_files0 = len(test_files0)
p_test_files1 = (num_test_files1)/(num_test_files1+num_test_files0)
p_test_files0 = (num_test_files0)/(num_test_files1+num_test_files0)


class1_inputs, class0_inputs = make_input_vectors(all_unique_words, class1_train_files, class0_train_files)




lam = [0, 0.1, 0.2, 0.5, 1, 3]
print("Here, we will keep iterations constant at 200, and change the value of lambda.")
print("-------------------------------------------------------------------------")
for i in range(0, 6):
    max_iter = 200
    result = train_parameters(class1_inputs, class0_inputs, 0.01, lam[i], max_iter)
    ham_acc = test(result, all_unique_words, test_files1, 1)
    spam_acc = test(result, all_unique_words, test_files0, 0)
    print("Ham accuracy for lambda = {} is {}%".format(lam[i], round((ham_acc*100), 2)))
    print("Spam accuracy for lambda = {} is {}%".format(lam[i], round((spam_acc*100), 2)))
    print("Overall weighted-average accuracy for lambda = {} is {}%".format(lam[i], round(((ham_acc*p_test_files1 + spam_acc*p_test_files0)*100), 2)))
    print("-----------------")

iterations = [20, 50, 100, 200, 500]
print("Here, we will keep lambda constant at 0.1, and change the number of iterations.")
print("-------------------------------------------------------------------------")
for i in range(0, 5):
    lam = 0.1
    result = train_parameters(class1_inputs, class0_inputs, 0.01, lam, iterations[i])
    ham_acc = test(result, all_unique_words, test_files1, 1)
    spam_acc = test(result, all_unique_words, test_files0, 0)
    print("Ham accuracy for {} iterations is {}%".format(iterations[i], round((ham_acc*100), 2)))
    print("Spam accuracy for {} iterations is {}%".format(iterations[i], round((spam_acc*100), 2)))
    print("Overall weighted-average accuracy for {} iterations is {}%".format(iterations[i], round(((ham_acc*p_test_files1 + spam_acc*p_test_files0)*100), 2)))
    print("-----------------")



with open('stopwords.txt', 'r') as file:
    data = file.read()

stopwords = re.split('[\n]', data)

print("Now we test with stop words removed.")
print("-------------------------------------------------------------------------")
print("Total number of unique words before removing stopwords = {}".format(len(all_unique_words)))

for stopword in stopwords:
    if stopword in all_unique_words:
        all_unique_words.remove(stopword)


print("Total number of unique words after removing stopwords = {}".format(len(all_unique_words)))
print("-------------------------------------------------------------------------")

class1_inputs, class0_inputs = make_input_vectors(all_unique_words, class1_train_files, class0_train_files)

lam = [0, 0.1, 0.2, 0.5, 1, 3]
print("Here, we will keep iterations constant at 200, and change the value of lambda. (Stopwords removed)")
print("-------------------------------------------------------------------------")
for i in range(0, 6):
    max_iter = 200
    result = train_parameters(class1_inputs, class0_inputs, 0.01, lam[i], max_iter)
    ham_acc = test(result, all_unique_words, test_files1, 1)
    spam_acc = test(result, all_unique_words, test_files0, 0)
    print("Ham accuracy for lambda = {} is {}%".format(lam[i], round((ham_acc*100), 2)))
    print("Spam accuracy for lambda = {} is {}%".format(lam[i], round((spam_acc*100), 2)))
    print("Overall weighted-average accuracy for lambda = {} is {}%".format(lam[i], round(((ham_acc*p_test_files1 + spam_acc*p_test_files0)*100), 2)))
    print("-----------------")

iterations = [20, 50, 100, 200, 500]
print("Here, we will keep lambda constant at 0.1, and change the number of iterations. (Stopwords removed)")
print("-------------------------------------------------------------------------")
for i in range(0, 5):
    lam = 0.1
    result = train_parameters(class1_inputs, class0_inputs, 0.01, lam, iterations[i])
    ham_acc = test(result, all_unique_words, test_files1, 1)
    spam_acc = test(result, all_unique_words, test_files0, 0)
    print("Ham accuracy for {} iterations is {}%".format(iterations[i], round((ham_acc*100), 2)))
    print("Spam accuracy for {} iterations is {}%".format(iterations[i], round((spam_acc*100), 2)))
    print("Overall weighted-average accuracy for {} iterations is {}%".format(iterations[i], round(((ham_acc*p_test_files1 + spam_acc*p_test_files0)*100), 2)))
    print("-----------------")



