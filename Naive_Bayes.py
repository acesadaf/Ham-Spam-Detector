import glob
import os
import re 
import math

def word_list_maker(address):
    file_list = glob.glob(os.path.join(os.getcwd(), address, "*.txt"))
    AllFiles = []

    for file_path in file_list:
        with open(file_path, errors = 'ignore') as f_input:
            AllFiles.append(f_input.read())

    split_version = []

    for file in AllFiles:
        split_version.append(re.split('[ \n]', file))

    word_counter = {}
    unique_words = []
    for current_file in split_version:
        for word in current_file:
            if(word not in unique_words):
                unique_words.append(word)
                word_counter[word] = 1
            else:
                val = word_counter.get(word)
                val+=1
                word_counter.update({word:val})

    return len(file_list), unique_words, word_counter

def generator_for_testing(address):
    file_list = glob.glob(os.path.join(os.getcwd(), address, "*.txt"))
    AllFiles = []

    for file_path in file_list:
        with open(file_path, errors = 'ignore') as f_input:
            AllFiles.append(f_input.read())
 

    split_version = []

    for file in AllFiles:
        split_version.append(re.split('[ \n]', file))
    
    return split_version


def NaiveBayes(classes, number_of_class_i, dictionary_class_i, test_file):
    total = sum(number_of_class_i)
    prior_of_class = [] #will store the prior probability of class i.
    total_words_in_class = [] #will store the total number of words in class i.
    unique_words_in_class = [] #Will store nested lists, each of which contains all the unique words in class i.
    number_of_unique_words_in_class = [] #Will store the number of unique words in class i
    posterior = {} #will store the posterior probability of class i given a certain test_file. This is the output
    
    
    for dictionary in dictionary_class_i:
        unique_words_in_class.append(list(dictionary.keys()))
        number_of_unique_words_in_class.append(len(list(dictionary.keys())))
        total_words_in_class.append(sum(list(dictionary.values())))
        
    for idx, val in enumerate(number_of_class_i):
        prior_of_class.append(math.log2(val/total))

    for idx, prior in enumerate(prior_of_class):
        posterior[classes[idx]] = prior #initializing the posterior with the priors.
    
    for word in test_file:
        for idx, dictionary in enumerate(dictionary_class_i):
            if word in unique_words_in_class[idx]:
                laplaced_probability = ((dictionary.get(word) + 1) / (total_words_in_class[idx] + number_of_unique_words_in_class[idx]))
            else:
                laplaced_probability = 1 / (total_words_in_class[idx] + number_of_unique_words_in_class[idx])
            
            value = posterior.get((classes[idx]))
            updated_value = value + math.log2(laplaced_probability)
            posterior[classes[idx]] = updated_value

    return posterior


def tester(classes, numbers_of_class, dicts_of_class, address, test_class):
    test_files = generator_for_testing(address)
    total = 0
    correct = 0
    for test_file in test_files:
        results = NaiveBayes(classes, numbers_of_class, dicts_of_class, test_file)
        maximum = max(results.values())
        for key, value in results.items():
            if value == maximum:
                maxkey = key

        if maxkey == test_class:
            correct += 1
        total += 1

    print("Accuracy for {} = {}%".format(test_class, round((correct/total)*100, 2)))
    return len(test_files), (correct/total)


number_of_ham, ham_uniq_words, ham_word_counter = word_list_maker("train\ham")
number_of_spam, spam_uniq_words, spam_word_counter = word_list_maker("train\spam")


numbers_of_class = []
dicts_of_class = []
numbers_of_class.append(number_of_ham)
numbers_of_class.append(number_of_spam)
dicts_of_class.append(ham_word_counter)
dicts_of_class.append(spam_word_counter)
classes = ["ham", "spam"]


print("Naive Bayes Classifier for ham and spam")
print("-----------------------------")
no_of_test_files1, ham_acc = tester(classes, numbers_of_class, dicts_of_class, "test\ham", "ham")
no_of_test_files0, spam_acc = tester(classes, numbers_of_class, dicts_of_class, "test\spam", "spam")
weighted_average = ((no_of_test_files1*ham_acc) + (no_of_test_files0*spam_acc))/(no_of_test_files0 + no_of_test_files1) 
print('Weighted average is {}%'.format(round(weighted_average*100, 2)))
    
with open('stopwords.txt', 'r') as file:
    data = file.read()


stopwords = re.split('[\n]', data)

for stopword in stopwords:
    if stopword in dicts_of_class[0]:
        del dicts_of_class[0][stopword]
    if stopword in dicts_of_class[1]:
        del dicts_of_class[1][stopword]

print("With stopwords removed, we now get:")
no_of_test_files1, ham_acc = tester(classes, numbers_of_class, dicts_of_class, "test\ham", "ham")
no_of_test_files0, spam_acc = tester(classes, numbers_of_class, dicts_of_class, "test\spam", "spam")
weighted_average = ((no_of_test_files1*ham_acc) + (no_of_test_files0*spam_acc))/(no_of_test_files0 + no_of_test_files1) 
print('Weighted average is {}%'.format(round(weighted_average*100, 2)))


    






