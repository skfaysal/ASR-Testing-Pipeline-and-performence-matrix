import pandas as pd
import numpy as np
import jiwer

# measure wer
def measure_wer(ground_truth_mul,hypothesis_mul):
    

    # multiple sentences

    # ground_truth_mul = ['আমার পায়ে কিঞ্চিত ব্যাথা অনুভব করছি','আমার পায়ে সামান্য ব্যাথা']

    # hypothesis_mul = ['আমার মাথায় কিঞ্চিত ব্যাথা অনুভব করছি','আমার পেটে সামান্য ব্যাথা']

    error = jiwer.wer(ground_truth_mul, hypothesis_mul)

    print("The Word Error Rate for {} sentences is {:0.2f}%: ".format(len(ground_truth_mul), error*100))


    return error


def measure_all_matrix(ground_truth_mul,hypothesis_mul):
    
    measures = jiwer.compute_measures(
        ground_truth_mul, hypothesis_mul)

    wer = measures['wer']
    mer = measures['mer']
    wil = measures['wil'] #word information lost

    wip = measures['wip'] 

    hits = measures['hits'] #are the correct word matches between reference and transcription

    substitutions = measures['substitutions']

    deletions = measures['deletions']

    insertions = measures['insertions']

    
    return wer, mer, wil, wip, hits, substitutions, deletions, insertions



# create lists of actual texts and predicted texts
def create_lists(actual_label_df,predicted_df):
        
    actual_text = []
    predicted_text = []
    c=0
    for label in actual_label_df.iterrows():
        
        for predict in predicted_df.iterrows():
            # print("\n ACtual: ",label[1]['FileName'])
            # print("\n Predict:",predict[1]['FileName'])

            if label[1]['FileName'] == predict[1]['FileName']:
                c+=1
                print("Matching found and appending: ",c)
                actual_text.append(label[1]['Actual Text'])
                predicted_text.append(predict[1]['PredictedText'])

                break

    return actual_text,predicted_text


def word_error_matrix(ground_truth_mul,hypothesis_mul):

    actual_word = []
    miss_predict_word = []
    count_miss_pred = 0
    for act_list,pred_list in zip(ground_truth_mul,hypothesis_mul):
        # print("Actual word: ",act_list.split())
        # print("Corresponding predicted word: ",pred_list.split())
        # print("\n")

        for act_word,pred_word in zip(act_list.split(),pred_list.split()):
            if act_word != pred_word:
                print("\nThe actual word is:  {} where predicted is: {}".format(act_word,pred_word))
                actual_word.append(act_word)
                miss_predict_word.append(pred_word)
                count_miss_pred+=1


    return actual_word,miss_predict_word,count_miss_pred


def _preprocess(text):

    bangla_punc = '''।’!()[]{};:'"\,<>./?@#$%^&*_~'''
    transformation_compose = jiwer.Compose([
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.RemoveWhiteSpace(replace_by_space=True),
        jiwer.SentencesToListOfWords(word_delimiter=" ")
    ])

    no_punc = text.translate(str.maketrans('', '', bangla_punc))
    # punc_removed = jiwer.RemovePunctuation()(no_punc)
    mul_space_removed = jiwer.RemoveMultipleSpaces()(no_punc)
    white_space_removed = jiwer.RemoveWhiteSpace(replace_by_space=True)(mul_space_removed)
    list_of_words = jiwer.SentencesToListOfWords(word_delimiter=" ")(white_space_removed)


    return list_of_words



# read and process actual label csv
actual_label_df = pd.read_csv('./asr_bengali/asr_bengali_filename_text.csv',sep='\t')

actual_label_df.columns = ['FileName','Actual Text']

print(actual_label_df)

# read and process predicted csv

predicted_df = pd.read_csv('./asr_bengali/asr_predicted_text.csv',sep='\t',error_bad_lines=False)

print(predicted_df)


# get matched lists
# actual_text,predicted_text = create_lists(actual_label_df,predicted_df)

# calculate error                        
# error = measure_wer(actual_text,predicted_text)



actual_text ='আমার পায়ে কিঞ্চিত ব্যাথা অনুভব করছি'

predicted_text = 'আমার মাথায় কিঞ্চিত ব্যাথা অনুভব করছি'

transformation_compose = jiwer.Compose([
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.SentencesToListOfWords(word_delimiter=" ")
])



wer, mer, wil, wip, hits, substitutions, deletions, insertions = measure_all_matrix(
    _preprocess(actual_text),_preprocess(predicted_text))

for actual_text,predicted_text in zip(actual_text,predicted_text):

    print(actual_text,predicted_text)

    differences = list(set(_preprocess(actual_text))-set(_preprocess(predicted_text)))

    # print("...........: ",differences)

text = '৫-৬জন যুবক'

list_of_words = _preprocess(text)

print('\n[INFO]..',list_of_words)


import string

print(string.punctuation)



print("wer: ",wer)
print("mer: ", mer)
print("wil: ",wil)
print("\n")
print("wip: ",wip)
print("hits: ", hits)
print("substitutions: ",substitutions)
print("deletions: ", deletions)
print("insertions: ",insertions)

# calculate error word by word get miss pred words ans count
# actual_word,miss_predict_word,count_miss_pred = word_error_matrix(actual_text,predicted_text)

# # print("ACtual 40 words: ",actual_word)
# # print("Miss predicted 40 words: ",miss_predict_word)
# print("Total miss pred words: ",count_miss_pred)

# with open("actual_word.txt", "w") as output:
#     output.write(str(actual_word))

# with open("file.txt", "w") as output:
#     output.write(str(miss_predict_word))