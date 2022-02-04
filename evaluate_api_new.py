import pandas as pd
import numpy as np
import jiwer


def _preprocess(text):

    """
    Preprocess a given text as: removing punctuation, multiple spaces, white spaces and convert the text into list of words

    :param text : The text to be processed
    :type text : str

    """

    bangla_punc = '''।’!()[]{};:'"\,<>./?@#$%^&*_~'''

    no_punc = text.translate(str.maketrans('', '', bangla_punc))
    # punc_removed = jiwer.RemovePunctuation()(no_punc)
    mul_space_removed = jiwer.RemoveMultipleSpaces()(no_punc)
    white_space_removed = jiwer.RemoveWhiteSpace(replace_by_space=True)(mul_space_removed)
    list_of_words = jiwer.SentencesToListOfWords(word_delimiter=" ")(white_space_removed)

    return list_of_words

def measure_all_matrix(filename, actual_text,pred_text,ground_truth,hypothesis):
    """
    Measures:
    wer : Word Error Rate
    mer : Match Error Rate
    wil : Word Information Lost
    wip : Word Information Preserved
    hits: correct word matches between reference and transcription
    substitutions : substituted words
    inserion : inserted words
    deletions: deleted words
    actual_word: ACtual words which are miss predicted
    miss_pred_words: Miss predicted words

    :param filename : audio filenam
    :param actual_text : Actual raw text
    :param pred_text : Predicted raw text
    :param ground_truth : Processed text / list of words for measurement
    :param hypothesis : processed predicted text / list of words


    :return result_dict : A dictionary contained all the results 
    """

    
    result_dict = {}

    measures = jiwer.compute_measures(
    ground_truth, hypothesis)

    result_dict['FileName'] = str(filename)
    result_dict['ActualText'] = str(actual_text)
    result_dict['ActualWords'] = str(ground_truth)


    result_dict['PredictedText'] = str(pred_text)
    result_dict['PredWords'] = str(hypothesis)

    result_dict['wer'] = float(format(measures['wer'],'.2f'))
    result_dict['mer'] = float(format(measures['mer'],'.2f'))
    result_dict['wil'] = float(format(measures['wil'],'.2f')) #word information lost

    result_dict['wip'] = float(format(measures['wip'],'.2f')) 

    result_dict['hits'] = measures['hits'] #are the correct word matches between reference and transcription

    result_dict['substitutions'] = measures['substitutions']

    result_dict['deletions'] = measures['deletions']

    result_dict['insertions'] = measures['insertions']
    
    result_dict['actual_word'] = list(set(ground_truth)-set(hypothesis))

    result_dict['miss_pred_word'] = list(set(hypothesis)-set(ground_truth))


    return result_dict

actual_label_df = pd.read_csv('./bn_bd/line_index.tsv',sep='\t')

actual_label_df.columns = ['FileName','ActualText']


# read and process predicted csv
predicted_df = pd.read_csv('./bn_bd/filename_text.csv',sep='\t',error_bad_lines=False)


# merge two dataframe
merged_df = pd.merge(actual_label_df, predicted_df, on='FileName')

print(merged_df)

final_df = pd.DataFrame()

for rows in merged_df.iterrows():
    # print(rows[1]['ActualText'])
    # print(rows[1]['PredictedText'])

    # get processed list of words for both label and prediction
    ground_truth_processed = _preprocess(rows[1]['ActualText'])

    hypothesis_processed = _preprocess(rows[1]['PredictedText'])

    # get stats
    result_dict = measure_all_matrix(rows[1]['FileName'],rows[1]['ActualText'],rows[1]['PredictedText'],
        ground_truth_processed,hypothesis_processed)

   # print(len(result_dict['differences']))

    final_df = final_df.append(result_dict, ignore_index=True)


print(final_df)

print("[INFO] Avg Word Error Rate: {:0.2f}%".format((sum(final_df['wer'])/len(final_df['wer']))*100))

final_df.to_csv('./FinalREsult_bn_bd.csv',sep='\t',encoding='utf-8',index=False)

    