import jiwer_my as ji


def measure_all_matrix(ground_truth_mul,hypothesis_mul):
    
    measures = ji.compute_measures(ground_truth_mul, hypothesis_mul)

    wer = measures['wer']
    mer = measures['mer']
    wil = measures['wil'] #word information lost

    wip = measures['wip'] 

    hits = measures['hits'] #are the correct word matches between reference and transcription

    substitutions = measures['substitutions']

    deletions = measures['deletions']

    insertions = measures['insertions']

    processed_truth = measures['processed_truth']

    processed_hypothesis = measures['processed_hypothesis']

    editops = measures['editops']



    
    return editops,processed_truth,processed_hypothesis, substitutions, deletions, insertions



actual_text = ['আমার পায়ে কিঞ্চিত ব্যাথা অনুভব করছি','আমার পায়ে সামান্য ব্যাথা']

predicted_text = ['আমার মাথায় কিঞ্চিত ব্যাথা অনুভব করছি','আমার পেটে সামান্য ব্যাথা এবং কোমড়ে ব্যাথা']


editops,processed_truth,processed_hypothesis, substitutions, deletions, insertions = measure_all_matrix(
    actual_text,predicted_text)

differences = list(set(actual_text)-set(predicted_text))

print("substitutions: ",type(substitutions))
print("deletions: ", deletions)
print("insertions: ",insertions)

print("Differences: ",di)

print("processed_truth: ", processed_truth)
print("processed_hypothesis: ",processed_hypothesis)

print("editops: ",editops)
