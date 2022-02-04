import numpy as np  # useful for many scientific computing in Python
import pandas as pd # primary data structure library
from PIL import Image # converting images into arrays
import matplotlib.pyplot as plt # for visualizing the data
from wordcloud import WordCloud, STOPWORDS


import cv2
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from bnlp.corpus import stopwords, punctuations

def append_new_line(file_name, text_to_append):
    """Append given text as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) >=0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)

df = pd.read_csv('./FinalREsult.csv',sep='\t',error_bad_lines=False)

words = list(df['miss_pred_word'])


def clean(text):
    text = re.sub('[%s]' % re.escape(punctuations), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('\xa0', '', text)
    return text


cleaned_text = df['miss_pred_word'].apply(lambda x: clean(str(x)))
# print(cleaned_text)

refined_sentence = " ".join(cleaned_text)

print(refined_sentence)

def get_mask(img_path):
    img = cv2.imread(img_path, -1)
    if img.shape[2] == 3:
        return img
    return cv2.bitwise_not(img[:, :, 3])

mask = get_mask("./pikachu.png")
regex = r"[\u0980-\u09FF]+"
wc = WordCloud(width=800, height=400,mode="RGBA",background_color=None,colormap="hsv",   mask=mask,stopwords = stopwords,
font_path="./kalpurush.ttf",regexp=regex).generate(refined_sentence)
plt.figure(figsize=(15, 7))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()
result = wc.to_file("Bengali_word_cloud.png")







# list_of_words = []

# for i in words:
 
#     if len(i)>2:

#         bangla_punc = '[]"'
#         i = i.translate(str.maketrans('', '', bangla_punc))
#         print(i)
#         append_new_line('word_cloud.txt', i)
#         # list_of_words.append(i)




# # print(list_of_words)

# stopwords = set(STOPWORDS)
# # instantiate a word cloud object
# bangla_wc = WordCloud(
#     background_color='white',
#     max_words=2000,
#     stopwords=stopwords
# )


# alice_novel = open('word_cloud.txt', 'r').read()

# # print(alice_novel)
# print(type(alice_novel))
# # # generate the word cloud
# bangla_wc.generate(alice_novel)

# import matplotlib.pyplot as plt
# # display the word cloud
# plt.imshow(bangla_wc, interpolation='bilinear')
# plt.axis('off')
# plt.show()