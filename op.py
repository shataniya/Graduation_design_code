import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from jieba import lcut
import json

def read_file_to_array(file_path):
    f = open(file_path, "r")
    file_array = f.readlines()
    array = []
    for item in file_array:
        item = item.replace("\n", "")
        array.append(item)
    return array

mali_array = read_file_to_array("Malicious.txt")

barrage_array = read_file_to_array("barrage.txt")
# barrage_array = read_file_to_array("test.txt")
# print(barrage_array)

def cut_text(text):
    cut_array = lcut(text)
    return " ".join(cut_array)

def cut_array(text_array):
    cut = []
    for item in text_array:
        item = cut_text(item)
        cut.append(item)
    return cut

def write_file(data, filename):
    f = open(filename+".json", "w")
    f.write(json.dumps(str(data)))
    print(f"{filename}.json have built...")

barrage_cut_array = cut_array(barrage_array)
# print(barrage_cut_array)

stop_word = read_file_to_array("中文停用词表.txt")

cv = CountVectorizer(max_df=0.8, min_df=3, stop_words=frozenset(stop_word))
# cv = CountVectorizer()
vect = cv.fit_transform(barrage_cut_array)
# print(cv.get_feature_names())
# write_file(cv.get_feature_names(), "op")
# print(vect.toarray())
# write_file(vect.toarray(), "op")

# tf = TfidfVectorizer(max_df=0.8, min_df=3, stop_words=frozenset(stop_word))
# tfvect = tf.fit_transform(barrage_cut_array)
# print(tf.get_feature_names())

# test_array = read_file_to_array("test.txt")
# test_cut_array = cut_array(test_array)
# word_vector = cv.transform(test_cut_array)
# print(word_vector.toarray())

