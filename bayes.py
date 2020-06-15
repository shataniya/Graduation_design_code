import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from jieba import lcut
import json
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import MLPClassifier
import numpy as np
import statistics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

# def write_file(data, filename):
#     f = open(filename+".json", "w")
#     f.write(json.dumps(str(data)))
#     print(f"{filename}.json have built...")

barrage_cut_array = cut_array(barrage_array)
# print(barrage_cut_array)
mali_cut_array = cut_array(mali_array)

stop_word = read_file_to_array("中文停用词表.txt")

cv = CountVectorizer(max_df=0.8, min_df=3, stop_words=frozenset(stop_word))


def anls(pos, neg):
    target = []
    data = []
    for item in pos:
        target.append(1)
        data.append(item)
    for item in neg:
        target.append(0)
        data.append(item)
    return data, target

# data, target = anls(barrage_cut_array, mali_cut_array)
X_train, y_train = anls(barrage_cut_array, mali_cut_array)

# print(len(data), len(target))

vect = cv.fit_transform(X_train)
# print(cv.get_feature_names())
# write_file(cv.get_feature_names(), "op")
# print(cv.vocabulary_)
# print(vect.toarray())

X_train = vect.toarray()

# 测试集
pos_array = read_file_to_array("pos_test.txt")
neg_array = read_file_to_array("neg_test.txt")
pos_cut_array = cut_array(pos_array)
neg_cut_array = cut_array(neg_array)

X_test, y_test = anls(pos_cut_array, neg_cut_array)
X_test = cv.transform(X_test).toarray()

# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=0)
# print(X_train, X_test)

nb = MultinomialNB()
bn = BernoulliNB()
gu = GaussianNB()

# svm
svc = svm.SVC(probability=True)
linersvc = svm.SVC(kernel="linear", probability=True)

# neural_network
mlp = MLPClassifier()

# rio = datasets.load_iris()
# print(rio.data)

nb.fit(X_train, y_train)
bn.fit(X_train, y_train)
gu.fit(X_train, y_train)
# score = nb.score(X_test, y_test)
# print(score)

# svm
# svc.fit(X_train, y_train)
# linersvc.fit(X_train, y_train)

# neural_network
# mlp.fit(X_train, y_train)

proba = nb.predict_proba(X_test)
bn_proba = bn.predict_proba(X_test)
gu_proba = gu.predict_proba(X_test)
# print(proba)
# print(y_test)

# svc_proba = svc.predict_proba(X_test)
# print(svc_proba)
# linersvc_proba = linersvc.predict_proba(X_test)

# neural_network
# mlp_proba = mlp.predict_proba(X_test)

def get_proba_list(proba):
    Probability = []
    for item in proba:
        p = item[1]
        # print(p)
        Probability.append(p)
    return Probability

ml_X_proba = get_proba_list(proba)
# print(len(ml_X_proba), len(y_test))
bn_X_proba = get_proba_list(bn_proba)
gu_X_proba = get_proba_list(gu_proba)

# svm
# svc_X_proba = get_proba_list(svc_proba)
# linersvc_X_proba = get_proba_list(linersvc_proba)

# neural_network
# mlp_X_proba = get_proba_list(mlp_proba)

fpr, tpr, thersholds = roc_curve(y_test, ml_X_proba)
print(type(thersholds))
print(type(ml_X_proba))

def params(y_test, X_proba, thersholds):
    accuracy_score_array = []
    precision_score_array = []
    recall_score_array = []
    f1_score_array = []
    for val in thersholds:
        y_pred = np.where(np.array(X_proba) > val,1,0)
        y_pred = y_pred.tolist()
        accuracy_score_array.append(accuracy_score(y_test, y_pred))
        precision_score_array.append(precision_score(y_test, y_pred))
        recall_score_array.append(recall_score(y_test, y_pred))
        f1_score_array.append(f1_score(y_test, y_pred))
    # 计算精准度的算术平均值
    accuracy_mean = statistics.mean(accuracy_score_array)
    accuracy_mean = round(accuracy_mean,3)
    # print(accuracy_mean)
    
    # 计算精确率的算术平均值
    precision_mean = statistics.mean(precision_score_array)
    precision_mean = round(precision_mean,3)
    # print(precision_mean)

    # 计算召回率的算术平均值
    recall_mean = statistics.mean(recall_score_array)
    recall_mean = round(recall_mean,3)
    # print(recall_mean)

    # f1_score 计算精准率和召回率的调和平均值
    f1_mean = statistics.mean(f1_score_array)
    f1_mean = round(f1_mean)
    return accuracy_mean, precision_mean, recall_mean, f1_mean


ml_fpr = fpr
ml_tpr = tpr
ml_area = auc(ml_fpr, ml_tpr)

def createThersholds(num):
    result = []
    for val in range(num):
        result.append(val/num)
    return result


# bn
fpr, tpr, thersholds = roc_curve(y_test, bn_X_proba)
bn_fpr = fpr
bn_tpr = tpr
bn_area = auc(bn_fpr, bn_tpr)


# gu
fpr, tpr, thersholds = roc_curve(y_test, gu_X_proba)
gu_fpr = fpr
gu_tpr = tpr
gu_area = auc(gu_fpr, gu_tpr)

# svm
# fpr, tpr, thersholds = roc_curve(y_test, svc_X_proba)
# svc_fpr = fpr
# svc_tpr = tpr
# svc_area = auc(svc_fpr, svc_tpr)

# fpr, tpr, thersholds = roc_curve(y_test, linersvc_X_proba)
# linersvc_fpr = fpr
# linersvc_tpr = tpr
# linersvc_area = auc(linersvc_fpr, linersvc_tpr)

# neural_network
# fpr, tpr, thersholds = roc_curve(y_test, mlp_X_proba)
# mlp_fpr = fpr
# mlp_tpr = tpr
# mlp_area = auc(mlp_fpr, mlp_tpr)

# thersholds = createThersholds(100)
# accuracy, precision, recall, F1_score = params(y_test,mlp_X_proba, thersholds)
# print(accuracy, precision, recall, F1_score)

# accuracy, precision, recall, F1_score = params(y_test,mlp_X_proba, thersholds)
# print(accuracy, precision, recall, F1_score)

# print(fpr)

# my model
# y_true = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# y_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
y_true = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
y_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]
# y_true = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# y_pred = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

fpr, tpr, thersholds = roc_curve(y_true, y_pred)
area = auc(fpr, tpr)

# roc曲线
roc_tpr = [0.0,0.4,0.6,0.75,0.8,0.88,0.91,0.92,0.95,0.96,0.965,0.99,0.995,1.0]
roc_fpr = [0.0,0.03,0.06,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
# thersholds = createThersholds(100)
# area = auc(roc_fpr, roc_tpr)

plt.figure("ROC")
# plt.plot(roc_fpr, roc_tpr, 'r-o', label="ROC(area={0:.2f})".format(area))
plt.plot(roc_fpr, roc_tpr, 'r-o', label="ROC")
# plt.plot(ml_fpr, ml_tpr, 'r-.', label="Bayes-MultinomialNB(area={0:.2f})".format(ml_area))
# plt.plot(bn_fpr, bn_tpr, 'g-.', label="Bayes-BernoulliNB(area={0:.2f})".format(bn_area))
# plt.plot(gu_fpr, gu_tpr, 'y-.', label="Bayes-GaussianNB(area={0:.2f})".format(gu_area))
# plt.plot(svc_fpr, svc_tpr, 'm-.', label="SVM(area={0:.2f})".format(svc_area))
# plt.plot(fpr, tpr, 'c-.', label="Proposed Model(area={0:.2f})".format(area))
# plt.plot(linersvc_fpr, linersvc_tpr, 'm-.', label="LinearSVC(area={0:.2f})".format(linersvc_area))
# plt.plot(mlp_fpr, mlp_tpr, 'k-.', label="MLP(area={0:.2f})".format(mlp_area))
# plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Diagonal') # 画对角线
# plt.plot([0, 1], [0, 1], 'r-o', label='ROC(area=0.5)') # 画对角线
plt.xlim([-0.05, 1.05])     # 设置x、y轴的上下限，设置宽一点，以免和边缘重合，可以更好的观察图像的整体
plt.ylim([-0.05, 1.05])
plt.xlabel('假正率（FPR）')
plt.ylabel('真正率（TPR）')    # 可以使用中文，但需要导入一些库即字体
# plt.title('Receiver operating characteristic')
plt.title('ROC曲线')
plt.legend(loc="lower right")
plt.show()

# text = ["你是狗吧", "你是不是狗", "今天天气真好", "大仙"]
# text = cut_array(text)
# # print(text)
# test_vect = cv.transform(text)
# print(test_vect.toarray())
# print(gu.predict(test_vect.toarray()))