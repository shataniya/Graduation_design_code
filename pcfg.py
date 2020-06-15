#coding=utf-8
import re
# 分词
import jieba

# PCFG句法分析
from nltk.parse import stanford
import os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
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

# 获取字典
nn_array = read_file_to_array('NN.txt')

# 建立违规模型
violation_model = read_file_to_array('violation_model.txt')

# 0 是合格弹幕， 1 是恶意弹幕
def good_barrage(file_path):
    good_data = read_file_to_array(file_path)
    good_label = []
    for item in good_data:
        good_label.append(0)
    return good_data, good_label

def bad_barrage(file_path):
    bad_data = read_file_to_array(file_path)
    bad_label = []
    for item in bad_data:
        bad_label.append(1)
    return bad_data, bad_label

def anls_barrage(good_path, bad_path):
    good_data, good_label = good_barrage(good_path)
    bad_data, bad_label = bad_barrage(bad_path)
    return good_data, bad_data, good_label, bad_label

def combine_barrage(good_path, bad_path):
    good_data, bad_data, good_label, bad_label = anls_barrage(good_path, bad_path)
    good_data.extend(bad_data)
    good_label.extend(bad_label)
    data = good_data
    target = good_label
    return data, target

# 判断是不是违规模型
def is_violation_model(model_string):
    if model_string in violation_model:
        return True
    else:
        return False

# 定义一个添加模型的函数
def cutText(string):
    seg_list = jieba.cut(string, cut_all=False, HMM=True)
    seg_str = ' '.join(seg_list)
    return seg_str

# cn_name
def cn_label(label):
    if label == 'ROOT':
        return '要处理文本的语句'
    elif label == 'IP':
        return '简单从句'
    elif label == 'NP':
        return '名词短语'
    elif label == 'VP':
        return '动词短语'
    elif label == 'PU':
        return '断句符'
    elif label == 'LCP':
        return '方位词短语'
    elif label == 'PP':
        return '介词短语'
    elif label == 'CP':
        return 'XX的'
    elif label == 'DNP':
        return 'xx的'
    elif label == 'ADVP':
        return '副词短语'
    elif label == 'ADJP':
        return '形容词短语'
    elif label == 'DP':
        return '限定词短语'
    elif label == 'QP':
        return '量词短语'
    elif label == 'NN':
        return '常用名词'
    elif label == 'NR':
        return '固有名词'
    elif label == 'NT':
        return '时间名词'
    elif label == 'PN':
        return '代词'
    elif label == 'VV':
        return '动词'
    elif label == 'VC':
        return '是'
    elif label == 'CC':
        return '不是'
    elif label == 'VE':
        return '有'
    elif label == 'VA':
        return '表语形容词'
    elif label == 'AS':
        return '了'
    elif label == 'VRD':
        return '动补复合词'
    elif label == 'DT':
        return '限定词'
    else:
        return label

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


if __name__ == '__main__':

    def checkMali(string):
        
        # seg_list = jieba.cut(string, cut_all=False, HMM=True)
        # seg_str = ' '.join(seg_list)
        seg_str = cutText(string)

        # print(seg_str)
        root = './'
        parser_path = root + 'stanford-parser.jar'
        model_path =  root + 'stanford-parser-3.9.2-models.jar'

        # 指定JDK路径
        if not os.environ.get('JAVA_HOME'):
            JAVA_HOME = '/usr/lib/jvm/jdk1.8'
            os.environ['JAVA_HOME'] = JAVA_HOME

        # PCFG模型路径
        pcfg_path = 'edu/stanford/nlp/models/lexparser/chinesePCFG.ser.gz'

        parser = stanford.StanfordParser(
            path_to_jar=parser_path,
            path_to_models_jar=model_path,
            model_path=pcfg_path
        )

        

        # 判断是不是关键词性
        def islabel(label):
            if label == 'PN':
                return True
            elif label == 'AD':
                return True
            elif label == 'NN':
                return True
            elif label == 'M':
                return True
            elif label == 'P':
                return True
            elif label == 'VV':
                return True
            else:
                return False

        # 判断是不是PN NN
        def is_nn_label(label):
            if label == 'PN':
                return True
            elif label == 'NN':
                return True
            else:
                return False

        def info(seg_str):
            sentence = parser.raw_parse(seg_str)
            for line in sentence:
                line.draw()
                # print(line.pos())
                return line.pos()

        def fil_label(model_object):
            val_array = []
            for val, key in model_object:
                if is_nn_label(key):
                    # 说明是PN 或者 NN
                    val_array.append(val)
            return val_array

        def checkval(val_array):
            flag = True
            for val in val_array:
                if val in nn_array:
                    flag = True
                else:
                    flag = False
            return flag

        def label(line):
            key_list = []
            val_list = []
            for val, key in line:
                if islabel(key):
                    key_list.append(key)
                    tpl = tuple([val ,key])
                    val_list.append(tpl)
            model_string = ' '.join(key_list)
            model_object = val_list
            return model_string, model_object

        # 对句法进行检测，化简，提取主要内容
        def simple_model(model_string):
            string = re.findall(r'(.*?NN) PN', model_string)
            if len(string):
                return string[0]

            string = re.findall(r'(.*?NN) AD', model_string)
            if len(string):
                return string[0]
            return model_string

        def check(line):
            model_string, model_object = label(line)
            # print(model_string)
            model_string = simple_model(model_string)
            # print(model_string)
            if is_violation_model(model_string):
                val_array = fil_label(model_object)
                val_flag = checkval(val_array)
                return val_flag
            else:
                return False
                

        line = info(seg_str)
        flag = check(line)
        if flag == True:
            return 1
        else:
            return 0
        # return flag
    
    def check_barrage(barrage):
        barrage_label = []
        for item in barrage:
            # print(item)
            label = checkMali(item)
            # print(label)
            barrage_label.append(label)
        return barrage_label

    # data, target = combine_barrage('barrage.txt', 'Malicious.txt')
    # barrage_label = check_barrage(data)
    # fpr, tpr, thersholds = roc_curve(target, barrage_label)
    # area = auc(fpr, tpr)
    # print(target)
    # print(barrage_label)
    # print(thersholds)

    # accuracy, precision, recall, F1_score = params(target,barrage_label, thersholds)
    # print(accuracy, precision, recall, F1_score)

    # plt.figure("ROC")
    # plt.plot(fpr, tpr, 'r-.', label="NLP(area={0:.2f})".format(area))
    # plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='diagonal') # 画对角线
    # plt.xlim([-0.05, 1.05])     # 设置x、y轴的上下限，设置宽一点，以免和边缘重合，可以更好的观察图像的整体
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')    # 可以使用中文，但需要导入一些库即字体
    # plt.title('Receiver operating characteristic for NLP')
    # plt.legend(loc="lower right")
    # plt.show()

    string = '你就是个废物'
    flag = checkMali(string)
    print(flag)

    # malitxt = read_file_to_array('neg_test.txt')
    # result = ''
    # for item in malitxt:
    #     target = checkMali(item)
    #     result = result + f'{target}\n'
    # f = open('result.txt', 'w')
    # f.write(result)


    
