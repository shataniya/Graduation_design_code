import matplotlib.pyplot as plt

y = [0.51
, 0.52
, 0.54
, 0.54
, 0.54
, 0.57
, 0.58
, 0.58
, 0.58
, 0.58
, 0.62
, 0.64
, 0.64
, 0.64
, 0.65
, 0.66
, 0.67
, 0.68
, 0.69
, 0.70
, 0.70
, 0.71
, 0.73
, 0.75
, 0.79
, 0.81
, 0.82
, 0.86
, 0.89
, 0.89
, 0.90
, 0.91]

def curve(num):
    N = len(y)
    line = []
    for item in y:
        line.append(num)
    return line

topline = curve(0.9)
bottomline = curve(0.5)

x = list(range(1,33))
plt.figure("Unit model curve")
plt.plot(x, y, 'r-.', label="Unit model accuracy curve")
plt.plot(x, topline, '-.', label="Accuracy=0.9")
plt.plot(x, bottomline, '-.', label="Accuracy=0.5")
plt.xlim([0, 33])     # 设置x、y轴的上下限，设置宽一点，以免和边缘重合，可以更好的观察图像的整体
plt.ylim([0, 1])
plt.xlabel('Number of unit models')
plt.ylabel('Accuracy')    # 可以使用中文，但需要导入一些库即字体
plt.title('Unit model number growth accuracy curve')
plt.legend(loc="lower right")
plt.show()