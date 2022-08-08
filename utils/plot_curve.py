# -*- coding: utf-8 -*-
# @Time    : 2022/6/24 17:21
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : plot_curve.py
# @Software: PyCharm
import matplotlib.pyplot as plt


def plot_curve(x, y, x_label="x", x_step=1, y_label="y", title_name="curve"):
    """

    Args:
        x: list
        y: list
        x_label: str, x轴标签
        x_step: 对x输入数据进行挑选
        y_label: str, y轴标签
    """
    plt.plot(y)
    plt.xlabel(x_label, fontsize=14)  # 设置x坐标系的名称
    plt.ylabel(y_label, fontsize=14)  # 设置y坐标系的名称
    plt.tick_params(labelsize=5)  # 设置坐标字体大小
    # 对x轴的数据进行抽样显示
    step = x_step
    xticks_ = list(range(0, len(x), step))
    xticks_label = [x[each] for each in xticks_]
    plt.xticks(xticks_, xticks_label)
    # plt.grid(alpha=0.3)
    plt.title(title_name)
    plt.show()


if __name__ == '__main__':
    x = [1, 2, 4, 10]
    y = [2, 3, 4, 4]
    plot_curve(x, y)
