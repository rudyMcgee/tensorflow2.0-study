#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File demo  
@Time 2019/12/11 上午11:12
@Author wushib
@Description None

第一步使用figure创建画板
第二步使用subplots创建多个子图形，或者直接使用plt创建一个图形
第三步使用plot画图，使用legend标明图例
"""
import matplotlib.pyplot as plt
import numpy as np


# 创建画板,可以指定边界色，背景色，大小，dpi
fig = plt.figure()
fig.suptitle("matplotlib")
# 添加坐标轴
"""
ax221 = fig.add_subplot(221, frameon=True)
ax222 = fig.add_subplot(222, frameon=True)
ax223 = fig.add_subplot(223, frameon=True)
ax224 = fig.add_subplot(224, frameon=True)
ax221.set(xlabel="1x", ylabel="1y", title="1", xlim=[0.5, 2.5], ylim=[1, 5])
ax222.set(xlabel="2x", ylabel="2y", title="2")
ax223.set(xlabel="3x", ylabel="3y", title="3")
ax224.set(xlabel="4x", ylabel="4y", title="4")
"""
"""
axes = fig.subplots(nrows=2, ncols=2)
axes[0, 0].set(xlabel="1x", ylabel="1y", title="1")
axes[0, 1].set(xlabel="2x", ylabel="2y", title="2")
axes[1, 0].set(xlabel="3x", ylabel="3y", title="3")
axes[1, 1].set(xlabel="4x", ylabel="4y", title="4")

x = np.linspace(0, 6)
axes[0, 0].plot(x, np.sin(x), label="sin()")
axes[0, 1].plot(x, np.cos(x), label="cos()")
axes[1, 0].plot(x, np.tan(x), label="tan()")
axes[1, 1].plot(x, np.sinh(x), label="sinh()")
axes[0, 0].legend()
axes[0, 1].legend()
axes[1, 0].legend()
axes[1, 1].legend()
"""
"""
x = np.linspace(0, 6)
plt.plot(x, np.sin(x), label="sin", color="red")
plt.plot(x, np.cos(x), label="cos", marker="+")
plt.plot(x, np.tan(x), label="tan")
plt.scatter(np.random.randn(20), np.random.randn(20))
plt.legend()
plt.show()
"""


"""
===============操作图片
"""
import matplotlib.image as mpi

img = mpi.imread("./bit.png")
print(img.shape)
# 表示任意行，任意列中，第n深度的数据,相当于保留一个数据通道
img = img[:, :, 0]
print(img.shape)
imgplot=plt.imshow(img)
# 调整img颜色风格
imgplot.set_cmap('hot')
# 显示rgb条
plt.colorbar()
plt.show()



