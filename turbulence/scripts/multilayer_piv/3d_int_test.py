import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import library.basics.formatarray as fa

def f(x,y,z):
    a, b = np.random.rand(1), np.random.rand(1)
    return 2 * (x+a)**3 + 3 * (y+b)**2 - z


def g(x, y):
    return 2 * x ** 3 + 3 * y ** 2
n=602361
x, y, z = np.random.rand(n), np.random.rand(n), np.random.rand(n)
xmin, xmax, ymin, ymax, zmin, zmax = np.min(x), np.max(x), np.min(y), np.max(y), np.min(z), np.max(z)
points = zip(x,y,z)
values = f(x,y,z)


xmin, xmax, ymin, ymax, zmin, zmax = np.min(x), np.max(x), np.min(y), np.max(y), np.min(z), np.max(z)
xx, yy, zz = np.mgrid[xmin:xmax:100j, ymin:ymax:100j, zmin:zmax:100j]
data_int = griddata(points, values, (xx, yy, zz), method='linear')

print len(points), len(values), data_int.shape
nnans = fa.count_nans(data_int)
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

print data_int[..., 0]
for i in range(10):
    plt.figure()
    plt.pcolormesh(xx[..., i], yy[..., i], data_int[..., i])
    plt.colorbar()
plt.show()


#
# points = zip(x,y)
# values = g(x,y)
#
#
# xx, yy = np.mgrid[xmin*1.2:xmax*0.8:100j, ymin*1.2:ymax*0.8:100j]
# data_int = griddata(points, values, (xx, yy), method='linear')
#
# print data_int.shape
# #
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
#
# print data_int
# plt.pcolormesh(xx, yy, data_int)
# plt.colorbar()
# plt.show()