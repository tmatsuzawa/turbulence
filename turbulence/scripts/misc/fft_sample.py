from __future__ import division
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import library.tools.rw_data as rw
import library.tools.process_data as process
import library.display.graph as graph
import numpy as np

datapath = '/Volumes/labshared3-1/kris/4_8_18/time_evolution_data.json'
data = rw.read_json(datapath)

for key in data:
    if key == 'trForDr':
        time = data[key]
        time = time[:-1]
    if key == 'dr':
        diameter = data[key]
        diameter = diameter[:-1]
    print key

time_int, diameter_int = process.interpolate_1Darrays(time, diameter, xnum=100)
#
# time_diff = np.diff(time)
# time_diff_avg = np.nanmean(time_diff)
# std = np.nanstd(time_diff)
# print time_diff_avg, std
#
#
# fig, ax = graph.scatter(time, diameter)
# graph.show()
#
# N = len(diameter)
# dt = 1./200
#
# df = fft(diameter)
# f = np.linspace(0.0, 1.0/(2.0*dt), N//2)
#
# fig, ax = graph.plot(f, 1.0/(5.*N) * np.abs(df[0:N//2]))
# graph.setaxes(ax, 0, 5, 0, 10)
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

#data = np.random.rand(301) - 0.5

#data = diameter_int
x=np.linspace(0,20,10)
data = np.sin(x)

ps = np.abs(np.fft.fft(data))**2

time_diff = np.diff(time)
time_diff_avg = np.nanmean(time_diff)
std = np.nanstd(time_diff)

time_step = time_diff_avg
freqs = np.fft.fftfreq(data.size, time_step)
idx = np.argsort(freqs)
ps[:]=[x / max(ps) for x in ps]

fig1, ax1 = graph.plot(freqs[idx], ps[idx])
graph.setaxes(ax1, -10,10,0,1)
plt.show()
#
# # Number of sample points
# N = len(diameter)
# # sample spacing
# dt = 1.0 / 200.0
# t = np.linspace(0.0, N*dt, N)
# x = np.sin(5.0 * 2.0*np.pi*t) + 0.5*np.sin(1.0 * 2.0*np.pi*t)
# xf = fft(x)
# print len(xf)
# f = np.linspace(0.0, 1.0/(2.0*dt), N//2)
#
#
# fig2, ax2 = graph.scatter(t, x)
# graph.show()
#
#
#
# plt.plot(f, 10.0/N * np.abs(xf[0:N//2]))
# plt.xlabel('frequency [Hz]')
# plt.ylabel('$x(f)$')
# plt.show()
