import numpy as np
import datetime
import dxf
import dxf_auxiliary_functions as af
import matplotlib.pyplot as plt

'''Make all the parts for the vortex collision box
'''

# Paths and naming
now = datetime.datetime.now()
date = '%02d%02d%02d_%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute)
# the directory where we save the dxf file, change to be wherever you want to save it
rootdir = './dxfparts/'

###################################
# Parameters in mm
###################################
# global params
size = 325.0
snug = 0.25
thickness = 5.5
wtooth = 15. + snug  # 6.36, used to be thickness + snug
wspace = 15. - snug  # 6.34, used to be thickness - snug
wtooth_trap = wtooth - snug
wspace_trap = wspace + snug
htooth = 6.2  # must be smaller than thickness
# INDEPENDENT OF SIZE
###################################
# slug add-ons
dslug_screw = 245. * 0.5  # 4.5 inches
rslug_screw = 7.9 * 0.5
# vortex add-ons
rscrew = 2.00  # actual screws are about 3.89 mm in diameter, make this through-fit
rhole = 25.00  # radius of the vortex holes
dscrew = 30.0  # distance from the center of the vortex holes to the screws for vortex add-ons
rpiston = 158.496 * 0.50  # the radius of the slug hole    195.0*0.5 ->  158.496*0.5
rinner_vortex = 10.0  # inner radius of the vortex hole mask
router_vortex = 35.0  # outer radius of the vortex hole mask
# DEPENDENT ON SIZE
###################################
# top and bottom
s0 = size * 0.5
# side panel dimensions
q1 = s0 * 0.6  # height of window + 2*thickness
q3 = q1 * 0.50  # inset length for angled window
q5 = q1 * 0.3  # lip around each window
# trapezoid vortex hole --> really a truncated triangle
t0 = s0 * 1.0  # red
t2 = 4.  # size * 0.02
offset0 = 0.
offset1 = 0.

# BOOSTS and TWEAKS
# plug
plug_toothboost = 1.00
plug_hprimeboost = 1.0
plug_yboost = 2.0
plug_xboost = 3.0
# trap
trap_boost = 1.04  # used to be 1.04
trap_addtop = 2.  # 1.0
trap_addbot = 5.
# window
window_boostx = 1.035
window_xtoothboost = 1.0
# overhang
overhang_boost = 1.0
overhang_snugtooth = 1.0

# parameters which shall not be tuned
# Note that q0 is the hypotenuse of a 45-45-90 triangle with sides (size-q1-2*q5)/2
q0 = np.sqrt(2. * ((size - q1 - 2. * q5) * 0.5) ** 2)

# trapezoid
t1 = s0 - q0
print 's0 = ', s0
print 'q0 = ', q0
print 't1 = ', t1
# windows
w0 = np.sqrt(2) * q3 + t1
w1 = q1 - 2. * thickness


###################################
# initialize
###################################
# the dxf object is called 'dd'
dd = dxf.DXF()

###################################
# vortex add-ons
###################################
# Make the vortex mask sitting inside the slug hole
dd.circle([0., 0.], rinner_vortex)
dd.circle([0., 0.], router_vortex)

# Make the screw holes
tt = np.pi * 1./6. + np.array([0., 2. * np.pi / 3., - 2. * np.pi / 3.])
xxv = dscrew * np.cos(tt)
yyv = dscrew * np.sin(tt)
for kk in range(len(tt)):
    dd.circle(np.array([xxv[kk], yyv[kk]]), rscrew)

# # Make screw holes for hydrolysis grids --> actually we are letting the already present screw holes
# # play double duty! No need
# # get radial distance of little screw holes for hydrolysis meshes based on equipartition in angle space
# # chord_legnth is the same as space_between_holes
# chord_length = 51.9615  # mm, this is from hydrolysis_grids.py
# # chord_length = 2 R sin(theta/2), where theta = pi/3, and
# # R = chord_length / (2 * sin(theta/2)), while
# # dy = R cos(theta/2), where dy is the y coordinate relative to the center of the hole, so
# # dy = 0.5 * np.sqrt(4. * radius_hydrolysis_holes ** 2 - chord_length ** 2)
# radius_hydrolysis_holes = chord_length * 0.5 / np.sin(np.pi / 6.)
# # place the hydrolysis screw holes 'below' which is at pi/2 (above in xy plane here)
# dy = 0.5 * np.sqrt(4. * radius_hydrolysis_holes ** 2 - chord_length ** 2)
# # now add the two screw holes to dd
# dd.circle(np.array([chord_length * 0.5, dy]), rscrew)
# dd.circle(np.array([-chord_length * 0.5, dy]), rscrew)

###################################
# top and bottom with slug hole (rpiston)
###################################
center = np.array([0., 0.], dtype=float)
# make screw holes for mounting
for xx in [-25.4 * 1.5, 0, 25.4 * 1.5]:
    for yy in [dslug_screw, 0, -dslug_screw]:
        if abs(xx) > 59 or abs(yy) > 59:
            dd.circle(np.array([xx, yy]), rslug_screw)
            dd.circle(np.array([yy, xx]), rslug_screw)
            dd.circle(np.array([yy, - xx]), rslug_screw)
            # Add rotated by 45 degrees
            xtmp = xx * np.cos(np.pi * 0.25) + yy * np.sin(np.pi * 0.25)
            ytmp = yy * np.cos(np.pi * 0.25) - xx * np.sin(np.pi * 0.25)
            dd.circle(np.array([xtmp, ytmp]), rslug_screw)
            dd.circle(np.array([ytmp, xtmp]), rslug_screw)
            dd.circle(np.array([ytmp, -xtmp]), rslug_screw)
            # Add rotated by -45 degrees
            xtmp = xx * np.cos(np.pi * 0.25) - yy * np.sin(np.pi * 0.25)
            ytmp = yy * np.cos(np.pi * 0.25) + xx * np.sin(np.pi * 0.25)
            dd.circle(np.array([xtmp, ytmp]), rslug_screw)
            dd.circle(np.array([ytmp, xtmp]), rslug_screw)
            dd.circle(np.array([ytmp, -xtmp]), rslug_screw)

dd.circle(center.tolist(), rpiston)
#
dd.circle(center.tolist(), dslug_screw+4.*rslug_screw)

#
#          size
#   <---------------->
#           s1          (diagonal is width)
#          _____
#         /     \             ^
#       /         \  s0       |
#     /             \         |
#    |               | s1     |
#    |               |        | size
#   ^ \             /         |
#   |   \ s0      /           |
# s2|     \     /             |
#   v<---> -----              v
#      s2    s1
#
s2 = s0 / np.sqrt(2.)
s1 = size - s2 * 2
# width is determined by size and s0 only --> we need this for later
width = s0 + np.sqrt(2.) * s1

top = np.array([[-size * 0.5, -size * 0.5 + s2],
                [-size * 0.5, -size * 0.5 + s2 + s1],
                [-s1 * 0.5, size * 0.5],
                [s1 * 0.5, size * 0.5],
                [size * 0.5, -size * 0.5 + s2 + s1],
                [size * 0.5, -size * 0.5 + s2],
                [s1 * 0.5, -size * 0.5],
                [-s1 * 0.5, -size * 0.5]])
top += center
lineseg_inds = None
# dd.polyline(top.tolist(), closed=True)
top = af.polygon2sawgon(top, wtooth, wspace, htooth=htooth, lineseg_inds=lineseg_inds, outward=True, invert=False)
dd.polyline(top.tolist(), closed=True)

###################################
# sides
###################################
center += np.array([-0.65 * size, -0.75 * size])
#
#           q2 ~ s1
#       6  _____  7           ^     ^
#         /     \             |     |
#       /         \  q0       |     | (size - q1 - 2 * q5)/2
#   5 /             \  8      |     v
# q5 |__q3 3   10  __|        |
#       |         | q1 ~ (w1 + 2 thickness)
#  1  __| 2    11 |__  12     | size
#  0 |               |        |
# ^   \             /  13     |
# |     \ q0      /           |
# |q4     \     /             |
# v  <---> -----  14          v
#      q4    q2
#
q2 = s1
q4 = q0 * np.cos(np.pi * 0.25)
# # Check that size - q1 == 2 * q0 * np.sin(np.pi * 0.25) = 2 * q0 / sqrt(2) = sqrt(2) * q0
# print 'dy = ', size - q1 - 2 * q5
# print 'dy = ', 2 * q0 * np.sin(np.pi * 0.25)
# sys.exit()
side = np.array([[-q2 * 0.5 - q4, q0 * np.sin(np.pi * 0.25)],               # 0
                 [-q2 * 0.5 - q4, q0 * np.sin(np.pi * 0.25) + q5],          # 1
                 [-q2 * 0.5 - q4 + q3, q0 * np.sin(np.pi * 0.25) + q5],     # 2
                 # ensure thickness is not square-waved
                 [-q2 * 0.5 - q4 + q3, q0 * np.sin(np.pi * 0.25) + q5 + thickness],         # 3
                 [-q2 * 0.5 - q4 + q3, q0 * np.sin(np.pi * 0.25) + q5 + q1 - thickness],    # 4
                 [-q2 * 0.5 - q4 + q3, q0 * np.sin(np.pi * 0.25) + q1 + q5],                # 5
                 [-q2 * 0.5 - q4, q0 * np.sin(np.pi * 0.25) + q1 + q5],                     # 6
                 [-q2 * 0.5 - q4, q0 * np.sin(np.pi * 0.25) + q1 + 2. * q5],                # 7
                 [-q2 * 0.5, size],                                                         # 8
                 [q2 * 0.5, size],                                                          # 9
                 [q2 * 0.5 + q4, q0 * np.sin(np.pi * 0.25) + q1 + 2. * q5],                 # 10
                 [q2 * 0.5 + q4, q0 * np.sin(np.pi * 0.25) + q1 + q5],                      # 11
                 [q2 * 0.5 + q4 - q3, q0 * np.sin(np.pi * 0.25) + q1 + q5],                 # 12
                 # ensure thickness insets are not square-waved
                 [q2 * 0.5 + q4 - q3, q0 * np.sin(np.pi * 0.25) + q1 + q5 - thickness],     # 13
                 [q2 * 0.5 + q4 - q3, q0 * np.sin(np.pi * 0.25) + q5 + thickness],          # 14
                 [q2 * 0.5 + q4 - q3, q0 * np.sin(np.pi * 0.25) + q5],                      # 15
                 [q2 * 0.5 + q4, q0 * np.sin(np.pi * 0.25) + q5],                           # 16
                 [q2 * 0.5 + q4, q0 * np.sin(np.pi * 0.25)],                                # 17
                 [q2 * 0.5, 0],                                                             # 18
                 [-q2 * 0.5, 0]]) - np.array([0, 0.5 * size])                               # 19
side += center
# dd.polyline(side.tolist(), closed=True)
lineseg_inds = [[3, 4], [7, 8], [8, 9], [9, 10], [13, 14], [17, 18], [18, 19], [19, 0]]
side = af.polygon2sawgon(side, wspace, wtooth, htooth=htooth, lineseg_inds=lineseg_inds, outward=False)
dd.polyline(side.tolist(), closed=True)

# Also get qghost for trapezoid dimensions
qghost = np.sqrt(2. * ((size - q2) * 0.5) ** 2)

###################################
# trapezoidal vortex holes
###################################
#              t0
# 0 ______________________1
#  /                      \ t2   ^
# 9 \                     / 2    |
#     \                 /        |
#    t3 \             /          | htrap
#         \  6 _ 5  /            |
#          ---| |---             v
#        8   7  4   3
#          <------->
#              t1
#
# Ghost trapezoid:
#              s0
#    ___________________________
#    \                         /
#      \                     /
#        \                 /
#          \             /    q0
#            \    _    /
#             ---| |---
#
#             <------->
#                 t1 = s0 - q0
#
center = np.array([size, -size * 0.4])
traptip_angle = -np.pi / 3.
# print 't1 = ', t1
# t1 = w0 - 2. * np.cos(np.pi * 0.25) * q3
# print 't1 = ', t1
# sys.exit()
htrap = np.sqrt(3) * 0.5 * q0

# For the central hole, position it a distance disth_fromtop from the top of the trapezoid.
# Note: the top of the trapezoid has intrusions (invert=True), instead of protrusions, for teeth.
print 'width = ', width
print 'size = ', size
disth_fromtop = np.sqrt(2. * ((np.sqrt(2.) * size - width) * 0.25) ** 2)
holec = center + np.array([0., htrap * 0.5 - disth_fromtop])
dd.circle(holec.tolist(), rhole)
# make the screws
# xyv = af.rotate_vectors_2d(np.dstack((np.array(xxv), np.array(yyv)))[0], np.pi * 0.25)
xyv = np.dstack((np.array(xxv), np.array(yyv)))[0] + holec
for kk in range(len(tt)):
    dd.circle(np.array([xyv[kk, 0], xyv[kk, 1]]), rscrew)

# make the trapezoid --> this is before the g
# trap = np.array([[-t0 * 0.5, htrap * 0.5],
#                  [t0 * 0.5, htrap * 0.5],
#                  [t0 * 0.5 + t2 * np.cos(traptip_angle),
#                   htrap * 0.5 + t2 * np.sin(traptip_angle)],
#                  [t1 * 0.5, -htrap * 0.5],
#                  [wspace * 0.5, -htrap * 0.5],
#                  [wspace * 0.5, -htrap * 0.5 + htooth],
#                  [-wspace * 0.5, -htrap * 0.5 + htooth],
#                  [-wspace * 0.5, -htrap * 0.5],
#                  [-t1 * 0.5, -htrap * 0.5],
#                  [-t0 * 0.5 - t2 * np.cos(traptip_angle),
#                   htrap * 0.5 + t2 * np.sin(traptip_angle)]])
# Note: traverse trapghost counterclockwise to have intrusions
# trapghost = 0.5 * np.array([[-s0, htrap],
#                             [-t1, -htrap],
#                             [-wspace, -htrap],
#                             [-wspace, -htrap + 2. * htooth],
#                             [wspace, -htrap + 2. * htooth],
#                             [wspace, -htrap],
#                             [t1, -htrap],
#                             [s0, htrap]])
# lineseg_inds = [[0, 1], [6, 7], [7, 0]]

# boost on the top line only
trapghost = 0.5 * np.array([[-s0, htrap],  # add trap_addtop to this one
                            [s0, htrap],  # add trap_addtop to this one too
                            [s0, htrap],
                            [t1, -htrap],
                            [t1, -htrap],  # add trap_addbot to this one and following
                            [wspace / trap_boost, -htrap],
                            [wspace / trap_boost, -htrap + 2. * htooth],
                            [-wspace / trap_boost, -htrap + 2. * htooth],
                            [-wspace / trap_boost, -htrap],
                            [-t1, -htrap],  # last one to add trap_addbot to.
                            [-t1, -htrap],
                            [-s0, htrap]])
lineseg_inds = [[0, 1], [2, 3], [10, 11]]

# # Boost to magnify whole part
# trapghost = 0.5 * np.array([[-s0, htrap],
#                             [s0, htrap],
#                             [t1, -htrap],
#                             [wspace, -htrap],
#                             [wspace, -htrap + 2. * htooth],
#                             [-wspace, -htrap + 2. * htooth],
#                             [-wspace, -htrap],
#                             [-t1, -htrap]])
# lineseg_inds = [[0, 1], [1, 2], [7, 0]]
trapghost *= trap_boost
# Now that we have multiplied by boost, translate lines by trap_add***
trapghost[0:2, 1] += trap_addtop
trapghost[4:10, 1] += trap_addbot
trapghost = af.rotate_vectors_2d(trapghost, np.pi)
trapghost += center

# Reverse the order
# lineseg_inds = (len(trap) - np.array(lineseg_inds) - 1)[::-1]
# lineseg_inds[:, 0], lineseg_inds[:, 1] = lineseg_inds[:, 1], lineseg_inds[:, 0].copy()
# print 'lineseg_inds = ', lineseg_inds
# Reverse the order for trapghost
# lineseg_inds = (len(trapghost) - np.array(lineseg_inds) - 1)[::-1]
# lineseg_inds[:, 0], lineseg_inds[:, 1] = lineseg_inds[:, 1], lineseg_inds[:, 0].copy()
# print 'lineseg_inds = ', lineseg_inds
# sys.exit()
# trap = af.polygon2sawgon(trapghost, wtooth, wspace, htooth=htooth, lineseg_inds=lineseg_inds, outward=True,
#                          invert=False, offsets=[0, offset0, offset1], check=False)
trapghost = af.polygon2sawgon(trapghost, wtooth_trap, wspace_trap, htooth=htooth, lineseg_inds=lineseg_inds,
                              outward=[True, False, False],
                              invert=True, offsets=[0, offset0, offset1], check=False)

dd.polyline(trapghost.tolist(), closed=True)

# Add corners
trap_extras = [np.array([[t0 * 0.5, htrap * 0.5],
                         [t0 * 0.5 + t2 * np.cos(traptip_angle),
                          htrap * 0.5 + t2 * np.sin(traptip_angle)]]),
               np.array([[-t0 * 0.5, htrap * 0.5],
                         [-t0 * 0.5 - t2 * np.cos(traptip_angle),
                          htrap * 0.5 + t2 * np.sin(traptip_angle)]])]

for extra in trap_extras:
    extra = af.rotate_vectors_2d(extra, np.pi)
    extra += center
    dd.polyline(extra.tolist(), closed=False)

###################################
# windows
###################################
center += np.array([-size * 0.65, -size * 0.35])
#
#       2       3/5
# 1 ____._______.____ 4/6   ^
#  | w00    w01      |      |
#  |                 |      |
#  |                 |      |
#  | w1              |      |  w1
#  |                 |      |
#  | w00   w01       |      |
#  |____._______.____|      v
# 0     7/11   6/8   5/7
#
#  <----------------->
#           w0
#

# Used to auto-gen the two teeth
# w01 = wtooth + 0.2 * wspace
# w00 = (w0 - w01) * 0.5
# window = np.array([[0, 0],
#                    [0, w1],
#                    [w00, w1],
#                    [w00 + w01, w1],
#                    [w0, w1],
#                    [w0, 0],
#                    [w00 + w01, 0],
#                    [w00, 0]])
# lineseg_inds = [[0, 1], [2, 3], [4, 5], [6, 7]]
# window = af.polygon2sawgon(window, wtooth, wspace, htooth=htooth, lineseg_inds=lineseg_inds, outward=True)

w01 = wtooth
w00 = (w0 * window_boostx - w01) * 0.5
w0b = w0 * window_boostx
window = np.array([[0, 0],
                   [0, w1],
                   [w00, w1],
                   [w00, w1 + htooth],
                   [w00 + w01, w1 + htooth],
                   [w00 + w01, w1],
                   [w0b, w1],
                   [w0b, 0],
                   [w00 + w01, 0],
                   [w00 + w01, -htooth],
                   [w00, -htooth],
                   [w00, 0]])
window -= np.array([w0b * 0.5, w1 * 0.5])
# window[:, 0] = window_boostx * window[:, 0]
lineseg_inds = [[0, 1], [6, 7]]
htooth_prime = htooth * np.sqrt(2.)
window = af.polygon2sawgon(window, wtooth, wspace, htooth=htooth,
                           lineseg_inds=lineseg_inds, outward=False, invert=True)
window += center
dd.polyline(window.tolist(), closed=True)

###################################
# protrusion plug
###################################
center += np.array([-size * 0.05, -size * 0.3])
#
#         3 ____ 4
#          |    |  htooth_prime
# 1  ______|    |______ 6
#   |     2     5      |
#   |                  |  q5
# 0 |______11  8 ______|
#          |    |        7
#        10 ---- 9  htooth
#
#
wptooth = wtooth * plug_toothboost
htooth_prime = htooth * plug_hprimeboost
t1p = t1 + plug_xboost
plug = np.array([[0, 0],
                 [0, q5 + plug_yboost],
                 [(t1p - wptooth) * 0.5, q5 + plug_yboost],
                 [(t1p - wptooth) * 0.5, q5 + htooth_prime + plug_yboost],
                 [(t1p + wptooth) * 0.5, q5 + htooth_prime + plug_yboost],
                 [(t1p + wptooth) * 0.5, q5 + plug_yboost],
                 [t1p, q5 + plug_yboost],
                 [t1p, 0],
                 [(t1p + wptooth) * 0.5, 0],
                 [(t1p + wptooth) * 0.5, -htooth],
                 [(t1p - wptooth) * 0.5, -htooth],
                 [(t1p - wptooth) * 0.5, 0]])

plug += center
dd.polyline(plug.tolist(), closed=True)

###################################
# protrusion overhang
###################################
center += np.array([-size * 0.1, -size * 0.15])
#
#               t1prime = np.sqrt(2. * (t1 / np.sqrt(2) + t)**2)
#            _____________
#          /               \
#        /                   \   q3
#      /                       \
#    /                           \
#    \_______.____________._______/ t
#        w00     w01         w00
#
#
t1prime = np.sqrt(2. * (t1 / np.sqrt(2.) + thickness) ** 2)

# check tprime by computing a different way
# t1prime1 = w0 + 2. * np.cos(np.pi * 0.25) * (thickness - q3)  # violet
# print 't1prime = ', t1prime
# print 't1prime1 = ', t1prime1

# here redefine w01 to be wspace instead of wtooth
w01 = wspace * overhang_snugtooth
w00 = (w0 - w01) * 0.5

# overhang --> 'oh'
oh2x = np.cos(np.pi * 0.25) * (q3 - thickness)
oh2y = np.sin(np.pi * 0.25) * (q3 + thickness)
overhang = np.array([[0, 0],
                     [-thickness * np.cos(np.pi * 0.25), thickness * np.sin(np.pi * 0.25)],
                     [oh2x, oh2y],
                     [oh2x + (t1prime - w01) * 0.5, oh2y],
                     [oh2x + (t1prime - w01) * 0.5, oh2y - htooth],
                     [oh2x + (t1prime + w01) * 0.5, oh2y - htooth],
                     [oh2x + (t1prime + w01) * 0.5, oh2y],
                     [oh2x + t1prime, oh2y],
                     [w01 + 2. * w00 + thickness * np.cos(np.pi * 0.25), thickness * np.sin(np.pi * 0.25)],
                     [w01 + 2. * w00, 0.],
                     [w01 + w00, 0.],
                     [w01 + w00, htooth],
                     [w00, htooth],
                     [w00, 0.]])
overhang *= overhang_boost
overhang += center
# plt.plot(overhang[:, 0], overhang[:, 1], 'b.-')
# plt.show()
dd.polyline(overhang.tolist(), closed=True)

###################################
# Save it
###################################
specstr = '_size{0:0.1f}'.format(size).replace('.', 'p') + '_s0{0:0.3f}'.format(s0).replace('.', 'p')
specstr += '_t0{0:0.3f}'.format(t0).replace('.', 'p')
specstr += '_off{0:0.3f}'.format(offset0).replace('.', 'p')
specstr += '_windowbx{0:0.4f}'.format(window_boostx).replace('.', 'p')
specstr += '_windowxtb{0:0.4f}'.format(window_xtoothboost).replace('.', 'p')
specstr += '_plugxb{0:0.4f}'.format(plug_xboost).replace('.', 'p')
filename = rootdir + date + '_box' + specstr + '.dxf'
dd.save(filename)
filename = rootdir + date + '_box' + specstr + '_params.txt'
pdict = {'size': size,
         'wtooth': wtooth,
         'wspace': wspace,
         'wtooth': wtooth,
         'rscrew': rscrew,
         'rhole': rhole,
         'rpiston': rpiston,
         'dscrew': dscrew,
         'rinner_vortex': rinner_vortex,
         'router_vortex': router_vortex,
         't0': t0,
         't1': t1,
         't2': t2,
         's0': s0,
         'q1': q1,
         'q3': q3,
         'q5': q5,
         'w0': w0,
         'w1': w1,
         'offset0': offset0,
         'offset1': offset1,
         'plug_toothboost': plug_toothboost,
         'plug_hprimeboost': plug_hprimeboost,
         'plug_yboost': plug_yboost,
         'trap_boost': trap_boost,
         'trap_addtop': trap_addtop,
         'trap_addbot': trap_addbot,
         'window_boostx': window_boostx,
         'window_xtoothboost': window_xtoothboost,
         'overhang_boost': overhang_boost,
         'overhang_snugtooth': overhang_snugtooth,
         }
af.save_dict(pdict, filename, 'parameters for this box dxf', keyfmt='auto', valfmt='auto', padding_var=7)

print 'saved to filename: ', filename




