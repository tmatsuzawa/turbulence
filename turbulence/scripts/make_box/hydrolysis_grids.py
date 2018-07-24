import numpy as np
import dxf
import datetime
import matplotlib.pyplot as plt
import sys
import cPickle as pkl

'''Make the pieces for creating bubbles with parallel grids.
Changed from original design to make bottom grids (octopus hands) smaller 
'''

# Paths and naming
now = datetime.datetime.now()
date = '%02d%02d%02d_%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute)
# the directory where we save the dxf file, change to be wherever you want to save it
rootdir = './dxfparts/'

############################
# Parameters
############################
# lips for top vortex holes
wlip = 72.  # mm
hlip = 20.  # mm
rhole = 4.22 * 0.5  # mm 4.62 ->4.22 TM 1/22/18
# distance between holes is space_between_holes = (wlip - dhole * 2), so
# dhole = 0.5 * (space_between_holes - wlip)
space_between_holes = 58.6992  # mm 51.9615->58.6992mm
radius_hydrolysis_holes = space_between_holes * 0.5 / np.sin(np.pi / 3.)
print 'radius_hholes = ', radius_hydrolysis_holes
dhole = 0.5 * (wlip - space_between_holes)  # mm
hcorner_cutout = 2.0  # mm
wcorner_cutout = 16.0  # mm
wcutout = wlip - wcorner_cutout * 2  # mm
hcutout = hlip - hcorner_cutout * 2  # mm
radius_arc = 18.  # mm
distpt_to_edge = radius_hydrolysis_holes * np.cos(np.pi / 3.) - hlip * 0.5
chord_length = 2. * np.sqrt(radius_arc ** 2 - distpt_to_edge ** 2)
hh = hlip * 0.5 - (radius_hydrolysis_holes * np.cos(np.pi / 3.) - radius_arc)
print 'hh = ', hh
chord_length = 2. * np.sqrt(hh * (2. * radius_arc - hh))
print 'distpt_to_edge = ', distpt_to_edge
print 'distpt_to_edge = ', 0.5 * np.sqrt(4. * radius_arc ** 2 - chord_length ** 2)
angular_span = 2 * np.arcsin(chord_length / (2. * radius_arc))
print 'distpt_to_edge = ', radius_arc * np.cos(angular_span * 0.5)
############################
# plates for lower holes
wplate = 80.  # mm
hplate = 50.  # mm
dcenter = 8.0  # mm
wbuff = 40.  # mm
dfasten = hplate * 0.5
rfastenhole = rhole

############################
# create the arc at the base of the lip
tt = np.linspace(np.pi * 0.5 - angular_span * 0.5, np.pi * 0.5 + angular_span * 0.5, 100)
arc = np.dstack((np.cos(tt), np.sin(tt)))[0]
arc -= np.array([0., arc[0, 1]])
# check it
# plt.plot(arc[:, 0], arc[:, 1], 'b.')
# plt.axis('scaled')
# plt.show()
# sys.exit()

############################
# create masks vortex ring hole
rinner = 25.5906 #mm
router = 37.0 #mm
phi = np.linspace(0, np.pi, 3)

#rfastenhole = rhole #hole for screws




############################
# the dxf object is called 'dd'
dd = dxf.DXF()
outline = np.array([[0, 0], [0, hlip], [wlip, hlip], [wlip, 0]])
outline = np.vstack((outline, radius_arc * arc + np.array([wlip * 0.5, 0.])))
dd.polyline(outline.tolist(), closed=True)
cutout = np.array([[wcorner_cutout, hcorner_cutout], [wcorner_cutout, hcorner_cutout + hcutout],
                   [wcorner_cutout + wcutout, hcorner_cutout + hcutout], [wcorner_cutout + wcutout, hcorner_cutout]])
cutout = np.vstack((cutout, (radius_arc + hcorner_cutout) * arc + np.array([wlip * 0.5, hcorner_cutout])))
dd.polyline(cutout.tolist(), closed=True)

dd.circle([dhole, hlip * 0.5], rhole)
dd.circle([wlip - dhole, hlip * 0.5], rhole)
# check it
plt.plot(outline[:, 0], outline[:, 1], 'r.-')
plt.plot(cutout[:, 0], cutout[:, 1], 'b.-')
plt.axis('equal')
# plt.show()
# sys.exit()

holecenter = [wlip * 0.5, hlip * 0.5 - radius_hydrolysis_holes * np.cos(np.pi/3.)]
dd.circle(holecenter, radius_hydrolysis_holes)
dd.circle(holecenter, 12.7953)  ##mask inner radius 10.->12.7953
# check it
tmpangles = np.linspace(0, 2. * np.pi, 100)
plt.plot(holecenter[0] + radius_hydrolysis_holes * np.cos(tmpangles),
         holecenter[1] + radius_hydrolysis_holes * np.sin(tmpangles), 'c-')
plt.plot(holecenter[0] + 10. * np.cos(tmpangles),
         holecenter[1] + 10. * np.sin(tmpangles), 'g-')
plt.show()
# sys.exit()

############################
# Now we make the bottom plate for cutting out the lower grids
translate = np.array([wlip * 1.2, 0.])
outline = np.array([[0, 0], [0, hplate], [wplate, hplate], [wplate, 0]]) + translate
dd.polyline(outline.tolist(), closed=True)

dd.circle([dcenter + translate[0], dcenter], rhole)
dd.circle([wplate - dcenter + translate[0], dcenter], rhole)
dd.circle([dcenter + translate[0], hplate - dcenter], rhole)
dd.circle([wplate - dcenter + translate[0], hplate - dcenter], rhole)

############################
# Make the bottom plate for fastening the lower grids
translate += np.array([wplate * 1.2, 0.])
outline = np.array([[0, 0], [0, hplate], [wplate + wbuff, hplate],
                    [wplate + wbuff, 0]]) + translate
dd.polyline(outline.tolist(), closed=True)

# Holes for fastening the grid
dd.circle([wbuff + dcenter + translate[0], dcenter], rhole)
dd.circle([wbuff + wplate - dcenter + translate[0], dcenter], rhole)
dd.circle([wbuff + dcenter + translate[0], hplate - dcenter], rhole)
dd.circle([wbuff + wplate - dcenter + translate[0], hplate - dcenter], rhole)

# Holes for fastening the plaquette
dd.circle([dfasten + translate[0], dfasten], rhole)

############################
# Make the bottom plate for spacing the lower grids
translate += np.array([wplate + wbuff * 2.0, 0.])
wd = wplate * 0.15
frac0 = 0.8
frac1 = 0.55
frac2 = 0.5 - (frac1 - 0.5)
frac3 = 1 - frac0
outline = np.array([[0, 0], [0, hplate], [wplate, hplate],
                    [wplate, hplate * frac0], [wd, hplate * frac0],
                    [wd, hplate * frac1], [wplate, hplate * frac1],
                    [wplate, hplate * frac2], [wd, hplate * frac2],
                    [wd, hplate * frac3], [wplate, hplate * frac3],
                    [wplate, 0]]) + translate
dd.polyline(outline.tolist(), closed=True)

# Holes for fastening the grid
dd.circle([dcenter + translate[0], dcenter], rhole)
dd.circle([wplate - dcenter + translate[0], dcenter], rhole)
dd.circle([dcenter + translate[0], hplate - dcenter], rhole)
dd.circle([wplate - dcenter + translate[0], hplate - dcenter], rhole)


###################################
# Save it
###################################
specstr = '_wlip{0:0.1f}'.format(wlip).replace('.', 'p')
specstr += '_hlip{0:0.1f}'.format(hlip).replace('.', 'p')
specstr += '_rarc{0:0.1f}'.format(radius_arc).replace('.', 'p')
filename = rootdir + date + '_grids' + specstr + '.dxf'
dd.save(filename)
print 'saved dxf to ' + filename

# Save the parameters for the code
params = {'wlip': wlip,
          'hlip': hlip,
          'rhole': rhole,
          'dhole': dhole,
          'wcutout': wcutout,
          'hcutout': hcutout,
          'hcorner_cutout': hcorner_cutout,
          'wcorner_cutout': wcorner_cutout,
          'radius_arc': radius_arc,
          'angular_span': angular_span,
          'wplate': wplate,
          'hplate': hplate,
          'dcenter': dcenter,
          'wbuff': wbuff,
          'dfasten': dfasten,
          'rfastenhole': rfastenhole,
          }

filename = rootdir + date + '_grids' + specstr + '.pkl'
with open(filename, "wb") as fn:
    pkl.dump(params, fn)

print 'saved params to ' + filename
