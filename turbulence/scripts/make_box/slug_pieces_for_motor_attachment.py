import numpy as np
import dxf
import datetime

'''Make the pieces for attaching to the motor in order to make a slug
'''

# Paths and naming
now = datetime.datetime.now()
date = '%02d%02d%02d_%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute)
# the directory where we save the dxf file, change to be wherever you want to save it
rootdir = './dxfparts/'

# Parameters
rplate = 100 * 0.5  # mm
rslug = 155.0 * 0.5  # mm 150->155mm
rscrew = 3.92
rthread = 0.245 * 0.5 * 25.4  # mm
dthread = 35.0  # mm
nthreads = 5

# the dxf object is called 'dd'
dd = dxf.DXF()
dd.circle([0., 0.], rslug)
dd.circle([0., 0.], rplate)
# Make hole in center for attachment to rod
dd.circle([0., 0.], rscrew)

thetas = np.linspace(0, 2. * np.pi, nthreads + 1)[0:-1]
for tt in thetas:
    dd.circle([dthread * np.cos(tt), dthread * np.sin(tt)], rthread)


###################################
# Save it
###################################
specstr = '_nthreads' + str(nthreads) + '_rplate{0:0.1f}'.format(rplate).replace('.', 'p')
specstr += '_rslug{0:0.1f}'.format(rslug).replace('.', 'p')
specstr += '_rscrew{0:0.3f}'.format(rscrew).replace('.', 'p')
specstr += '_rthread{0:0.3f}'.format(rthread).replace('.', 'p')
specstr += '_dthread{0:0.2f}'.format(dthread).replace('.', 'p')
filename = rootdir + date + '_slug' + specstr + '.dxf'
dd.save(filename)
