import glob
import numpy as np
from PIL import Image
import basics.dataio as dio
import ilpm.movies as bmov
import argparse

'''
Example usage:
python trace_flows.py
'''

parser = argparse.ArgumentParser(description='Track videos of gyros.')
parser.add_argument('-check', '--check', help='Display intermediate results', action='store_true')
parser.add_argument('-overwrite', '--overwrite', help='Overwrite previous flow tracing results', action='store_true')
parser.add_argument('-brighten', '--brighten', help='Brighten all images by this factor', type=float, default=1.0)
args = parser.parse_args()


fps = 30
root = '/Volumes/labshared3/takumi/2018_07_22_wide/'
indir = root + 'Tiff_folder/' \
               'PIV_fv_vp_left_macro55mm_fps2000_Dp56p57mm_D12p8mm_piston10p5mm_freq3Hz_v300mms_setting1_File/'

outdir = root + 'trace_flows/'
if not os.path.exists(outdir):
    os.path.mkdir(outdir)

tiffs = glob.glob(indir + '*.tiff')
step = 10
todo = np.arange(0, len(tiffs), step)

# If the frames are not already saved, or if we are to overwrite, go through and sum adjacent frames
if len(glob.glob(outdir + 'trace_flows*.png')) < len(todo) or args.overwrite:
    for (start, kk) in zip(todo, np.arange(len(todo))):
        print 'start=' + str(start) + ', index = ' + str(kk) + '/' + str(len(todo))
        end = start + 25
        count = 0
        for tiff in tiffs[start:end]:
            im = np.asarray(Image.open(tiff)).astype('float')
            count += 1
            if count == 1:
                imsum = im
            else:
                imsum += im

        imsum *= args.brighten / float(count)
        imsum = imsum.astype('uint8')
        imsum[imsum > 255] = 255

        result = Image.fromarray(imsum)
        result.save(outdir + 'trace_flows_{0:06d}'.format(kk) + '.png')

# Make movie
imgname = outdir + 'trace_flows_'
movname = root + 'trace_flows'
bmov.make_movie(imgname, movname, indexsz='06', framerate=fps)
