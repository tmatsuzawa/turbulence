import sys
sys.path.append('/Users/stephane/Documents/git/takumi/turbulence')
import turbulence.display.graphes as graphes
import numpy as np
import matplotlib.pyplot as plt

import turbulence.display.graphes as graphes

x = np.linspace(0,1,101)
y = [xx**2 for xx in x]
plt.plot(x,y)
#graphes.legendeTest('X', 'Y', 'title', display=False, cplot=False)
graphes.legende('X', 'Y', 'title', display=False, cplot=False)


# figlabel = {}
# figlabel.update(graphes.figure_label('X', 'Y', 'title', display=False, cplot=False, include_title=False))

plt.show()
print 'test'