#!/usr/bin/env python
bsum = sum
from numpy import *
#from boundary import Boundary, hex_grid, circle, join_points_3D
from mesh import *
from boundary import hex_grid, Boundary
import sys

def norm_ang(a):
    return (a + pi) % (2 * pi) - pi

def bezier(p0, p1, p2, p3, np=10, include_ends=False):
    if include_ends:
        t = (arange(np+2, dtype='d')) / (np+1)
    else:
        t = (arange(np, dtype='d') + 1) / (np+1)
    t.shape = t.shape + (1,)
    
    pt = lambda n: (1-t)**(3 - n) * t**n
        
    return pt(0) * p0 + 3 * (pt(1) * p1 + pt(2) * p2) + pt(3) * p3

def torus_knot(p, q, a, np=100, phi_s=None, phi_e=None):
    if phi_s is None:
        phi = 2 * pi * arange(np) / np
    else:
        phi = linspace(phi_s, phi_e, np)
        
    r = 1 + a * sin(q * phi)
    
    x = r * cos(p * phi)
    y = r * sin(p * phi)
    z = -a * cos(q * phi)
    
    return array((x, y, z)).T

def circle(c=(0, 0), r=1, np=100):
    theta = arange(np, dtype='d') / np * 2 * pi
    p = zeros((np, len(c))) #c might be 3D!
    p[:] = c
    p[:, 0] += r * cos(theta)
    p[:, 1] += r * sin(theta)
    return p

def rot_z(x, a):
    rx = x.copy()
    rx[:, 0] = cos(a) * x[:, 0] - sin(a) * x[:, 1]
    rx[:, 1] = cos(a) * x[:, 1] + sin(a) * x[:, 0]
    
    return rx
    
def path_connection(p1, s1, p2, s2, np):
    d1 = p1[-1] - p1[-2]
    d1 /= sqrt(sum(d1**2))

    d2 = p2[0] - p2[1]
    d2 /= sqrt(sum(d2**2))
    
    return bezier(p1[-1], p1[-1] + s1 * d1, p2[0] + s2 * d2, p2[0], np, False)
    
def rad_connection(p1, p2, round=None, np=10, nrp=0, connect_start=True, connect_end=True):
    sp = p1[-1]
    ep = p2[0]
    
    stheta = arctan2(sp[1], sp[0])
    etheta = arctan2(ep[1], ep[0])
    etheta = stheta + norm_ang(etheta - stheta)
    dir_theta = sign(etheta - stheta)
    
    sr = sqrt(sum(sp[:2]**2))
    er = sqrt(sum(ep[:2]**2))
    dir_r = sign(er - sr)
    
    if round is not None:
        stheta -= dir_r * round / sr
        etheta += dir_r * round / er
        sr -= dir_theta * round
        er += dir_theta * round
    
    r = linspace(sr, er, np)
    theta = linspace(stheta, etheta, np)
    
    x = r * cos(theta)
    y = r * sin(theta)
    
    if p1.shape[1] == 3:
        z = linspace(sp[2], ep[2], np)
        rp = array((x, y, z)).T
    else:
        rp = array((x, y)).T

    if round is None: return rp
    
    return vstack((
        (path_connection(p1, round, rp, round, nrp) if connect_start else zeros((0, p1.shape[1]))),
        rp,
        (path_connection(rp, round, p2, round, nrp) if connect_end else zeros((0, p1.shape[1]))),
    ))   
    
if __name__ == '__main__':
    make_plot = False

    if make_plot:
        from pylab import plot, show, subplot, figure, fill
        import mpl_toolkits.mplot3d.axes3d as p3
    
        def list_plot(p, *args, **kwargs): plot(p[:, 0], p[:, 1], *args, **kwargs)
        def list_plot3d(p, *args, **kwargs):
            ax = p3.Axes3D(figure())
            ax.plot3D(p[:, 0], p[:, 1], p[:, 2], *args, **kwargs)

    else:
        def dummy(*args, **kwargs): pass
        list_plot = dummy
        list_plot3d = dummy

#    list_plot(bezier(p0, p1, p2, p3, 100, True))
#    list_plot(bezier(p0, p2, p1, p3, 100))

# The numbers below are given in INCHES.
    chop1 = 0.125
    chop2 = 0.125
    connection_overshoot = 0.125
    p = 2
    q = 3
    a = 0.25
    torus_scale = 0.75
    torus_points = 200
    grid_size = 0.05
    
    #chop1 = 0.1
    #chop2 = 0.035
    #p = 2
    #q = 5
    #a = 0.33
    #torus_scale = 0.75
    #torus_points = 100
    #grid_size = 0.05


    flat_radius = 30. / 25.4
    outer_radius = 40./ 25.4
    coarsen = 1
    thickness = 1./8

    hole_size = 2. / 25.4   #
    hole_radius = 33.5 / 25.4
    num_holes = 3
    flatten_steps = int(5 * flat_radius/grid_size)
    
    JUST_PLOT = False
    
    path = circle()
    
    #th1 = chop1
    #th2 = 2 * pi / q - chop2
    #
    ##Torus Section
    #ts = [rot_z(
    #    torus_knot(p, q, a, torus_points, th1, th2),
    #        n * 2 * pi / q) for n in range(q)]
    #
    ##Torus Bridge Section
    #tbs = [rot_z(
    #    torus_knot(p, q, a, torus_points//3, th2, th1 + 2 * pi / q + connection_overshoot),
    #        n * 2 * pi / q) for n in range(q)]
    #
    ##Connection section
    #cs = [
    #    rad_connection(ts[n], ts[(n+1)%q], (chop1 + chop2)/2., torus_points//3, torus_points//20, connect_start=False)
    #        for n in range(q)
    #]
    #
    ##Bridge outer edge
    #boe = [
    #    rad_connection(tbs[n], cs[n], np=torus_points//3)
    #        for n in range(q)
    #]
    #
    #bs = [
    #    vstack((tbs[n], boe[n][2:]))[::-1] * torus_scale
    #    for n in range(q)
    #]
    #
    #path = []
    #for n in range(q):
    #    path.append(ts[n])
    #    path.append(cs[n])
    #    list_plot(bs[n], 'mo-')
    #    
    #    
    #path = vstack(path) * torus_scale
    ##path = path + (0, 0, a)
    #list_plot(path, 'rx-')
    
    NP = 300
    helix_windings = 4
    helix_amplitude = 0.25
    mean_radius = 12.8 / 25.4
    
    theta = arange(0, NP) * 2*pi / NP
    phi = helix_windings * theta
    a = helix_amplitude #* exp(-10 * cos(theta / 2)**2)
    r = 1 + a * cos(phi)
    path = array([r * cos(theta), r * sin(theta), a * sin(phi)]).T * mean_radius
    path[..., 2] -= path[..., 2].min()
    
    path_rmax = max(sqrt((path[:, :2]**2).sum(1)))
    
#    list_plot(path, 'bx-')


    if JUST_PLOT:
        show()
        sys.exit()
    
    r_out = flat_radius - 0.5 * grid_size
    r_fine_grid = path_rmax * 1.25
    
    grid_coarse = hex_grid(1.5 * r_out, grid_size * 2)
    r = sqrt((grid_coarse**2).sum(1))
    grid_coarse = grid_coarse[where((r < r_out) & (r > r_fine_grid))]

    grid_fine = hex_grid(1.5 * r_fine_grid, grid_size)
    r = sqrt((grid_fine**2).sum(1))
    grid_fine = grid_fine[where(r <= r_fine_grid)]
    
    inner = circle(r=flat_radius, np=200)
    fixed_points = arange(len(inner))
    grid = vstack((grid_coarse, grid_fine))
    list_plot(grid, 'yx')
    
    outer = circle(r=outer_radius, np=200)
    list_plot(outer, 'bx-')
    
    #show()
    #sys.exit()
    
    b = Boundary(outer[::coarsen]) - Boundary(path[::coarsen])
    
    for phi in 2 * pi * arange(num_holes) / num_holes:
        hole = circle((hole_radius * sin(phi), hole_radius * cos(phi)), hole_size / 2., 50)
        list_plot(hole, 'bx-')
        b -= Boundary(hole)
        
    #b = Boundary(outer) - Boundary(outer / 2.)
    
    mesh = b.make_mesh2(grid, fixed_points=inner, min_dist=grid_size/2., relax=flatten_steps, extrude=thickness, dir=-1)
    DATA_ROOT = os.path.expanduser('./airfoil2_meshes')
    if not os.path.exists(DATA_ROOT): os.mkdir(DATA_ROOT)
    out_name = os.path.join(DATA_ROOT, os.path.splitext(sys.argv[0])[0])
    
    mesh.points *= 25.4
    
    mesh.save(out_name + '.ply')
    mesh.save(out_name + '.stl')

        
    #mesh.draw_triangles(plot)
    
    
    #list_plot(b.clip_points(p), 'ro')
    
 #   list_plot(, 'bo')
  
#    list_plot(s1)
#    list_plot(s2)
#    list_plot(path_connection(s1, 0.1, s2, 0.1, 100), 'x')
#    list_plot(rad_connection(s1, s2, chop/2., 100, 10))

    if make_plot: show()