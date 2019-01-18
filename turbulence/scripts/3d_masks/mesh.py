#!/usr/bin/env python
from numpy import *
from scipy import interpolate
import os, sys
import struct
import boundary

import pickle

def struct_read(f, fs):
    s = struct.Struct(fs)
    return s.unpack(f.read(s.size))

class Mesh(object):
    def __init__(self, points=zeros((0, 3)), triangles=zeros((0, 3))):
        if type(points) == str:
            ext = os.path.splitext(points)[1].lower()

            if ext == '.ply':
                f = open(points, 'rt')
                if not f.readline().startswith('ply'):
                    if not f.readline().startswith('format ascii'):
                        raise ValueError("File is not ascii formatted PLY!  I don't know how to read it.")
                
                    line = ''
                    while not line.startswith('end_header'):
                        line = f.readline()
                        if line.startswith('element'):
                            parts = line.split()
                            if parts[1] == 'vertex':
                                self.num_v = int(parts[2])
                            elif parts[1] == 'face':
                                self.num_f = int(parts[2])
                                
                    if not hasattr(self, 'num_v'):
                        raise ValueError("Couldn't find number of vertices in PLY file.")
        
                    if not hasattr(self, 'num_f'):
                        raise ValueError("Couldn't find number of faces in PLY file.")
                    
                    self.points = zeros((self.num_v, 3))
                    for i in range(self.num_v):
                        line = f.readline()
                        try: X = map(float, line.split())
                        except: X = []
                        if len(X) != 3:
                            raise ValueError("Entry in PLY file (%s) doesn't look like a vertex!" % line)
                        else:
                            self.points[i] = X
                    
                    self.triangles = zeros((self.num_f, 3), dtype='u4')
                    for i in range(self.num_f):
                        line = f.readline()
                        try: X = map(int, line.split())
                        except: X = []
                        if len(X) != 4 and X[0] != 3:
                            raise ValueError("Entry in PLY file (%s) doesn't look like a triangle!\n(This script does not accept quads, etc.)" % line)
                        else:
                            self.triangles[i] = X[1:]
                            
            elif ext == '.stl':
                f = open(points, 'rb')
                
                if f.read(5).lower() == 'solid':
                    raise ValueError("ASCII STL reading not implemented!")
                
                f.seek(80)
                num_triangles = struct_read(f, 'I')[0]
                print num_triangles
                    
                self.points = zeros((num_triangles*3, 3), dtype='d')
                self.triangles = arange(num_triangles * 3, dtype='u4').reshape((num_triangles, 3))
                self.corner_normals = zeros_like(self.points)
                
                j = 0
                for i in range(num_triangles):
                    nx, ny, nz, ax, ay, az, bx, by, bz, cx, cy, cz, att = struct_read(f, '12fH')
#                    print nx, ax, bx, cy, att
                    for k, p in enumerate([(ax, ay, az), (bx, by, bz), (cx, cy, cz)]):
                        jj = j + k
                        
                        self.points[jj] = p                        
                        self.corner_normals[jj] = (nx, ny, nz)
                        
                    j += 3
                
            else:
                raise ValueError("Only STL/PLY files supported.")
                
            
        
        else:    
            self.points = array(points)
            self.triangles = array(triangles, dtype='u4')
            
    def inverted(self):
        return Mesh(self.points.copy(), self.triangles[:, ::-1].copy())
            
    def translate(self, offset):
        return Mesh(self.points + offset, self.triangles.copy())

    def scale(self, s):
        s = asarray(s)
        if not s.shape:
            s = s * ones(3)
        return Mesh(self.points * s, self.triangles.copy())
        
    def draw_triangles(self, draw_func, with_z=False, close=True, *args, **kwargs):
        for t in self.triangles:
            if close:
                t = hstack((t, t[0:1]))
            if with_z:
                x, y, z = self.points[t, :3].T
                draw_func(x, y, z, *args, **kwargs)
            else:
                x, y = self.points[t, :2].T
                draw_func(x, y, *args, **kwargs)
                 
    def copy(self):
        return Mesh(self.points.copy(), self.triangles.copy())
                    
    def volume(self):
        px, py, pz = self.tps(0).T
        qx, qy, qz = self.tps(1).T
        rx, ry, rz = self.tps(2).T
        
        return (px*qy*rz + py*qz*rx + pz*qx*ry - px*qz*ry - py*qx*rz - pz*qy*rx).sum() / 6.
        
    def is_closed(self, tol=1E-12):
        x, y, z = self.points.T
        m2 = self.copy()
        m2.points += 2 * array((max(x) - min(x), max(y) - min(y), max(z) - min(z)))
        v1 = self.volume()
        v2 = m2.volume()
        return abs((v1 - v2) / v1) < tol
        
    def __add__(self, other):
        if hasattr(other, 'points') and hasattr(other, 'triangles'):
            return Mesh(
                points = vstack((self.points, other.points)),
                triangles = vstack((self.triangles, other.triangles + len(self.points)))
            )
            
        else: raise TypeError('Can only add a Mesh to another Mesh')
                    
    def tps(self, n):
        return self.points[self.triangles[:, n]]
                    
    def make_normals(self, normalize=True):
        n = cross(self.tps(2) - self.tps(0), self.tps(1) - self.tps(0))
        if normalize:
            n = norm(n)
        
        self.normals = n
        return n
    
    def make_corner_normals(self):
        if not hasattr(self, 'normals'): self.make_normals()
        
        self.corner_normals = zeros_like(self.points)        
        for i, n in enumerate(self.normals):
            for j in self.triangles[i]:
                self.corner_normals[j] += n
                
        self.corner_normals = norm(self.corner_normals)
        return self.corner_normals
        
    def force_z_normal(self, dir=1):
        if not hasattr(self, 'normals'): self.make_normals()
        
        inverted = where(sign(self.normals[:, 2]) != sign(dir))[0]
        self.triangles[inverted] = self.triangles[inverted, ::-1]
        self.normals[inverted] *= -1

    def save(self, fn, ext=None):
        if ext is None:
            ext = os.path.splitext(fn)[-1][1:]
        
        ext = ext.lower()
        if ext == 'ply':
            self.save_ply(fn)
        elif ext == 'stl':
            self.save_stl(fn)
        else:
            raise ValueError('Extension should be "stl" or "ply".')
            
    def save_stl(self, fn, header=None):
        output = open(fn, 'wb')
        if header is None:
            header = '\x00\x00This is an STL file. (http://en.wikipedia.org/wiki/STL_(file_format))'

        e = '<'

        output.write(header + ' ' * (80 - len(header)))
        output.write(struct.pack(e + 'L', len(self.triangles)))
        
        self.make_normals()
        
        for t, n in zip(self.triangles, self.normals):
            output.write(struct.pack(e + 'fff', *n))
            for p in t:
                x = self.points[p]
                output.write(struct.pack(e + 'fff', *x))
            output.write(struct.pack(e + 'H', 0))
                
        output.close()
            
    def save_ply(self, fn):
        output = open(fn, 'wt')
        output.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
element face %d
property list uchar int vertex_indices
end_header
''' % (len(self.points), len(self.triangles)))
        
        for p in self.points:
            output.write('%10.5f %10.5f %10.5f\n' % tuple(p))
            
        for t in self.triangles:
            output.write('3 %5d %5d %5d\n' % tuple(t))

        output.close()
        
    def relax_z(self, fixed=None, steps=5):
        oz = self.points[:, 2]
        N = len(self.points)
        K = dict() #Stiffness matrix
        
        dist = lambda p1, p2: sqrt(sum((self.points[p1, :2] - self.points[p2, :2])**2))
        
        for t in self.triangles:
            #Triangle side lengths
            a = dist(t[1], t[2])
            b = dist(t[2], t[0])
            c = dist(t[0], t[1])
            s = (a + b + c) / 2
            A = sqrt(s * (s-a) * (s-b) * (s-c)) #Heron's formula
            #print A
            
            p1, p2 = t[1], t[2]
            if p1 > p2: p1, p2 = p2, p1
            pair = (p1, p2)
            K[pair] = K.get(pair, 0.) + (-a**2 + b**2 + c**2) / A

            p1, p2 = t[2], t[0]
            if p1 > p2: p1, p2 = p2, p1
            pair = (p1, p2)
            K[pair] = K.get(pair, 0.) + (a**2 - b**2 + c**2) / A

            p1, p2 = t[0], t[1]
            if p1 > p2: p1, p2 = p2, p1
            pair = (p1, p2)
            K[pair] = K.get(pair, 0.) + (a**2 + b**2 - c**2) / A
            

        
        #nc = histogram(nl.flatten(), bins=range(-1, N+2))[0][1:-1]
        #nc[where(nc == 0)] = 1
        tK = zeros(N)
        for (p1, p2), W in K.iteritems():
            tK[p1] += W
            tK[p2] += W
        
        tK[where(tK == 0)] = 1
        
        for n in range(steps):
            z = zeros(N)
            
            for (p1, p2), W in K.iteritems():
                z[p1] += W * oz[p2]
                z[p2] += W * oz[p1]
                
            z /= tK
            
            if fixed is not None:
                z[fixed] = self.points[fixed, 2]
                
            oz = z
                
        self.points[:, 2] = z
        
    def rot_x(self, angle):
        return Mesh(rot_x(self.points, angle), self.triangles)
    def rot_y(self, angle):
        return Mesh(rot_y(self.points, angle), self.triangles)
    def rot_z(self, angle):
        return Mesh(rot_z(self.points, angle), self.triangles)
        
    def merge_points(self, tol=1E-10, verbose=False):
        new = zeros((len(self.points), 3))
        p_map = zeros(len(self.points), dtype='i')
        
        if verbose:
            print 'Merging %d points...' % len(self.points)
        
        j = 0
        for i, p in enumerate(self.points):
            if j == 0:
                new[j] = p
                p_map[i] = j
                j += 1
            else:
                dist = m1(new[:j] - p)
                j_min = argmin(dist)
                min_dist = dist[j_min]
                if min_dist < tol:
                    p_map[i] = j_min
                else:
                    new[j] = p
                    p_map[i] = j
                    j += 1
                    
                    
        print '   Done.  Eliminated %d redundant points.' % (len(self.points) - j)
        self.points = new[:j]
        self.triangles = p_map[self.triangles]
        
    def project(self, X, x, y, z=None):
        if z is None: z = cross(x, y)
        
        np = zeros_like(self.points)
        for i, a in enumerate((x, y, z)):
            np += self.points[:, i:i+1] * a
            
        np += X
            
        return Mesh(np, self.triangles.copy())
    
def closed_path_interp(X, threshold=2., interp_func=interpolate.interp1d, interp_args=(), interp_kwargs={'axis':0}):
    d = threshold * mag(X[1] - X[0])
    lead = True
    
    for i, x in enumerate(X[2:]):
        if mag(x - X[0]) < d:
            if lead is False:
                print "Found path closure at %d points" % i
                break
        else:
            lead = False
    else:
        print "Warning: didn't find path closure, joining first and last point..."
    
    X = vstack((X[:i], X[0:1]))

    l = zeros(len(X))
    for i in range(1, len(X)):
        l[i] = mag(X[i] - X[i-1]) + l[i-1]

    total_l = l[-1]
    interp = interp_func(l, X, *interp_args, **interp_kwargs)
    return total_l, lambda x: interp(x % total_l)
    
def Gram_Schmidt(*vecs):
    ortho = []
    
    for v in vecs:
        v = array(v)
        for b in ortho:
            v -= b * dot(v, b)
        ortho.append(norm(v))
        
    return ortho

def make_cap(p, dir=1, offset=0):
    x, y = Gram_Schmidt(p[1] - p[0], p[2] - p[0])
    
    p2 = array([dot(p, x), dot(p, y), zeros(len(p))]).T
    b = boundary.Boundary(p2)
    
    m = b.mesh_cap(dir=dir, ignore_wind=True)
    
    return m.triangles + offset
        
    
        
def make_tube(points, nc, cap=False, invert_caps=False):
    N = int(len(points) // nc)
    if nc * N != len(points):
        raise ValueError('Number of points in a tube must be a multiple of the number around the circumference!')
    tris = []

    
    for i in range(N - (1 if cap else 0)):
        i0 = i * nc
        i1 = ((i + 1) % N) * nc
    
        for j0 in range(nc):
            j1 = (j0 + 1) % nc
            tris += [(i0 + j0, i0 + j1, i1 + j0), (i1 + j0, i0 + j1, i1 + j1)]    
        
    if cap:
        dir = -1 if invert_caps else 1
        
        tris += list(make_cap(points[:nc], dir=-1 * dir))
        tris += list(make_cap(points[-nc:], dir=dir, offset = len(points) - nc))
        
    return Mesh(points, tris)
        
def arglocalmin(x):
    return list(where((x < shift(x)) * (x <= shift(x, -1)))[0])    
    
def cone(x0, x1, r0, r1=None, a=None, points=10):
    N = norm(array(x1) - x0)
    if a is None: a = array((1, 0, 0)) if abs(N[0]) < 0.9 else array((0, 1, 0))
    A = norm(a - proj(a, N))
    B = cross(N, A)
    phi = arange(points) * 2 * pi / points
    phi.shape += (1,)
    
    C = cos(phi) * A + sin(phi) * B
    
    if r1 is None:
        p = vstack((array((x1)), x0 + C * r0))
        t = [(0, n + 1, (n+1)%points + 1) for n in range(points)] + \
            [(1, 1+(n+1)%points, n+1) for n in range(1, points-1)]
        
    else:
        p = vstack((x0 + C * r0, x1 + C * r1))
        t = [(n, (n+1)%points + points, n + points) for n in range(points)] + \
            [(n, (n+1)%points, (n+1)%points + points) for n in range(points)] + \
            [(0, (n+1)%points, n) for n in range(1, points-1)] + \
            [(0 + points, n + points, (n+1)%points + points) for n in range(1, points-1)]     
        
    return Mesh(p, t)
    
def column(x0, x1, radius_function=lambda x: 1+2*(x-x**2), rp=20, lp=20):
    N = norm(array(x1) - x0)
    if a is None: a = array((1, 0, 0)) if abs(N[0]) < 0.9 else array((0, 1, 0))
    A = norm(a - proj(a, N))
    B = cross(N, A)
    phi = arange(points) * 2 * pi / points
    phi.shape += (1,)
    
    C = cos(phi) * A + sin(phi) * B
    
    if r1 is None:
        p = vstack((array((x1)), x0 + C * r0))
        t = [(0, n + 1, (n+1)%points + 1) for n in range(points)] + \
            [(1, 1+(n+1)%points, n+1) for n in range(1, points-1)]
        
    else:
        p = vstack((x0 + C * r0, x1 + C * r1))
        t = [(n, (n+1)%points + points, n + points) for n in range(points)] + \
            [(n, (n+1)%points, (n+1)%points + points) for n in range(points)] + \
            [(0, (n+1)%points, n) for n in range(1, points-1)] + \
            [(0 + points, n + points, (n+1)%points + points) for n in range(1, points-1)]     
        
    return Mesh(p, t)
    
def arrow(x0, x2, points=10):
    x1 = (array(x0) + x2) / 2.
    l = mag(array(x0) - x2)
    return cone(x0, x1, l/10., l/10., points=points) + \
           cone(x1, x2, l/4., points=points)

def circle(c=zeros(3), r=1, np=100):
    theta = arange(np, dtype='d') / np * 2 * pi
    p = zeros((np, len(c))) #c might be 3D!
    p[:] = c
    p[:, 0] += r * cos(theta)
    p[:, 1] += r * sin(theta)
    return p

def rot_x(x, a):
    x = array(x)
    rx = x.copy()
    rx[..., 1] = cos(a) * x[..., 1] - sin(a) * x[..., 2]
    rx[..., 2] = cos(a) * x[..., 2] + sin(a) * x[..., 1]
    
    return rx
    
def rot_y(x, a):
    x = array(x)
    rx = x.copy()
    rx[..., 2] = cos(a) * x[..., 2] - sin(a) * x[..., 0]
    rx[..., 0] = cos(a) * x[..., 0] + sin(a) * x[..., 2]
    
    return rx
 
def rot_z(x, a):
    x = array(x)
    rx = x.copy()
    rx[..., 0] = cos(a) * x[..., 0] - sin(a) * x[..., 1]
    rx[..., 1] = cos(a) * x[..., 1] + sin(a) * x[..., 0]
    
    return rx
    
def shift(a, n=1):
    return a[(arange(len(a)) + n) % len(a)]
    
def mag(x, axis=None):
    x = asarray(x)
    if len(x.shape) == 1:
        return sqrt((x**2).sum())
    else:
        if axis is None: axis = len(x.shape) - 1
        m = sqrt((x**2).sum(axis))
        ns = list(x.shape)
        ns[axis] = 1
        return m.reshape(ns)
    
def m1(x):
    return sqrt((x**2).sum(1))
    
def D(x):
    x = array(x)
    return 0.5 * (shift(x, +1) - shift(x, -1)) 
    
def norm(x):
    return x / mag(x)    

def path_frame(x, curvature=False):
    dr = D(x)
    ds = mag(dr)
    T = dr / ds
    N = norm(D(T))
    B = cross(T, N)
    
    if curvature:
        return T, N, B, mag(D(T) / ds)    
    else:
        return T, N, B
    
def N_2D(x):
    return norm(cross((0, 0, 1), norm(D(x))))

def proj(a, b):
    b = norm(b)
    return a.dot(b) * b
    
def bezier(p0, p1, p2, p3, np=10, include_ends=False):
    if include_ends:
        t = (arange(np+2, dtype='d')) / (np+1)
    else:
        t = (arange(np, dtype='d') + 1) / (np+1)
    t.shape = t.shape + (1,)
    
    pt = lambda n: (1-t)**(3 - n) * t**n
        
    return pt(0) * p0 + 3 * (pt(1) * p1 + pt(2) * p2) + pt(3) * p3
    

rot_axes = {'x':rot_x, 'y':rot_y, 'z':rot_z}
    
def surface_of_revolution(path, start=None, end=None, np=100, axis='z'):
    if start is None:
        theta = arange(np, dtype='d') / np * 2 * pi
        cap = False
    else:
        theta = linspace(start, end, ceil(abs(end - start) / (2*pi) * np) + 1)
        cap = True
        
    p = []
    rot = rot_axes[axis.lower()]
    for t in theta:
        p += list(rot(path, t))
        
    return make_tube(p, len(path), cap=cap)
    
def path_connection(p1, s1, p2, s2, np):
    d1 = p1[-1] - p1[-2]
    d1 /= sqrt(sum(d1**2))

    d2 = p2[0] - p2[1]
    d2 /= sqrt(sum(d2**2))
    
    return bezier(p1[-1], p1[-1] + s1 * d1, p2[0] + s2 * d2, p2[0], np, False)
    
def arc(c, r, a1, a2, np=20, max_error=None):
    if max_error is not None:
        np = max(abs(a1 - a2) / 2. * sqrt(r / max_error), 3)

    phi = linspace(a1, a2, np)
    
    x = zeros((np, len(c)))
    x[:] = c
    x[..., 0] += r * cos(phi)
    x[..., 1] += r * sin(phi)
    
    return x
    
def join_meshes(m1, m2, boundaries):
    np = len(m1.points)
    m = m1 + m2
    
    edge_tris = []
    n0 = 0
    for p in boundaries.paths:
        pip = len(p)
        
        for j in range(pip):
            j1 = n0 + j
            j2 = n0 + (j + 1) % pip
            j3 = j1 + np
            j4 = j2 + np
            
            edge_tris += [(j3, j2, j1), (j2, j3, j4)]
        
        n0 += pip
        
    m.triangles = vstack((m.triangles, edge_tris))
    
    return m

def circular_array(x, delta, n, rot=rot_z, offset=0):
    return vstack([rot(x, i * delta + offset) for i in arange(n)])

def oriented_tube(path, outline):
    z = array((0, 0, 1), dtype='d')
    T, N, B = path_frame(path)
    NP = norm(N * (1-z))
    x = outline[:, 0:1]
    y = outline[:, 1:2]
    
    points = []
    for p, n in zip(path, NP):
        points += list(p + n * x + z * y)

    return make_tube(points, len(outline))

def oriented_offset(path, offset):
    z = array((0, 0, 1), dtype='d')
    T, N, B = path_frame(path)
    NP = norm(N * (1-z))

    x, y = offset
    return path + NP * x + z * y

    
def arglocalmin(x):
    return list(where((x < shift(x)) * (x <= shift(x, -1)))[0])

def arglocalmax(x):
    return list(where((x >= shift(x)) * (x > shift(x, -1)))[0])
    
def argclosest(x, y):
    return argmin(mag(x - y))
    
def vector_angle(v1, v2, t):
    v1 = norm(v1)
    v2 = norm(v2)
    
    s = dot(cross(v1, v2), t) 
    c = dot(v1, v2)
    
    return arctan2(s, c)
    
def trace_line(path, thickness=0.05, sides=15):
    num_points = len(path)
    T = norm(D(path))
    
    N = [eye(3)[T[0].argmin()]]
    
    for t in T:
        N.append(norm(N[-1] - dot(N[-1], t) * t))
    
    np = norm(N[-1] - dot(N[-1], T[0]) * T[0]) #Wrap around to measure angle

    N = N[1:]
    
    angle = vector_angle(N[0], np, T[0])
        
    B = cross(T, N)

    theta = arange(num_points) * angle / num_points
    c = cos(theta).reshape((num_points, 1))
    s = sin(theta).reshape((num_points, 1))
    
    N, B = c * N - s * B, s * N + c * B


    phi = 2 * pi * arange(sides) / sides
    phi.shape += (1,)
    x = sin(phi)
    y = cos(phi)

    
    #normals = -vstack([n * x + b * y for n, b in zip(N, B)])
    points = vstack([p + thickness * (n * x + b * y) for p, n, b in zip(path, N, B)])
    
    return make_tube(points, sides)
    

if __name__ == '__main__':
    import sys
    
    m = Mesh(sys.argv[1])
    x, y, z = m.points.T
    
    
    
    print 'Points: %d' % len(m.points)
    print 'Triangles: %d' % len(m.triangles)

    #m.triangles[0] = m.triangles[0, ::-1] #Flip a triangle to test the closed shape detection
    print 'Volume: %f (%s)' % (m.volume(), 'mesh appears closed and properly oriented' if m.is_closed(tol=1E-12) else 'MESH DOES NOT APPEAR CLOSED AND ORIENTED PROPERLY)\n   (A translated copy had a different calculated volume at 1 part in 10^12;\n    this could be a rounding error.')
    print 'Print price: $%.2f (assuming units=mm and $0.30/cc)' % (m.volume() / 1000. * 0.30)
    print 'X extents: (%f, %f)' % (min(x), max(x))
    print 'Y extents: (%f, %f)' % (min(y), max(y))
    print 'Z extents: (%f, %f)' % (min(z), max(z))
    
    from enthought.mayavi import mlab
    mlab.triangular_mesh(x, y, z, m.triangles)
    mlab.show()   