from mtolib.maxtree import MaxTree
from mtolib.preprocessing import smooth_image

import numpy as np

from math import sqrt, exp, log, floor, ceil, pi
from matplotlib import pyplot as plt

from scipy.special import gammaincinv, gammainc
from scipy.optimize import curve_fit, fsolve, least_squares
from scipy.signal import fftconvolve
from scipy.interpolate import Akima1DInterpolator
from scipy.ndimage import zoom
from scipy.stats import zscore

MT_HAVE_SIGNIFICANT_DESCENDANT = 4
MT_HAVE_DESCENDANT = 16

PI_DIV = 1 / pi

def get_attributes(mto_struct, tree : MaxTree, object_id, psf=None):
    """
    tree        : The Max-Tree to search in
    center_id   : The id of the Max-Tree node representing the object.
    """
    xs = []
    ys = []
    ysi = []
    zsi = []

    len_major = []
    len_minor = []

    bg = tree.nodes[object_id].parent
    V_prev = tree.node_attributes[bg].volume
    I = 0

    V_tot = tree.node_attributes[object_id].volume
    R_tot = sqrt(tree.nodes[object_id].area * PI_DIV)
    V_e = V_tot / 2
    I_e = 0
    I_e_guess = 0
    R_e = R_tot

    area = tree.nodes[bg].area
    node_id = object_id
    while node_id != -1:
        node = tree.nodes[node_id]
        atts = tree.node_attributes[node_id]
        R = sqrt(node.area * PI_DIV)

        len_major.append(atts.len_major)
        len_minor.append(atts.len_minor)

        I += (V_prev - atts.volume) / area
        V_prev = atts.volume
        area = node.area
        V = atts.volume + I * node.area
        if V > V_e:
            R_e = R
        if I_e_guess == 0 and V < V_e:
            I_e_guess = I

        xs.append(R)
        ys.append(V)
        ysi.append(I)

        node_id = one_up(mto_struct, node_id)
    
    I += V_prev / area

    n = 3.5
    b_n = gammaincinv(2*n, 0.5)
    I_e = I * exp(-b_n)
    
    print('I', I, 'V_e', V_e, 'I_e', I_e_guess, 'R_e', R_e)
    while object_id != -1:
        R = sqrt(tree.nodes[object_id].area * PI_DIV)
        Ix = sersic_intensity (4, R, 100, 1e-11)
        zsi.append(Ix)
        object_id = one_up(mto_struct, object_id)
    
    xs0 = [R / R_e for R in xs]

    plt.plot(len_major, len_minor)
    plt.show()

    elongation = [major / minor for major, minor in zip(len_major, len_minor) if major * minor > 10]
    xse = xs0[:len(elongation)]
    plt.plot(xse, elongation)
    plt.show()
    return

    plot(xs0, ysi, zsi)

    """
    print('BLUR TEST')
    weight = False
    n, R_e, I_0, blur_fit = sersic_blur_fit(xs, ysi, I, R_tot, psf=psf, weight=weight)
    err0 = np.sum((blur_fit - ysi) ** 2)
    plot(xs, ysi, blur_fit)
    n, R_e, I_0, blur_fit = sersic_blur_fitR(xs, ysi, I, R_tot, psf=psf, weight=weight)
    err1 = np.sum((blur_fit - ysi) ** 2)
    plot(xs, ysi, blur_fit)

    print('BLUR FIT 2')
    I_0_fit, blur_fit = sersic_blur_fit2(xs, ysi, I, R_tot, psf=psf)
    b_n = -np.log(I_e_guess / I_0_fit)
    n_guess = fsolve(lambda n: gammaincinv(2*n, 0.5) - b_n, (b_n + 1/3) * 0.5)[0]
    print('n_guess', n_guess)
    I_0_fit, blur_fit = sersic_blur_fit2(xs, ysi, I, R_tot, n=n_guess, I_init=I_0_fit, psf=psf)

    print('BLUR FIT R')
    n, R_e, I_0, blur_fit = sersic_blur_fitR(xs, ysi, I, R_tot, n_init=n_guess, I_init=I_0_fit, psf=psf, supersample=1, weight=False)
    err2 = np.sum((blur_fit - ysi) ** 2)
    plot(xs, ysi, blur_fit)

    n, R_e, I_0, blur_fit = sersic_blur_fit(xs, ysi, I, R_tot, n_init=n, R_init=R_e, I_init=I_0, psf=psf)
    err3 = np.sum((blur_fit - ysi) ** 2)
    plot(xs, ysi, blur_fit)
    
    print(err0, err1)
    print(err2, err3)
    #"""

    xs0 = [R / R_e for R in xs]
    plt.plot(xs0[:len(elongation)], elongation)
    plt.xlabel('R / R_e')
    plt.ylabel('Elongation')
    plt.show()

    linear_model(xse, elongation)

def plot(xs, ys_img, ys_fit, x2 = None):
    if x2 is None:
        x2 = xs
    plt.plot(x2, ys_fit, color='#ff7f0e', label='Sérsic fit')
    plt.plot(xs, ys_img, color='#1f77b4', label='Image data')
    plt.xlabel('R')
    plt.ylabel('I')
    plt.legend()
    plt.show()

def tukey(r):
    r = np.asarray(r)
    return np.where(np.abs(r) <= 1, (1/6) * (1 - (1 - r**2)**3), (1/6))

def test(xs, a, b, c, p, q, ys):
    return [a * exp(-b * (x-c)**2) + p*x + q for x in xs]

def linear_model(xs, ys):
    func = lambda params: tukey([params[0] * x + params[1] - y for x, y in zip(xs, ys)])
    res_robust = least_squares(func, [0, 1], bounds=([0,0],[np.inf, np.inf]))
    p, q = res_robust.x
    plot(xs, ys, [p * x + q for x in xs])

    func = lambda xs, a, b, c: [a * exp(-b * (x-c)**2) + p*x + q for x in xs]
    func2 = lambda xs, a, b, c: test(xs, a, b, c, p, q, ys)

    popt, _ = curve_fit(func2, xs, ys, bounds=([0, 0, 0], [np.inf, np.inf, 2]), p0=[1, 4, 1])
    a, b, c = popt

    plt.plot(xs, ys, label='Elongation')
    plt.plot(xs, func(xs, a, b, c), label='Fitted function')
    plt.xlabel('R / R_e')
    plt.ylabel('Elongation')
    plt.legend()
    plt.show()

def sersic_intensity(n, R, R_e, I_e):
    b_n = gammaincinv(2*n, 0.5)
    ex = -b_n * ((R/R_e) ** (1/n) - 1)
    return I_e * np.exp(ex)

ss_psf = None
ss = None

def sersic_intensity_blur(n, xlen, R_e, I_e, psf=None, supersample=1):
    xlen *= supersample
    if psf is None:
        psf = np.load('mock/psf.npy')
    if supersample != 1:
        global ss_psf
        global ss
        if ss == supersample:
            psf = ss_psf
        else:
            psf = zoom(psf, supersample, order=3)
            ss_psf = psf
            ss = supersample
    sersic = get_plane_distance(np.zeros([psf.shape[0], psf.shape[1] + xlen]), floor(psf.shape[0] / 2), floor(psf.shape[1] / 2)) / supersample # generate distance plane
    sersic = sersic_intensity(n, sersic, R_e, I_e)
    sersic = fftconvolve(sersic, psf, mode='valid') # We only need a single vector, since it is the same in all directions from the center
    return sersic[0]

def sersic_intensity_blur2(Rs, n, R_e, I_0, psf=None, supersample=1, avglen=None):
    b_n = gammaincinv(2*n, 0.5)
    I_e = I_0 * exp(-b_n)
    if avglen is None:
        res = sersic_intensity_blur(n, ceil(max(Rs)), R_e, I_e, psf=psf, supersample=supersample)
        indices = np.arange(0, len(res) / supersample, 1 / supersample)
        spline = Akima1DInterpolator(indices, res)
        return spline(Rs)
    return sersic0(Rs, n, R_e, I_e, psf=psf, supersample=supersample, avglen=avglen)

def sersic_intensity_blurR(Rs, ys, n, I_0, R_tot, psf=None, supersample=1, avglen=None, cap=False):
    b_n = gammaincinv(2*n, 0.5)
    I_e = I_0 * exp(-b_n)
    R_e = R_tot / 2
    for R, I_cur in zip(Rs, ys):
        if I_cur > I_e:
            break
        if cap and R < R_tot / 10:
            break
        R_e = R
    return sersic0(Rs, n, R_e, I_e, psf=psf, supersample=supersample, avglen=avglen)

def sersic0(Rs, n, R_e, I_e, psf=None, supersample=1, avglen=None):
    if avglen is None:
        res = sersic_intensity_blur(n, ceil(max(Rs)), R_e, I_e, psf=psf, supersample=supersample)
        indices = np.arange(0, len(res) / supersample, 1 / supersample)
        spline = Akima1DInterpolator(indices, res)
        return spline(Rs)
    avgidx = ceil(Rs[-avglen])
    res1 = sersic_intensity_blur(n, avgidx, R_e, I_e, psf=psf, supersample=supersample)
    res2 = sersic_intensity_blur(n, ceil(max(Rs)), R_e, I_e, psf=psf, supersample=1)
    r1 = np.arange(0, int(len(res1) / supersample), 1 / supersample)
    r2 = range(int(len(res1) / supersample), len(res2))
    indices = np.concatenate((r1, np.array(r2)))
    res = np.concatenate((res1, res2[(avgidx+1):]))
    spline = Akima1DInterpolator(indices, res)
    return spline(Rs)

def sersic_blur_fit(xs, ys, I, R_tot, n_init=1, R_init=None, I_init=None, psf=None, weight=True):
    if R_init is None:
        R_init = R_tot / 2
    if I_init is None:
        I_init = I

    ys = [y * 1e14 for y in ys]
    I *= 1e14
    I_init *= 1e14

    weights = np.ones(len(xs))
    if weight:
        weights[-1] = 2

    popt, _ = curve_fit(sersic_intensity_blur2, xs, ys, bounds=([0, R_init / 2, I],[12, R_init * 2, np.inf]), p0=[n_init, R_init, I_init], sigma=weights)
    n, R_e, I_0 = popt
    print('fit n', n, 'R_e', R_e, 'I_0', I_0 * 1e-14)
    return n, R_e, I_0 * 1e-14, sersic_intensity_blur2(xs, n, R_e, I_0 * 1e-14, psf=psf)

def sersic_blur_fitR(xs, ys, I, R_tot, n_init=1, I_init=None, psf=None, supersample=1, lad=False, weight=True):
    if I_init is None:
        I_init = I
    ys = [y * 1e14 for y in ys]
    I *= 1e14
    I_init *= 1e14
    func = lambda Rs, n, I_0: sersic_intensity_blurR(Rs, ys, n, I_0, R_tot, psf=psf, supersample=supersample, avglen=None, cap=True)
    
    weights = np.ones(len(xs))
    if weight:
        weights[-1] = 2

    popt, _ = curve_fit(func, xs, ys, bounds=([0, I],[12, np.inf]), p0=[n_init, I_init], sigma=weights)
    n, I_0 = popt

    b_n = gammaincinv(2*n, 0.5)
    I_e = I_0 * exp(-b_n)
    R_e = R_tot / 2
    for R, I_cur in zip(xs, ys):
        if I_cur > I_e:
            break
        R_e = R

    print('fit n', n, 'R_e', R_e, 'I_0', I_0 * 1e-14)
    return n, R_e, I_0 * 1e-14, sersic_intensity_blur2(xs, n, R_e, I_0 * 1e-14, psf=psf, avglen=None)

def sersic_blur_fit2(xs, ys, I, R_tot, n=4, I_init=None, psf=None):
    ys = [y * 1e14 for y in ys]
    I *= 1e14
    if I_init is None:
        I_init = 10*I

    func = lambda Rs, I_0: sersic_intensity_blurR(Rs, ys, n, I_0, R_tot, psf=psf)
    weights = np.ones(len(xs))
    weights[-1] = 1e-10
    popt, _ = curve_fit(func, xs, ys, p0=[10*I], sigma=weights)
    I_0 = popt[0]

    #"""
    b_n = gammaincinv(2*n, 0.5)
    I_e = I_0 * exp(-b_n)
    R_e = R_tot / 2
    for R, I_cur in zip(xs, ys):
        if I_cur > I_e:
            break
        R_e = R
    #"""

    print('fit I_0', I_0 * 1e-14, 'R_e', R_e)
    return I_0 * 1e-14, sersic_intensity_blur2(xs, n, R_e, I_0 * 1e-14, psf=psf)

def get_plane_distance(data, xc, yc):
    nx, ny = len(data[0]), len(data)
    x, y = np.ogrid[-yc:float(ny) - yc, -xc:float(nx) - xc]
    return np.sqrt(x ** 2. + y ** 2.)

def biggest_object(tree : MaxTree, object_ids):
    objects = set(object_ids.flatten())
    biggest = -1
    area = -1
    for id in objects:
        nodeArea = tree.nodes[id].area
        if nodeArea > area:
            biggest = id
            area = nodeArea
    return biggest

def one_up(mto_struct, node_id):
    if check_flag(mto_struct, node_id, MT_HAVE_SIGNIFICANT_DESCENDANT):
        return mto_struct.main_branches[node_id]
    if check_flag(mto_struct, node_id, MT_HAVE_DESCENDANT):
        return mto_struct.main_power_branches[node_id]
    return -1

def check_flag(mto_struct, idx, flag):
    return mto_struct.flags[idx] & flag != 0