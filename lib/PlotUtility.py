""" Auxiliary functions for plotiing """

__author__ = "Vitaly Vorobyev"
__email__ = "vit.vorobiev@gmail.com"
__date__ = "April 2017"

import numpy as np
import matplotlib.pyplot as plt

from DalitzPhaseSpace import phsp_edge

def show_phase_space(phsp, rtype1, rtype2, num=None, color=None):
    """ Plot phase space edge """
    if color is None:
        color = 'blue'
    mr1, mr2 = phsp_edge(phsp, rtype1, rtype2)
    fig = plt.figure(num=num, figsize=(8, 7))
    plt.plot(mr1, mr2, linestyle='-', color=color)
    plt.gca().set_xlabel(r'$m^{2}_{\mathrm{' + rtype1 + r'}}\ (GeV^{2}/c^{4})$')
    plt.gca().set_ylabel(r'$m^{2}_{\mathrm{' + rtype2 + r'}}\ (GeV^{2}/c^{4})$')
    plt.axis('equal')
    plt.tight_layout()
    # plt.xlim(0, 1.05*max(mr1))
    # plt.ylim(0, 1.05*max(mr2))
    return fig

def plot_density_countours(model, rtype1='AB', rtype2='BC', num=None, figsize=None, nlevels=25, cmap=plt.cm.PuBu):
    """ Contour plot for a model """
    msq1g, msq2g, dens = model.grid_dens(rtype1, rtype2, 250)
    fig = plt.figure(num=num, figsize=figsize)
    levels = np.linspace(0, max(dens.flatten()), 25)
    cntr = plt.contourf(msq1g, msq2g, dens, cmap=plt.cm.PuBu, levels=levels)
    # cntr = plt.contourf(msq1g, msq2g, dens, levels=levels)
    plt.colorbar(cntr)
    return fig

def plot_ddist(data, rt1='AB', rt2='BC', bins=200, num=None):
    """ 2D hist of Dalitz distribution """
    fig = plt.figure(num=num, figsize=(8,6))
    ax = fig.add_subplot(111)
    hist = ax.hist2d(data[rt1], data[rt2], bins=bins, cmap=plt.cm.PuBu)
    # hist = ax.hist2d(mab, mbc, bins=bins)
    fig.colorbar(hist[3], ax=ax, pad=0.02)
    return fig

def scatter_dalitz(data, rt1='AB', rt2='BC', num=None):
    """ Scatter Dalitz plot for small samples """
    fig = plt.figure(num=num, figsize=(8,6))
    plt.scatter(data[rt1], data[rt2], marker='.')
    # ax = fig.add_subplot(111)
    # hist = ax.hist2d(mab, mbc, bins=bins, cmap=plt.cm.PuBu)
    # hist = ax.hist2d(mab, mbc, bins=bins)
    # fig.colorbar(hist[3], ax=ax, pad=0.02)
    return fig

def projections(data, color='steelblue', dens=False, alpha=0.6,
                weights=None, figax=None):
    """ Three Dalitz plot projections """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 3)) if figax is None else figax
    mab_sq, mac_sq, mbc_sq = [data[key] for key in ['AB', 'AC', 'BC']]
    ax1.hist(np.sqrt(mab_sq), bins=100, density=dens, edgecolor="none", color=color, alpha=alpha, weights=weights)
    ax2.hist(np.sqrt(mac_sq), bins=100, density=dens, edgecolor="none", color=color, alpha=alpha, weights=weights)
    ax3.hist(np.sqrt(mbc_sq), bins=100, density=dens, edgecolor="none", color=color, alpha=alpha, weights=weights)
    return (fig, (ax1, ax2, ax3))

def effPlot(eff, x, y, num=None):
    """ Dalitz plot efficiency plot """
    fig = plt.figure(num=num, figsize=(8, 7))
    ax = fig.add_subplot(111)
    X, Y = np.meshgrid(x, y)
    plt.tight_layout()
    # mask = (eff > 0)
    im = ax.pcolormesh(X, Y, eff, cmap=plt.cm.PuBu)
    fig.colorbar(im, ax=ax, pad=0.02)
    return fig
