import numpy as np
import chaosfunctions as cf
import integrate as ig
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc

rc('text', usetex = True)
rc('font', family='serif')
rc('savefig', dpi = '300')
rc('xtick', labelsize=15)
rc('ytick', labelsize=15)
rc('xtick.major', size=2.5)
rc('ytick.major', size=2.5)
rc('xtick', direction='in')
rc('ytick', direction='in')

shear = 1
sf = 0.9

def pf(xx, amplitude):
    return cf.torodal_field_squared_q(xx, sf=0.9, shear=shear)+cf.perturb_kink(xx, width = 0.2, amplitude=amplitude, sf=sf, shear = shear)

numstreams=50
numsteps = 40

cmap =  plt.get_cmap('gnuplot')
randoms = np.random.random(numstreams)


streampoints = ig.linepoints(np.array((1.005,0,0)), np.array((1.54568542494, 0, .59568542494)), numstreams )
#perturbations=[0.01, 0.1, 1.0, 2, 5, 10, 0.6]
perturbations=np.linspace(0,1, 40)



for num, amplitude in enumerate(perturbations):
    print('starting poincare plot {}'.format(num))
    fig = plt.figure()
    fig.set_size_inches(10,14)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3.5,1], hspace =0.1)
    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[1,0])
    ax1.set_aspect('equal')

    xx = np.linspace(0, 0.8, 200)
    ax2.plot(xx, sf + shear*xx**2, zorder=1)
    ax2.set_yticklabels([r'$\frac{2}{3}$', r'$1$', r'$1~\frac{1}{3}$'])

    streamlines = ig.stream_multi(streampoints, vvfn=pf, tol=1e-6,  amplitude=amplitude)

    RR = []
    zz = []
    for snum, streamline in enumerate(streamlines):
        ppoints = streamline.makePoincare(normal =  np.array([0,1,0]))
        ax1.scatter(ppoints[:,0], ppoints[:,1], s=.3, color = cmap(randoms[snum]))
        RR.append(streampoints[snum][0])
        zz.append(streampoints[snum][2])
    ax1.plot(RR, zz, color='k')

    twists = []
    locs = []
    for snum, streamline in enumerate(streamlines):
        twists.append(streamline.getTwist())
        locs.append(np.sqrt(np.sum((streampoints[snum]-np.array((1,0,0)))**2)))
    ax2.scatter(locs, 1/np.array(twists), color=cmap(randoms), zorder = 2)
    ax1.set_ylim(-1, 1)

    ax1.text(.2, .8, r'$\epsilon = {:.4f}$'.format(amplitude), fontsize=25)
    ax1.xaxis.set_label_position("top")
    ax1.xaxis.tick_top()
    ax1.set_xlabel(r'$R$', fontsize = 25)
    ax1.set_ylabel(r'$z$', fontsize = 25)

    ax2.set_xlabel(r'$a$', fontsize = 25)
    ax2.set_ylabel(r'$q$', fontsize = 25)
    ax2.set_ylim(0.5, 1.52)
    ax2.set_xlim(0., .8)
    ax2.set_yticks([2/3, 1, 1+1/3])
    ax2.axhline(1.0, c='k')
    ax2.axhline(2/3, c='c')

    plt.savefig('twistprof_ampl_{}.png'.format(num+5), bbox_inches='tight')
    plt.close()



