"""Usage:
python plot.py filename wf_number orbital(s) normalize plot_xdir [plot_xmin plot_xmax [plot_ydir plot_ymin plot_ymax]]
"""
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import nemodata as nd

filename = sys.argv[1]
wfnumber = int(sys.argv[2])
orbitals = sys.argv[3]  # specific string (below) for wavefunction, otherwise comma separated list of orbitals
normalize = int(sys.argv[4])
plot_xdir = sys.argv[5]
if len(sys.argv) > 6:
    plot_xmin = float(sys.argv[6])
    plot_xmax = float(sys.argv[7])
    if len(sys.argv) > 8:
        plot_ydir = sys.argv[8]
        plot_ymin = float(sys.argv[9])
        plot_ymax = float(sys.argv[10])

# read atomic coordinates
r_fn = filename + '.nd_rAtom_0'
print('Reading ' + r_fn)
r_atom = nd.parse(r_fn)

# test if the wavefunction or the orbitals are to be plotted
is_wf = (orbitals == 'wf' or
         orbitals == 'wavefn' or
         orbitals == 'wavefunction')

# read wavefunction
if is_wf:
    data_fn = filename + '.nd_wf_' + str(wfnumber)
else:
    data_fn = filename + '.nd_evec_' + str(wfnumber)
print('Reading ' + data_fn)
state = nd.parse(data_fn)
# optional normalization
if normalize:
    if is_wf:
        state.data = state.data / np.sum(state.data)
    else:
        state.data = state.data / np.linalg.norm(state.data)

# the atomic coefficients are the sum of the relevant orbital coefficients
if is_wf:
    psi2 = state.data
else:
    print('Calculating probability density')
    nd.evec_reshape(state)
    psi2 = np.zeros(state.data.shape)
    if orbitals == 'all':
        orbital_list = state.data.dtype.names
    else:
        orbital_list = orbitals.split(',')
    for orbital in orbital_list:
        # fast mod-squared complex number
        psi2 += state.data[orbital].real**2 + state.data[orbital].imag**2

print('Processing wavefunctions...')
# get the right limits
mask = np.ones(state.data.shape, dtype=bool)
if len(sys.argv) > 6:
    xmin = min(plot_xmin, plot_xmax)
    xmax = max(plot_xmin, plot_xmax)
    mask &= (r_atom.data[plot_xdir] >= xmin) & (r_atom.data[plot_xdir] <= xmax)
    if len(sys.argv) > 8:
        ymin = min(plot_ymin, plot_ymax)
        ymax = max(plot_ymin, plot_ymax)
        mask &= (r_atom.data[plot_ydir] >= ymin) & (r_atom.data[plot_ydir] <= ymax)
# project wavefunction onto the axis/plane
if len(sys.argv) > 8:
    proj = nd.project(r_atom.data[mask], psi2[mask], plot_xdir, plot_ydir)
else:
    proj = nd.project(r_atom.data[mask], psi2[mask], plot_xdir)

# save data
save_fn = sys.argv[1] + '_' + sys.argv[2]
if not is_wf:
    save_fn += '[' + orbitals + ']'
save_fn += '_' + sys.argv[5]
if len(sys.argv) > 6:
    save_fn += sys.argv[6] + '-' + sys.argv[7]
    if len(sys.argv) > 8:
        save_fn += '_' + sys.argv[8] + sys.argv[9] + '-' + sys.argv[10]

np.savetxt(save_fn + '_data.txt', proj, fmt='%.10g')

# plot
print('Plotting...')
#plt.rcParams['figure.figsize'] = [26, 12]
plt.rcParams['figure.dpi'] = 300.0

if len(sys.argv) > 8:
    if len(sys.argv) > 11 and sys.argv[11] == 'log':
        normalization = mpl.colors.LogNorm(vmin=0.001*max(proj[:, 2]),
                                           vmax=max(proj[:, 2]))
        save_fn += '_log'
    else:
        #normalization = mpl.colors.PowerNorm(gamma=0.2)
        normalization = mpl.colors.PowerNorm(gamma=0.2, vmin=0, vmax=0.02)
    plt.tripcolor(proj[:, 0], proj[:, 1], proj[:, 2],
                  shading='gouraud',
                  norm=normalization,
                  cmap='inferno')
    plt.gca().set_aspect('equal', 'box')
    image_ratio = (ymax - ymin) / (xmax - xmin)
    plt.colorbar(fraction=0.046*image_ratio)
else:
    plt.bar(proj[:, 0], proj[:, 1])

plt.xlabel(sys.argv[5] + ' (nm)')
if len(sys.argv) > 6:
    plt.xlim(plot_xmin, plot_xmax)
    if len(sys.argv) > 8:
        plt.ylim(plot_ymin, plot_ymax)
        plt.ylabel(sys.argv[8] + ' (nm)')
#plt.title(sys.argv[1] + ', state #' + sys.argv[2])

plt.savefig(save_fn + '.png', bbox_inches='tight')
