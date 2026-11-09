#!/usr/bin/python3

# Python wrapper written by T. Merle to interpolate:
# (i)  MARCS model atmospheres (standard solar-scaled chemical composition)
# (ii) non-LTE departure coefficients for a given species
# based on the Fortran interpolation code of T. Masseron
# for the NLTE version of Turbospectrum of B. Plez
#
# interp_nlte_3.py = interp_nlte_2.py + protection against holes in the departure coefficient grid.
# The grid of departure coefficients does not contain every MARCS model (e.g. s4500_g+1.0_m1.0_t02_z-3.00 for Fe).
# The Fortran code does not notice it: the index of the missing corner is left uninitialised and a random
# departure coefficient record is interpolated. Here, all eight corners are checked beforehand, and if one is
# missing the bracket is widened by one grid step along one axis until the eight corners exist (the smallest
# widening is preferred). The Fortran output is also checked to make sure the eight records that were read are the expected ones.
#
# interp_nlte_4.py = interp_nlte_3.py for any element with a grid in dc/<element>/ (dc.bin + master.dat), including the
# grids with several abundances per model atmosphere (all of them except Fe). The departure coefficients are not
# interpolated in A(X), each of the eight corners uses one record. The way the Fortran code picks that record cannot be
# trusted with such grids: it takes the first record within 0.099 dex (else 0.199 dex) rather than the closest one; a gap
# in the abundances of a model leaves the index uninitialised (a random record, or a crash); its upper limit of the
# abundances starts at 0 (abu_max = 0.), so the abundances of a model that are all negative are not clamped either (same
# outcome); and with at most 20000 records in the master file it replaces A(X) by [Fe/H] + 7.50 (made for the Fe grid).
# So for these grids the record is chosen here, the one with the abundance closest to A(X) for each corner, and the Fortran
# code gets a small master file with only those records, listed at the requested A(X), which leaves it nothing to get
# wrong. The records it reports are compared with the choice, as in interp_nlte_3.py, and the abundance used at each corner is
# printed. A grid with one abundance per model (Fe) goes through the Fortran code as before. The line that
# starts with DC_INFO is read by the calling script.

import json
import os
import re
import sys
import pathlib as pl
import subprocess as sp
import numpy as np
import matplotlib.pyplot as plt

#User defined parameters
opath = os.environ.get('INTERP_OPATH', '/home/student/NICO/data/interpolated/HE22081239') #Output directory (env var INTERP_OPATH overrides it)
satms = os.listdir('st/s') #Spherical model atmospheres
patms = os.listdir('st/p') #Parallel model atmospheres
bla = True  #Verbosity
plot = os.environ.get('INTERP_PLOT', '1') == '1' #Control plot saved in pdf (env var INTERP_PLOT=0 disables it)
interactive_plot = plot and os.environ.get('INTERP_INTERACTIVE', '1') == '1' #Display interactive plot (works only if plot = True)
widen_axis = os.environ.get('INTERP_WIDEN_AXIS', '') #Restrict the bracket widening to one axis: T, g or Z (empty = any)
temp = False #Write temporary files (for checking)

os.mkdir(opath) if not os.path.exists(opath) else None

if len(sys.argv) < 7:
    print(f'Usage: interp_nlte.py mass teff logg [Fe/H] element abundance')
    print(f'Example: interp_nlte.py 1.0 5125 2.21 -0.67 Ba 3.90')
    print(f'Mass values must be 0.0 for plane-parallel models, or in [0.5, 1.0, 2.0, 5.0] for spherical ones.')
    quit(1)

#Desired interpolated parameters from the command line
mass  = float(sys.argv[1])
teff0 = int(float(sys.argv[2]))
logg0 = float(sys.argv[3])
feoh0 = float(sys.argv[4])
elemt = sys.argv[5]
abund = float(sys.argv[6])


def find_aofe(feoh):
    '''Determine the enrichment in alpha elements in function of the metallicity
    '''
    if feoh >= +0.00:
        return +0.00
    elif feoh <= -1.00:
        return +0.40
    else:
        return -0.4 * feoh

def build_fn(teff, logg, feoh, interpolated=False, verbose=True):
    '''Build MARCS model atmosphere filename as a function of the effective temperature, surface gravity and metallicity
       Some original MARCS models (.mod) are missing and were interpolated (.int) to fullfil the grid
       By default, microturbulence velocity (vmic) is
       1.0 for plan parallel models
       2.0 for spherical models
    '''

    aofe = find_aofe(feoh)

    if mass == 0.0:
        geo = 'p'
        vmic = '1'
    else:
        geo = 's'
        vmic = '2'

    chain = f"{geo}{teff}_g{logg:+3.1f}_m{mass}_t0{vmic}_st_z{feoh:+4.2f}_a{aofe:+4.2f}_c+0.00_n+0.00_o{aofe:+4.2f}_r+0.00_s+0.00"
    #print(chain)

    #Just to get the name
    if interpolated:
        return f'{chain}.int'

    #For plane-parallel models
    if geo == 'p':
        if f'{chain}.mod' in patms:
            fn = f'st/{geo}/{chain}.mod'
        elif f'{chain}.int' in patms:
            fn = f'st/{geo}/{chain}.int'
        else:
            if verbose:
                print(f'{chain}.mod and .int does not exist')
            fn = ''
    #For spherical models
    elif geo == 's':
        if f'{chain}.mod' in satms:
            fn = f'st/{geo}/{chain}.mod'
        elif f'{chain}.int' in satms:
            fn = f'st/{geo}/{chain}.int'
        else:
            if verbose:
                print(f'{chain}.mod and .int does not exist')
            fn = ''

    return fn

def corner_id(param):
    '''Identifier of a model atmosphere in the master file of the departure coefficients (filename without extension)
    '''
    return build_fn(*param, interpolated=True)[:-len('.int')]

def read_master(fn):
    '''Read the master file of the departure coefficients, returns
       {model identifier: [(A(X), pointer in the binary file, record number, other columns), ...]} in the order of the file
       The record number is the index used by the Fortran code: the lines after the header, comments included
       The other columns are the ones between the identifier and A(X)
    '''
    ids = {}
    with open(fn) as fd:
        next(fd)
        for irec, line in enumerate(fd, start=1):
            if line.startswith("'"):
                _, model, rest = line.split("'", 2)
                fields = rest.split()
                ids.setdefault(model, []).append((float(fields[-2]), int(fields[-1]), irec, fields[:-2]))
    return ids

#Initialize atmospheric parameters
if mass in [0.0, 1.0]:
    #MARCS model parameters for plan-parallel (m0.0) and spherical with 1.0 M☉ (m1.0)
    teffs = np.array([2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4250, 4500, 4750, 5000, 5250, 5500, 5750, 6000, 6250, 6500, 6750, 7000, 7250, 7500, 7750, 8000])
    loggs = np.array([-0.5, +0.0, +0.5, +1.0, +1.5, +2.0, +2.5, +3.0, +3.5, +4.0, +4.5, +5.0, +5.5])
    feohs = np.array([-5.00, -4.00, -3.00, -2.50, -2.00, -1.50, -1.00, -0.75, -0.50, -0.25, +0.00, +0.25, +0.50, +0.75, +1.00])
elif mass in [0.5, 2.0, 5.0]:
    #MARCS model parameters for spherical with 0.5, 2.0 and 5.0 M☉ (m0.5, m2.0, m5.0)
    teffs = np.array([2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 5000, 6000, 7000, 8000])
    loggs = np.array([+0.0, +1.0, +2.0, +3.0])
    feohs = np.array([-5.00, -4.00, -3.00, -2.50, -2.00, -1.50, -1.00, -0.75, -0.50, -0.25, +0.00, +0.25, +0.50, +0.75, +1.00])
else:
    print(f'Mass should be one of these values: 0.0 for plan parallel, or [0.5, 1.0, 2.0, 5.0] for spherical models.')
    quit(1)

#Look for the closest parameters
if teff0 in teffs:
    teff1 = teff0
else:
    teff1 = teffs[np.argmin(abs(teffs-teff0))]

if logg0 in loggs:
    logg1 = logg0
else:
    logg1 = loggs[np.argmin(abs(loggs-logg0))]

if feoh0 in feohs:
    feoh1 = feoh0
else:
    feoh1 = feohs[np.argmin(abs(feohs-feoh0))]

#Selection of min and max for each parameter that encompasses the desired values
t1, g1, z1 = teffs[np.argmin(abs(teffs-teff0))], loggs[np.argmin(abs(loggs-logg0))], feohs[np.argmin(abs(feohs-feoh0))]

if t1 < teff0:
    i = +1
else:
    i = -1
t2 = teffs[np.argmin(abs(teffs-teff0))+i]

if g1 < logg0:
    i = +1
else:
    i = -1
g2 = loggs[np.argmin(abs(loggs-logg0))+i]

if z1 < feoh0:
    i = +1
else:
    i = -1
z2 = feohs[np.argmin(abs(feohs-feoh0))+i]

tmin, tmax = min(t1, t2), max(t1, t2)
gmin, gmax = min(g1, g2), max(g1, g2)
zmin, zmax = min(z1, z2), max(z1, z2)

#In case we are on the grid points
if teff1 == teff0:
    tmin = tmax = teff0
if logg1 == logg0:
    gmin = gmax = logg0
if feoh1 == feoh0:
    zmin = zmax = feoh0


#Section that deal with the NLTE departure coefficients
ifn1 = f"dc/{elemt}/dc.bin"
ifn2 = f"dc/{elemt}/master.dat"
if not (os.path.exists(ifn1) and os.path.exists(ifn2)):
    print(f'NLTE departure coefficients for {elemt} not found: {ifn1} and {ifn2} are needed.')
    quit(1)
n_ifn2 = sum(1 for _ in open(ifn2)) # Rapidity and concision (see https://www.developpez.net/forums/d427131/autres-langages/python/general-python/compter-nombre-ligne-d-fichier-lire/)
n_ifn2 -= 1                         # To remove the first line which is the header
dc_ids = read_master(ifn2)          # {model identifier: records}, used to check that the departure coefficients exist
n_abund = max(len(recs) for recs in dc_ids.values()) #Largest number of abundances for one model
with open(ifn2) as fd:
    is_subset = 'SUBSET' in fd.readline()   #Small grid cut out of a full one by make_dc_subset.py


def cube_params(cube):
    '''The eight corners of the cube ((tmin, tmax), (gmin, gmax), (zmin, zmax)), sorted for the Fortran code of T. Masseron
       (tmin, gmin, zmin), (tmin, gmin, zmax), (tmin, gmax, zmin), (tmin, gmax, zmax), (tmax, gmin, zmin), ...
    '''
    return [(t, g, z) for t in cube[0] for g in cube[1] for z in cube[2]]

def missing_corners(cube):
    '''List of (corner, reason) for the corners of the cube that have no model atmosphere or no departure coefficients
    '''
    bad = []
    for param in cube_params(cube):
        if build_fn(*param, verbose=False) == '':
            bad.append((param, 'no model atmosphere'))
        elif corner_id(param) not in dc_ids:
            bad.append((param, 'no departure coefficients'))
    return bad

def widened_cubes(cube):
    '''Cubes obtained by moving one edge of the bracket outward by one grid step, as (span ratio, axis, name, cube)
       The span ratio is the width of the bracket along the widened axis over its original width
    '''
    cands = []
    for iax, (name, grid) in enumerate([('T', teffs), ('g', loggs), ('Z', feohs)]):
        lo, hi = cube[iax]
        if lo == hi or (widen_axis and widen_axis != name): #On a grid point (or axis not allowed): nothing to widen
            continue
        ilo, ihi = int(np.argmin(abs(grid-lo))), int(np.argmin(abs(grid-hi)))
        for nlo, nhi in [(ilo-1, ihi), (ilo, ihi+1)]:
            if nlo < 0 or nhi >= len(grid):
                continue
            new = list(cube)
            new[iax] = (grid[nlo], grid[nhi])
            cands.append(((grid[nhi]-grid[nlo])/(hi-lo), iax, name, tuple(new)))
    return cands

#Check that the departure coefficients exist for the eight corners, widen the bracket if not
cube = ((tmin, tmax), (gmin, gmax), (zmin, zmax))
bad = missing_corners(cube)
if bad:
    print(f'\nWARNING: {len(bad)} of the 8 corners of the bracket cannot be used for {elemt}:')
    for param, reason in bad:
        print(f'  Teff={param[0]} logg={param[1]:+3.1f} [Fe/H]={param[2]:+4.2f}: {reason}')

    cands = [c for c in widened_cubes(cube) if not missing_corners(c[3])]
    if not cands:
        print(f'No bracket widened by one grid step along one axis has all its corners. Try other stellar parameters.')
        if is_subset:
            print(f'dc/{elemt} is only a subset of the full grid (see its README.md): add these parameters to POINTS in make_dc_subset.py and run that again.')
        quit(1)

    ratio, iax, name, new = min(cands, key=lambda c: (round(c[0], 6), c[1])) #Smallest widening, ties go to T, then logg, then [Fe/H]
    print(f'Widening the bracket along {name} (x{ratio:.1f}):')
    print(f'  Teff   {cube[0][0]} - {cube[0][1]}  ->  {new[0][0]} - {new[0][1]}')
    print(f'  logg   {cube[1][0]:+3.1f} - {cube[1][1]:+3.1f}  ->  {new[1][0]:+3.1f} - {new[1][1]:+3.1f}')
    print(f'  [Fe/H] {cube[2][0]:+4.2f} - {cube[2][1]:+4.2f}  ->  {new[2][0]:+4.2f} - {new[2][1]:+4.2f}')
    (tmin, tmax), (gmin, gmax), (zmin, zmax) = new

#List of the eight models atmospheres sorted for the Fortran code of T. Masseron
params = cube_params(((tmin, tmax), (gmin, gmax), (zmin, zmax)))

fns = []
for param in params:
    fns.append(build_fn(*param))

if '' in fns:
    print(f'Some model atmospheres are missing, the interpolation cannot be done.')
    quit(1)

def closest_record(param):
    '''The record of this corner with the abundance closest to A(X) (the first of the file for equal distances)
    '''
    return min(dc_ids[corner_id(param)], key=lambda rec: round(abs(rec[0] - abund), 6))

records = [closest_record(param) for param in params]

#With several abundances per model the Fortran code only sees the chosen records, listed at the requested abundance
#(more than 20000 lines: below that it would replace the abundance by [Fe/H] + 7.50)
master_used, n_master_used, expected_index = ifn2, n_ifn2, [rec[2] for rec in records]
if n_abund > 1:
    master_used, n_master_used = 'master_selected.dat', 20001
    chosen = {corner_id(param): rec for param, rec in zip(params, records)}
    line_of = {model: iline for iline, model in enumerate(chosen, start=1)}
    expected_index = [line_of[corner_id(param)] for param in params]
    with open(master_used, 'w') as fd:
        fd.write('# the records of one interpolation, selected by interp_nlte_4.py\n')
        for model, rec in chosen.items():
            fd.write(f"'{model}' {' '.join(rec[3])} {abund:.4f} {rec[1]}\n")
        fd.writelines("'PADDING' 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1\n" for _ in range(n_master_used - len(chosen)))

if temp:
    #Save the list of model atmospheres required to perform the interpolations
    np.savetxt('atml.dat', fns, fmt='%s', header='Last cube model atmospheres')

    #Save the atmospheric parameters in a separate file for examination
    ofd = open('atm_par_total.dat', 'a')
    for param in params:
        print(f'{param[0]} {param[1]:5.2f} {param[2]:5.2f}', file=ofd)


if bla:
    print(f"\nModel atmosphere interpolation:")
    print(f"For mass = {mass}")
    print(f"Input parameters:   {teff0}, {logg0}, {feoh0}")
    print(f"Closest parameters: {teff1}, {logg1}, {feoh1}")
    print(f"Teff_min, Teff_max:      {tmin},  {tmax}")
    print(f"logg_min, logg_max:      {gmin:+3.1f},  {gmax:+3.1f}")
    print(f"[Fe/H]_min, [Fe/H]_max: {zmin:+4.2f}, {zmax:+4.2f}")
    print(f'\nNumber of effective model atmospheres: {len(fns)}')

    for fn in fns:
        print(fn)

    print(f'\nNLTE departure coefficients interpolation for {elemt} with abundance {abund}:')
    print(f'Departure coefficient binary file: {ifn1}')
    print(f'Master file:                       {ifn2}')

#Output filenames
pl.Path(opath).mkdir(exist_ok=True, parents=True)
ofn1 = build_fn(teff0, logg0, feoh0, interpolated=True)
ofn2 = ofn1.replace('.int', '.alt')
ofn3 = ofn1.replace('.int', f'_{elemt}_{abund:4.2f}_dc.dat')
ofn4 = ofn1.replace('.int', '.pdf')

#print(ofn1)

#Interpolation is done here
cmd = 'interpol_modeles_nlte<<EOF\n"{}"\n"{}"\n"{}"\n"{}"\n"{}"\n"{}"\n"{}"\n"{}"\n"{}"\n"{}"\n"{}"\n"{}"\n"{}"\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\nEOF'.format(*fns, ofn1, ofn2, ofn3, ifn1, master_used, n_master_used, teff0, logg0, feoh0, abund, ".FALSE.", ".FALSE.", "toto")

if bla:
    print(f'\n{cmd}')

proc = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)

#Display errors if any
if proc.returncode != 0:
    print(proc.stdout.splitlines())
    print(proc.stderr.splitlines())
    quit(1)

#Check that the departure coefficient records read by the Fortran code are the expected ones
#For each corner the Fortran code prints 'Teff logg [Fe/H] A(X) <record number>' when it selects the records, then
#'NLTE: <i> <model identifier (cut at 50 characters)> <pointer>' (asterisks for pointers of more than 10 digits)
stdout = proc.stdout.decode(errors='replace').splitlines()
iarray = next((i for i, line in enumerate(stdout) if 'new NLTE array' in line), len(stdout))
num = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?' #List-directed output: 0.0000000E+00, 7.99999982E-02 for small numbers
picked = []
for line in stdout[:iarray]:
    match = re.match(rf'\s*{num}\s+{num}\s+{num}\s+({num})\s+(\d+)\s*$', line)
    if match:
        picked.append((float(match.group(1)), int(match.group(2))))
picked = picked[-8:]

found = {}
for line in stdout:
    match = re.match(r'\s*NLTE:\s+(\d)\s+(\S+)\s+(\d+|\*+)\s*$', line)
    if match:
        found[int(match.group(1))] = (match.group(2), int(match.group(3)) if match.group(3).isdigit() else None)

wrong = []
for icorner, (param, rec, index) in enumerate(zip(params, records, expected_index), start=1):
    model = corner_id(param)
    got = found.get(icorner)
    if (len(picked) != 8 or picked[icorner-1][1] != index or got is None or got[0] != model[:50]
            or (got[1] is not None and got[1] != rec[1])):
        wrong.append(icorner)

if bla or wrong:
    print(f'\nDeparture coefficient records read by the interpolation code:')
    for icorner, (param, rec) in enumerate(zip(params, records), start=1):
        model = corner_id(param)
        got = found.get(icorner, ('-', 0))
        ptr_read = 'overflow' if got[1] is None else got[1]
        print(f'{icorner} {"WRONG" if icorner in wrong else "ok   "} expected {model[:50]} A={rec[0]:+.2f} {rec[1]:>11d}  read {got[0]} {ptr_read:>11}')

if wrong:
    print(f'\nERROR: the departure coefficients read are not the expected ones, the outputs are discarded.')
    sp.run(f'rm -f {ofn1} {ofn2} {ofn3} fort.25 modele.sm master_selected.dat', shell=True)
    quit(1)

#Models whose closest abundance is not next to A(X): A(X) is outside their range, or in a gap of their abundances
corners = [f'T{param[0]} g{param[1]:+3.1f} z{param[2]:+4.2f}' for param in params]
far = [n_abund > 1 and abs(rec[0] - abund) > 0.1 for rec in records]
if any(far):
    print(f'\nWARNING: no departure coefficients within 0.1 dex of A({elemt}) = {abund:.2f} at {len({c for c, off in zip(corners, far) if off})} of the {len(set(corners))} models, the closest abundance was used there:')
    for corner, rec, off in dict.fromkeys(zip(corners, [r[0] for r in records], far)):
        if off:
            print(f'  {corner}: A({elemt}) = {rec:+.2f}')

print('DC_INFO ' + json.dumps(dict(element=elemt, requested=abund, used=[rec[0] for rec in records], far=far, corners=corners,
                                   abundances_per_model=n_abund, subset=is_subset)))

#Control plot
if plot:
    plt.rc('font', size=10)
    plt.figure(figsize=(12,6))
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)
    for atm, param in zip(fns, params):
        cmd = "sed -n '26,81 p' {} > modelatm.dat".format(atm)
        proc = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
        if proc.returncode != 0:
            print(proc.stdout.splitlines())
            print(proc.stderr.splitlines())
            quit(1)
        tau, t, pe, pg, pr = np.loadtxt('modelatm.dat', dtype=float, unpack=True, usecols=(2, 4, 5, 6, 7))
        ax1.plot(tau, t, lw=0.5, label='T={} K logg={} [Fe/H]={}'.format(param[0], param[1], param[2]))
        ax2.plot(tau, pe, lw=0.5)
        ax3.plot(tau, pg, lw=0.5)
        ax4.plot(tau, pr, lw=0.5)
    for ax in [ax3, ax4]:
        ax.set_xlabel('log$\\tau_\\mathrm{Rosseland}$')
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim(min(tau), max(tau))
    ax1.set_ylabel('$T$ [K]')
    ax2.set_ylabel('$P_\\mathrm{{e}}$ [cgs]')
    ax3.set_ylabel('$P_\\mathrm{{g}}$ [cgs]')
    ax4.set_ylabel('P$_\\mathrm{{rad}}$ [cgs]')
    for ax in [ax2, ax3, ax4]:
        ax.set_yscale('log')
    try:
        tau, t, pe_log, pg_log = np.genfromtxt('{}'.format(ofn1), unpack=True, usecols=(0, 1, 2, 3), skip_header=1, skip_footer=8)
        ax1.plot(tau, t, 'k-', label='T={} K logg={} [Fe/H]={:>5.2f}'.format(teff0, logg0, feoh0))
        ax2.plot(tau, 10**pe_log, 'k-')
        ax3.plot(tau, 10**pg_log, 'k-')
    except OSError: #MULTI format not yet supported for the control plot
        pass
    ax1.legend(fontsize=8)
    ax4.text(-5.5, 1.2*max(pr), s='/!\\ No $P_\\mathrm{rad}$ variable in the interpolated atmosphere')

    plt.suptitle(ofn1)

    plt.savefig(ofn4)

    if interactive_plot:
        plt.show()


#Move the output files in the output directory
cmd = f'mv {ofn1} {ofn2} {ofn3}'
if plot:
    cmd += f' {ofn4}'
cmd += f' {opath}'
#print(cmd)
proc = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)

#Final displays
print(f'\nInterpolated model atmosphere:        {opath}/{ofn1}')
print(f'Alternative interp. model atmosphere: {opath}/{ofn2}')
print(f'Interpolated departure coefficients:  {opath}/{ofn3}')

if plot:
    print(f'Atmosphere control plot:              {opath}/{ofn4}')

#Remove temporary files
cmd = 'rm -f modelatm.dat fort.25 modele.sm master_selected.dat'
proc = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
