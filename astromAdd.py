
import sys
from os.path import exists
from os import makedirs
from os import listdir
from os.path import isfile, join
from pathlib import Path

from astropy.io import ascii
from astropy.io import fits
from astropy.table import Table
from astropy.table import MaskedColumn

from functools import reduce
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.offsetbox as offsetbox


def main():
    """
    Takes an input data file with (x,y) coordinates and assigns (alpha, delta)
    astrometry, given a 'corr.fits' file previously obtained from the
    astrometry.net service.

    It can also work with a list of stars manually cross-matched and stored
    in a '_corr.fits' file with column names:
    field_ra, field_dec, field_x, field_y

    """

    col_IDs, astrom_gen, regs_filt, nanvals = params_input()

    # Process all files inside 'in/' folder.
    cl_files = get_files(astrom_gen)
    if not cl_files:
        print("No input cluster files found")

    mypath = Path().absolute()
    for clust_file in cl_files:

        cl_name = clust_file.split('/')[1][:-4]
        print("\nProcessing: {}".format(cl_name))
        # Read cluster photometry.
        phot, x_p, y_p, Vmag = photRead(clust_file, nanvals, col_IDs)

        if astrom_gen is True:
            # astrometry.net file
            astrometryFeed(cl_name, x_p, y_p, Vmag, regs_filt, col_IDs)
            print("astrometry.net file generated.")
            break

        # Read astrometry.net correlated coordinates.
        cr_m_data = astronetRead(clust_file[:-4] + '_corr.fits')

        # Cross-matched coordinates
        ra, dec = cr_m_data['field_ra'], cr_m_data['field_dec']
        x, y = cr_m_data['field_x'], cr_m_data['field_y']

        log = open(join(mypath, 'out', cl_name + '.log'), "w")

        # Pixels to RA,DEC (several methods)
        # ra, dec = px2Eq(x_p, y_p, cr_m_data)
        # ra, dec = transf2(x_p, y_p, cr_m_data)
        # ra, dec = nudgeTransf(x_p, y_p, cr_m_data)
        ra_p, dec_p, ra_x, dec_y, ra_coeffs, dec_coeffs, fit_error =\
            jarnoTransf(x_p, y_p, x, y, ra, dec, log)

        # Add columns to photometry
        phot.add_column(ra_p)
        phot.add_column(dec_p)

        # Write to final file
        out_name = clust_file.replace('in', 'out')
        phot.write(out_name, format='csv', overwrite=True)

        makePlot(
            x, y, ra, dec, x_p, y_p, ra_p, dec_p, ra_x, dec_y, ra_coeffs,
            dec_coeffs, fit_error, Vmag, out_name)

        log.close()
        print("End")


def params_input():
    """
    Read input parameters from 'params_input.dat' file.
    """
    with open('params_input.dat', "r") as f_dat:
        # Iterate through each line in the file.
        for l, line in enumerate(f_dat):
            if not line.startswith("#") and line.strip() != '':
                reader = line.split()
                if reader[0] == 'CI':
                    col_IDs = reader[1:]
                if reader[0] == 'AT':
                    astrom_gen = True if reader[1] == 'True' else False
                    regs_filt = list(map(float, reader[2:]))
                if reader[0] == 'NN':
                    nanvals = reader[1:]

    return col_IDs, astrom_gen, regs_filt, nanvals


def get_files(astrom_gen):
    '''
    Store the paths and names of all the input clusters stored in the
    input folder.

    The format for the input file names is:

    - cluster.xyz
    - cluster_corr.fits

    where 'cluster' is the cluster's name, and 'xyz' is any extension.
    '''

    cl_files = []
    for f in listdir('in/'):
        # Don't store '*corr.fits' files
        if isfile(join('in/', f)) and not f.endswith('.fits'):

            if astrom_gen is False:
                # But check that they do exist in the folder
                if isfile(join('in/', f[:-4] + '_corr.fits')):
                    cl_files.append(join('in/', f))
                else:
                    sys.exit("Missing '*corr.fits' file for {}".format(f))
            else:
                cl_files.append(join('in/', f))

    # Remove readme file it is still there.
    try:
        cl_files.remove('in/README.md')
    except ValueError:
        pass

    return cl_files


def photRead(final_phot, nanvals, col_IDs):
    """
    Select a file with photometry to read and compare with APASS.
    """
    id_id, x_id, y_id, V_id = col_IDs

    fill_msk = [('', '0')] + [(_, '0') for _ in nanvals]
    # Read IDs as strings, not applying the 'fill_msk'
    phot = ascii.read(
        final_phot, converters={id_id: [ascii.convert_numpy(np.str)]})
    # Store IDs
    id_data = phot[id_id]
    # Read rest of the data applying the mask
    phot = ascii.read(final_phot, fill_values=fill_msk)
    # Replace IDs column
    phot[id_id] = id_data

    # # Final calibrated photometry
    # fill_msk = [(_, np.nan) for _ in nanvals]
    # phot = Table.read(
    #     final_phot, fill_values=fill_msk, format="ascii",
    #     converters={id_col: [ascii.convert_numpy(np.str)]})

    # # Mask stars with no valid V magnitude.
    # try:
    #     Vmsk = ~phot['V']._mask
    #     phot = phot[Vmsk]
    # except AttributeError:
    #     pass

    x_p, y_p, Vmag = phot[x_id], phot[y_id], phot[V_id]

    # Remove meta data to avoid https://github.com/astropy/astropy/issues/7357
    phot.meta = {}

    return phot, x_p, y_p, Vmag


def astrometryFeed(cl_name, x_p, y_p, v_p, regs_filt, col_IDs):
    """
    Create file with the proper format to feed astrometry.net
    """
    x_id, y_id, V_id = col_IDs[1:]

    t = Table([x_p, y_p, v_p], names=(x_id, y_id, V_id))
    t.sort(V_id)

    # Define central region limits.
    xmin, xmax, ymin, ymax = regs_filt
    mask = [xmin < t[x_id], t[x_id] < xmax, ymin < t[y_id], t[y_id] < ymax]
    total_mask = reduce(np.logical_and, mask)
    xm, ym = t[x_id][total_mask], t[y_id][total_mask]

    ascii.write(
        [xm, ym], 'out/' + cl_name + "_astrometry.dat",
        delimiter=' ', format='fixed_width_no_header', overwrite=True)


def astronetRead(astro_cross):
    """
    Read astrometry.net crossed-matched data data.
    """

    # astrometry.net cross-matched data
    try:
        hdul = fits.open(astro_cross)
        cr_m_data = hdul[1].data
    except IOError:
        # In case the file is not a real 'fits', but a text file with stars
        # manually matched, with an added 'fits' extension.
        cr_m_data = Table.read(astro_cross, format='ascii')

    return cr_m_data


def transf2(x_p, y_p, cr_m_data):
    """
    This is a more complicated transformation that uses an intermediate step
    transforming to the standard system (X,Y). Sources:

    https://web.archive.org/web/20090510004846/http://gtn.sonoma.edu/data_reduction/astrometry.php
    http://www.astro.utoronto.ca/~astrolab/files/Example_Lab_Report.pdf

    June 2019
    As far as I could tell, there are no visible benefits from using this
    transformations instead of simply transforming from (x,y) to (ra,dec)
    with the px2Eq() function.
    """

    ra, dec = cr_m_data['field_ra'], cr_m_data['field_dec']
    x, y = cr_m_data['field_x'], cr_m_data['field_y']

    def standardCoords(ra_rad, dec_rad, ra0, dec0):
        """
        """
        X = (np.cos(dec_rad) * np.sin(ra_rad - ra0)) /\
            (np.cos(dec0) * np.cos(dec_rad) * np.cos(ra_rad - ra0) +
                np.sin(dec_rad) * np.sin(dec0))
        Y = (np.cos(dec0) * np.sin(dec_rad) - np.cos(dec_rad) * np.sin(dec0) *
             np.cos(ra_rad - ra0)) /\
            (np.sin(dec0) * np.sin(dec_rad) + np.cos(dec_rad) * np.cos(dec0) *
             np.cos(ra_rad - ra0))
        return X, Y

    def radecCoords(x_X, y_Y, ra_rad, dec_rad, ra0, dec0):
        """
        """
        ra = ra0 + np.arctan(x_X / (np.cos(dec0) - y_Y * np.sin(dec0)))
        dec = np.arcsin((np.sin(dec0) + y_Y * np.cos(dec0)) /
                        np.sqrt(1 + x_X**2 + y_Y**2))
        return ra, dec

    ra_rad, dec_rad = np.deg2rad(ra), np.deg2rad(dec)
    ra0, dec0 = (max(ra_rad) + min(ra_rad)) / 2.,\
        (max(dec_rad) + min(dec_rad)) / 2.

    # Transform celestial (ra,dec) to standard coordinates
    X, Y = standardCoords(ra_rad, dec_rad, ra0, dec0)

    def func(X, a, b, c, d, e, f):
        x, y = X
        return a + b * x + c * y + d * x * y + e * x ** 2 + f * y ** 2

    # Find transformation from (x,y) to (X,Y)
    p0 = (0., 0., 0., 0., 0., 0.)
    x2X = curve_fit(func, (x, y), X, p0)[0]
    y2Y = curve_fit(func, (x, y), Y, p0)[0]

    # Transform pixel (x,y) coordinates into the standard (X,Y) system
    x_X = func((x_p, y_p), *x2X)
    y_Y = func((x_p, y_p), *y2Y)

    # Finally, convert from the standard system to (ra, dec)
    ra, dec = radecCoords(x_X, y_Y, ra_rad, dec_rad, ra0, dec0)
    ra_p, dec_p = np.rad2deg(ra), np.rad2deg(dec)
    ra_p = MaskedColumn(ra_p, name='ra')
    dec_p = MaskedColumn(dec_p, name='dec')

    return ra_p, dec_p


def px2Eq(x_p, y_p, cr_m_data):
    """
    Transform pixels to (ra, dec) using the correlated astrometry.net file.
    """
    ra, dec = cr_m_data['field_ra'], cr_m_data['field_dec']
    x, y = cr_m_data['field_x'], cr_m_data['field_y']

    def func(X, a, b, c, d, e, f):
        x, y = X
        return a + b * x + c * y + d * x * y + e * x ** 2 + f * y ** 2

    # Find transformation from (x,y) to (ra,dec)
    p0 = (100., 0., 0., 0., 0., 0.)
    x2ra = curve_fit(func, (x, y), ra, p0)[0]
    y2de = curve_fit(func, (x, y), dec, p0)[0]

    print(("ra   = {:.7f} + {:.7f} * x + {:.7f} * y + {:.7f} * xy +"
           " {:.7f} * x^2 + {:.7f} * y^2 +").format(*x2ra))
    print(("dec = {:.7f} + {:.7f} * x + {:.7f} * y + {:.7f} * xy + "
           "{:.7f} * x^2 + {:.7f} * y^2 +").format(*y2de))

    # Transform pixel (x,y) coordinates into the (ra,dec) system
    ra_p = func((x_p, y_p), *x2ra)
    dec_p = func((x_p, y_p), *y2de)

    ra_p = MaskedColumn(ra_p, name='ra')
    dec_p = MaskedColumn(dec_p, name='dec')

    return ra_p, dec_p


def nudgeTransf(x_p, y_p, cr_m_data):
    """
    This works but gives *very* poor results (July 2019)
    """
    ra, dec = cr_m_data['field_ra'], cr_m_data['field_dec']
    x, y = cr_m_data['field_x'], cr_m_data['field_y']

    import nudged

    dom = list(map(list, zip(x, y)))
    ran = list(map(list, zip(-ra, dec)))

    trans = nudged.estimate(dom, ran)

    cc = list(map(list, list(np.array([x_p, y_p]).T)))
    ra_p, dec_p = np.array(trans.transform(cc)).T
    ra_p = -ra_p

    print(trans.get_matrix())
    print(trans.get_rotation())
    print(trans.get_scale())
    print(trans.get_translation())

    # plt.subplot(221);plt.title("dom");plt.scatter(x, y)
    # plt.subplot(222);plt.title("ran");plt.scatter(ra, dec)
    # plt.subplot(223);plt.title("(x, y)");plt.scatter(x_p, y_p)
    # plt.subplot(224);plt.title("(x_t, y_t)");plt.scatter(ra_p, dec_p)
    # plt.show()

    ra_p = MaskedColumn(ra_p, name='ra')
    dec_p = MaskedColumn(dec_p, name='dec')

    return ra_p, dec_p


def jarnoTransf(x_p, y_p, x, y, ra, dec, log):
    """
    Source: https://elonen.iki.fi/code/misc-notes/affine-fit/

    Slightly modified to work better with (ra, dec) and simplified using
    np.linalg.solve() instead of the original Gauss-Jordan method.
    """
    from_pt = np.array([x, y]).T
    to_pt = np.array([ra, dec]).T
    trn = Affine_Fit(from_pt, to_pt)

    print("\nTransformation is:")
    print(trn.To_Str())
    print("\nTransformation is:", file=log)
    print(trn.To_Str(), file=log)
    ra_coeffs, dec_coeffs = trn.coeffs()

    # Transform cross-matched (x, y), for plotting
    ra_x, dec_y = trn.Transform(np.array([x, y]))

    # Fitting error as mean of distances.
    err = np.sqrt((ra - ra_x)**2 + (dec - dec_y)**2).mean()
    print("Fitting error = {:.9f}\n".format(err))
    print("Fitting error = {:.9f}".format(err), file=log)

    ra_p, dec_p = trn.Transform(np.array([x_p, y_p]))
    ra_p = MaskedColumn(ra_p, name='ra')
    dec_p = MaskedColumn(dec_p, name='dec')

    return ra_p, dec_p, ra_x, dec_y, ra_coeffs, dec_coeffs, err


def Affine_Fit(from_pts, to_pts):
    """
    Fit an affine transformation to given point sets.
    More precisely: solve (least squares fit) matrix 'A' and 't' from
    'p ~= A*q+t', given vectors 'p' and 'q'; where:
    q = from_pts, p = to_pts.
    Works with arbitrary dimensional vectors (2d, 3d, 4d...).

    Written by Jarno Elonen <elonen@iki.fi> in 2007.
    Placed in Public Domain.

    Based on paper "Fitting affine and orthogonal transformations
    between two sets of points, by Helmuth Späth (2003).
    """

    # num of dimensions
    dim = len(from_pts[0])
    if len(from_pts) < dim:
        raise ValueError("Too few points => under-determined system.")

    # Make an empty (dim) x (dim+1) matrix and fill it
    c = np.zeros((dim + 1, dim))
    for j in range(dim):
        for k in range(dim + 1):
            for i in range(len(from_pts)):
                qt = list(from_pts[i]) + [1]
                c[k][j] += qt[k] * to_pts[i][j]

    # Make an empty (dim+1) x (dim+1) matrix and fill it
    Q = np.zeros((dim + 1, dim + 1))
    for qi in from_pts:
        qt = list(qi) + [1]
        for i in range(dim + 1):
            for j in range(dim + 1):
                Q[i][j] += qt[i] * qt[j]

    # Solve Q * x = c system
    M = np.linalg.solve(Q, c)

    class Transformation:
        """
        Result object that represents the transformation from affine fitter.
        """

        def To_Str(self):
            res = ""
            for j, dd in enumerate(('ra', 'dec')):
                _str = "{} = ".format(dd)
                for i, xy in enumerate(('x', 'y')):
                    _str += "{} * {:g} + ".format(xy, M[i][j])
                _str += "{:.7f}".format(M[dim][j])
                res += _str + "\n"
            return res

        def coeffs(self):
            """
            This works for 2D only.
            """
            c_ra = (M[2][0], M[0][0], M[1][0])
            c_dec = (M[2][1], M[0][1], M[1][1])
            return np.array([c_ra, c_dec])

        def Transform(self, pt):
            res = [0.0 for a in range(dim)]
            for j in range(dim):
                for i in range(dim):
                    res[j] += pt[i] * M[i][j]
                res[j] += M[dim][j]
            return res

    return Transformation()


def makePlot(
    x, y, ra, dec, x_p, y_p, ra_p, dec_p, ra_x, dec_y, ra_coeffs, dec_coeffs,
        fit_error, Vmag, out_name):
    """
    """
    # plt.style.use('seaborn-darkgrid')
    fig = plt.figure(figsize=(18, 18))
    gs = gridspec.GridSpec(3, 3)

    plt.subplot(gs[0])
    plt.title("Cross-matched (x, y)")
    plt.scatter(x, y, s=2. * star_size(Vmag))
    plt.grid(which='major', axis='both', linestyle='--', color='grey', lw=.5)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(gs[1])
    plt.title("Cross-matched (ra, dec)")
    plt.scatter(-ra, dec, s=2. * star_size(Vmag), c='r')
    plt.grid(which='major', axis='both', linestyle='--', color='grey', lw=.5)
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\delta$')

    ax = plt.subplot(gs[2])
    ax.axis('off')  # Remove axis from frame.
    t1 = r'$Transformation$'
    t2 = r'$\alpha = {:.9f} + {:.9f} * x + {:.9f} * y$'.format(*ra_coeffs)
    t3 = r'$\delta = {:.9f} + {:.9f} * x + {:.9f} * y$'.format(*dec_coeffs)
    t4 = r'$Fitting\;error = {:.9f}$'.format(fit_error)
    text = t1 + '\n\n' + t2 + '\n' + t3 + '\n\n' + t4
    ob = offsetbox.AnchoredText(
        text, pad=1, loc=6, borderpad=-5, prop=dict(size=10))
    ob.patch.set(alpha=0.85)
    ax.add_artist(ob)

    plt.subplot(gs[3])
    plt.title("Original (x, y)")
    plt.scatter(x_p, y_p, s=star_size(Vmag))
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(gs[4])
    plt.title("Transformed (x, y)")
    plt.scatter(-ra_p, dec_p, s=star_size(Vmag), c='r')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\delta$')

    plt.subplot(gs[5])
    plt.title(
        "Cross-matched & transformed (x, y) overlayed on (ra, dec)",
        fontsize=10)
    plt.scatter(
        -ra_x, dec_y, s=4. * star_size(Vmag), c='b', marker='+', lw=.75,
        label="(x, y)")
    plt.grid(which='major', axis='both', linestyle='--', color='grey', lw=.5)
    plt.scatter(
        -ra, dec, s=4. * star_size(Vmag), c='r', marker='x', lw=.75,
        label=r"($\alpha,\,\delta$)")
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\delta$')
    plt.legend()

    # plt.close()
    fig.tight_layout()
    plt.savefig(out_name[:-4] + '.png', dpi=150, bbox_inches='tight')


def star_size(mag, N=None, min_m=None):
    '''
    Convert magnitudes into intensities and define sizes of stars in
    finding chart.
    '''
    # Scale factor.
    if N is None:
        N = len(mag)
    if min_m is None:
        min_m = np.nanmin(mag)
        # print("min mag used: {}".format(min_m))
    factor = 500. * (1 - 1 / (1 + 150 / N ** 0.85))
    return 0.1 + factor * 10 ** ((np.array(mag) - min_m) / -2.5)


if __name__ == '__main__':
    # Generate output dir if it doesn't exist.
    if not exists('out'):
        makedirs('out')
    main()
