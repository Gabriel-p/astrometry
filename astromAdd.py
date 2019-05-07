
import sys
from os.path import exists
from os import makedirs
from os import listdir
from os.path import isfile, join

from astropy.io import fits
from astropy.table import Table
from astropy.table import MaskedColumn

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def main():
    """
    Takes an input data file with (x,y) coordinates and assigns (alpha, delta)
    astrometry, given a 'corr.fits' file previously obtained from the
    astrometry.net service.

    Generated from the 'apass.py' script in the 'photpy' repo.

    Assumes that all input files have column headers in the format:
    id x y V eV BV eBV UB eUB VI eVI
    """

    nanvals = ('99.999', '1000.000', '1000')

    # Process all files inside 'input/' folder.
    cl_files = get_files()
    if not cl_files:
        print("No input cluster files found")

    for clust_file in cl_files:

        print("\nProcessing: {}".format(clust_file.split('/')[1][:-4]))
        # Read cluster photometry.
        phot, x_p, y_p, Vmag = photRead(clust_file, nanvals)

        # Read APASS data and astrometry.net correlated coordinates.
        cr_m_data = astronetRead(clust_file[:-4] + '_corr.fits')

        # Pixels to RA,DEC
        ra, dec = px2Eq(x_p, y_p, cr_m_data)

        # Add columns to photometry
        phot.add_column(ra)
        phot.add_column(dec)

        # Write to final file
        out_name = clust_file.replace('input', 'output')
        phot.write(out_name, format='csv', overwrite=True)

        makePlot(x_p, y_p, ra, dec, Vmag, out_name)

        sys.exit()


def get_files():
    '''
    Store the paths and names of all the input clusters stored in the
    input folder.

    The format for the input file names is:

    - cluster.xyz
    - cluster_corr.fits

    where 'cluster' is the cluster's name, and 'xys' is any extension.
    '''

    cl_files = []
    for f in listdir('input/'):
        # Don't store '*corr.fits' files
        if isfile(join('input/', f)) and not f.endswith('.fits'):
            # But check that they do exist in the folder
            if isfile(join('input/', f[:-4] + '_corr.fits')):
                cl_files.append(join('input/', f))
            else:
                sys.exit("Missing '*corr.fits' file for {}".format(f))

    # Remove readme file it is still there.
    try:
        cl_files.remove('input/README.md')
    except ValueError:
        pass

    return cl_files


def photRead(final_phot, nanvals):
    """
    Select a file with photometry to read and compare with APASS.
    """
    # Final calibrated photometry
    nv = [(_, np.nan) for _ in nanvals]
    phot = Table.read(final_phot, fill_values=nv, format="ascii")

    # phot = ascii.read(final_phot, fill_values=('99.999', np.nan))

    x_p, y_p, Vmag = phot['x'], phot['y'], phot['V']

    # Remove meta data to avoid https://github.com/astropy/astropy/issues/7357
    phot.meta = {}

    return phot, x_p, y_p, Vmag


def astronetRead(astro_cross):
    """
    Read astrometry.net crossed-matched data data.
    """

    # astrometry.net cross-matched data
    hdul = fits.open(astro_cross)
    cr_m_data = hdul[1].data

    return cr_m_data


def transf2(x_p, y_p, ra, dec, x, y):
    """
    This is a more complicated transformation that uses tan intermediate step
    transforming to the standard system (X,Y). Sources:

    https://web.archive.org/web/20090510004846/http://gtn.sonoma.edu/data_reduction/astrometry.php
    http://www.astro.utoronto.ca/~astrolab/files/Example_Lab_Report.pdf

    As far as I could tell, there are no visible benefits from using this
    transformations instead of simply transforming from (x,y) to (ra,dec).
    """
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

    # Find transformation from (x,y) to (X,Y)
    p0 = (0., 0., 0., 0., 0., 0.)
    x2X = curve_fit(func, (x, y), X, p0)[0]
    y2Y = curve_fit(func, (x, y), Y, p0)[0]

    # Transform pixel (x,y) coordinates into the standard (X,Y) system
    x_X = func((x_p, y_p), *x2X)
    y_Y = func((x_p, y_p), *y2Y)

    # Finally, convert from the standard system to (ra, dec)
    ra, dec = radecCoords(x_X, y_Y, ra_rad, dec_rad, ra0, dec0)

    return np.rad2deg(ra), np.rad2deg(dec)


def func(X, a, b, c, d, e, f):
    x, y = X
    return a + b * x + c * y + d * x * y + e * x ** 2 + f * y ** 2


def px2Eq(x_p, y_p, cr_m_data):
    """
    Transform pixels to (ra, dec) using the correlated astrometry.net file.
    """
    ra, dec = cr_m_data['field_ra'], cr_m_data['field_dec']
    x, y = cr_m_data['field_x'], cr_m_data['field_y']

    # centx, centy = (max(x) + min(x)) / 2., (max(y) + min(y)) / 2.
    # Q1 = (centx <= x) & (centy <= y)
    # Q2 = (x < centx) & (centy <= y)
    # Q3 = (x < centx) & (y < centy)
    # Q4 = (centx <= x) & (y < centy)
    # for msk in (Q1, Q2, Q3, Q4):

    # Find transformation from (x,y) to (ra,dec)
    p0 = (100., 0., 0., 0., 0., 0.)
    x2ra = curve_fit(func, (x, y), ra, p0)[0]
    y2de = curve_fit(func, (x, y), dec, p0)[0]

    print(("X  = {:.7f} + {:.7f} * x + {:.7f} * y + {:.7f} * xy +"
           " {:.7f} * x^2 + {:.7f} * y^2 +").format(*x2ra))
    print(("Y = {:.7f} + {:.7f} * x + {:.7f} * y + {:.7f} * xy + "
           "{:.7f} * x^2 + {:.7f} * y^2 +").format(*y2de))

    # Transform pixel (x,y) coordinates into the (ra,dec) system
    ra = func((x_p, y_p), *x2ra)
    dec = func((x_p, y_p), *y2de)

    ra = MaskedColumn(ra, name='ra')
    dec = MaskedColumn(dec, name='dec')

    return ra, dec


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


def makePlot(x_p, y_p, ra, dec, Vmag, out_name):
    plt.figure(figsize=(16, 8))
    plt.subplot(121)
    plt.scatter(x_p, y_p, s=star_size(Vmag))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.subplot(122)
    plt.scatter(-ra, dec, s=star_size(Vmag))
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\delta$')
    plt.savefig(out_name[:-4] + '.png', dpi=150)
    plt.close()


if __name__ == '__main__':
    # Generate output dir if it doesn't exist.
    if not exists('output'):
        makedirs('output')
    main()
