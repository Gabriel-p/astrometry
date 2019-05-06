
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
        phot, x_p, y_p = photRead(clust_file, nanvals)

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

    x_p, y_p, = phot['x'], phot['y']
    # v_p, bv_p  = phot['V'], phot['BV']
    # b_p = bv_p + v_p

    # Remove meta data to avoid https://github.com/astropy/astropy/issues/7357
    phot.meta = {}

    return phot, x_p, y_p


def astronetRead(astro_cross):
    """
    Read astrometry.net crossed-matched data data.
    """

    # astrometry.net cross-matched data
    hdul = fits.open(astro_cross)
    cr_m_data = hdul[1].data

    return cr_m_data


def func(X, a, b, c):
    x, y = X
    return a * x + b + c * y


def px2Eq(x_p, y_p, cr_m_data):
    """
    Transform pixels to (ra, dec) using the correlated astrometry.net file.
    """
    x0, y0 = cr_m_data['field_ra'], cr_m_data['field_dec']
    x1, y1 = cr_m_data['field_x'], cr_m_data['field_y']
    p0 = (0., 100., 0.)
    x2ra = curve_fit(func, (x1, y1), x0, p0)[0]
    y2de = curve_fit(func, (y1, x1), y0, p0)[0]

    print("ra  = {:.5f} * x + {:.5f} + {:.7f} * y".format(*x2ra))
    print("dec = {:.5f} * y + {:.5f} + {:.7f} * x".format(*y2de))
    ra = x2ra[0] * x_p + x2ra[1] + x2ra[2] * y_p
    dec = y2de[0] * y_p + y2de[1] + y2de[2] * x_p

    ra = MaskedColumn(ra, name='ra')
    dec = MaskedColumn(dec, name='dec')

    return ra, dec


if __name__ == '__main__':
    # Generate output dir if it doesn't exist.
    if not exists('output'):
        makedirs('output')
    main()
