
# astrometry

Takes an input data file with (x,y) coordinates and assigns (alpha, delta)
astrometry, given a `*_corr.fits` file previously obtained from the
[astrometry.net](http://nova.astrometry.net/) service.

It can also work with a list of stars manually cross-matched and stored
in a '_corr.fits' file with column names:

    field_ra, field_dec, field_x, field_y

It can also generate the file needed to feed the `astrometry.net` service.

![alt text](out.png)

## Requirements

    Python 3, astropy, numpy scipy, matplotlib
