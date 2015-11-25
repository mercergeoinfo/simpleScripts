# Python scripts for spatial work

The scripts and functions contained here are pretty messy for the most part.
They have been written to solve specific problems as these have arisen and therefore lack consistency of design.
I am attempting to clean up the code and make it more efficient but this process will take some time.

## Files and their uses

```AM_Func.py``` is a collection of all sorts of stuff.
It is where my own code starts out and is not really for use as a library.

```standard.py``` is a collection of functions for handling files and data in a general sense.

```spatialfiles.py``` is a collection of functions for working with spatial data.
It relies on gdal.

```kriging.py``` is a messy collection of functions for performing ordinary kriging.
It is based on Martin Trauth's "MATLAB Recipes for Earth Science", published by Springer.

```Croper.py``` crops multiple images to the same size and at the same pixel location.
Output from say matplotlib can be trimmed if needed so that all images match in size and form.