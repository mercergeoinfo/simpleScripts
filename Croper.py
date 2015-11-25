#!/Users/andrew/anaconda/bin/python
###!/Library/Frameworks/Python.framework/Versions/Current/bin/python
###!/usr/bin/env python
from __future__ import division
import sys
import os
from PIL import Image
# ANDREW MERCER
# 12/04/2014
# V.1
#
## FUNCTION DEFINITIONS
#
## MAIN PROGRAM
# Set file folder name to current folder
flistfolder = './'
# Get file type from user. PIL can handle other types but checking all that is more
# than I am willing to do for this quick script.
filetype = ''
while filetype not in ['jpg','png','tif','gif','tiff']:
    filetype = raw_input('Enter file type: ')
# Create file list containing all and only jpg files in folder
FileList = filter(lambda x: x.endswith(filetype),os.listdir(flistfolder))
# Get crop area
left = ''
right = ''
upper = ''
lower = ''
print "Enter coordinates for crop area. 0,0 is upper left of image\n"
# Use isinstance as you may want to set to 0. This checks that the variable is set to
# the integer value of 0 and not just empty. Also checks against non-numerical input.
while not isinstance(left, int):
            leftin = raw_input('\nEnter left coordinate: ')
            try:
                left = int(leftin)
                print 'Left set at: ',left
            except:
                left = ''
while not isinstance(right, int):
            rightin = raw_input('\nEnter right coordinate: ')
            try:
                right = int(rightin)
                print 'Right set at: ',right
            except:
                right = ''
while not isinstance(upper, int):
            upperin = raw_input('\nEnter upper coordinate: ')
            try:
                upper = int(upperin)
                print 'Upper set at: ',upper
            except:
                upper = ''
while not isinstance(lower, int):
            lowerin = raw_input('\nEnter lower coordinate: ')
            try:
                lower = int(lowerin)
                print 'Lower set at: ',lower
            except:
                lower = ''
# Loop through file list
if not os.path.exists('./Crop'):
    os.makedirs('./Crop')
for f in FileList:
    # Create new file name for edited file
    file, ext = os.path.splitext(f)
    fed = file+'_e'+ext
    fout = os.path.join('./Crop',fed)
    # Open image with PIL
    try:
        im = Image.open(f)
    except IOError:
        continue
    # Crop and save image
    im.crop((left,upper,right,lower)).save(fout)

