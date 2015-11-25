#!/Users/andrew/anaconda/bin/python
# -*- coding: utf-8 -*-
#from __future__ import division
import sys
import os
import math
import numpy as np
from scipy.stats.stats import nanmean
#import scipy.stats as stats
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
#from osgeo import gdal_array
#from osgeo import gdalconst
#import gdal
#import gdalconst
import csv
import cPickle as pickle
import matplotlib
import matplotlib.pyplot as plt
#import matplotlib.dates as dates
import pylab
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages
#import time
import itertools
import gc
#pylab.ion()
pylab.ioff()
from datetime import datetime
#
#
# ANDREW MERCER
# v2.0
# 11/03/2014
#
#
####################################DATA IN AND OUT#######################################
def envset(*args):
	'''Set up working folders and files via parameters found in a text file
	Settings file should look like this (omit or # those not needed).
	Paths can be absolute (from root /Users/username) or relative (./path).
	**********************************************************************
	Input Data Folder =./InData/
	Output Folder =./OutData
	Probing = SG_Probe_2013.csv
	Density = SG_Density_2013.csv
	Stakes = SG_Stakes_2013.csv
	DEM=./InData/2010_DEM2.tif
	#Data Files =SG_Probe_2013.csv
	Plot DEM? =y
	Plot results? =y
	#Base =
	Detrend = n'''
	env = os.environ
	curLoc = env['PWD']
	curList = os.listdir('./')
	if len(args)>0:
		if os.path.exists(args[0]):
			envFile = args[0]
	else:
		print '\nList of files in ', curLoc,'\n'
		for i in curList:
			if not os.path.isdir(i):
				print i
		envFile = './AMset.txt'
		while not os.path.exists(envFile):
			envFile = raw_input('\nEnter name of file containing environment settings(txt): ')
	settings = {}
	InFile = open(envFile,'rb')
	# Check contents and set up dictionary
	for row in InFile:
		line = row.strip().split('=')
		print line
		settings[line[0].strip()]=line[1].strip()
	# Folder containing data files
	if 'Input Data Folder' not in settings:
		settings['Input Data Folder'] = './Indata/.'
	inDataDir = settings['Input Data Folder']
	if not os.path.exists(inDataDir):
		sys.exit('Input Data Folder setting incorrect')
	# Root folder for output data
	outDir = settings['Output Folder']
	if not os.path.exists(outDir):
		os.makedirs(outDir)
	# Name namstr used for output subfolder, set first to day_time
	namstr = datetime.now().strftime('%j_%H%M%S')
	destDir = os.path.join(outDir,namstr)
	if 'Data Files' in settings:
		fileList = settings['Data Files'].split(',')
		settings['Data Files'] = fileList
		for inDataName in fileList:
			if not os.path.exists(os.path.join(inDataDir,inDataName)):
				sys.exit('Data File setting incorrect')
	if 'Probing' in settings:
			if not os.path.exists(os.path.join(inDataDir,settings['Probing'])):
				sys.exit('Probing setting incorrect')
	if 'Density' in settings:
		print settings['Density']
		densityList = settings['Density'].split(',')
		settings['Density'] = densityList
		print settings['Density']
		print densityList
		for densityName in densityList:
			if not os.path.exists(os.path.join(inDataDir,densityName)):
				sys.exit('Density setting incorrect')
	if 'Stakes' in settings:
			if not os.path.exists(os.path.join(inDataDir,settings['Stakes'])):
				sys.exit('Stakes setting incorrect')
	# Create output folder for session
	if not os.path.exists(destDir):
		os.makedirs(destDir)
	settings['Write Folder']=destDir
	if 'DEM' in settings:
		demFile = settings['DEM']
		if not os.path.exists(demFile):
			print settings['DEM']
			sys.exit('DEM file setting incorrect')
	if 'MSK' in settings:
		demFile = settings['MSK']
		if not os.path.exists(demFile):
			print settings['MSK']
			sys.exit('MSK (mask) file setting incorrect')
	if 'Pickle' in settings:
		pickleFile = settings['Pickle']
		if not os.path.exists(pickleFile):
			sys.exit('No such pickle file')
	settings['This File'] = envFile
	print '\nSettings are: '
	for i in settings.keys():
		print i,': ',settings[i]
	print '\n'*2
	return settings, envFile
#
#
def cont():
	'''Simple function to break script flow'''
	print '\n'*2,'*'*10
	answers = ['c','y','s','e']
	a = ''
	while a not in answers:
		a = raw_input('\nDo you wish to continue (y or c), skip (s) or end (e)?: ')
	if a == 'e':
		sys.exit(0)
	elif a == 's':
		ans = 1
	elif a == 'c' or a == 'y':
		ans = 0
	print '\n'*2
	return ans
#
#
def namer(pathstring):
	'''Get name and extension of a file
	To use: name,ext,path,namefull = namer(pathstring) '''
	## Called by: kriging
	namefull = os.path.basename(pathstring)
	path = os.path.dirname(pathstring)
	ext = namefull.split('.')[1]
	name = namefull.split('.')[0]
	return name,ext,path,namefull
#
#
def fileList(folder,ending='*'):
	'''Return list of specific file type in folder other than current folder. If no ending provided all files listed.
	To use: filelist = fileList(folder, ending)'''
	matchstring = '*.' + ending
	filelist = fnmatch.filter(os.listdir(folder),matchstring)
	print "From %s the following matched %s" % (folder, matchstring)
	for f in filelist:
		print f
	return filelist
#
#
def import2vector(fileName, dateString = '%d/%m/%y %H:%M:%S'):
	'''Imports the data as vectors in a dictionary. dateString is optional and can be set to match datetime format
	To use: db = import2vector(filename) or db = import2vector(filename, dateString = '%d/%m/%y %H:%M:%S')'''
	# Open file
	InFile = open(fileName,'rb')
	line = InFile.next()
	Name = os.path.basename(fileName).split('.')[0]
	# Get headers
	Headers = line.strip().split(',')
	# Create dictionary for data
	data = {}
	data['Data'] = {}
	data['Description'] = {}
	data['Description']['Source'] = Name
	i=0
	# Set up list of Nan values
	nanlist = ['NAN','NaN','nan','NULL','null','-9999',-9999,'']
	# Read through data file
	for i in Headers:
		data['Data'][i] = []
	for row in InFile:
		if row != "\n":
			# Split read line of data into list
			dataIn = row.strip().split(',')
			for i in range(len(dataIn)):
				# Check for NaN and empty values
				if dataIn[i] in nanlist:
					 dataIn[i] = np.nan
				else:
					# Try date formatted data conversion
					try:
						dataIn[i] = datetime.strptime(dataIn[i],dateString)
					except:
						# Try converting to float
						try:
							dataIn[i] = float(dataIn[i])
						except:
							# Leave as string
							dataIn[i] = dataIn[i]
				# Add to vector
				data['Data'][Headers[i]].append(dataIn[i])
	for i in Headers:
		# Convert to numpy arrays
		data['Data'][i] = np.array(data['Data'][i])
		try:
			# Create posts containing basic statistics for each numerical column (vector)
			data['Description'][(str(i)+'_min')] = np.nanmin(data['Data'][i])
			data['Description'][(str(i)+'_max')] = np.nanmax(data['Data'][i])
			data['Description'][(str(i)+'_mean')] = np.nanmean(data['Data'][i])
			data['Description'][(str(i)+'_stdDev')] = np.nanstd(data['Data'][i])
			data['Description'][(str(i)+'_median')] = np.median(data['Data'][i])
		except:
			print "\nStatistics not computable for %s\n" % str(i)
	return data
#
#
def importTovector(fileName):
	"""Imports the data as vectors in a dictionary."""
	InFile = open(fileName,'rb')
	ibutCheck = '1-Wire/iButton Part Number: DS1922L'
	headPar = 'Date/Time,Unit,Value'
	# Check if file has iButton header
	line = InFile.next()
	if ibutCheck in line:
		# If iButton header found get name and skip to data headers
		fullName, Name = iButtonName(fileName)
		while headPar not in line:
			print line
			line = InFile.next()
		print 'Skipped'
	# If no iButton header found take name from file name
	else: Name = os.path.basename(fileName).split('.')[0]
	Headers = line.strip().split(',')
	data = {}
	data['Source'] = Name
	i=0
	#
	nanlist = ['NAN','NaN','nan','NULL','null','-9999',-9999]
	for i in Headers:
		data[i] = []
	for row in InFile:
		if row != "\n":
			# Split read line of data into list
			dataIn = row.strip().split(',')
			for i in range(len(dataIn)):
				if dataIn[i] in nanlist:
				#if dataIn[i] == 'nan' or dataIn[i] == 'NaN' or dataIn[i] == 'NULL' :
					 dataIn[i] = np.nan
					 #data[Headers[i]].append(dataIn[i])
				elif dataIn[i] == "":
					dataIn[i] = np.nan
					#data[Headers[i]].append(dataIn[i])
				else:
					try:
						dataIn[i] = datetime.strptime(dataIn[i],'%d/%m/%y %H:%M:%S')
					except:
						try:
							dataIn[i] = float(dataIn[i])
						except:
							dataIn[i] = dataIn[i]
				data[Headers[i]].append(dataIn[i])
	for i in Headers:
		data[i] = np.array(data[i])
	return data
#
#
# Search file header for iButton name
def iButtonName(fileName):
	'''Used by import2vector to check name of iButton when appropriate'''
	namePar = '1-Wire/iButton Registration Number:'
	inFile = open(fileName,'rb')
	line = inFile.next()
	while namePar not in line:
		line = inFile.next()
	fullName = line.strip().split(': ')[1]
	print fullName
	shortName = fullName[-8:-2]
	print shortName
	return fullName, shortName
#
#
def InDataArray(fileName):
	'''Simplified csv data reader that returns 3 vectors'''
	## Called by: kriging
	print '\n'*2,'*'*10
	print 'Read ',fileName
	InFile = open(fileName,'rb')
	Headers = InFile.next().strip().split(',')
	Xchc = ['Easting','SWEREF99_E','RT90_E','E','X','East','EASTING','EAST']
	Ychc = ['Northing','SWEREF99_N','RT90_N','N','Y','North','NORTHING','NORTH']
	Zchc = ['Elevation','ELEV','Z','mwe','Bw','Bs','Bn','Dw']
	print 'File Headers:\n', Headers
	Xcol = ''
	Ycol = ''
	Zcol = ''
	for a in Xchc:
		if a in Headers:
			print a
			a_ans = raw_input('Use as x? (y/any other key): ')
			if a_ans == 'y':
				Xcol = a
				break
	for b in Ychc:
		if b in Headers:
			print b
			b_ans = raw_input('Use as y? (y/any other key): ')
			if b_ans == 'y':
				Ycol = b
				break
	for c in Zchc:
		if c in Headers:
			print c
			c_ans = raw_input('Use as z? (y/any other key): ')
			if c_ans == 'y':
				Zcol = c
				break
	while Xcol not in Headers:
		Xcol = raw_input('Enter column for "x": ')
	while Ycol not in Headers:
		Ycol = raw_input('Enter column for "y": ')
	while Zcol not in Headers:
		Zcol = raw_input('Enter column for "z": ')
	#
	# Dirty little hack to get name of z for kriging return
	global ZcolName
	ZcolName = Zcol
	#
	inFile = open(fileName,'rb')
	X = []
	Y = []
	Z = []
	j = 1
	for line in csv.DictReader(inFile, delimiter=','):
		catch = 0
		#print j,': ',' x: ',line[Xcol].strip(),' y: ',line[Ycol].strip(),' z: ',line[Zcol].strip()
		try:
			float(line[Zcol].strip())
		except ValueError:
			catch = 1
			print 'Line ',j,': ',line, ' not processed'
			j = j+1
			continue
		#if not (line[Zcol].strip()) or isnan(float(line[c3].strip())) == True:
			#catch = 1
			#print 'Line ',j,': ',line, ' not processed'
		if catch == 0:
			X.append(float(line[Xcol].strip()))
			Y.append(float(line[Ycol].strip()))
			Z.append(float(line[Zcol].strip()))
		j = j+1
	return X,Y,Z
	#
#
#
def rasterImport(file):
	'''Import a raster file and return grid plus meta data.
	To use: data, meta, metadata = rasterImport(file)'''
	time_one = datetime.now()
	print '\nImporting ',file
	#print time_one.strftime('at day:%j %H:%M:%S')
	# register all of the GDAL drivers
	gdal.AllRegister()
	# Open file
	raster = gdal.Open(file)
	if raster is None:
		print 'Could not open ',file,'\n'
		sys.exit(1)
	# Get coordinate system parameters
	projec = raster.GetProjection()
	srs=osr.SpatialReference(wkt=projec)
	transf = raster.GetGeoTransform()
	ul_x = transf[0]
	ul_y = transf[3]
	xres = transf[1]
	yres = transf[5]
	# get image size
	rows = raster.RasterYSize
	cols = raster.RasterXSize
	dims = {'xres':xres,'yres':yres,'rows':rows,'cols':cols}
	# Calculate corners
	ll_x = ul_x
	ll_y = ul_y + (rows * yres)
	ur_x = ul_x + (cols * xres)
	ur_y = ul_y
	lr_x = ur_x
	lr_y = ll_y
	corners = {'ll':(ll_x,ll_y),'ul':(ul_x,ul_y),'ur':(ur_x,ur_y),'lr':(lr_x,lr_y)}
	#
	driveshrt = raster.GetDriver().ShortName
	driver = gdal.GetDriverByName(driveshrt)
	metadata = driver.GetMetadata()
	metakeys = metadata.keys()
	# Read the file band to a matrix called band_1
	band = raster.GetRasterBand(1)
	# Access data in rastern band as array
	data = band.ReadAsArray(0,0,cols,rows)
	#
	# gdal interpolation creates "upside down files, hence this
	Yres=yres
	if Yres/np.fabs(Yres)==-1:
		Yres=-1*Yres
		data = np.flipud(data)
	#
	# get nodata value
	nandat = band.GetNoDataValue()
	# get minimum and maximum value
	mindat = band.GetMinimum()
	maxdat = band.GetMaximum()
	if mindat is None or maxdat is None:
			(mindat,maxdat) = band.ComputeRasterMinMax(1)
	dats = [nandat,mindat,maxdat]
	sumvals = data[np.where(np.logical_not(data == nandat))]
	sumval = sumvals.sum()
	numvals = len(sumvals)
	avval = sumval/numvals
	volvals = sumval*(math.fabs((transf[1]*transf[5])))
	meta = {'transform':transf,'projection':projec,'corners':corners,'dimension':dims,'dats':dats,'sumval':sumval,'numvals':numvals,'avval':avval,'volvals':volvals}
	if srs.IsProjected:
		print 'projcs 1: ',srs.GetAttrValue('projcs')
		print 'projcs 2: ',srs.GetAuthorityCode('projcs')
		print 'projcs 3: ',srs.GetAuthorityName('projcs')
	print 'geogcs 1:: ',srs.GetAttrValue('geogcs')
	print 'Sum of pixel values = ',sumval,' in range ',mindat,' to ',maxdat
	print 'From ',numvals,' of ',rows,' X ',cols,' = ',rows*cols,' pixels, or ',((1.0*numvals)/(rows*cols))*100,'%'
	print 'Average then is ',avval
	print 'Pixels size, x:',transf[1],' y:',transf[5]
	print '"Volume" is then the sum * 1 pixel area: ',volvals
	return data, meta, metadata
#
#
def rasterExport(outdata,meta,filnm):
	'''Export array as single layer GeoTiff. Uses metadata from rasterImport and numpy nan as mask.
	To use: rasterExport(outdata,meta,filnm)'''
	inrows =  meta['dimension']['rows']
	incols =  meta['dimension']['cols']
	datdriver = gdal.GetDriverByName( "GTiff" )
	datout = datdriver.Create(filnm,incols,inrows,1,gdal.GDT_Float32)
	datout.SetGeoTransform(meta['transform'])
	datout.SetProjection(meta['projection'])
	outdata_m = np.flipud(outdata)
	outdata_m = np.where(np.isnan(outdata_m),meta['dats'][0],outdata_m)
	datout.GetRasterBand(1).WriteArray(outdata_m)
	datout.GetRasterBand(1).SetNoDataValue(meta['dats'][0])
	datout.GetRasterBand(1).ComputeStatistics(True)
	datout = None
	return 0
#
#
def DemImport(demfile):
	'''Import a DEM file and return grid plus meta data. Written for kriging.
	To use: demdata,Xg,Yg,Xg1,Yg1,rx,ry,demmeta = DemImport(demfile)'''
	## Called by: kriging
	time_one = datetime.now()
	print '\nImporting ',demfile
	#print time_one.strftime('at day:%j %H:%M:%S')
	# register all of the GDAL drivers
	gdal.AllRegister()
	# Open file
	dem = gdal.Open(demfile)
	if dem is None:
		print 'Could not open ',demfile,'\n'
		sys.exit(1)
	# Get coordinate system parameters
	projec = dem.GetProjection()
	transf = dem.GetGeoTransform()
	ul_x = transf[0]
	ul_y = transf[3]
	xres = transf[1]
	yres = transf[5]
	# get image size
	demrows = dem.RasterYSize
	demcols = dem.RasterXSize
	# Calculate corners
	ll_x = ul_x
	ll_y = ul_y + (demrows * yres)
	#
	driveshrt = dem.GetDriver().ShortName
	driver = gdal.GetDriverByName(driveshrt)
	#metadata = driver.GetMetadata()
	# Read the dem band to a matrix called band_1
	demband = dem.GetRasterBand(1)
	# Access data in rastern band as array
	demdata = demband.ReadAsArray(0,0,demcols,demrows)
	# gdal interpolation creates "upside down files, hence this
	Yres=yres
	if Yres/np.fabs(Yres)==-1:
		Yres=-1*Yres
		demdata = np.flipud(demdata)
	#
	# get nodata value
	demnandat = demband.GetNoDataValue()
	# get minimum and maximum value
	demmindat = demband.GetMinimum()
	demmaxdat = demband.GetMaximum()
	if demmindat is None or demmaxdat is None:
			(demmindat,demmaxdat) = demband.ComputeRasterMinMax(1)
	#
	## Create grid to krig to.
	xstart = int(round(ll_x))
	xstop = xstart + int(demcols)*int(round(xres))
	ystart = int(round(ll_y))
	ystop = ystart + int(demrows)*int(round(Yres))
	# Vectors of values along grid axes
	demRx = range(xstart,xstop,int(round(xres)))
	demRy = range(ystart,ystop,int(round(Yres)))
	# Make grid 2D arrays
	Xg1,Yg1 = np.meshgrid(demRx,demRy)
	# Convert grids to 1D vectors. NOTE THE STUPID NAMES Xg1 IS NOT 1D
	Yg=Yg1.reshape((-1,1))
	Xg=Xg1.reshape((-1,1))
	#
	rx = len(demRx)
	ry = len(demRy)
	# Collect metadata to list for return
	demmeta = []
	demmeta.append(['projection','geotransform','driver','rows','columns','nanvalue','min','max'])
	demmeta.append(projec)
	demmeta.append(transf)
	demmeta.append(driver)
	demmeta.append(demrows)
	demmeta.append(demcols)
	demmeta.append(demnandat)
	demmeta.append(demmindat)
	demmeta.append(demmaxdat)
	#for i in demmeta: print i
	#print 'demmeta:\n',demmeta[0],'\n'
	return demdata,Xg,Yg,Xg1,Yg1,rx,ry,demmeta
#
#
def picksave(data,fileName,location):
	'''Pickle data into a pickle database for easy retrieval into Python'''
	# Save data dictionary to pickle database
	fileName = fileName + '.p'
	pickleFile = os.path.join(location,fileName)
	with open(pickleFile, 'wb') as fp:
		pickle.dump(data, fp)
	return pickleFile
#
#
def pickload(pickleFile):
	'''Load data from a pickle database'''
	with open(pickleFile, 'rb') as fp:
					data = pickle.load(fp)
	return data
#
#
def rehash1(headName,colName,b):
	'''Rehash data dictionary for plotting. Written to extract yearly values for
	ablation stakes to an array containing the year and value as vectors'''
	bKeys=sorted(b.keys())
	indic = []
	for i in bKeys:
		xList=b[i][headName]
		for x in xList:
			if x not in indic:
				indic.append(x)
	indic = sorted(indic)
	c={}
	for i in indic:
		x = []
		y = []
		for j in bKeys:
			if i in b[j][headName]:
				datLoc = (np.where(b[j][headName]==i))[0][0]
				datVal = b[j][colName][datLoc]
				try:
					j = datetime.strptime(j,'%d/%m/%y %H:%M:%S')
				except:
					try:
						if float(j).is_integer(): j = int(j)
					except:
						try:
							j = float(j)
						except:
							j = j
				x.append(j)
				y.append(float(datVal))
		c[i]=[x,y]
	cKeys = c.keys()
	return c, cKeys
#
#
def datawrite(outdata,demdata,meta,name,outDir):
	'''Write an array of grid data to a georeferenced raster file'''
	#meta = ['projection','geotransform','driver','rows','columns','nanvalue']
	filnm = os.path.join(outDir,(name + '.tif'))
	datdriver = gdal.GetDriverByName( "GTiff" )
	datout = datdriver.Create(filnm,meta[5],meta[4],1,gdal.GDT_Float32)
	datout.SetGeoTransform(meta[2])
	datout.SetProjection(meta[1])
	nanmask = demdata != meta[6]
	outdata_m = np.flipud(outdata * nanmask)
	outdata_m = np.where(outdata_m==0,-9999,outdata_m)
	datout.GetRasterBand(1).WriteArray(outdata_m)
	datout.GetRasterBand(1).SetNoDataValue(-9999)
	datout.GetRasterBand(1).ComputeStatistics(True)
	datout = None
	print "datawrite returns: ",filnm
	return filnm
#
def getClipboardData():
	'''Get contents of clipboard'''
	# from http://www.macdrifter.com/2011/12/python-and-the-mac-clipboard.html
	p = subprocess.Popen(['pbpaste'], stdout=subprocess.PIPE)
	retcode = p.wait()
	data = p.stdout.read()
	return data
#
def setClipboardData(data):
	'''Set contents of clipboard'''
	# from http://www.macdrifter.com/2011/12/python-and-the-mac-clipboard.html
	p = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
	p.stdin.write(data)
	p.stdin.close()
	retcode = p.wait()
	return 0
#
#############################PLOTTING#####################################################
#
#
## Plot data all on one
def plotAllone(PlotTable,xcol,ycols,plotName,ata,xlimit,destDir):
	"""Plot data in the vector dictionary on one single plot"""
	matplotlib.rcParams['axes.grid'] = True
	matplotlib.rcParams['legend.fancybox'] = True
	matplotlib.rcParams['figure.figsize'] = 18, 9
	matplotlib.rcParams['savefig.dpi'] = 300
	# Set figure name and number for pdf ploting
	plotName = plotName + '_a.pdf'
	pp1 = PdfPages(os.path.join(destDir,plotName))
	#
	oldlab=str(ycols[0])
	lnclr = ['r','g','b','y','c','m','k','r','g','b','y','c','m','k','r','g','b','y','c','m','k','r','g','b','y','c','m','k','r','g','b','y','c','m','k']
	lnsty = ['-','-','-','-','-','-','-','--','--','--','--','--','--','--','-.','-.','-.','-.','-.','-.','-.',':',':',':',':',':',':',':',':',':',':',':',':',':',':']
	#
	cnt = 0
	fig1 = plt.figure(ata)
	ax1 = fig1.add_subplot(111)#new
	for nm in ycols:
		if str(nm)==oldlab:
			lab = str(nm)
		else:
			lab = str(PlotTable)+' data'
		oldlab = lab
		#plt.plot(vectors[PlotTable][xcol],vectors[PlotTable][nm],color=lnclr[cnt],linestyle=lnsty[cnt],label = str(nm))
		ax1.plot(vectors[PlotTable][xcol],vectors[PlotTable][nm],color=lnclr[cnt],linestyle=lnsty[cnt],label = str(nm))
		matplotlib.pyplot.axes().set_position([0.05, 0.05, 0.70, 0.85])
		#plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
		ax1.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
		plt.xlabel(str(xcol))
		#ax1.xlabel(str(xcol))
		plt.ylabel(lab)
		#ax1.ylabel(lab)
		ax1.set_xlim(xlimit)
		plt.title('All y columns by '+str(xcol))
		cnt = cnt + 1
		fig1.autofmt_xdate()
	plt.suptitle(PlotTable,fontsize=16, fontweight='bold')
	pp1.savefig()
	pp1.close()
	plt.close()
	ata = ata + 1
	return ata
#
#
def plotAllsep(PlotTable2,xcol2,ycols2,plotName,ata,xlimit,destDir):
	"""Plot data in the vector dictionary on individual plots for each vector"""
	matplotlib.rcParams['axes.grid'] = True
	matplotlib.rcParams['legend.fancybox'] = True
	matplotlib.rcParams['figure.figsize'] = 18, 9
	matplotlib.rcParams['savefig.dpi'] = 300
	# Set figure name and number for pdf ploting
	pdfName = plotName + '_i.pdf'
	pp2 = PdfPages(os.path.join(destDir,plotName))
	#
	if int(len(ycols2)) == 1:
		pc = 1
	else:
		pc = 2
	# Number of figure (plot) rows
	pr = 2
	pltnr = 1
	# Multiple
	fig = plt.figure(ata)
	for nm in ycols2:
		lab = str(nm)
		ay=plt.subplot(pr,pc,pltnr)
		#ay.set_ylim(-15, 20)
		plt.plot(vectors[PlotTable2][xcol2],vectors[PlotTable2][nm],color='black',linestyle='-',label = str(nm))
		plt.title(str(nm)+' by '+str(xcol2))
		#plt.legend(loc='upper right')
		ay.set_xlim(xlimit)
		plt.xlabel(str(xcol))
		plt.ylabel(str(nm))
		fig.subplots_adjust(bottom=0.1, hspace=0.5)
		if pltnr == 1:
			plt.suptitle(PlotTable2,fontsize=16, fontweight='bold')
		pltnr=pltnr+1
		if pltnr == 5:
			fig.autofmt_xdate()
			pp2.savefig(plt.figure(ata))
			plt.close()
			pltnr = 1
			ata = ata + 1
			fig = plt.figure(ata)
	fig.autofmt_xdate()
	pp2.savefig(plt.figure(ata))
	pp2.close()
	plt.close()
	ata = ata + 1
	return ata
#
#
## Plot data all on one
def plotTogether(dataDict,xName,yName,plotName,destDir):
	"""Plot data in the vector dictionary on one single plot"""
	matplotlib.rcParams['axes.grid'] = True
	matplotlib.rcParams['legend.fancybox'] = True
	#matplotlib.rcParams['figure.figsize'] = 18, 9 #Mine
	matplotlib.rcParams['figure.figsize'] = 16.54, 11.69 #A3
	#matplotlib.rcParams['figure.figsize'] = 11.69, 8.27 #A4
	matplotlib.rcParams['savefig.dpi'] = 300
	# Set figure name and number for pdf ploting
	plotName = plotName + '_a.pdf'
	pp1 = PdfPages(os.path.join(destDir,plotName))
	#
	cKeys = sorted(dataDict.keys())
	lnclr = ['r','g','b','y','c','m','k']
	lnsty = ['-','--','-.',':']
	clrcnt = 0
	stycnt = 0
	fig1 = plt.figure(1)
	ax1 = fig1.add_subplot(111)#
	xmin = sys.maxint
	xmax = -1*(sys.maxint-1)
	nritm = len(cKeys)
	if nritm < 31:
		legcolnr = 1
		radj = 0.9
	elif nritm < 61:
		legcolnr = 2
		radj = 0.8
	else:
		legcolnr = 3
		radj = 0.7
	for nm in cKeys:
		if np.min(dataDict[nm][0]) < xmin: xmin = np.min(dataDict[nm][0])
		if np.max(dataDict[nm][0]) > xmax: xmax = np.max(dataDict[nm][0])
		ax1.plot(dataDict[nm][0],dataDict[nm][1],color=lnclr[clrcnt],linestyle=lnsty[stycnt],label = str(nm))
		plt.xticks(dataDict[nm][0],dataDict[nm][0])
		matplotlib.pyplot.axes().set_position([0.05, 0.05, 0.05, 0.05])
		ax1.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.,ncol=legcolnr)
		plt.xlabel(xName)
		plt.ylabel(yName)
		ax1.set_xlim((xmin,xmax))
		plt.title(plotName)
		clrcnt = clrcnt + 1
		if clrcnt == len(lnclr):
			clrcnt = 0
			stycnt = stycnt + 1
			if stycnt == len(lnsty):
				stycnt = 0
	#fig1.autofmt_xdate()
	#plt.suptitle(PlotTable,fontsize=16, fontweight='bold')
	plt.subplots_adjust(right=radj)
	pp1.savefig(bbox_inches='tight')
	pp1.close()
	plt.close()
	return 0
#
#
def maplot(x,y,z,c,m,name,outDir):
	plt.figure()
	plt.autoscale(enable=True, axis='both', tight=False)
	plt.plot(x,y,m,color=c,hold=True)
	zrange = [np.min(z),np.max(z)]
	zmess = 'Z Range = '+ str(zrange[0]) + ' to ' + str(zrange[1])
	plt.xlabel(zmess)
	plt.title(name)
	indatplt = name + '_dataplot.png'
	plt.savefig(os.path.join(outDir,indatplt))
	#NEW
	plt.close()
	return 0
#
#
def histoplot(x,bins,name,outDir):
	plt.figure()
	plt.hist(x,bins)
	plt.xlabel(name)
	plt.ylabel('Count')
	plt.title(name)
	indatplt = name + '_histoplot.png'
	plt.savefig(os.path.join(outDir,indatplt))
	#NEW
	plt.close()
	return 0
#
#
def map3d(x,y,z,c,valmin,valmax,name,outDir):
	'''Plot 3D map (kriging)'''
	plt.figure(figsize=(16,8),facecolor='w')
	h = plt.pcolor(x,y,z,cmap=c, vmax = valmax, vmin = valmin)
	plt.autoscale(enable=True, axis='both', tight=False)
	plt.savefig(os.path.join(outDir, (name + '_data3d.png')))
	#NEW
	plt.close()
	return 0
#
#
def semvar(Z,D,G,name,outDir):
	'''Plot variogram (kriging)'''
	indx=[]
	for i in range(len(Z)):indx.append(i+1)
	C,R = np.meshgrid(indx,indx)
	I = (R > C)
	Dpl = D * I
	Gpl = G * I
	plt.figure()
	plt.plot(Dpl,Gpl,'.')
	plt.xlabel('Lag Distance')
	titletext = name + ' Semivariogram'
	plt.title(titletext)
	plt.ylabel('Variogram')
	plt.savefig(os.path.join(outDir,(name + '_semivar.png')))
	#NEW
	plt.close()
	return 0
#
#
def varestplt(DE,GE,GEest,GErsqrd,nuggest,sillest,rangeest,name,outDir,G_mod_export,G_mod_export_rsqrd,model):
	'''Plot variogram estimate (kriging)'''
	## Called by: modelPlotter
	plt.figure()
	plt.plot(DE,GE,'.',hold=True)
	b = [0, max(DE)]
	c = [sillest,sillest]
	plt.plot(b,c,'--r',hold=True)
	y1 = 1.1 * max(GE)
	plt.ylim(0,y1)
	plt.plot(DE,GEest,':b',hold=True)
	plt.plot(DE,G_mod_export,'--k',hold=True)
	plt.xlabel('Averaged distance between observations')
	plt.ylabel('Averaged semivariance')
	titletext = name + ' Variogram Estimator and ' + model + ' model'
	plt.title(titletext)
	nuggtext = 'Nugget estimate =\n %s' % (str(nuggest))
	silltext = 'Sill estimate =\n %s' % (str(sillest))
	rangetext = 'Range estimate = \n %s' % (str(rangeest))
	GErsqrdtext = 'Estimate R^2 =\n %s' % (str(GErsqrd))
	G_modrsqrdtext = 'Model R^2 =\n %s' % (str(G_mod_export_rsqrd))
	plt.text(max(DE)*0.8,max(GE)*0.1,nuggtext,bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})
	plt.text(max(DE)*0.8,max(GE)*0.9,silltext,bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})
	plt.text(max(DE)*0.8,max(GE)*0.3,rangetext,bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})
	plt.text(max(DE)*0.8,max(GE)*0.5,GErsqrdtext,bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})
	plt.text(max(DE)*0.8,max(GE)*0.7,G_modrsqrdtext,bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})
	plt.savefig(os.path.join(outDir,(name + '_' + model+ '.png')))
	#NEW
	plt.close()
	return 0
#
#
def krigplot(Xg1,Yg1,X,Y,Z,DEMmdZ_,name,outDir):
	'''Plot kriged data as map'''
	# Get range of values for colour scale
	#print name,type(DEMmdZ_)
	currentmin = DEMmdZ_.min()
	currentmax = DEMmdZ_.max()
	plt.figure(figsize=(16,8),facecolor='w')
	if currentmin >-1.5 and currentmax <1.5:
		h = plt.pcolor(Xg1,Yg1,DEMmdZ_,cmap=plt.cm.RdBu, vmax = 1.5, vmin = -1.5)
		plt.scatter(X, Y, c=Z, cmap=plt.cm.RdBu, vmax = 1.5, vmin = -1.5, hold=True)
	elif currentmin >-3.0 and currentmax <3.0:
		h = plt.pcolor(Xg1,Yg1,DEMmdZ_,cmap=plt.cm.RdBu, vmax = 3.0, vmin = -3.0)
		plt.scatter(X, Y, c=Z, cmap=plt.cm.RdBu, vmax = 3.0, vmin = -3.0, hold=True)
	elif currentmin >-5.0 and currentmax <5.0:
		h = plt.pcolor(Xg1,Yg1,DEMmdZ_,cmap=plt.cm.RdBu, vmax = 5.0, vmin = -5.0)
		plt.scatter(X, Y, c=Z, cmap=plt.cm.RdBu, vmax = 5.0, vmin = -5.0, hold=True)
	elif currentmax <6:
		h = plt.pcolor(Xg1,Yg1,DEMmdZ_,cmap=plt.cm.Reds, vmax = 5, vmin = 0)
		plt.scatter(X, Y, c=Z, cmap=plt.cm.RdBu, vmax = 5, vmin = 0, hold=True)
	else:
		soddincmmap = plt.cm.Blues
		soddincmmap.set_under(color='k')
		h = plt.pcolor(Xg1,Yg1,DEMmdZ_,cmap=soddincmmap, vmax = int(currentmax), vmin = int(currentmin))
		plt.scatter(X, Y, c=Z, cmap=plt.cm.RdBu, vmax = int(currentmax), vmin = int(currentmin), hold=True)
	titletext = name + ' Kriging Estimate'
	plt.title(titletext)
	plt.xlabel('x-Coordinates')
	plt.ylabel('y-Coordinates')
	plt.colorbar()
	plt.axis("equal")
	#plt.plot(X,Y,'.k',hold=True)
	#plt.scatter(X, Y, c=Z, hold=True)
	krigtext = 'Min = %f\nMax = %f' % (currentmin,currentmax)
	plt.text(min(X)+((max(X)-min(X))*0.9),min(Y)+((max(Y)-min(Y))*0.9),krigtext,bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})
	krigplt = name + '_krigplot.png'
	plt.savefig(os.path.join(outDir,krigplt))
	#NEW
	plt.close()
	return
#
#
def krigvarplot(Xg1,Yg1,SK,name,outDir):
	'''Plot variance (kriging)'''
	plt.figure(figsize=(10,5),facecolor='w')
	h = plt.pcolor(Xg1,Xg2,SK)
	plt.title(' Kriging Variance')
	plt.xlabel('x-Coordinates')
	plt.ylabel('y-Coordinates')
	plt.colorbar()
	plt.axis("equal")
	plt.plot(X,Y,'ok',hold=True)
	krigvarplt = name + '_krigvarplot.png'
	plt.savefig(os.path.join(outDir,krigvarplt))
	#NEW
	plt.close()
	return
#
#
def funcPlot(xvec,yvec,p,name='plot',outfolder='./',xname='Elevation',yname='Ablation'):
	'''Plot a function'''
	xp = np.linspace(int(np.nanmin(xvec)), int(np.nanmax(xvec)), 100)
	yp = p(xp)
	yvecp = p(xvec)
	Rfit = rSquared(xvec,yvec,yvecp)
	titletext = 'Gradient fit: ' + str(Rfit)
	#xp = np.linspace(1000, 2000, 100)
	plt.figure(figsize=(10,5),facecolor='w')
	plt.plot(xvec, yvec, '.', xp, yp, '-')
	plt.ylim(min(yp)*0.9,max(yp)*1.1)
	plt.xlabel(xname)
	plt.ylabel(yname)
	plt.title(titletext)
	#plt.show()
	plt.savefig(os.path.join(outfolder,(name +'.png')))
	#NEW
	plt.close()
	return 0
#
#
def logTranHist(Zin,outfolder,name):
	''' Plot histograms of data and log transformed data'''
	## Called by: kriging
	# Should data be log transformed?
	plt.figure(1)
	plt.hist(Zin, bins=20, histtype='stepfilled', normed=True, color='b', label='Untransformed')
	plt.title("Untransformed Data")
	plt.xlabel("Value")
	plt.ylabel("Number")
	plt.legend()
	plt.savefig(os.path.join(outfolder,(name +'Hist.png')))
	plt.close()
	if not np.min(Zin) < 0:
		Zlog = np.log(Zin)
		checklist =[]
		for i in range(len(Zlog)):
			if np.isnan(Zlog[i]) | np.isinf(Zlog[i]):
				checklist.append(i)
		Zlog = np.delete(Zlog, checklist)
		plt.figure(2)
		plt.hist(Zlog, bins=20, histtype='stepfilled', normed=True, color='r', label='Log transformed')
		plt.title("Log Transform of data")
		plt.xlabel("Value")
		plt.ylabel("Number")
		plt.legend()
		plt.savefig(os.path.join(outfolder,(name +'logHist.png')))
		plt.close()
	else:
		print "WARNING: UNABLE TO LOG TRANSFORM DATA"
		return Zin
	return Zlog
################################KRIGING AND ANALYSIS#########################################
#
#
def fitFunc(xvec,yvec,fitOrder=1):
	"""Create least squares fitted function through data"""
	z = np.polyfit(np.array(xvec), np.array(yvec), fitOrder)
	p = np.poly1d(z)
	return p
#
#
def gradientAbl(xvec,yvec,dem,name='gradient.tif',outfolder='./',g=1):
	'''Create an ablation gradient then apply the gradient to DEM to get ablation by elevation.
	To use outfile = gradientAbl(xvec,yvec,dem,name,outfolder) '''
	Adata,AXg,AYg,AXg1,AYg1,Arx,Ary,Ademmeta = DemImport(dem)
	AdataMasked = np.ma.masked_where(Adata == Ademmeta[6], Adata)
	z = np.polyfit(xvec, yvec, int(g))
	p = np.poly1d(z)
	outdata = p(AdataMasked)
	outfile = datawrite(outdata,Adata,Ademmeta,name,outfolder)
	return outfile,z,p
#
#
## Taken from http://stackoverflow.com/questions/7997152/python-3d-polynomial-surface-fit-order-dependent
def polyfit2d(x, y, z, order=4):
	'''Fit a polynomial to a surface (for detrending kriging data)'''
	print '\nCreate polynomial trend surface'
	ncols = (order + 1)**2
	G = np.zeros((x.size, ncols))
	ij = itertools.product(range(order+1), range(order+1))
	for k, (i,j) in enumerate(ij):
		G[:,k] = x**i * y**j
	#m, mresids,mrank,ms = np.linalg.lstsq(G, z)
	m,_,_,_ = np.linalg.lstsq(G, z)
	print m
	return m
#
#
## Taken from http://stackoverflow.com/questions/7997152/python-3d-polynomial-surface-fit-order-dependent
def polyval2d(x, y, m):
	'''Apply polynomial to 3D data (for detrending kriging data)'''
	print '\nApply polynomial trend surface'
	order = int(np.sqrt(len(m))) - 1
	ij = itertools.product(range(order+1), range(order+1))
	z = np.zeros_like(x)
	for a, (i,j) in zip(m, ij):
		#print "a: %2.4f i: %2.4f j: %2.4f" % (a, i, j)
		z += a * x**i * y**j
	print "polyval"
	print(np.shape(z))
	return z
#
def detrend(X,Y,Z,Xg,Yg,Xg1,Yg1,outDir,name):
	'''Detrend data for kriging'''
	# To Use: Zdtr, Z_retrend = detrend(X,Y,Z,Xg,Yg,Xg1,Yg1,outDir,name)
	## Call function to create trend polynomial
	detrend_m = polyfit2d(np.array(X),np.array(Y),np.array(Z),4)
	## Call function to detrend data
	print "Grid"
	nxy = np.shape(Xg1)
	print(nxy)
	nx = nxy[1]
	ny = nxy[0]
	xx, yy = np.meshgrid(np.linspace(X.min(), X.max(), nx),
						 np.linspace(Y.min(), Y.max(), ny))
	Zg_trend = polyval2d(xx, yy, detrend_m)
	Zg_trend = np.flipud(Zg_trend)
	#Zg_trend = np.fliplr(Zg_trend) # NOT RIGHT EITHER
	print "Shape Zg_trend: ", np.shape(Zg_trend)
	print "Shape Xg1: ", np.shape(Xg1)
	#
	print "Data vector"
	Z_trend = polyval2d(np.array(X), np.array(Y), detrend_m)
	Zdtr = Z - Z_trend
	#
	plt.imshow(Zg_trend, extent=(X.min(), X.max(), Y.min(), Y.max()))
	plt.scatter(X, Y, c=Z)
	#plt.show()
	plt.savefig(os.path.join(outDir,(name+'_trend')))
	plt.close()
	return Zdtr, Zg_trend
#
def lagCheck(D, **kwargs):
	'''Calculate number and size of lags and check with user'''
	if ('lagrc' in kwargs):
		lagrc = kwargs['lagrc']
		print 'lag size: ',lagrc
	if ('lagnrrc' in kwargs):
		lagnrrc = kwargs['lagnrrc']
		print 'lag no.: ',lagnrrc
	#
	lag, max_lags = lagcalc(D)
	#
	try:
		print 'Lag overridden from %f to %f'%(lag,lagrc)
		lag = lagrc
	except:
		pass
	try:
		print 'Number of lags overridden from %f to %f'%(max_lags,lagnrrc)
		max_lags = lagnrrc
	except:
		pass
	## __kriging_lagcheck__
	lagan = []
	while not lagan:
		print 'Lag size set at: ',lag
		lagin = raw_input('\nEnter value for lag: ')
		try:
			lagan = float(lagin)
			lag = lagan
			print 'Lag set at: ',lag
		except:
			if lagan == []:
				lagan = lag
				print 'Lag set at: ',lag
			else:
				lagan = []
	maxlagan = []
	while not maxlagan:
		print 'Lag number set at: ',max_lags
		maxlagin = raw_input('\nEnter number of lags: ')
		try:
			maxlagan = float(maxlagin)
			max_lags = maxlagan
			print 'Lag number set at: ',max_lags
		except:
			if maxlagan == []:
				maxlagan = max_lags
				print 'No of lags set to: ',max_lags
			else:
				maxlagan = []
	return lag, max_lags
#
def paramGet(parEst, par, parName):
	'''Report current vale of parameter and ask for confirmation or new value'''
	# To use: parEst, par = paramGet(parEst, par, parName)
	paran = ''
	while not isinstance(paran, float):
		print '%s estimate: %2.4f' % (parName, parEst)
		print '%s set at: %2.4f' % (parName, par)
		parin = raw_input('\nEnter value for %s: ' % (parName))
		try:
			paran = float(parin)
			par = paran
			print '%s reset at: %2.4f' % (parName, par)
		except:
			if paran == '':
				paran = par
				print '%s set at: %2.4f' %(parName, par)
			else:
				paran = ''
	return parEst, par
#
def variogramOnly(Datafile,DEM,modelSel=0,*chc,**kwargs):
	'''Create variogram only'''
	if ('lagrc' in kwargs):
		lagrc = kwargs['lagrc']
		print 'lag size: ',lagrc
	if ('lagnrrc' in kwargs):
		lagnrrc = kwargs['lagnrrc']
		print 'lag no.: ',lagnrrc
	if ('sillrc' in kwargs):
		sillrc = kwargs['sillrc']
		print 'sill: ',sillrc
	if ('nuggrc' in kwargs):
		nuggrc = kwargs['nuggrc']
		print 'nugget: ',nuggrc
	if ('rangrc' in kwargs):
		rangrc = kwargs['rangrc']
		print 'range: ',rangrc
	if ('alpha' in kwargs):
		alpha = kwargs['alpha']
		print 'alpha: ',alpha
	else:
		alpha = 2.0
		print 'alpha: ',alpha
	print '\nStart kriging...'
	time_zero = datetime.now()
	#print time_zero.strftime('... at day:%j %H:%M:%S'),'\n'
	# Get DEM
	demdata,Xg,Yg,Xg1,Yg1,rx,ry,demmeta = DemImport(DEM)
	print "Shape Xg: ", np.shape(Xg)
	print "Shape Yg: ", np.shape(Yg)
	print "Shape Xg1: ", np.shape(Xg1)
	print "Shape Yg1: ", np.shape(Yg1)
	gc.collect()
	# Get data for variogram
	if len(chc) == 1:
		Xin,Yin,Zin = InDataArray(Datafile,chc)
	else:
		Xin,Yin,Zin = InDataArray(Datafile)
	#
	# Set output directory
	namstr,ext_,outDir,full_ = namer(Datafile)
	namstr = namstr + '_' + ZcolName
	outDir = os.path.join(outDir,namstr)
	if not os.path.exists(outDir):
		os.makedirs(outDir)
	#
	Zminvec = []
	Zmaxvec = []
	Zminvec.append(np.min(Zin))
	Zmaxvec.append(np.max(Zin))
	#
	# REMOVE NaN and -inf values from data
	checklist =[] # REMOVE NaN
	for i in range(len(Zin)): # REMOVE NaN
		if np.isnan(Zin[i]) | np.isinf(Zin[i]): # REMOVE NaN
			checklist.append(i) # REMOVE NaN
	Z = np.delete(Zin, checklist) # REMOVE NaN
	X = np.delete(Xin, checklist) # REMOVE NaN
	Y = np.delete(Yin, checklist) # REMOVE NaN
	Zminvec.append(np.min(Z))
	Zmaxvec.append(np.max(Z))
	Zp = Z[:] # Copy Z for plot
	yn = ['y','n']
	#
	# DETREND
	detq = '' # DETREND
	while detq not in yn: # DETREND
		detq = raw_input('Detrend data? ') # DETREND
	if detq == 'y': # DETREND
		Z, Z_retrend = detrend(X,Y,Z,Xg,Yg,Xg1,Yg1,outDir,namstr) # DETREND
		tfilename = datawrite(Z_retrend,demdata,demmeta,namstr+'_detrend',outDir) # DETREND
		DEMmaskedtrend = np.ma.masked_where(demdata == demmeta[6], Z_retrend) # DETREND
		filnm2 = namstr + '_detrendp' # DETREND
		krigplot(Xg1,Yg1,X,Y,Z,DEMmaskedtrend,filnm2,outDir) # DETREND
		print "Trend surface plotted to ", tfilename # DETREND
	Zminvec.append(np.min(Z))
	Zmaxvec.append(np.max(Z))
	#
	# TRANSLATE TO POSITIVE
	shifa = '' # TRANSLATE TO POSITIVE
	print "Minimum value in dataset %2.4f" % (np.min(Z)) # TRANSLATE TO POSITIVE
	if np.min(Z) <= 0: # TRANSLATE TO POSITIVE
		while shifa not in yn: # TRANSLATE TO POSITIVE
			shifa = raw_input("Translate data to positive values only (for log transforms)? ") # TRANSLATE TO POSITIVE
			if shifa == 'y': # TRANSLATE TO POSITIVE
				minval = np.min(Z) # TRANSLATE TO POSITIVE
				if minval > 0: # TRANSLATE TO POSITIVE
					print "No translation needed" # TRANSLATE TO POSITIVE
				elif minval <= 0: # TRANSLATE TO POSITIVE
					# shift = np.finfo(float).eps - minval # TRANSLATE TO POSITIVE
					shift = 1 - minval # TRANSLATE TO POSITIVE
					Z = Z + shift # TRANSLATE TO POSITIVE
		Zminvec.append(np.min(Z)) # TRANSLATE TO POSITIVE
		Zmaxvec.append(np.max(Z)) # TRANSLATE TO POSITIVE
	#
	# LOG TRANSFORM
	Zlog = logTranHist(Z,outDir,namstr) # LOG TRANSFORM
	print '\nCHECK IN\n',outDir,'\nFOR DATA HISTOGRAMS.\n' # LOG TRANSFORM
	trana = '' # LOG TRANSFORM
	while trana not in yn: # LOG TRANSFORM
		trana = raw_input('Log transform Z value? ') # LOG TRANSFORM
	if trana == 'y': # LOG TRANSFORM
		Z = Zlog # LOG TRANSFORM
	Zminvec.append(np.min(Z))
	Zmaxvec.append(np.max(Z))
	#
	#
	## __kriging_report1__
	# Report data summary to to user interface
	#stat1 = 'Data file: '+full_+'\n'
	#stat2 = 'Data ranges\n'
	stat3 = 'X: '+str(np.min(X))+' '+str(np.max(X))+'\n'
	stat4 = 'Y: '+str(np.min(Y))+' '+str(np.max(Y))+'\n'
	stat5 = 'Z min: '
	for i in Zminvec:
		stat5 = stat5 + str(i) + ' '
	stat5 = stat5 + '\n'
	stat5a = 'Z max: '
	for i in Zmaxvec:
		stat5a = stat5a + str(i) + ' '
	stat5a = stat5a + '\n'
	print stat1,stat2,stat3,stat4
	print stat5,stat5a
	print '*'*10
	#
	# Create grids of raw data
	X1,X2 = np.meshgrid(X,X)
	Y1,Y2 = np.meshgrid(Y,Y)
	Z1,Z2 = np.meshgrid(Z,Z)
	#
	print "Grids\n", "*"*10
	## __variogram_Dcalc__
	# Calculate distances in XY plane (pythagoras)
	D = np.sqrt((X1 - X2)**2 + (Y1 -Y2)**2)
	#Dang = np.arctan2((X1 - X2),(Y1 -Y2))*180/np.pi
	#Dang[Dang<0]+=360
	print "D\n", "*"*10
	## __variogram_Gcalc__
	# Calculate experimental variogram
	G = 0.5*(Z1 - Z2)**2
	print "G\n", "*"*10
	#
	modelGood = ""
	modelGoodAns = ['Yes','yes','y','Y']
	while modelGood not in modelGoodAns:
		# Calculate lags and maximum lag
		try:
			lag, max_lags = lagCheck(D, lagrc, lagnrrc)
		except:
			lag, max_lags = lagCheck(D)
		#
		# Create experimental variogram
		print "D:\n", np.shape(D)
		print "G:\n", np.shape(G)
		DE,GE,GEest,nuggest,sillest,rangeest,GErsqrd = expvar(D,G,lag,max_lags)
		#
		# If values provided as optional arguments reset here.
		try:
			print 'Sill overridden from %f to %f'%(sillest,sillrc)
			sillest = sillrc
		except:
			pass
		try:
			print 'Nugget overridden from %f to %f'%(nuggest,nuggrc)
			nuggest = nuggrc
		except:
			pass
		try:
			print 'Range overridden from %f to %f'%(rangeest,rangrc)
			rangeest = rangrc
		except:
			pass
		#
		## __variogram_varEstAppr__
		# Plot variogram estimates for appraisal
		nugget = float(nuggest)
		sill = float(sillest)
		range_ = float(rangeest)
		# Create variogram models
		outDirEst = outDir+'/initVar'
		if not os.path.exists(outDirEst):
			os.makedirs(outDirEst)
		G_mod, modelTypes = modelPlotter(nugget,sill,D,range_,lag,max_lags,DE,GE,GEest,GErsqrd,namstr,outDirEst,alpha)
		semvar(Z,D,G,namstr,outDir)
		#
		print 'VARIOGRAM ESTIMATES AND MODEL PLOTS SAVED TO:\n',outDir,'\nPLEASE REVIEW BEFORE PROCEEDING.'
		#
		## __variogram_varModSetup__
		# Ask user to set sill, nugget and lag
		contAns = ''
		while contAns not in ['yes','Yes','YES']:
			sillest, sill = paramGet(sillest, sill, "Sill")
			nuggest, nugget = paramGet(nuggest, nugget, "Nugget")
			rangeest, range_ = paramGet(rangeest, range_, "Range")
			#
			for i in range(len(modelTypes)):
				print i, ': ',modelTypes[i]
			ansa = []
			while ansa not in range(len(modelTypes)):
				ansa = int(raw_input('Which model to use? '))
			print ansa, ' ',modelTypes[ansa]
			modelnr = int(ansa)
			G_mod, modSel = modelSelect(modelnr,nugget,sill,D,range_,alpha)
			G_mod_export,G_mod_export_rsqrd,G_mod_poly1d,G_mod_exp = VarModPrep(D,lag,max_lags,DE,GE,G_mod)
			varestplt(DE,GE,GEest,GErsqrd,nugget,sill,range_,namstr,outDir,G_mod_export,G_mod_export_rsqrd,modSel)
			print 'Variogram model plots saved to:\n',outDir,'\nPLEASE REVIEW BEFORE PROCEEDING.\n\n'
			contAns = raw_input('Enter "yes" to accept this model or any other key to repeat choices: ')
		modelGood = raw_input("Are you sure you want to continue with this model? ")
	return 0
#
#
def kriging(Datafile,DEM,modelSel=0,*chc,**kwargs):
	'''Main kriging starter function'''
	if ('lagrc' in kwargs):
		lagrc = kwargs['lagrc']
		print 'lag size: ',lagrc
	if ('lagnrrc' in kwargs):
		lagnrrc = kwargs['lagnrrc']
		print 'lag no.: ',lagnrrc
	if ('sillrc' in kwargs):
		sillrc = kwargs['sillrc']
		print 'sill: ',sillrc
	if ('nuggrc' in kwargs):
		nuggrc = kwargs['nuggrc']
		print 'nugget: ',nuggrc
	if ('rangrc' in kwargs):
		rangrc = kwargs['rangrc']
		print 'range: ',rangrc
	if ('alpha' in kwargs):
		alpha = kwargs['alpha']
		print 'alpha: ',alpha
	else:
		alpha = 2.0
		print 'alpha: ',alpha
	print '\nStart kriging...'
	time_zero = datetime.now()
	#print time_zero.strftime('... at day:%j %H:%M:%S'),'\n'
	# Get DEM
	demdata,Xg,Yg,Xg1,Yg1,rx,ry,demmeta = DemImport(DEM)
	print "Shape Xg: ", np.shape(Xg)
	print "Shape Yg: ", np.shape(Yg)
	print "Shape Xg1: ", np.shape(Xg1)
	print "Shape Yg1: ", np.shape(Yg1)
	gc.collect()
	# Get data to krige
	if len(chc) == 1:
		Xin,Yin,Zin = InDataArray(Datafile,chc)
	else:
		Xin,Yin,Zin = InDataArray(Datafile)
	#
	# Set output directory
	namstr,ext_,outDir,full_ = namer(Datafile)
	namstr = namstr + '_' + ZcolName
	outDir = os.path.join(outDir,namstr)
	if not os.path.exists(outDir):
		os.makedirs(outDir)
	#
	Zminvec = []
	Zmaxvec = []
	Zminvec.append(np.min(Zin))
	Zmaxvec.append(np.max(Zin))
	#
	# REMOVE NaN and -inf values from data
	checklist =[] # REMOVE NaN
	for i in range(len(Zin)): # REMOVE NaN
		if np.isnan(Zin[i]) | np.isinf(Zin[i]): # REMOVE NaN
			checklist.append(i) # REMOVE NaN
	Z = np.delete(Zin, checklist) # REMOVE NaN
	X = np.delete(Xin, checklist) # REMOVE NaN
	Y = np.delete(Yin, checklist) # REMOVE NaN
	Zminvec.append(np.min(Z))
	Zmaxvec.append(np.max(Z))
	Zp = Z[:] # Copy Z for plot
	yn = ['y','n']
	#
	# DETREND
	detq = '' # DETREND
	while detq not in yn: # DETREND
		detq = raw_input('Detrend data? ') # DETREND
	if detq == 'y': # DETREND
		Z, Z_retrend = detrend(X,Y,Z,Xg,Yg,Xg1,Yg1,outDir,namstr) # DETREND
		tfilename = datawrite(Z_retrend,demdata,demmeta,namstr+'_detrend',outDir) # DETREND
		DEMmaskedtrend = np.ma.masked_where(demdata == demmeta[6], Z_retrend) # DETREND
		filnm2 = namstr + '_detrendp' # DETREND
		krigplot(Xg1,Yg1,X,Y,Z,DEMmaskedtrend,filnm2,outDir) # DETREND
		print "Trend surface plotted to ", tfilename # DETREND
	Zminvec.append(np.min(Z))
	Zmaxvec.append(np.max(Z))
	#
	# TRANSLATE TO POSITIVE
	shifa = '' # TRANSLATE TO POSITIVE
	print "Minimum value in dataset %2.4f" % (np.min(Z)) # TRANSLATE TO POSITIVE
	if np.min(Z) <= 0: # TRANSLATE TO POSITIVE
		while shifa not in yn: # TRANSLATE TO POSITIVE
			shifa = raw_input("Translate data to positive values only (for log transforms)? ") # TRANSLATE TO POSITIVE
			if shifa == 'y': # TRANSLATE TO POSITIVE
				minval = np.min(Z) # TRANSLATE TO POSITIVE
				if minval > 0: # TRANSLATE TO POSITIVE
					print "No translation needed" # TRANSLATE TO POSITIVE
				elif minval <= 0: # TRANSLATE TO POSITIVE
					# shift = np.finfo(float).eps - minval # TRANSLATE TO POSITIVE
					shift = 1 - minval # TRANSLATE TO POSITIVE
					Z = Z + shift # TRANSLATE TO POSITIVE
		Zminvec.append(np.min(Z)) # TRANSLATE TO POSITIVE
		Zmaxvec.append(np.max(Z)) # TRANSLATE TO POSITIVE
	#
	# LOG TRANSFORM
	Zlog = logTranHist(Z,outDir,namstr) # LOG TRANSFORM
	print '\nCHECK IN\n',outDir,'\nFOR DATA HISTOGRAMS.\n' # LOG TRANSFORM
	trana = '' # LOG TRANSFORM
	while trana not in yn: # LOG TRANSFORM
		trana = raw_input('Log transform Z value? ') # LOG TRANSFORM
	if trana == 'y': # LOG TRANSFORM
		Z = Zlog # LOG TRANSFORM
	Zminvec.append(np.min(Z))
	Zmaxvec.append(np.max(Z))
	#
	#
	## __kriging_report1__
	# Report data summary to to user interface
	stat1 = 'Data file: '+full_+'\n'
	stat2 = 'Data ranges\n'
	stat3 = 'X: '+str(np.min(X))+' '+str(np.max(X))+'\n'
	stat4 = 'Y: '+str(np.min(Y))+' '+str(np.max(Y))+'\n'
	stat5 = 'Z min: '
	for i in Zminvec:
		stat5 = stat5 + str(i) + ' '
	stat5 = stat5 + '\n'
	stat5a = 'Z max: '
	for i in Zmaxvec:
		stat5a = stat5a + str(i) + ' '
	stat5a = stat5a + '\n'
	print stat1,stat2,stat3,stat4
	print stat5,stat5a
	print '*'*10
	#
	## __kriging_meshgrids__
	# Create grids of raw data
	X1,X2 = np.meshgrid(X,X)
	Y1,Y2 = np.meshgrid(Y,Y)
	Z1,Z2 = np.meshgrid(Z,Z)
	#
	print "Grids\n", "*"*10
	## __kriging_Dcalc__
	# Calculate distances in XY plane (pythagoras)
	D = np.sqrt((X1 - X2)**2 + (Y1 -Y2)**2)
	#Dang = np.arctan2((X1 - X2),(Y1 -Y2))*180/np.pi
	#Dang[Dang<0]+=360
	print "D\n", "*"*10
	## __kriging_Gcalc__
	# Calculate experimental variogram
	G = 0.5*(Z1 - Z2)**2
	print "G\n", "*"*10
	#
	modelGood = ""
	modelGoodAns = ['Yes','yes','y','Y']
	while modelGood not in modelGoodAns:
		# Calculate lags and maximum lag
		try:
			lag, max_lags = lagCheck(D, lagrc, lagnrrc)
		except:
			lag, max_lags = lagCheck(D)
		#
		# Create experimental variogram
		print "D:\n", np.shape(D)
		print "G:\n", np.shape(G)
		DE,GE,GEest,nuggest,sillest,rangeest,GErsqrd = expvar(D,G,lag,max_lags)
		#
		# If values provided as optional arguments reset here.
		try:
			print 'Sill overridden from %f to %f'%(sillest,sillrc)
			sillest = sillrc
		except:
			pass
		try:
			print 'Nugget overridden from %f to %f'%(nuggest,nuggrc)
			nuggest = nuggrc
		except:
			pass
		try:
			print 'Range overridden from %f to %f'%(rangeest,rangrc)
			rangeest = rangrc
		except:
			pass
		#
		## __kriging_varEstAppr__
		# Plot variogram estimates for appraisal
		nugget = float(nuggest)
		sill = float(sillest)
		range_ = float(rangeest)
		# Create variogram models
		outDirEst = outDir+'/initVar'
		if not os.path.exists(outDirEst):
			os.makedirs(outDirEst)
		G_mod, modelTypes = modelPlotter(nugget,sill,D,range_,lag,max_lags,DE,GE,GEest,GErsqrd,namstr,outDirEst,alpha)
		print 'VARIOGRAM ESTIMATES AND MODEL PLOTS SAVED TO:\n',outDir,'\nPLEASE REVIEW BEFORE PROCEEDING.'
		#
		## __kriging_varModSetup__
		# Ask user to set sill, nugget and lag
		contAns = ''
		while contAns not in ['yes','Yes','YES']:
			sillest, sill = paramGet(sillest, sill, "Sill")
			nuggest, nugget = paramGet(nuggest, nugget, "Nugget")
			rangeest, range_ = paramGet(rangeest, range_, "Range")
			#
			for i in range(len(modelTypes)):
				print i, ': ',modelTypes[i]
			ansa = []
			while ansa not in range(len(modelTypes)):
				ansa = int(raw_input('Which model to use? '))
			print ansa, ' ',modelTypes[ansa]
			modelnr = int(ansa)
			G_mod, modSel = modelSelect(modelnr,nugget,sill,D,range_,alpha)
			G_mod_export,G_mod_export_rsqrd,G_mod_poly1d,G_mod_exp = VarModPrep(D,lag,max_lags,DE,GE,G_mod)
			varestplt(DE,GE,GEest,GErsqrd,nugget,sill,range_,namstr,outDir,G_mod_export,G_mod_export_rsqrd,modSel)
			print 'Variogram model plots saved to:\n',outDir,'\nPLEASE REVIEW BEFORE PROCEEDING.\n\n'
			contAns = raw_input('Enter "yes" to accept this model or any other key to repeat choices: ')
		modelGood = raw_input("Are you sure you want to continue with this model? ")
	# Krige data
	Zkg,G_mod_export,G_mod_export_rsqrd,model = krig(nugget,sill,range_,D,Z,X,Y,Xg,Yg,rx,ry,lag,max_lags,DE,GE,modelnr,alpha)
	print "Kriged data shape: ", np.shape(Zkg)
	print "Kriged data min: %2.2f mean: %2.2f max: %2.2f" % (np.min(Zkg), np.mean(Zkg), np.max(Zkg))
	#
	# REVERSE LOG TRANSFORM
	if trana == 'y': # REVERSE LOG TRANSFORM
		Zkg = np.exp(Zkg) # REVERSE LOG TRANSFORM
		print "Reverse log transform\n", np.min(Zkg), np.mean(Zkg), np.max(Zkg) # REVERSE LOG TRANSFORM
	#
	# REVERSE TRANSLATION
	if shifa == 'y': # REVERSE TRANSLATION
		Zkg = Zkg - shift # REVERSE TRANSLATION
		print "Reverse translation\n", np.min(Zkg), np.mean(Zkg), np.max(Zkg) # REVERSE TRANSLATION
	#
	try:
		Z_ = Zkg.reshape(ry,rx)
		#SK = s2_k.reshape(ry,rx)
		# REVERSE DETREND
		if detq == 'y':  # REVERSE DETREND
			Z_ = Z_ + Z_retrend  # REVERSE DETREND
			print "Reverse detrending\n", np.min(Z_), np.mean(Z_), np.max(Z_) # REVERSE DETREND
		DEMmaskedZ_ = np.ma.masked_where(demdata == demmeta[6], Z_)
		plotson = 'y'
	except:
		plotson = 'n'
	#
# Write output to files
#	 outDir = os.path.join(outDir,namstr)
#	 if not os.path.exists(outDir):
#		 os.makedir(outDir)
	name = namstr + '_kriged_data'
	if plotson == 'y':
		print "Data kriged: ",outDir,name
		outfilename = datawrite(Z_,demdata,demmeta,name,outDir)
	else:
		print plotson
		outfilename = 'Not kriged'
	# Write summary to file
	txtName = namstr + '_vario.txt'
	outtxtName = os.path.join(outDir,txtName)
	varioFile = open(outtxtName, "ab")
	lagout = 'Lag: '+str(lag)+' nr. of lags: '+str(max_lags)+'\n'
	DEout = 'DE: '+str(DE)+'\n'
	GEout = 'GE: '+str(GE)+'\n'
	GEestout = 'GEest: '+str(GEest)+'\n'
	Nuggout = 'Nugget, estimate: '+str(nugget)+' '+str(nuggest)+'\n'
	Sillout = 'Sill, estimate: '+str(sill)+' '+str(sillest)+'\n'
	Rangout = 'Range, estimate: '+str(range_)+' '+str(rangeest)+'\n'
	Alphout = 'Alpha set to:'+str(alpha)
	GErout = 'GE R^2: '+str(GErsqrd)+'\n'
	outlist = [stat1, stat2, stat3, stat4, stat5, stat5a, lagout, DEout, GEout, GEestout, modSel+'\n', Nuggout, Sillout, Rangout, Alphout, GErout]
	for out in outlist:
		varioFile.write(out)
	varioFile.close()
	## PLOT RESULTS
	DEMplotReq = 'n'
	if DEMplotReq == 'y':
		DEMmasked = np.ma.masked_where(demdata == demmeta[6], demdata)
		name = 'DEM'
		map3d(Xg1,Yg1,DEMmasked,plt.cm.RdBu,DEMmasked.min(),DEMmasked.max(),name,outDir)
	#
	## Plot figure of data to krige
	name = namstr + '_in'
	maplot(X,Y,Zp,'r','o',name,outDir)
	## Plot variogram estimators
	name = namstr + '_data'
	varestplt(DE,GE,GEest,GErsqrd,nugget,sill,range_,name,outDir,G_mod_export,G_mod_export_rsqrd,model)
	## Plot semi-variograms
	name = namstr + '_data'
	semvar(Z,D,G,name,outDir)
	if plotson == 'y':
	## Plot kriged data
		name = namstr + '_data'
		krigplot(Xg1,Yg1,X,Y,Zp,DEMmaskedZ_,name,outDir)
	## Plot histogram
		#name = namstr + '_raw'
		#bins = int(len(Zkg)/10)
		#histoplot(Zkg,bins,name,outDir)
	#
	time_i = datetime.now()
	sys.stdout.flush()
	#print time_i.strftime('Day %j %H:%M:%S')
	print 'Total time = ', time_i - time_zero
	print 'Data column kriged: ', ZcolName
	gc.collect()
	print '\a'
	return outfilename
#
#
def lagcalc(D1):
	'''Calculate lag (kriging)'''
	print '\nCalculating lag'
	# Replace zeros in diagonal with nan
	D2 = np.copy(D1)
	np.fill_diagonal(D2,np.nan)
	# nanmin needs to know which axis to check along, otherwise gives min of whole
	lag_c =np.mean(np.nanmin(D2,axis=1))
	# Get maximum distance and divide by 2
	hmd_c = np.nanmax(D1)/2
	# Set number of lags equal to 1/2 max distance / lag length
	max_lags_c = np.floor(hmd_c/lag_c)
	print lag_c, ' ',max_lags_c,'\n'
	return lag_c, max_lags_c
#
#
def OLDexpvar(D,G,lag,max_lags):
	'''Calculate variogram for kriging'''
	## Called by: kriging
	print '\nCalculating variogram'
	# Set seperation distances and calculate variogram
	LAGS = np.ceil(D/lag)
	DE=[]
	PN=[]
	GE=[]
	for i in range(1, int(max_lags)):
		SEL = (LAGS == i); #Selection matrix
		DE.insert(i, np.mean(np.mean(D[SEL]))) #Mean lag
		PN.insert(i, sum(sum(SEL == 1))/2) #Number of pairs
		GE.insert(i, np.mean(np.mean(G[SEL]))) #Variogram estimator
		#print 'DE: %f, PN: %f, GE: %f = '% (np.mean(np.mean(D[SEL])), sum(sum(SEL == 1))/2, np.mean(np.mean(G[SEL])))
	# polyfit to estimate
	GEpoly = np.polyfit(DE,GE,3,full=True)
	# Create function from polyfitted line
	GEpoly1d = np.poly1d(GEpoly[0])
	GEest = GEpoly1d(DE)
	nuggest = GEpoly1d(0)
	GEpoly1d1 = GEpoly1d.deriv(1)
	GEpoly1d2 = GEpoly1d.deriv(2)
	# Get roots of first derivative (inflection points)
	GEinfl = GEpoly1d1.r
	sillest = GEpoly1d(GEinfl.real[0])
	rangeest = GEinfl.real[0]
	rangeest = abs(rangeest)
	# Calculate R^2 for polyfit
	GErsqrd = rSquared(DE,GE,GEest)
	print 'Nugget: ',nuggest,' sill: ',sillest,' range: ',rangeest,'\n'
	return DE,GE,GEest,nuggest,sillest,rangeest,GErsqrd
#
def expvar(D, G, lag, max_lags):
	'''Calculate variogram'''
	## Called by:
	print '\nCalculating variogram'
	###############################
	# Set seperation distances and calculate variogram
	LAGS = np.ceil(D/lag)
	DE=[]
	PN=[]
	GE=[]
	for i in range(1, int(max_lags)):
		SEL = (LAGS == i); #Selection matrix
		DE.insert(i, np.mean(np.mean(D[SEL]))) #Mean lag
		PN.insert(i, sum(sum(SEL == 1))/2) #Number of pairs
		GE.insert(i, np.mean(np.mean(G[SEL]))) #Variogram estimator
		#print 'DE: %f, PN: %f, GE: %f = '% (np.mean(np.mean(D[SEL])), sum(sum(SEL == 1))/2, np.mean(np.mean(G[SEL])))
	###############################
	# polyfit to estimate
	GEpoly = np.polyfit(DE,GE,3,full=True)
	# Create function from polyfitted line
	GEpoly1d = np.poly1d(GEpoly[0])
	GEest = GEpoly1d(DE)
	print "Coefficients of function through data:\n"
	print np.poly1d(GEpoly1d)
	print "DE	GE	GEest\n"
	for counter in range(len(DE)):
		print "%3.4f	 %3.4f	%3.4f" % (DE[counter], GE[counter], GEest[counter])
	nuggest = GEpoly1d(0)
	GEpoly1d1 = GEpoly1d.deriv(1)
	GEpoly1d2 = GEpoly1d.deriv(2)
	###############################
	# Get roots of first derivative (inflection points)
	GEinfl = GEpoly1d1.r
	print "\nRoot of first derivative:\n", GEinfl
	print "Real: ", GEinfl.real
	print "Imaginary: ", GEinfl.imag
	if GEpoly1d(GEinfl.real[0]) > 0:
		sillest = GEpoly1d(GEinfl.real[0])
		rangeest = GEinfl.real[0]
	else:
		sillest = GEpoly1d(GEinfl.real[1])
		rangeest = GEinfl.real[1]
	rangeest = abs(rangeest)
	###############################
	# Calculate R^2 for polyfit
	GErsqrd = rSquared(DE,GE,GEest)
	###############################
	print 'Nugget: ',nuggest,' sill: ',sillest,' range: ',rangeest,'\n'
	return DE, GE, GEest, nuggest, sillest, rangeest, GErsqrd
#
def rSquared(x,y,f):
	'''Calculate coefficient of determination, R^2'''
	x = np.array(x)
	y = np.array(y)
	f = np.array(f)
	ybar = np.mean(y)
	ydif = y - ybar
	ydifsq = ydif * ydif
	SStot = np.sum(ydifsq)
	fydif = f - y
	fydifsq = fydif * fydif
	SSres = np.sum(fydifsq)
	Rsqrd = 1 - (SSres/SStot)
#	print 'R^2 function:\n'
#	 print 'X: ',x
#	 print '\nY: ',y
#	 print '\nf: ',f
#	 print 'Mean y: ',ybar,'\n'
#	 print 'y-difs: ',ydif,'\n',ydifsq,'\n'
#	 print 'SStot: ',SStot,'\n'
#	 print 'f-difs: ',fydif,'\n',fydifsq,'\n'
#	 print 'SSres: ',SSres,'\n'
	print 'R^2: ',Rsqrd,'\n'
	return Rsqrd
#
#
def krig_Org(nuggest,sillest,rangeest,D,Z,X,Y,Xg,Yg,rx,ry,lag,max_lags,DE,GE,modelnr,alpha=2.0):
	'''Kriging function. Kriging routine based on Trauth "Matlab Recipes for Earth Sciences", 2nd edition'''
	## Called by: kriging
	#####
	nugget = float(nuggest)
	sill = float(sillest)
	range_ = float(rangeest)
	#
	# Create variogram model
	G_mod, modSel = modelSelect(modelnr,nugget,sill,D,range_,alpha,lag)
	print 'Use ',modSel
	#
	# Prepare variogram for printing
	G_mod_export,G_mod_export_rsqrd,G_mod_poly1d,G_mod_exp = VarModPrep(D,lag,max_lags,DE,GE,G_mod)
	#####
	# Create empty matrices to accpet data (krigged estimates and variance)
	Zg = np.empty(Xg.shape)
	#s2_k = np.empty(Xg.shape)
	#
	# Manipulate model for application to data
	n = len(X)
	naddr = np.ones((1,n))
	naddc = np.ones((n+1,1))
	G_mod=np.vstack((G_mod,naddr))
	G_mod=np.hstack((G_mod,naddc))
	G_mod[n,n] = 0
	print "G_mod shape: ", np.shape(G_mod)
	G_modDet = np.linalg.det(G_mod)
	print "G_mod determinant: ", G_modDet
	print
	try:
		G_inv = np.linalg.inv(G_mod)
	except:
		try:
			G_inv = np.linalg.pinv(G_mod)
		except:
			sys.exit('Bad Model')
	## Krig values at grid points
	kto = int(len(Xg))
	for k in range(0, kto):
		DOR = ((X - Xg[k])**2 + (Y - Yg[k])**2)**0.5
		G_R,_ = modelSelect(modelnr,nugget,sill,DOR,range_,alpha,lag)
		G_R = np.append(G_R,1)
		# Lagrange multiplier
		E = np.dot(G_inv,G_R)
		Zg[k] = sum(E[0:n]*Z)
		#s2_k[k] = sum(E[0:n]*G_R[0:n])+E[n]
	Zg.reshape(ry,rx)
	#for i in Zg: print i
	#Z_ = Zg.reshape(ry,rx)
	#SK = s2_k.reshape(ry,rx)
	return Zg,G_mod_export,G_mod_export_rsqrd, modSel
#
#
def krig(nuggest,sillest,rangeest,D,Z,X,Y,Xg,Yg,rx,ry,lag,max_lags,DE,GE,modelnr,alpha=2.0):
	'''Kriging function. Kriging routine based on Trauth "Matlab Recipes for Earth Sciences", 2nd edition'''
	## Called by: kriging
	#####
	nugget = float(nuggest)
	sill = float(sillest)
	range_ = float(rangeest)
	#
	# Create variogram model
	G_mod, modSel = modelSelect(modelnr,nugget,sill,D,range_,alpha,lag)
	print 'Use ',modSel
	#
	# Prepare variogram for printing
	G_mod_export,G_mod_export_rsqrd,G_mod_poly1d,G_mod_exp = VarModPrep(D,lag,max_lags,DE,GE,G_mod)
	#####
	# Create empty matrices to accpet data (krigged estimates and variance)
	Zg = np.empty(Xg.shape)
	#s2_k = np.empty(Xg.shape)
	#

	## Krig values at grid points
	kto = int(len(Xg))
	for k in range(0, kto):
		# Use only those points with "range"
		Xl = []
		Yl = []
		Zl = []
		for i in range(len(X)):
			if ( ((X[i] - Xg[k])**2 + (Y[i] -Yg[k])**2)**0.5 < float(range_) ):
				Xl.append(X[i])
				Yl.append(Y[i])
				Zl.append(Z[i])
		if len(Xl) > 0:
			Xl1,Xl2 = np.meshgrid(Xl,Xl)
			Yl1,Yl2 = np.meshgrid(Yl,Yl)
			Dl = np.sqrt((Xl1 - Xl2)**2 + (Yl1 -Yl2)**2)
			G_modl, modSell = modelSelect(modelnr,nugget,sill,Dl,range_,alpha,lag)
			# Manipulate model for application to data
			n = len(Xl)
			naddr = np.ones((1,n))
			naddc = np.ones((n+1,1))
			G_modl=np.vstack((G_modl,naddr))
			G_modl=np.hstack((G_modl,naddc))
			G_modl[n,n] = 0
			try:
				G_inv = np.linalg.inv(G_modl)
			except:
				G_modDet = np.linalg.det(G_modl)
				print "G_mod shape: ", np.shape(G_modl)
				print "G_mod determinant: ", G_modDet
				sys.exit('Bad Model')
			DOR = ((Xl - Xg[k])**2 + (Yl - Yg[k])**2)**0.5
			G_R,_ = modelSelect(modelnr,nugget,sill,DOR,range_,alpha,lag)
			G_R = np.append(G_R,1)
			# Lagrange multiplier
			E = np.dot(G_inv,G_R)
			Zg[k] = sum(E[0:n]*Zl)
			#s2_k[k] = sum(E[0:n]*G_R[0:n])+E[n]
		else:
			Zg[k] = np.nan
			#s2_k[k] = np.nan
	Zg.reshape(ry,rx)
	#for i in Zg: print i
	#Z_ = Zg.reshape(ry,rx)
	#SK = s2_k.reshape(ry,rx)
	return Zg,G_mod_export,G_mod_export_rsqrd, modSel
#
#
def Variogram(Datafile,DEM,chc):
	'''Create variogram, estimated variogram and models for analysis'''
	print 'Creating variograms\n'
	# Get DEM
	demdata,Xg,Yg,Xg1,Yg1,rx,ry,demmeta = DemImport(DEM)
	# Get data to krige
	if chc:
		X,Y,Z = InDataArray(Datafile,chc)
	else:
		X,Y,Z = InDataArray(Datafile)
	# Report data summary to to user interface
	namstr,ext_,outDir,full_ = namer(Datafile)
	histoplot(Z,int(len(Z)/5),namstr,outDir)
	stat1 = 'Data file: '+full_+'\n'
	stat2 = 'Data ranges\n'
	stat3 = 'X: '+str(np.min(X))+' '+str(np.max(X))+'\n'
	stat4 = 'Y: '+str(np.min(Y))+' '+str(np.max(Y))+'\n'
	stat5 = 'Z: '+str(np.min(Z))+' '+str(np.max(Z))+'\n'
	#
	# Create grids of raw data
	X1,X2 = np.meshgrid(X,X)
	Y1,Y2 = np.meshgrid(Y,Y)
	Z1,Z2 = np.meshgrid(Z,Z)
	#
	# Calculate distances in XY plane (pythagoras)
	D = np.sqrt((X1 - X2)**2 + (Y1 -Y2)**2)
	# Calculate experimental variogram
	G = 0.5*(Z1 - Z2)**2
	# Calculate lags and maximum lag
	lag, max_lags = lagcalc(D)
	# Create experimental variogram
	DE,GE,GEest,nuggest,sillest,rangeest,GErsqrd,GEpoly1d = expvar(D,G,lag,max_lags)
	# Fix negative range due to first inflexion point found left of zero axis
	rangeest = abs(rangeest)
	nugget = float(nuggest)
	sill = float(sillest)
	range_ = float(rangeest)
	#
	alpha = 0.5
	# Create variogram model
	# List of model types
	modelTypes = ['TRS','Linear','Exponential with Nugget','Spherical','Gaussian or stable']
	# Create each available model
	G_mod_0 = varMod_TRS(nugget,sill,D,range_)
	G_mod_1 = varMod_Linear(nugget,sill,D,range_)
	G_mod_2 = varMod_expNug(nugget,sill,D,range_)
	G_mod_3 = varMod_spheric(nugget,sill,D,range_)
	G_mod_4 = varMod_Gauss(nugget,sill,D,range_,alpha)
	# Prepare printing the models
	print modelTypes[0]
	G_mod_export_0,G_mod_export_rsqrd_0,G_mod_0_poly1d,G_mod_exp_0 = VarModPrep(D,lag,max_lags,DE,GE,G_mod_0)
	print modelTypes[1]
	G_mod_export_1,G_mod_export_rsqrd_1,G_mod_1_poly1d,G_mod_exp_1 = VarModPrep(D,lag,max_lags,DE,GE,G_mod_1)
	print modelTypes[2]
	G_mod_export_2,G_mod_export_rsqrd_2,G_mod_2_poly1d,G_mod_exp_2 = VarModPrep(D,lag,max_lags,DE,GE,G_mod_2)
	print modelTypes[3]
	G_mod_export_3,G_mod_export_rsqrd_3,G_mod_3_poly1d,G_mod_exp_3 = VarModPrep(D,lag,max_lags,DE,GE,G_mod_3)
	print modelTypes[4]
	G_mod_export_4,G_mod_export_rsqrd_4,G_mod_4_poly1d,G_mod_exp_4 = VarModPrep(D,lag,max_lags,DE,GE,G_mod_4)
	# Prepare text for report file
	mod_0txt = modelTypes[0] + ': ' + str(G_mod_export_rsqrd_0) + '\n'
	mod_1txt = modelTypes[1] + ': ' + str(G_mod_export_rsqrd_1) + '\n'
	mod_2txt = modelTypes[2] + ': ' + str(G_mod_export_rsqrd_2) + '\n'
	mod_3txt = modelTypes[3] + ': ' + str(G_mod_export_rsqrd_3) + '\n'
	mod_4txt = modelTypes[4] + ' (alpha = '+str(alpha)+'): ' + str(G_mod_export_rsqrd_4) + '\n'	#
	# Print the models
	matplotlib.rcParams['axes.grid'] = True
	#matplotlib.rcParams['legend.fancybox'] = True
	#matplotlib.rcParams['figure.figsize'] = 18, 9 #Mine
	#matplotlib.rcParams['figure.figsize'] = 16.54, 11.69 #A3
	matplotlib.rcParams['figure.figsize'] = 11.69, 8.27 #A4
	matplotlib.rcParams['savefig.dpi'] = 300
	plt.figure()
	plt.plot(DE,GE,'.',hold=True)
	sillx = [0, range_]
	silly = [sillest,sillest]
	plt.plot(sillx,silly,':k',hold=True)
	rangex = [range_,range_]
	rangey = [0,sillest]
	plt.plot(rangex,rangey,':k',hold=True)
	y1 = 1.5 * max(GE)
	if sillest > y1:
		sillloc = 1.2 * max(GE)
	else:
		sillloc = 1.1*sillest
	if range_ >= max(DE):
		xcoords = range(0,int(range_*1.2),int(range_/100))
	else:
		xcoords = range(0,int(max(DE)*1.2),int(range_/100))
	plt.ylim(0,y1)
	plt.plot(DE,GEest,'-k',hold=True)
	plt.plot(xcoords,G_mod_0_poly1d(xcoords),':r',hold=True,label=modelTypes[0])
	plt.plot(DE,G_mod_exp_0,'--r',hold=True,label=modelTypes[0])
	plt.plot(xcoords,G_mod_1_poly1d(xcoords),':g',hold=True,label=modelTypes[1])
	plt.plot(DE,G_mod_exp_1,'--g',hold=True,label=modelTypes[1])
	plt.plot(xcoords,G_mod_2_poly1d(xcoords),':b',hold=True,label=modelTypes[2])
	plt.plot(DE,G_mod_exp_2,'--b',hold=True,label=modelTypes[2])
	plt.plot(xcoords,G_mod_3_poly1d(xcoords),':k',hold=True,label=modelTypes[3])
	plt.plot(DE,G_mod_exp_3,'--k',hold=True,label=modelTypes[3])
	plt.plot(xcoords,G_mod_4_poly1d(xcoords),':k',hold=True,label=modelTypes[4])
	plt.plot(DE,G_mod_exp_4,'--m',hold=True,label=modelTypes[4])
	plt.xlabel('Averaged distance between observations')
	plt.ylabel('Averaged semivariance')
	titletext = namstr + ' Variogram estimator and models'
	plt.title(titletext)
	nuggtext = 'Nugget estimate =\n %s' % (str(nuggest))
	nuggtxt = nuggtext + '\n'
	silltext = 'Sill estimate =\n %s' % (str(sillest))
	silltxt = silltext + '\n'
	rangetext = 'Range estimate = \n %s' % (str(rangeest))
	rangetxt = rangetext + '\n'
	GErsqrdtext = 'Estimate R^2 =\n %s' % (str(GErsqrd))
	GErsqrdtxt = GErsqrdtext + '\n'
	plt.text(range_*0.5,max(GE)*0.1,nuggtext,bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})
	plt.text(range_,sillloc,silltext,bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})
	plt.text(range_,max(GE)*0.1,rangetext,bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})
	plt.text(max(DE)*0.8,max(GE)*0.5,GErsqrdtext,bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})
	plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.,ncol=1)
	plt.savefig(os.path.join(outDir,(namstr+'_variogramFigure.png')),bbox_inches='tight')
	# Write report file
	repnam = os.path.join(outDir,(namstr+'_variogramReport.txt'))
	report = open(repnam, "ab")
	report.write(stat1)
	report.write(stat2)
	report.write(stat3)
	report.write(stat4)
	report.write(stat5)
	report.write(nuggtxt)
	report.write(silltxt)
	report.write(rangetxt)
	report.write(GErsqrdtxt)
	report.write(mod_0txt)
	report.write(mod_1txt)
	report.write(mod_2txt)
	report.write(mod_3txt)
	report.write(mod_4txt)
	report.close()
	return 0
#
#
def modelPlotter(nugget,sill,D,range_,lag,max_lags,DE,GE,GEest,GErsqrd,namstr,outDir,alpha=2.0):
	'''Call variogram model creators and plot results to file.'''
	## Called by: kriging
	modelTypes = ['TRS','Linear','Exponential with Nugget','Spherical','Gaussian or stable']
	for modSel in modelTypes:
		if modSel == 'Exponential with Nugget':
			print modSel
			G_mod = varMod_expNug(nugget,sill,D,range_)
		elif modSel == 'Spherical':
			print modSel
			G_mod = varMod_spheric(nugget,sill,D,range_)
		elif modSel == 'Gaussian or stable':
			print modSel
			G_mod = varMod_Gauss(nugget,sill,D,range_,alpha)
		elif modSel == 'Linear':
			print modSel
			G_mod = varMod_Linear(nugget,sill,D,range_)
		elif modSel == 'TRS':
			print modSel
			G_mod = varMod_TRS(nugget,sill,D,range_)
		G_mod_export,G_mod_export_rsqrd,G_mod_poly1d,G_mod_exp = VarModPrep(D,lag,max_lags,DE,GE,G_mod)
		varestplt(DE,GE,GEest,GErsqrd,nugget,sill,range_,namstr,outDir,G_mod_export,G_mod_export_rsqrd,modSel)
	return G_mod, modelTypes
#
#
def modelSelect(modelnr,nugget,sill,D,range_,lag,alpha=2.0):
	modelTypes = ['TRS','Linear','Exponential with Nugget','Spherical','Gaussian or stable']
	try:
		modSel = modelTypes[modelnr]
	except:
		modSel = 'Exponential with Nugget'
	if modSel == 'Exponential with Nugget':
		G_mod = varMod_expNug(nugget,sill,D,range_)
	elif modSel == 'Spherical':
		G_mod = varMod_spheric(nugget,sill,D,range_)
	elif modSel == 'Gaussian or stable':
		G_mod = varMod_Gauss(nugget,sill,D,range_,alpha)
	elif modSel == 'Linear':
		G_mod = varMod_Linear(nugget,sill,D,range_)
	elif modSel == 'TRS':
		G_mod = varMod_TRS(nugget,sill,D,range_)
	return G_mod, modSel
#
#
def VarModPrep(D,lag,max_lags,DE,GE,G_mod):
	'''Create variogram model for printing'''
	## Called by: modelPlotter, kriging
	# Export variogram for printing
	LAGS = np.ceil(D/lag)
	G_mod_copy = G_mod[:]
	G_mod_exp = []
	for i in range(1, int(max_lags)):
		SEL = (LAGS == i)
		G_mod_exp.insert(i, np.mean(np.mean(G_mod_copy[SEL])))
	G_mod_poly = np.polyfit(DE,G_mod_exp,3,full=True)
	# Create function from polyfitted line
	G_mod_poly1d = np.poly1d(G_mod_poly[0])
	G_mod_export = G_mod_poly1d(DE)
	# Calculate R^2 for polyfit
	G_mod_export_rsqrd = rSquared(DE,GE,G_mod_export)
	return G_mod_export,G_mod_export_rsqrd,G_mod_poly1d,G_mod_exp
#
#
def varMod_expNug(nugget,sill,D,range_):
	'''Variogram model from Trauth book.
	Variogram: exponential with nugget variance'''
	## Called by: modelPlotter, modelSelect
	#print '\nExponential with nugget\n'
	sill=sill-nugget
	G_mod = (nugget + sill*(1 - np.exp(-3*D/range_)))*(D>0)
	return G_mod
#
#
def varMod_spheric(nugget,sill,D,range_):
	'''Variogram: spherical'''
	## Called by: modelPlotter, modelSelect
	#print '\nSpherical\n'
	sill=sill-nugget
	G_mod_1 = (nugget + sill*(((3*D)/(2*range_)) - 0.5*((D/range_)**3)))*(D>0)*(D<range_)
	G_mod_2 = (nugget + sill) *(D>0)*(D>=range_)
	G_mod = G_mod_1 + G_mod_2
	return G_mod
#
#
def varMod_Gauss(nugget,sill,D,range_,alpha = 2.0):
	'''Variogram: Gaussian or stable. If alpha not passed it is set to 2 and the model is
	Gaussian.'''
	## Called by: modelPlotter, modelSelect
	#print '\nGaussian or stable model\n'
	#print 'Alpha = ',alpha
	sill=sill-nugget
	#G_mod = (nugget + sill*(1 - np.exp(-1*(D**alpha)/(((range_**2)/3)**alpha))))*(D>0)
	#r1 = (range_**2)/3
	#G_mod = (nugget + sill*(1 - np.exp(-1*D**alpha/r1**alpha)))*(D>0)
	G_mod = (nugget + sill*(1 - np.exp(-3*D**alpha/range_**alpha)))*(D>0)
	return G_mod
#
#
def varMod_Linear(nugget,sill,D,range_):
	'''Bounded linear model as used by TRS'''
	## Called by: modelPlotter, modelSelect
	#print 'Bounded linear model as used by TRS'
	#range_ = 500
	#sill = 1
	#sill=sill-nugget
	G_mod_1 = sill*(D/range_)*(D>0)*(D<range_)
	G_mod_2 = sill *(G_mod_1 == 0)
	G_mod = G_mod_1 + G_mod_2
	return G_mod
#
def varMod_TRS(nugget,sill,D,range_):
	'''Bounded linear model as used by TRS.
	According R. Petterssons kriging.m file in the massbalance toolbox for Matlab in lines 189 to 212
	linear model with no nugget, slope of 1 and power of 1. These are default values left unaltered.'''
	## Called by: modelPlotter, modelSelect
	#print 'Bounded linear model as used by TRS'
	range_ = range_
	slope = 1
	G_mod_1 = slope*(D/range_)*(D>0)*(D<range_)
	G_mod_2 = slope*(G_mod_1 == 0)
	G_mod = G_mod_1 + G_mod_2
	return G_mod
#
#
def xydiff(atype, bearing, distance):
	"""Calculate the x and y coordinates (relative) from HD and Bearing"""
	# Convert bearing to radians
	if atype == 'd':
		bearing = math.radians(bearing)
	elif atype == 'g':
		bearing = bearing * (math.pi / 200)
	elif atype == 'r':
		bearing = bearing
	#
	# Calculate difference in Easting
	ediff = math.sin(bearing)*distance
	# Calculate difference in Northing
	ndiff = math.cos(bearing)*distance
	return  ediff, ndiff
#
#
def hordist(atype,vangle,slpdist):
	"""Calculate the horisontal distance from the sloping distance and the vertical angle"""
	# Convert angle to radians
	if atype == 'd':
		vangle = math.radians(vangle)
	elif atype == 'g':
		vangle = vangle * (math.pi / 200)
	elif atype == 'r':
		vangle = vangle
	#
	# Calculate horisontal distance
	hdist = math.cos(vangle)*slpdist
	# Calculate height difference
	vdist = math.sin(vangle)*slpdist
	#
	return hdist, vdist
#
################################DATA MANIPULATION#########################################
#
#
def geopixsum(filename):
	'''Sum all the non NaN values in a raster file
	To Use:[sumval, area, average, countval] = geopixsum(filename) '''
	# register all of the GDAL drivers
	gdal.AllRegister()
	sumval = 'No File'
	# open the image
	try:
		inDs = gdal.Open(filename)
	except:
		print 'Could not open ',file,'\n'
	# get image size
	rows = inDs.RasterYSize
	cols = inDs.RasterXSize
	transf = inDs.GetGeoTransform()
	ul_x = transf[0]
	ul_y = transf[3]
	xres = transf[1]
	yres = transf[5]
	#print 'rows = ',rows,' cols = ',cols
	# read band 1 into data
	band1 = inDs.GetRasterBand(1)
	data = band1.ReadAsArray(0,0,cols,rows)
	print np.shape(data)
	# get nodata value
	nandat = band1.GetNoDataValue()
	print "NaN value: ", nandat
	sumvals = data[np.logical_not((np.isnan(data)) + (np.isinf(data)) + (data==nandat))]
	sumval = sumvals.sum()
	countval = len(sumvals)
	average = sumval/countval
	area = countval * abs(xres * yres)
	print "Sum = %2.3f, Area = %2.1f, Average = %2.3f, Number = %d" % (sumval, area, average, countval)
	inDs = None
	return [sumval, area, average, countval]
#
#
def demslicer(file,outfolder,slice=10):
	'''Slice a raster (usually a DEM) into bands according to pixel value
	To use: demslicer(file,outfolder,slice=10) '''
	print slice
	print '\nImporting ',file
	time_one = datetime.now()
	print time_one.strftime('at day:%j %H:%M:%S')
	demdata,Xg,Yg,Xg1,Yg1,rx,ry,demmeta = DemImport(file)
	demmeta.append(['projection','geotransform','driver','rows','columns','nanvalue','min','max'])
	# Get range for slices by rounding range of data to nearest slice size
	# round to nearest whole
	floordat = np.floor(demmeta[7])
	ceildat = np.ceil(demmeta[8])
	# Divide to multiple
	floordat_10 = floordat / slice
	ceildat_10 = ceildat / slice
	# Round to multiple
	floordat_10_f = np.floor(floordat_10)
	ceildat_10_c = np.ceil(ceildat_10)
	# Multiply back to original scale
	floord = floordat_10_f * slice
	ceild = ceildat_10_c * slice
	# Create slice range
	slicerange = range(int(floord),int(ceild),int(slice))
	#
	# Slice the data
	for i in slicerange:
		data0 = np.zeros(demdata.shape, demdata.dtype)
		# Select array members with value within range
		slice1 = np.where(np.logical_and(demdata>=i,demdata<(i+slice)))
		# Set array data0 members indexed by slice1 to 1
		data0[slice1] = 1
		# Create output file
		name = str(i) + 'to' + str(i+slice)
		filnm = datawrite(data0,demdata,demmeta,name,outfolder)
		print filnm
	print '\n'
	return 0
#
#
def rasXras(flist,flistfolder,Afile,outfolder):
	'''Multiply one raster file with a list of other raster files and calculate "volume".
	To use: filnm = rasXras(flist,flistfolder,Afile,outfolder) '''
	time_one = datetime.now()
	print time_one.strftime('Start rasXras at:%j %H:%M:%S')
#	 print flist
#	 print flistfolder
#	 print Afile
#	 print outfolder
	# Create/Open text file for storing results
	resnam = os.path.join(outfolder,(time_one.strftime('%j_%H%M%S')+'_results.csv'))
	f = open(resnam,'a')
	f.write('file,sum,count,pixel_avg,pixelsize,area,volume \n')
	# Get first raster file
	Adata,AXg,AYg,AXg1,AYg1,Arx,Ary,Ademmeta = DemImport(Afile)
	print Afile,' opened as first file.'
	# Go through list of other raster files
	for B in flist:
		print 'Test: ',B
		Bfile = os.path.join(flistfolder,B)
		Bdata,BXg,BYg,BXg1,BYg1,Brx,Bry,Bdemmeta = DemImport(Bfile)
		print Bfile,' opened.'
		print 'Rows (A,B): ',Arx,Brx,' Columns (A,B): ',Ary,Bry
		print 'xres (A,B): ',Ademmeta[2][1],Bdemmeta[2][1],' yres (A,B): ',Ademmeta[2][5],Bdemmeta[2][5]
		print 'No data (A): ',Ademmeta[6],' No data (B): ',Bdemmeta[6]
		# Check matching resolution
		if Arx != Brx or Ary != Bry:
			print B,' resolution mismatch with ', Afile
			continue
		elif Ademmeta[4] != Bdemmeta[4] or Ademmeta[5] != Bdemmeta[5]:
			print 'Size mismatch between ',B,' and ',Afile
			continue
		# Multiply first file with current file
		AdataMasked = np.ma.masked_where(Adata == Ademmeta[6], Adata)
		BdataMasked = np.ma.masked_where(Bdata == Bdemmeta[6], Bdata)
		outdata = AdataMasked * BdataMasked
		name = namer(B)[0]+'X'+namer(Afile)[0]
		#print "A: ",AdataMasked
		#print "B: ",BdataMasked
		#print "outdata: ", outdata
		#print "Adata: ", Adata
		print "Ademmeta: ",Ademmeta
		print "name: ",name
		print "outfolder: ",outfolder
		filnm = datawrite(outdata,Adata,Ademmeta,name,outfolder)
		sumval = np.sum(outdata)
		print 'Sum of values: ',sumval,'\n'
		avval = np.mean(outdata)
		print 'Mean value ',avval,'\n'
		countval = sumval/avval
		print 'Number of values: ',countval
		pixsize = abs(Ademmeta[2][1] * Ademmeta[2][5])
		area = pixsize * countval
		volume = avval * area
		# write values to text file
		blnk = ','
		endln = '\n'
		sumvalstr = str(sumval)
		countvalstr = str(countval)
		avvalstr = str(avval)
		pixsizestr = str(pixsize)
		areastr = str(area)
		volumestr = str(volume)
		results = B.split('.')[0] + blnk + sumvalstr + blnk + countvalstr + blnk + \
		avvalstr + blnk + pixsizestr + blnk + areastr + blnk + volumestr + endln
		f.write(results)
	f.close()
	print '\n'
	return filnm
#
#
def rasterAverage(flist,flistfolder,outfolder):
	'''Average all rasters together
	To use: filnmsum,filnmmean = rasterAverage(flist,flistfolder,outfolder) '''
	time_one = datetime.now()
	print time_one.strftime('Start rasterAverage at:%j %H:%M:%S')
	# Get first raster file
	Afile = os.path.join(flistfolder,flist[0])
	Adata,AXg,AYg,AXg1,AYg1,Arx,Ary,Ademmeta = DemImport(Afile)
	# Get second for creating mask to deal with NaN
	Bfile = os.path.join(flistfolder,flist[1])
	Bdata,BXg,BYg,BXg1,BYg1,Brx,Bry,Bdemmeta = DemImport(Bfile)
	Adata[Adata==Ademmeta[6]]=np.nan
	Bdata[Bdata==Bdemmeta[6]]=np.nan
	sumdata = np.ma.masked_array(np.nan_to_num(Adata), mask=np.isnan(Adata) & np.isnan(Bdata))
	counter = 1
	print '\n'*3
	print '*'*20
	# Go through list of other raster files
	for B in flist[1:]:
		print 'file ',counter+1,' of ',len(flist)
		Bfile = os.path.join(flistfolder,B)
		Bdata,BXg,BYg,BXg1,BYg1,Brx,Bry,Bdemmeta = DemImport(Bfile)
		print B,' data type: ',type(Bdata)
		print 'Rows (A,B): ',Arx,Brx,' Columns (A,B): ',Ary,Bry
		print 'xres (A,B): ',Ademmeta[2][1],Bdemmeta[2][1],' yres (A,B): ',Ademmeta[2][5],Bdemmeta[2][5]
		print 'No data (A): ',Ademmeta[6],' No data (B): ',Bdemmeta[6]
		# Check matching resolution
		if Arx != Brx or Ary != Bry:
			print B,' resolution mismatch with ', Afile
			continue
		elif Ademmeta[4] != Bdemmeta[4] or Ademmeta[5] != Bdemmeta[5]:
			print 'Size mismatch between ',B,' and ',Afile
			continue
		# Add current file to sum
		Bdata[Bdata==Bdemmeta[6]]=np.nan
		BdataMasked  = np.ma.masked_array(np.nan_to_num(Bdata), mask=sumdata.mask)
		sumdata = (sumdata + BdataMasked).filled(np.nan)
		sumdata = np.ma.masked_array(np.nan_to_num(sumdata), mask=np.isnan(sumdata))
		counter = counter + 1
	meandata = sumdata/counter
	#
	sumname = 'file_sum'
	filnmsum = datawrite(sumdata,Adata,Ademmeta,sumname,outfolder)
	meanname = 'file_mean'
	filnmmean = datawrite(meandata,Adata,Ademmeta,meanname,outfolder)
	return filnmsum,filnmmean
#
#
def rasSubras(flist,flistfolder,Afile,outfolder):
	'''Subtract one file from others in list
	To use: filnm = rasSubras(flist,flistfolder,Afile,outfolder) '''
	#time_one = datetime.now()
	#print time_one.strftime('Start raster subtraction at:%j %H:%M:%S')
	# Get first raster file
	Adata,AXg,AYg,AXg1,AYg1,Arx,Ary,Ademmeta = DemImport(Afile)
	Amean = nanmean(Adata)
	# Get second for creating mask to deal with NaN
#	 Bfile = os.path.join(flistfolder,flist[1])
#	 Bdata,BXg,BYg,BXg1,BYg1,Brx,Bry,Bdemmeta = DemImport(Bfile)
#	 Adata[Adata==Ademmeta[6]]=np.nan
#	 Bdata[Bdata==Bdemmeta[6]]=np.nan
#	 AdataMasked = np.ma.masked_array(np.nan_to_num(Adata), mask=np.isnan(Adata) & np.isnan(Bdata))
	counter = 1
	print '\n'*3
	print '*'*20
	# Go through list of other raster files
	for B in flist:
		print 'file ',counter+1,' of ',len(flist)
		Bfile = os.path.join(flistfolder,B)
		Bdata,BXg,BYg,BXg1,BYg1,Brx,Bry,Bdemmeta = DemImport(Bfile)
		print B,' data type: ',type(Bdata)
		print 'Rows (A,B): ',Arx,Brx,' Columns (A,B): ',Ary,Bry
		print 'xres (A,B): ',Ademmeta[2][1],Bdemmeta[2][1],' yres (A,B): ',Ademmeta[2][5],Bdemmeta[2][5]
		print 'No data (A): ',Ademmeta[6],' No data (B): ',Bdemmeta[6]
		# Check matching resolution
		if Arx != Brx or Ary != Bry:
			print B,' resolution mismatch with ', Afile
			continue
		elif Ademmeta[4] != Bdemmeta[4] or Ademmeta[5] != Bdemmeta[5]:
			print 'Size mismatch between ',B,' and ',Afile
			continue
		# Add current file to sum
		Bdata[Bdata==Bdemmeta[6]]=np.nan
		AdataMasked = np.ma.masked_array(np.nan_to_num(Adata), mask=np.isnan(Adata) & np.isnan(Bdata))
		BdataMasked  = np.ma.masked_array(np.nan_to_num(Bdata), mask=AdataMasked.mask)
		outdata = (BdataMasked - AdataMasked).filled(np.nan)
		outdata = np.ma.masked_array(np.nan_to_num(outdata), mask=np.isnan(outdata))
		counter = counter + 1
		outname = namer(B)[0]+'Sub'+namer(Afile)[0]
		filnm = datawrite(outdata,Adata,Ademmeta,outname,outfolder)
	#
	return filnm
#
#
def maskPosNeg(vector,Afile,name,outfolder):
	'''Create a mask raster from a vector over a raster then invert masked area to create mask of remaining area
	Created for masks of interpolated and extrapolated ares of glacier
	To use: maskPosNeg(vector,Afile,name,outfolder) '''
	# Get background raster (glacier dem) and calculate dimensions etc.
	Adata,AXg,AYg,AXg1,AYg1,Arx,Ary,Ademmeta = DemImport(Afile)
	transf = Ademmeta[2]
	ul_x = transf[0]
	ul_y = transf[3]
	xres = transf[1]
	yres = transf[5]
	# get image size
	demrows = Ademmeta[4]
	demcols = Ademmeta[5]
	# Calculate corners
	ll_x = ul_x
	ll_y = ul_y + (demrows * yres)
	ur_x = ll_x + (demcols * xres)
	ur_y = ul_y
	# Create names for output data
	Aname,Aext,Apath,Anamefull = namer(Afile)
	outFile1 = os.path.join(outfolder,(name + '_posMask_Temp.tif'))
	outFile2 = os.path.join(outfolder,(name + '_posMask.tif'))
	outFile3 = name + '_negMask'
	# Start by clipping to vector
	gdalMess1 = 'gdalwarp -te ' + str(ll_x) + ' ' + str(ll_y) + ' ' + str(ur_x) + ' ' + str(ur_y) + ' -tr ' +str(xres) + ' ' + str(yres) + ' -dstnodata -9999 -q -cutline ' + vector + ' -crop_to_cutline -of GTiff ' + Afile + ' ' + outFile1
	os.system(gdalMess1)
	# Then resize to original raster, I kid you not.
	gdalMess2 = 'gdalwarp -te ' + str(ll_x) + ' ' + str(ll_y) + ' ' + str(ur_x) + ' ' + str(ur_y) + ' -tr ' +str(xres) + ' ' + str(yres) + ' -dstnodata -9999 ' + outFile1 + ' ' + outFile2
	os.system(gdalMess2)
	print outFile2,' created.'
	# Remove temp file
	command = 'rm ' + outFile1
	os.system(command)
	#
	Bfile = outFile2
	Bdata,BXg,BYg,BXg1,BYg1,Brx,Bry,Bdemmeta = DemImport(Bfile)
	print Bfile,' data type: ',type(Bdata)
	print 'Rows (A,B): ',Arx,Brx,' Columns (A,B): ',Ary,Bry
	print 'xres (A,B): ',Ademmeta[2][1],Bdemmeta[2][1],' yres (A,B): ',Ademmeta[2][5],Bdemmeta[2][5]
	print 'No data (A): ',Ademmeta[6],' No data (B): ',Bdemmeta[6]
	# Check matching resolution
	if Arx != Brx or Ary != Bry:
		print Bfile,'\nResolution mismatch with ', Afile
	elif Ademmeta[4] != Bdemmeta[4] or Ademmeta[5] != Bdemmeta[5]:
		print '\nSize mismatch between ',Bfile,' and ',Afile
	# Do the masking and subtracting
	Bdata[Bdata==Bdemmeta[6]]=np.nan
	AdataMasked = np.ma.masked_array(np.nan_to_num(Adata), mask=np.isnan(Adata))# & np.isnan(Bdata))
	BdataMasked = np.ma.masked_array(np.nan_to_num(Bdata), mask=AdataMasked.mask)
	outdata = (AdataMasked - BdataMasked).filled(np.nan)
	outdata = np.ma.masked_array(np.nan_to_num(outdata), mask=np.isnan(outdata))
	filnm = datawrite(outdata,Adata,Ademmeta,outFile3,outfolder)
	print 'maskPosNeg done.'
	return 0
#
#
def gmttrans(infile):
	'''Quick and dirty transformation of csv columns between RT90 and SWEREF
	To use: gmttrans(infile) '''
	directions = ['r2s','s2r']
	for i in directions: print i
	direction = ''
	while direction not in directions:
		direction = raw_input('Enter r2s for RT90 2,gV to SWEREF99 TM or s2r for reverse: ')
	if direction == 'r2s':
		inSpatialRef = osr.SpatialReference()
		inSpatialRef.ImportFromEPSG(3021) #RT90 2,gV
		outSpatialRef = osr.SpatialReference()
		outSpatialRef.ImportFromEPSG(3006) #SWEREF99 TM
	if direction == 's2r':
		inSpatialRef = osr.SpatialReference()
		inSpatialRef.ImportFromEPSG(3006) #SWEREF99 TM
		outSpatialRef = osr.SpatialReference()
		outSpatialRef.ImportFromEPSG(3021) #RT90 2,gV
	# create Coordinate Transformation
	coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
	#driver = ogr.GetDriverByName('ESRI Shapefile')
	driver = ogr.GetDriverByName('GMT')
	indataset = driver.Open(infile, 0)
	if indataset is None:
		print ' Could not open file ',infile
		sys.exit(1)
	inlayer = indataset.GetLayer()
	outfileshortname,extension,outfilepath,namefull = namer(infile)
	if direction == 'r2s':
		outname = outfileshortname +'_tm'+'.gmt'
	if direction == 's2r':
		outname = outfileshortname +'_rn'+'.gmt'
	outfile = os.path.join(outfilepath,outname)
	print outfile
	outdataset = driver.CreateDataSource(outfile)
	if outfile is None:
		print ' Could not create file'
		sys.exit(1)
	outlayer = outdataset.CreateLayer(outfileshortname, geom_type=ogr.wkbPolygon)
	# get the FeatureDefn for the output shapefile
	featureDefn = outlayer.GetLayerDefn()
	#Loop through input features and write to output file
	infeature = inlayer.GetNextFeature()
	while infeature:
		#get the input geometry
		geometry = infeature.GetGeometryRef()
		#reproject the geometry, each one has to be projected seperately
		geometry.Transform(coordTransform)
		#create a new output feature
		outfeature = ogr.Feature(featureDefn)
		#set the geometry and attribute
		outfeature.SetGeometry(geometry)
		outlayer.CreateFeature(outfeature)
		#destroy the features and get the next input features
		outfeature.Destroy
		infeature.Destroy
		infeature = inlayer.GetNextFeature()
	#close the files
	indataset.Destroy()
	outdataset.Destroy()
	#create the prj projection file
	outSpatialRef.MorphToESRI()
	file = open(outfilepath + '\\'+ outfileshortname + '.prj', 'w')
	file.write(outSpatialRef.ExportToWkt())
	file.close()
	return 0
#
#
def pnt2Grid(templatefile,datafile,Zfield,p='2.0'):
	'''GDALGridCreate not supported in Python(!) so have to do this.
	Create interpolated rasters from point data. use vrtMaker to get csv with correct headers.
	To use: fullname = pnt2Grid(templatefile,datafile,Zfield,p="2.0")'''
	# Get shape of raster to map to
	_, tmeta, tmetadata = rasterImport(templatefile)
	ds=gdal.Open(templatefile)
	prj=ds.GetProjection()
	print "\nTemplate file is in \n",prj
	ll_x = tmeta['corners']['ll'][0]
	lr_x = tmeta['corners']['lr'][0]
	ll_y = tmeta['corners']['ll'][1]
	ur_y = tmeta['corners']['ur'][1]
	rows = tmeta['dimension']['rows']
	cols = tmeta['dimension']['cols']
	# Create name of new file
	name,ext,path,namefull = namer(datafile)
	nametif = name+'.tif'
	# fullname = os.path.join(path,nametif)
	fullname = nametif
	vrtName = name + '.vrt'
	# datafilevrt = os.path.join(path,vrtName)
	datafilevrt = vrtName
	command = 'cd '+path+'; gdal_grid -zfield '+Zfield+' -a invdist:power='+p+':smoothing=1.0 -txe '+str(ll_x)+' '+str(lr_x)+' -tye '+str(ll_y)+' '+str(ur_y)+' -outsize '+str(cols)+' '+str(rows)+' -of GTiff -ot Float64 -l '+name+' '+datafilevrt+' '+fullname+' --config GDAL_NUM_THREADS ALL_CPUS; cd -'
	#command = 'gdalwarp -s_srs '+prj+' -t_srs EPSG:'+out+' -r near -dstnodata -9999 -of GTiff' datafile
	print 'Sending\n',command,'\nto command line\n'
	os.system(command)
	return fullname
#
#
def vrtMaker(fileName):
	'''Read csv coordinate file and create vrt header
	To use: Zfield, namevrt = vrtMaker(datafile)'''
	print '\n'*2,'*'*10
	print 'Read ',fileName
	InFile = open(fileName,'rb')
	Headers = InFile.next().strip().split(',')
	InFile.close()
	warn = 0
	for i in range(len(Headers)):
		if len(Headers[i]) < 2:
			print '"gdal_grid" cannot parse single letter column headers.\n'
			print 'If you intend to pass this column as x,y or z then change the column name.'
			warn = 1
		if ' ' in Headers[i] or '.' in Headers[i]:
			print '"gdal_grid" cannot parse column headers with spaces or dots.\n'
			print 'If you intend to pass this column as x,y or z then change the column name.\n'
			warn = 1
	print 'File Headers:\n'
	for i in Headers:
		print i
	print '\n'
	if warn == 1:
		print 'THERE ARE BAD COLUMN HEADERS IN YOUR INPUT FILE\n'
		ans = cont()
		if ans == 1:
			print '\nLeaving makevrt function\n'
			return 0
	Xcol = ''
	Ycol = ''
	Zcol = ''
	useAns = ['y','n']
	useFirst = ''
	while useFirst not in useAns:
		useFirst = raw_input('Use first three fields as Easting, Northing and Elevation? (y/n): ')
	if useFirst == 'n':
		while Xcol not in Headers:
			Xcol = raw_input('Enter column for "Easting": ')
		while Ycol not in Headers:
			Ycol = raw_input('Enter column for "Northing": ')
		while Zcol not in Headers and Zcol !='none':
			print Zcol
			Zcol = raw_input('Enter column for Z or "none": ')
	elif useFirst == 'y':
		Xcol = 'field_1'
		Ycol = 'field_2'
		Zcol = 'field_3'
	#
	epsgList = ['3006','3021','7030']
	nameList = ['SWEREF99TM','RT90 2,5gV','WGS84']
	for i in range(len(nameList)):
		print nameList[i],' = ',epsgList[i]
	epsg = ''
	while epsg not in epsgList:
		epsg = raw_input('Enter epsg code for source file coordinate system: ')
	#
	namevrt = makevrt(fileName,epsg,Xcol,Ycol,Zcol)
	#
	return Zcol, namevrt
#
#
def makevrt(fileName,epsg,Xcol,Ycol,Zcol='none'):
	layername,ext,path,namefull = namer(fileName)
	line1 = '<OGRVRTDataSource>\n'
	line2 = '<OGRVRTLayer name="'+layername+'">\n'
	line3 = '<SrcDataSource>'+layername+'.csv</SrcDataSource>\n'
	line4 = '<GeometryType>wkbPoint</GeometryType>\n'
	line4a = '<LayerSRS>EPSG:'+epsg+'</LayerSRS>\n'
	if Zcol != 'none': line5 = '<GeometryField encoding="PointFromColumns" x="'+Xcol+'" y="'+Ycol+'" z="'+Zcol+'"/>\n'
	elif Zcol == 'none': line5 = '<GeometryField encoding="PointFromColumns" x="'+Xcol+'" y="'+Ycol+'"/>\n'
	else: return 1
	line6 = '</OGRVRTLayer>\n'
	line7 = '</OGRVRTDataSource>\n'
	#
	vrtName = layername + '.vrt'
	outName = os.path.join(path,vrtName)
	with open(outName,'ab') as OutFile:
		OutFile.write(line1)
		OutFile.write(line2)
		OutFile.write(line3)
		OutFile.write(line4)
		OutFile.write(line4a)
		OutFile.write(line5)
		OutFile.write(line6)
		OutFile.write(line7)
	return outName
#
#
def descstat(inTab,inCol):
	"""Create dictionary of basic descriptive statistics from vectors dictionary"""
	descr = {}
	#descr['lin'] = np.linspace(np.nanmin(vectors[i][j]), np.nanmax(vectors[i][j]),100)
	descr['n'] = len(inTab[inCol])
	descr['mean'] = nanmean(inTab[inCol])
	descr['min'] = np.nanmin(inTab[inCol])
	descr['max'] = np.nanmax(inTab[inCol])
	descr['range'] = descr['max'] - descr['min']
	descr['meandif'] = inTab[inCol] - descr['mean']
	descr['var'] = ( np.nansum( descr['meandif']**2 ) ) / descr['n']
	descr['std'] = descr['var']**0.5
	return descr
#
#
def descrp(arr_ay):
	"""Create dictionary of basic descriptive statistics from vectors dictionary"""
	descr = {}
	descr['n'] = len(arr_ay)
	descr['mean'] = nanmean(arr_ay)
	descr['min'] = np.nanmin(arr_ay)
	descr['max'] = np.nanmax(arr_ay)
	descr['range'] = descr['max'] - descr['min']
	descr['meandif'] = arr_ay - descr['mean']
	descr['var'] = ( np.nansum( descr['meandif']**2 ) ) / descr['n']
	descr['std'] = descr['var']**0.5
	return descr
#
#
def matdot(m1,m2):
	szm1=np.size(m1)
	szm2=np.size(m2)
	spm1=np.shape(m1)
	spm2=np.shape(m2)
	m1r=np.reshape(m1.T,(1,szm1))
	m2r=np.reshape(m2,(1,szm2))
	rows=spm1[0]
	cols=spm2[1]
	if len(spm1)==1:
		veclen=spm1[0]
	else:
		veclen=spm1[1]
	outvec = []
	for i in range(spm1[0]):
		for j in range(spm2[1]):
			b=[]
			for k in range(len(m1[i])):
				a = m1[i][k] * m2[k][j]
				b.append(a)
			outvec.append(np.nansum(b))
	newmat = np.reshape(outvec,(rows,cols))
	return newmat
#
#
def nancov(m):
	rows = float(np.shape(m)[0])
	print rows
	rowssqr =np.ones([rows,rows])
	mdivn = rowssqr/rows
	#mavg = mdivn.dot(m)
	mavg = matdot(mdivn,m)
	print 'Averages: ',mavg[0]
	mdiff = m-mavg
	mdiffT = mdiff.T
	#mdevscore = mdiffT.dot(mdiff)
	mdevscore = matdot(mdiffT,mdiff)
	print 'Deviation Score:\n',mdevscore
	mcov = mdevscore/(rows-1)
	print 'Covariance matrix:\n',mcov
	return mcov, mdevscore, mavg[0]
#
#
def mustard(file,kernel):
	'''Run a kernel (window,matrix) over a single layer raster. Uses rasterImport and rasterExport.
	Kernel must be numpy array of odd number dimensions.
	To use: outdata = mustard(file,kernel) '''
	# Example high pass filter
	# kernel = np.array([[-1.0,-1.0,-1.0],[-1.0,9.0,-1.0],[-1.0,-1.0,-1.0]])
	# Import data file
	data, meta, metadata = rasterImport(file)
	# Set no data values to numpy nan
	data[data==meta['dats'][0]]=np.nan
	# Get size of indata array
	inrows =  meta['dimension']['rows']
	incols =  meta['dimension']['cols']
	# Get size of kernel
	krows = np.shape(kernel)[0]
	kcols = np.shape(kernel)[1]
	# Check kernel is smaller than data grid.
	if krows >= inrows or kcols >= incols: sys.exit('Bad kernel. Too large.')
	# Check kernel has a central pixel
	if krows % 2 == 0: sys.exit('Bad kernel. Even number rows.')
	if kcols % 2 == 0: sys.exit('Bad kernel. Even number columns.')
	# Get central pixel location in kernel
	kmidrow = int(krows)/2
	kmidcol = int(kcols)/2
	# Create relative extent of kernel
	rowminext = -1*((krows-1)/2)
	rowmaxext = (krows-1)/2
	rowrange = range(rowminext,rowmaxext+1,1)
	colminext = -1*((kcols-1)/2)
	colmaxext = (kcols-1)/2
	colrange = range(colminext,colmaxext+1,1)
	# Set initial starting location of kernel on grid
	dati = kmidrow
	datj = kmidcol
	# Get number of rows to run kernel over
	gridrows = range(inrows + rowminext*2)
	# Get number of columns to run kernel over
	gridcols = range(incols + colminext*2)
	# Create output array filled with nan
	outdata = np.empty((inrows,incols))
	outdata[:] = np.nan
	# Start loop
	for row in gridrows:
		datj = kmidcol
		rowvec = np.ones((1,krows))*dati + rowrange
		for col in gridcols:
			if np.isnan(data[dati,datj]):
				datj = datj + 1
				outdata[dati,datj] = np.nan
			else:
				colvec = np.ones((1,kcols))*datj + colrange
				extract = np.empty((krows,kcols))
				for i in range(krows):
					for j in range(kcols):
						extract[i,j] = data[int(rowvec[0,i]),int(colvec[0,j])]
				pixval = np.nansum(extract * kernel)
				outdata[dati,datj] = pixval
				datj = datj + 1
		dati = dati + 1
	return outdata,meta
#
#
def kernelChoice(ans = ''):
	'''Returns a kernel to use with mustard as well as the name of the kernel.
	To use: kernel, ans = kernelChoice(ans = '') '''
	kernelDict ={
	'Average':np.array([[0.111,0.111,0.111],[0.111,0.111,0.111],[0.111,0.111,0.111]]),
	'HighPass1':np.array([[-0.7,-1.0,-0.7],[-1.0,6.8,-1.0],[-0.7,-1.0,-0.7]]),
	'HighPass2':np.array([[-1.0,-1.0,-1.0],[-1.0,9.0,-1.0],[-1.0,-1.0,-1.0]]),
	'PrewittX':np.array([[-1.0,0.0,1.0],[-1.0,0.0,1.0],[-1.0,0.0,1.0]]),
	'PrewittY':np.array([[1.0,1.0,1.0],[0.0,0.0,0.0],[-1.0,-1.0,-1.0]]),
	'SobelX':np.array([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]]),
	'SobelY':np.array([[1.0,2.0,1.0],[0.0,0.0,0.0],[-1.0,-2.0,-1.0]])
	}
	KerList = kernelDict.keys()
	KerList.sort()
	while ans not in KerList:
		print KerList
		ans = raw_input('Enter name of kernel: ')
	kernel = kernelDict[ans]
	print ans,':\n', kernel
	return kernel, ans
#
#
def curve(file,kernel):
	'''Copy of mustard, altered to calculate curvature according to Equation 4 from Park, S.; McSweeney, K. & Lowery, B. Identification of the spatial distribution of soils using a process-based terrain characterization Geoderma, 2001, 103, 249-272
	as suggested by http://casoilresource.lawr.ucdavis.edu/drupal/node/937
	Calls: rasterImport
	To Use: outdata, meta = curve(file,kernel) '''
	# Import data file
	data, meta, metadata = rasterImport(file)
	# Set no data values to numpy nan
	data[data==meta['dats'][0]]=np.nan
	# Get size of indata array
	inrows =  meta['dimension']['rows']
	incols =  meta['dimension']['cols']
	cellsize =  meta['dimension']['xres']
	# Get size of kernel
	krows = np.shape(kernel)[0]
	kcols = np.shape(kernel)[1]
	# Check kernel is smaller than data grid.
	if krows >= inrows or kcols >= incols: sys.exit('Bad kernel. Too large.')
	# Check kernel has a central pixel
	if krows % 2 == 0: sys.exit('Bad kernel. Even number rows.')
	if kcols % 2 == 0: sys.exit('Bad kernel. Even number columns.')
	# Get central pixel location in kernel
	kmidrow = int(krows)/2
	kmidcol = int(kcols)/2
	# Create relative extent of kernel
	rowminext = -1*((krows-1)/2)
	rowmaxext = (krows-1)/2
	rowrange = range(rowminext,rowmaxext+1,1)
	colminext = -1*((kcols-1)/2)
	colmaxext = (kcols-1)/2
	colrange = range(colminext,colmaxext+1,1)
	# Set initial starting location of kernel on grid
	dati = kmidrow
	datj = kmidcol
	# Get number of rows to run kernel over
	gridrows = range(inrows + rowminext*2)
	# Get number of columns to run kernel over
	gridcols = range(incols + colminext*2)
	# Create output array filled with nan
	outdata = np.empty((inrows,incols))
	outdata[:] = np.nan
	# Start loop
	for row in gridrows:
		datj = kmidcol
		rowvec = np.ones((1,krows))*dati + rowrange
		for col in gridcols:
			if np.isnan(data[dati,datj]):
				datj = datj + 1
				outdata[dati,datj] = np.nan
			else:
				colvec = np.ones((1,kcols))*datj + colrange
				extract = np.empty((krows,kcols))
				for i in range(krows):
					for j in range(kcols):
						extract[i,j] = data[int(rowvec[0,i]),int(colvec[0,j])]
					D = np.zeros(np.shape(extract))
					for i in range(krows):
						for j in range(kcols):
			# To here same as mustard
							D[i,j] = (((kmidrow - i)**2+(kmidcol - j)**2)**0.5) * cellsize
					D[kmidrow,kmidcol] = np.nan
					E = np.ones(np.shape(extract))
					E = E*extract[kmidrow,kmidcol]
					EmA = E - extract
					Asum1 = EmA/D
					Asum2 = np.nansum(Asum1)
					pixval = Asum2 / (np.size(extract)-1)
				outdata[dati,datj] = pixval
				datj = datj + 1
		dati = dati + 1
	return outdata, meta
#
def curveKernel():
	'''Creates an rXc kernel of ones. Called by curveDem for use with curve.
	Calls:
	To Use: kernel = curveKernel() '''
	print 'Define size of window to calculate curve from.\nMust have odd number of rows and columns'
	Ashape_r = ''
	Ashape_c = ''
	while not Ashape_r:
		ans = raw_input('Enter number of rows: ')
		try:
			Ashape_r = int(ans)
		except:
			Ashape_r = ''
	while not Ashape_c:
		ans = raw_input('Enter number of columns: ')
		try:
			Ashape_c = int(ans)
		except:
			Ashape_c = ''
	kernel = np.ones((Ashape_r,Ashape_c))
	return kernel
#
def curveDem(*infile):
	'''Create a curvature surface from a DEM.
	Calls: curveKernel, curve, namer, rasterExport
	To use: curveDem() '''
	if infile:
		file = infile[0]
	else:
		file = ''
	while os.path.isfile(file) is False:
		file = raw_input('Enter DEM file path and name: ')
	name,ext,path,namefull = namer(file)
	filnm = os.path.join(path,(name + '_curve.tif'))
	kernel = curveKernel()
	outdata, meta = curve(file,kernel)
	rasterExport(outdata,meta,filnm)
	return 0
#
def pt2fmt(pt):
	fmttypes = {
		GDT_Byte: 'B',
		GDT_Int16: 'h',
		GDT_UInt16: 'H',
		GDT_Int32: 'i',
		GDT_UInt32: 'I',
		GDT_Float32: 'f',
		GDT_Float64: 'f'
		}
	return fmttypes.get(pt, 'x')
#
def getRasterFile(file):
	'''Get georeferenced raster and open for use
	To use: raster, transf, bandcount = getRasterFile(file)'''
	# Called by:
	# register all of the GDAL drivers
	gdal.AllRegister()
	# Open file
	raster = gdal.Open(file, GA_ReadOnly)
	if raster is None:
		print 'Could not open ',file,'\n'
		sys.exit(1)
	# Get coordinate system parameters
	projec = raster.GetProjection()
	srs=osr.SpatialReference(wkt=projec)
	transf = raster.GetGeoTransform()
	bandcount = raster.RasterCount
	#
	return raster, transf, bandcount
#
def GetRasterVals(x,y, raster, transf, bandcount):
	'''Create vector of pixel values at location in raster
	To use: vals = GetRasterVals(x,y, raster, transf, bandcount)'''
	# Called by:
	# get image size
	#print x, y
	success, transfInv = gdal.InvGeoTransform(transf)
	if not success:
		print "Failed InvGeoTransform()"
		sys.exit(1)
	rows = raster.RasterYSize
	cols = raster.RasterXSize
#	 xdifpix = math.floor((x - transf[0])/transf[1])
#	 ydifpix = math.floor((y - transf[3])/transf[5])
#	 xpix = xdifpix -1
#	 ypix = ydifpix -1
	xpix, ypix = gdal.ApplyGeoTransform(transfInv, x, y)
	# Read the file band to a matrix called band_1
	vals = []
	for i in range(1,bandcount+1):
		band = raster.GetRasterBand(i)
		bandtype = gdal.GetDataTypeName(band.DataType)
		if band is None:
			continue
		# Access data in raster band as array
		#data = band.ReadAsArray(0,0,cols,rows)
		#vals[i] = (data[ypix,xpix])
		structval = band.ReadRaster(int(xpix), int(ypix), 1,1, buf_type = band.DataType )
		fmt = pt2fmt(band.DataType)
		intval = struct.unpack(fmt , structval)
		vals[i] = intval[0]
	return vals
#
def rasterDiff(Afile, Bfile, outfolder):
	'''Subtract one raster from another
	To use: filnm = rasSubras(Afile, Bfile, outfolder) '''
	A,Aext,Apath,Anamefull = namer(Afile)
	B,Bext,Bpath,Bnamefull = namer(Bfile)
	# Get first raster file
	Adata,AXg,AYg,AXg1,AYg1,Arx,Ary,Ademmeta = DemImport(Afile)
	Adata[Adata==Ademmeta[6]]=np.nan
	Avec = np.reshape(Adata,(Arx*Ary))
	AvecNan = Avec[np.logical_not(np.isnan(Avec))]
	# Get second
	Bdata,BXg,BYg,BXg1,BYg1,Brx,Bry,Bdemmeta = DemImport(Bfile)
	print '\n'*2
	print '*'*20
	print A,' data type: ', type(Adata)
	print B,' data type: ',type(Bdata)
	print 'Rows (A,B): ',Arx,Brx,' Columns (A,B): ',Ary,Bry
	print 'xres (A,B): ',Ademmeta[2][1],Bdemmeta[2][1],' yres (A,B): ',Ademmeta[2][5],Bdemmeta[2][5]
	print 'No data (A): ',Ademmeta[6],' No data (B): ',Bdemmeta[6]
	# Check matching resolution
	if Arx != Brx or Ary != Bry:
		print B,' resolution mismatch with ', Afile
	elif Ademmeta[4] != Bdemmeta[4] or Ademmeta[5] != Bdemmeta[5]:
		print 'Size mismatch between ',B,' and ',Afile
		return 1
	# Add current file to sum
	Bdata[Bdata==Bdemmeta[6]]=np.nan
	Bvec = np.reshape(Bdata,(Brx*Bry))
	BvecNan = Bvec[np.logical_not(np.isnan(Bvec))]
	#
	AdataMasked = np.ma.masked_array(np.nan_to_num(Adata), mask=np.isnan(Adata) & np.isnan(Bdata))
	BdataMasked  = np.ma.masked_array(np.nan_to_num(Bdata), mask=AdataMasked.mask)
	#
	ABcovmat = np.cov(AvecNan,BvecNan)
	ABcov = ABcovmat[0,1]
	Amean = np.nanmean(Adata)
	Astd = ABcovmat[0,0]**0.5
	print "Mean of A: %4.4f, %4.4f" % (Amean, Astd)
	Bmean = np.nanmean(Bdata)
	Bstd = ABcovmat[1,1]**0.5
	print "Mean of B: %4.4f, %4.4f" % (Bmean, Bstd)
	print "Covariance: %4.4f" % (ABcov)
	#
	outdata = (BdataMasked - AdataMasked).filled(np.nan)
	outmean = np.nanmean(outdata)
	outstd = np.nanstd(outdata)
	print "Mean of outdata: %4.4f, %4.4f" % (outmean, outstd)
	propagated = (ABcovmat[0,0] + ABcovmat[1,1] - 2*ABcovmat[0,1])**0.5
	print "StdDev of propagated: %4.4f" % (propagated)
	outdata = np.ma.masked_array(np.nan_to_num(outdata), mask=np.isnan(outdata))
	outname = B+'Sub'+A
	filnm = datawrite(outdata,Adata,Ademmeta,outname,outfolder)
	#
	return filnm
#
def pythag(Xdif,Ydif):
	'''Calculate hypotenuse of right angled triangle'''
	# Called by: pythag2p
	hypo = (Xdif**2 + Ydif**2)**0.5
	return hypo
#
def pythag2p(Xa,Ya,Xb,Yb):
	'''Calculate distance in plane between two points'''
	# Called by: inbindning, inskarning, freestation
	Xdif = Xb - Xa
	Ydif = Yb - Ya
	hypo = pythag(Xdif, Ydif)
	return hypo
#
def extender(ax,ay,bx,by,length):
	'''Extend a line between two by a set distance'''
	lenAB = pythag2p(ax,ay,bx,by)
	cx = bx + (bx - ax) / lenAB * length
	cy = by + (by - ay) / lenAB * length
	return cx,cy