#!/usr/bin/env python2.5
#
#  MakeDirectory.py
#  ss
#  This routine constructs the directory structure necessary for a fronts/gradient
#  workflow. For example, "SST" and "Median" are folders required when
#  running the FORTRAN code and these are not created by the FORTRAN code.
#
#  To run type:
#
#  python MakeDirectory.py --directory BASEDIRECTORY --start_year FIRSTYEAR --end_year LASTYEAR
#
#  Where BASEDIRECTORY is the directory in which the file structure is to be
#    created; e.g., /Volumes/RAID5/Pathfinder1km_L3/ Note no quotes around the
#    directory structure,
#   FIRSTYEAR is the first year in which there is data; e.g., 2001
#   LASTYEAR is, yup, you guessed it, the last year in which there is data.
#
#  MakeDirectory is issued from the directory in which it is found.
#
#  Created by Christian Buckingham on 7/23/09.
#  Copyright (c) 2009 University of Rhode Island. All rights reserved.
#
#  3/16/12 -DH and then PCC - modified to accept input, BASEDIR, FIRSTYEAR and LASTYEAR.
#
# 3/21/13 - PCC - Added directories for SobelSums and SobelHistograms
#
#  NOTES:
#  This should work on all unix machines, but not sure about PC; change "filesep" variable
#  below.

import os
import string
import re
import math

import optparse

parser = optparse.OptionParser(usage="%MakeDirectory [options] arg1 arg2", version="%MakeDirectory 1.0")

parser.add_option("-d", "--directory", dest="base_directory", type="string",
                  help="use as base directory")
parser.add_option("-s", "--start_year", dest="start_year", type="int",
                  help="starting year")
parser.add_option("-e", "--end_year", dest="end_year", type="int",
                  help="ending year")
parser.add_option("-v", "--verbose", dest="verbose", action="store_true", 
                  help="print status messages to stdout")

(options, args) = parser.parse_args()

if options.base_directory == None:
    parser.error("required --directory=<BASEDIR> option missing")
if options.start_year == None:
    parser.error("required --start_year=<YEAR> option missing")
if options.end_year == None:
    parser.error("required --end_year=<YEAR> option missing")

basedir=options.base_directory
filesep = '/' # unix, use different one '\' for PC

# Define folders that do not have years.

sfolders = ['Tmpdir','SupportingFiles','SupportingFiles/ArchiveInventories','SupportingFiles/LogFiles','SupportingFiles/SpawnLogs','SobelSums','SobelSums/DailyClimatologicalSobelSums','SobelSums/MonthlyClimatologicalSobelSums','SobelSums/MonthlySobelSums','SobelHistograms','SobelHistograms/DailyClimatologicalHistograms','SobelHistograms/MonthlyClimatologicalHistograms','SobelHistograms/MonthlyHistograms'] # list of folders to create
verbose = '+v' # '-v' turns text to screen on, '+v' off
    
yr_start = options.start_year
yr_stop = options.end_year

print 'BASEDIR: ' + basedir
print 'YR_START: ', yr_start
print 'YR_END: ', yr_stop

# Define folders that require years/months

folders = ['Fronts_1st_Pass','Fronts_2nd_Pass','Median','Merged','GeoSobel','Dummy','Sobel','Thinned','ZenithAngle','GeoLoc','SobelSums/NighttimeSobelSums','SobelHistograms/NighttimeHistograms'] # list of folders to create
verbose = '+v' # '-v' turns text to screen on, '+v' off

#print 'BASEDIR: ' + basedir

# Setup stuff.

def createDir(dirname,verbose):
    """Create a directory / First checks to see if dir exists ..."""
    # Import os module.
    import os
    
    status = 0 # directory not created
    
    #filesep = '/' # unix
    #filesep = "\" # pc
    #
    #if os.path.isdir("." + filesep + dirname + filesep):
    #    status = 2 # already exists
    #else:
    #    os.mkdir("." + filesep + dirname + filesep)
    
    if os.path.isdir(dirname):
        status = 2 # already exists
    else:
        os.mkdir(dirname)
        status = 1 # directory created
    
    if verbose == '-v':
        if status == 0:
            print 'Failure: directory not created ...' + dirname
        elif status == 2:
            print 'Directory already exists, skipping ... ' + dirname
        elif status == 1:
            print 'Creating directory ... ' + dirname
    
    return status

# Start by creating folders that do not require years and months

nfs = len(sfolders) # number of folders
print nfs
fds = range(nfs) # define indices to loop over
for ifd in fds: # loop over folders and create directories
    fd_tmp = sfolders[ifd] # define current folder
    dirname = basedir + filesep + fd_tmp + filesep # directory to be made
    status = createDir(dirname,verbose) # create directory

# Next create folders that require years.

nf = len(folders) # number of folders
print nf
fd = range(nf) # define indices to loop over
yr = range(yr_stop - yr_start + 1) # create a list of years, will add yr_start below
mo = range(12) # create a list of months, will add "1" below
for ifd in fd: # loop over folders and create directories
    fd_tmp = folders[ifd] # define current folder
    dirname = basedir + filesep + fd_tmp + filesep # directory to be made
    status = createDir(dirname,verbose) # create directory
        
    for iyr in yr: # loop over years and create directories
        yr_tmp = yr[iyr] + yr_start # add starting year to each element
        dirname = basedir + filesep + fd_tmp + filesep + \
                '%04d' % yr_tmp + filesep # directory to be made
        status = createDir(dirname,verbose) # create directory
        
        for imo in mo: # loop over months and create directories
            
            fd_tmp = folders[ifd] # define current folder
            yr_tmp = yr[iyr] + yr_start # add starting year to each element
            mo_tmp = mo[imo] + 1 # add 1 to each element
            #print mo[imo]
            dirname = basedir + filesep + fd_tmp + filesep + \
                '%04d' % yr_tmp + filesep + '%02d' % mo_tmp + filesep # directory to be made
            #print 'Creating directory ... ' + dirname
            status = createDir(dirname,verbose) # create directory, print text to screen

