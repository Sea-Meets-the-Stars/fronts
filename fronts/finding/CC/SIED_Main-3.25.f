c***********************************************************************
      program SIED_Main
c***********************************************************************
c     
c     This program will apply the single image edge detection algorithm
c     to a sequemce of images. If r_n, the run number, is 2, it will 
c     look for Cnt2 files as well as MT files. If the latter does not
c     exist for a particular input time, it will skip to the next time.
c     If the former exists it will also skip to the next time.
c     
c     The program has written to process data from the MSG or the
c     GOES archive at the Centre de Meteorologie Spatiale. To run
c     it on other archives will require some changes. 
c     
c     IMPORTANT
c     
c     LenX and LenY must be set to the image size in the archive. 
c     These variables are defined in ParameterStatments.f Chang them
c     there and they will carry over to this program and all of its
c     subroutines.
c     
c     For MSG, LenX is 3712 and LenY is 3712
c     For GOES (rewritten by AppQual) LenX is 1952 and LenY is 2400
c     For GOES (original CMS) LenX is 1980 and LenY is 2431
c     
c     There are also two parameters called xLenMeteo and xLenGoes.
c     These are used to make sure that you are processing the correct
c     archive. They are currently set to 3712 and 1980. If you are
c     only using one archive of data, search for these parameters
c     and bypass the part of the program that uses them.
c     
c     Written by Peter Cornillon, University of Rhode Island,
c     pcornillon@me.com 19 May 2009
c     
c     PCC 4/28/09 - removed variables not used in Main program. Changed
c     year, month, day, hour, minute, numclear and secondssince1970 
c     variables read from the inventory to Invxxx. Also changed prog
c     to SobelPeaksSied
c     
c     4/4/11 PCC - SIED version number ==> 1.01
c     Changed all logical*1 to logical in main program but only some
c     . in SIEDSubroutines.f
c     
c     5/11/11 PCC - SIED  version number ==> 1.1
c     Changed the portion of the program that checks for the existence
c     . of the input files. It no longer allows for either compressed
c     . or uncompressed; input files must be netCDF4 chunked.
c     
c     6/1/11 - PCC - Added Msg50 in calls to PrintArray2 and ElapsedTime
c     
c     6/6/11 - PCC - Changed length of SatName from 8 to 20 characters.
c     
c     6/7/11 - PCC - Major changes to simplify and correct handling of
c     strings. Replaced all length() with len(trim(). Removed all 
c     EndOfStrings. Added BlankFill before any definition of a string
c     variable to force the end of the string to be all blanks.
c     
c     7/18/11 - PCC - Added check for number of pixel elements, nep, 
c     for variable ppv. Abort if number exceeds MaxEdge.
c     
c     7/18/11 - PCC - Changed MaxEdge in ParameterStatements.f from 
c     400,000 to 1,000,000.
c     
c     Changes from 1.11 to 1.12
c     
c     1/2/12 - PCC - Changed all occurrences of 150 to 319 when 
c     referring to a string length - to 319 because 150 is too short
c     and because 319 is pretty unique.
c     
c     Version 1.12 to 1.13
c     
c     1/11/12 - PCC - Changed string lengths from 150 to 319 for file
c     names. Also added an explicit definition for uuid_gen.
c     
c     Changes from 1.13 to 1.14
c     
c     1/24/12 - PCC - Removed the check on which program is reading
c     the data so that the same RunControlFile can be used for a 
c     number of the programs.
c     
c     Changes from 1.14 to 1.20 
c     
c     2/26/12 - PCC - Modified to deal with arrays whose dimensions are
c     not multiples of 32. Pads the end of any dimension that is less
c     than a multiple of 32 to make it one.
c     
c     4/18/12 - PCC - removed SatID; it is no longer used. No impact.
c     
c     Changes from 1.2 to 2.00
c     
c     4/18/12 - PCC - CheckDimensions badly applied and the test on it
c     was wrong. Not sure how it worked when processing msg, but it
c     seems to have.
c     
c     5/2/12 - PCC - added pr_n = '1' before for loop over files to 
c     correct for a problem in searching for existing thinned files.
c     Also removed an assignment to pr_n later that was superfluous.
c     
c     5/3/12 - PCC - Corrected error in writing out 0,1 front array to
c     median SST file. Replaced FrontPicOut with frntpic and removed
c     all references to FrontPicOut.
c     Spiffed up the printout for when it skips a file.
c     Added a lines to SSTFileName and MedianFileName to blanks before
c     loading.
c     Replaced call to FileStatus with a simple inquire statement for
c     Cnt files.
c     Changed allocate x(LenX,LenY) to x(1:LenX,LenY) for all variables
c     x that were not explicitly allocated as 1 to Len.
c     Added code to test m, n for out of range in follow_main.
c     
c     5/4/12 - PCC - Added some debug statments and modified some other
c     to make the output easier to read.
c     Added stop if existence of thinned_1 changes from one file to the
c     next. If it starts as the 2nd pass it must continue that way. If
c     it starts as the first pass it must remain that way. Best to
c     finish all thinning and then do a cleanup run before 2nd pass of
c     SIED.
c     
c     Changes 2.00 ==> 2.01
c     
c     5/21/12 - PCC - Checked for existence of .cnt(1/2), but not of 
c     .cnt(1/2).gz. Fixed this. 
c     Cleaned up numbering of debug printout statements.
c     
c     Changes 2.01 ==> 2.02
c     
c     5/23/12 - PCC - removed ncID from passed parameter to ReadLatLon
c     in follow_main.
c     
c     Changes 2.02 ==> 2.03
c     
c     6/2/12 - PCC - Error in the calculation of the larger array. This
c     resulted in the thinned not being handle correctly. For a 
c     multiple of 32, the code developed dimensions that were 32
c     pixels larger than the input dimension.
c     Added code to write LenXA and LenYA to the output log.
c     Added some comments to SIEDSubroutines.
c     
c     Changes 2.03 ==> 2.04
c     
c     10/13/12 - PCC - Updated to CommonSubroutines-2.11.
c     
c     Changes 2.04 ==> 2.05
c     
c     11/24/12 - PCC - Updated to CommonSubroutines-2.12.
c     
c     Changes 2.05 ==> 2.06
c     
c     11/25/12 - PCC - The previous version was not saving the lat and
c     lon arrays between calls to sing_edge. This was OK when new lat
c     lon arrays were read for each image but failed when the same 
c     geometry applied to all images. Added code to save lat and lon
c     arrays between calls to sing_edge. This required modifyin the
c     main program and SIEDSubroutines.
c     I also moved the SIEDSubroutines into the main program so that 
c     if/when I change one or the other in the future, they are all
c     packaged together.
c     
c     Changes 2.06 ==> 2.07
c     
c     12/7/12 - PCC - The logic that tests for the existence of a 
c     thinned file was a bit off. A check was made to see if a 
c     thinned file exists for the file being processed and, if so, was 
c     there a thinned file for the previous file. It ignored this check
c     if this is the first file. being processed. The failure occurred 
c     when the first file in the archive had less than the threshhold 
c     number of good points, generally set at 100. I modified the code
c     to check for the first good file, not the first file in the test.
c     This is not a problem for the first pass of SIED on an archive.
c     
c     Changes 2.07 ==> 3.00
c     
c     12/20/12 - PCC - Changed the output from a flat ASCII file to a
c      netCDF file. This entaied the addition of the subroutine 
c     CreatNetCDFDimFile and modification to the portion that writes 
c      out the frontal data points. It also required reading in the 
c      Sobel gradient fields.
c     Also modified many of the debug print statements to include
c      a location xxx#nnn.
c     Moved the check on k in Follow_Main outside of the loops (23206 
c      and 23216) and changed the k to ccl. Not sure why it was 
c      checking on k in the loops rather than ccl before the loops.
c     Got rid of all references to lat, lons, dumlats and dumlons, no
c      longer used.
c     Documented variable names in Follow_Main.
c     Removed Lat_Array, LonArray from the main program, and sing_edge
c      and allocated these arrays in Follow_Main.
c     Removed code that checked for compressed versions of the thinned
c      file. Also changed ReadMergedThinned to close the file after 
c      reading --> CommonSubroutines 2.13
c     Changed test for a thinned file to a simple test rather than a 
c      call to FileStatus. FileStatus was used to check for files when
c      not using netCDF4 so files were compressed. No longer the case.
c     Changed ncID to ncMedID everywhere to make it more obvious.
c     Added iFirst, iLast, IMG to the line that prints out the file 
c      being processed. This is so that we know exactly where in the
c      processing loop the job is. Also added a date time variable
c      to print out at the end of the job.
c     Changed from 2.13 to 2.14 CommonSubroutines - new duration 
c      routine.
c
c     Changes 3.00 ==> 3.10
c
c     1/19/13 - PCC - Changed all nf_ to nf90_. For some reason, the
c       code with nf_ compiles on satdat3, but not on satdat4. Odd. 
c      Changed the fillvalue in CreateNetCDFDimFile from sstFillValue to
c       FillValueInt2.
c      Added SSTFillValue to ReadMedianFile and modified code to check
c       check for SSTFillValue when loading MedSSTA from MedSST. Through
c       version 3.00, it was assumed that the fill value for SST was 0.
c       This ain't so hot, hence the change. Remember however that the
c       histogramming routine in SIED assumes that the SST values range
c       from 0 to Range, so, when loading SSTA from SST, it replaces all
c       MedSST values of SSTFillValue with 0 in MedSSTA.
c      Changed CommonSubtroutines from 2.14 to 2.15. Added read to 
c       SSTFillValue to ReadMedianFile.
c      Commented out the parameter statement assigning SSTFillValue to
c       0 in parameterstatements.
c
c     1/20/13 - PCC - Changed from 2.15 to 2.16 CommonSubroutines - 
c       Added VariableName (also to parameterstatements and changed 
c       SSTFillValue to SSTFillValueIn. 
c      Changed SSTFillValue to SSTFillValueIn.
c
c     1/23/13 - PCC - moved definition of lat,lon arrays to the main
c       program.  Need to allocate these variables in the event that 
c       there is only one lat, lon file for this archive, otherwise 
c       it would be necessary to read in the array for every file. 
c
c     Changes 3.10 ==> 3.11
c
c     1/24/13 - PCC - Tweaked the wording for the  maximum_gradient_
c       location to include the maximum of the averaged gradients. No
c       impact on content.
c
c     1/25/13 - PCC - Changed the test of filpic in Follow_Main from
c       .eq. 0 to .eq. FillValueInt2 and assigned gradv to FillValueInt2
c       if it fails this test. 
c      Changed the assignment of MedSSTA from 0 to FillValueInt2 in the
c       calling program. This avoids the problem of 0 being a legitimate
c       value in a future version of the program. For now, it should 
c       have no impact since the input SST should not be less than or
c       equal to 0.
c      Added code to output the center-difference gradients. These are
c       used by Pmerge at present. Pmerge has access to both the Sobel
c       gradients and these, but rewriting it to use the Sobel gradients
c       would be a significant job and it would require reading in the
c       Sobel gradients.
c      Fixed some of the metadata definitions that had Kelvin hard-wired
c       into them. Now they use ParameterUnits.
c      Changed the range on the variables in the loop setting gradv to
c       fill values at the beginning of follow_main from i-2 to i+2 
c       ==> i-1 to i+1. This is to reduce the number of gradient values
c       set to fill value
c      CommonSubroutines 2.17 ==> 2.18 Changed formatting in PrintArray2
c
c     Changes 3.11 ==> 3.12
c
c     1/27/13 - PCC - Added a common for fill values, scale factors and 
c       offsets for gradient fields, lat/lon fields, sst fields and
c       zenithangle fields in ParameterStatements, changed calls to 
c       reading subroutines in CommonSubtroutines to accommodate the new
c       common statement and modified code in this programs to do so.
c      Added code to write the start and end times and the number of
c       files processed at the end of the main program to the log file.
c     1/28/13 -PCC - SatName moved to ArchiveDefinition common in 
c       ParameterStatements. This required removing SatName from 
c       subroutine calls and definitions for ReadInputs, CheckArchive 
c       and ReplaceBaseDir.
c     1/29/13 - PCC - Changed along-scan and cross-scan _gradient 
c     .  offsets from SSTOffset to 0.
c     . Closed MergedThinned after the call to read it. I had to leave
c     .  it open for use by the program thinned.
c     . CommonSubroutines 2.19 ==> 2.20.
c     . Changed the size of a window from 32 to WinSize, a variable
c     .  defined and set in ReadInputs in CommonStatements. 
c     . Added statements to calcluate the number of SIED windows in the 
c     .  1st and 2nd dimensions.
c
c     Version 3.12 ==> 3.13
c
c     3/16/13 - PCC - Changed Common Subroutines from 2.22 to 2.23
c
c     Version 3.13 ==> 3.14
c
c     3/16/13 - PCC - Added a debug statement - not processing correctly
c       for the 2nd pass.
c      Changed the way the program handles a missing thinned file. 
c       First, fixed the printout for the case when no thinned file 
c       exists for the first file processed in the run, but one is found
c       later. The printout was sort-of backwards for this case.
c       Second, added a 15 second wait if a thinned file existed for an
c       earlier median file but not for the one being processed and then
c       tested for existence again based on the idea that the link to 
c       the computer with the thinned file on it may have been 
c       momentarily down. If the file is still missing it writes out a
c       message and the run is aborted. 
c
c     Version 3.14 ==> 3.15
c
c     3/29/13 - PCC -Checked to see if zenith angle file exists. If it
c       doesn't, then don't open it, rather define a zenith angle array
c       of missing values. This change was made for Pathfinder and MODIS
c       although I did not make use of this for the AppQual v2.00 run so
c       the zenith angles output by SIED are bogus. 
c
c     Version 3.15 ==> 3.16
c
c     4/3/13 - PCC - CommonSubroutines 2.23 ==> 2.27
c      Added code to read in the zenith angle if present and to write it
c       out at each front location.
c
c     Version 3.16 ==> 3.17
c
c     4/3/13 - PCC - CommonSubroutines 2.27 ==> 2.29 - Fixed start time.
c
c     Version 3.17 ==> 3.18
c
c     7/3/13 - PCC - Added comment statements.
c     7/9/13 - PCC & JS - Added code to remove bilinear surface from the
c      input field when calculating the historgram and populations c
c      stuff.
c     Also added support for MinLength MinStep (minimum length and
c     separation of pops for fronts) in SIED for writing out in the
c     metadata and as common variables
c
c     Version 3.18 ==> 3.19
c
c     9/3/13 - PCC - CommonSubroutines 2.31 ==> 2.32.
c     
c     Version 3.19 ==> 3.20
c
c     9/3/13 - JPS - CommonSubroutines 2.32 ==> 2.33.
c     
c     Version 3.20 ==> 3.21
c
c     1/26/14 - PCC - CommonSubroutines 2.33 ==> 2.34.
c
c     Version 3.21 ==> 3.22
c
c     1/27/14 - PCC - Modified some of the printout for the file 
c       description in the output file.
c      Changed the write for MinClear_str from i2 to i3.
c
c     Version 3.22 ==> 3.23
c
c     1/29/14 - PCC - Added code to write out to the output netCDF file
c       if a plane was removed from each histogram window.
c      Put Summary, SummaryPart1, 2 and 3 in a common statement in 
c       ParameterStatements
c      Added a test for PlaneDebug to accommodate John's debug file.
c      CommonSubroutines 2.35 ==> 2.36.
c
c     Version 3.23 ==> 3.24
c     2/3/14 - JPS - Added code to ensure picttmp in histo feel between
c        0 and RANGE. Also added code to ensure proper value is passed out
c        through INDEX and MAX
c
c     7/21/15 - JPS - The way Peter was writing out the array pict in
c        the debug commands was causing run-time errors. I commented
c        them out for now. I have no idea how to fix it
c
c     Version 3.24 ==> 3.25
c
c     9/8/15 - JPS- Added a flag SIED_Full to be read in through the run
c          config file, which will tell SIED not to save the cross-front
c          SST information.
c.......................................................................

      use netcdf

      implicit none

c******Functions

c     SecondsSince - returns the number of hours since the reference.
c     . Arguments are year, month, day, hour, minute and the 
c     . values for these.
      real*8 SecondsSince

c     GenerateFileName - a function that will generate a filename for
c     . one of the output file types given the sudirectory name and
c     . the ending of the filename. These change for each program
c     . output.
      character*319 GenerateFileName

c     CheckDimensions - checks the dimensions of the variables in the
c     input netCDF file with LenX and LenY.

      logical CheckDimensions

c     Num2xxx - functions used to convert numbers to strings.
      character*20 Num2StrInt2, Num2StrInt4, Num2StrFloat, Num2StrG

c******Parameter statements

      include 'ParameterStatements.f'

c******LatLon common

c     LatLonFileName - the name of the file containing the latitud and
c     . longitude arrays.
      character*319 LatLonFileName
c     LatLonFileNameSave - the saved LatLonFileName used to see if the
c     . the filename is new for this SST field.
      character*319 LatLonFileNameSave

      common /LatLon/ LatLonFileName, LatLonFileNameSave

c******General variables

c     ArchiveID - is 1 for meteosat and 2 for goes
      integer ArchiveID

c$$$  c     CntCompressed - the compression flag for the Cnt1 or 2 file. 
c$$$  logical CntCompressed
c     CPUTime - CPU time used by this program to date
      real CPUTime

c     DimFileName - name of file with lat and lon located fronts. 
      character*319 DimFileName
c     Dum0 - a dummy paramete used in the call to check archive.
      integer Dum0

c     FileNameToUse - is the filename to be use for the iput file. It's
c     . either the filename in the inventory or a temporary filename.
      character*319 FileNameToUse
c     FirstGoodFile - a flag, 1 if this is the first good file, 0 
c     otherwise.
      integer FirstGoodFile
c     FrontsExists - Front file exists if .true. 
      logical FrontsExists
c     FrontsFileName - the output filename for fronts - uncompressed
      character*319 FrontsFileName
c     FrontsID - ID for the fronts field.
      integer FrontsID

c     i - do loop parameter
      integer i
c     iImg - the main loop parameter - the index for the file on which
c     . we are working.
      integer iImg
c     InvDay - day in month corresponding to an image in the archive.
      integer, allocatable :: InvDay(:)
c     InvFileName - the name of the input file read from the inventory
      character*319, allocatable :: InvFileName(:)
c     InvHour - hour in day corresponding to an image in the archive.
      integer, allocatable :: InvHour(:)
c     InvSecond - Second in minute corresponding to an image in the 
c     - archive.
      integer, allocatable :: InvSecond(:)
c     InvSecondsSince1970 - the number of second since 00:00 of 1  
c     . January 1970 corresponding to the time of a given image. Read 
c     . from the inventory
      real*8, allocatable :: InvSecondsSince1970(:)
c     InvMinute - minute of the hour corresponding to an image in the 
c     . archive.
      integer, allocatable :: InvMinute(:)
c     InvMonth - month of the year corresponding to an image in the 
c     . archive.
      integer, allocatable :: InvMonth(:)
c     InvNumClear - Number of clear pixels in the image read from the 
c     . inventory. 
      integer, allocatable :: InvNumClear(:)
c     InvYear - Year corresponding to an image in the archive.
      integer, allocatable :: InvYear(:)

c     j - do loop parameter
      integer j

c     LastChar1 - the location of the last character of a string. 
      integer LastChar1
c     LastChar2 - the last character of another string. 
      integer LastChar2
c     LastCPUTime - CPUTime the last time ElapsedTime was called.
      real LastCPUTime
c     Lat_Array - latitude values for image. Read in once if it does
c     not change or for each file if it does change.
      real*4, allocatable :: Lat_Array(:,:)
c     Lon_Array - longitude values for image. Read in once if it does
c     not change or for each file if it does change.
      real*4, allocatable :: Lon_Array(:,:)

c     MedianCompressed - the compression flag for the Median file. 
      logical MedianCompressed
c     MedianExists - file status for Median, .true. if the file exists
      logical MedianExists
c     MedianFileName - the name of the output file for the 3x3 median
c     . filter.
      character*319 MedianFileName
c     MedianTempFileName - filename used for the decompressed 
c     . temporary input file.
      character*319 MedianTempFileName 
c     MedFileOpen - .true. if the input Median file has been opened.
      logical MedFileOpen
c     MedID - ID for the median sst field.
      integer MedID
c     MedSST - 3x3 median filtered value of sst - input field
      integer*2, allocatable :: MedSST(:,:)
c     MedSSTA - 3x3 median filtered value of sst - augmented to have
c     dimensions that are a multiple of WinSize
      integer*2, allocatable :: MedSSTA(:,:)
c     MergedCompressed - the compression flag for the merged file.
      logical MergedCompressed
c     MergedExists - .true. if the merged file exists.
      logical MergedExists
c     MergedFileName - name of file with images of merged contours.
      character*319, allocatable :: MergedFileName(:)
c     MergeThinFileOpen - .true. if the merged file is open
      logical MergeThinFileOpen
c     Message - message to be written out for call to ElapsedTime.
      character*30 Message

c     ncMedID - NetCDF ID for the Median file.
      integer ncMedID
c     ncSSTID - NetCDFID for input SST file
      integer ncsstID
c     NumberOfFilesProcessed - the number of fields processed in this 
c     . run.
      integer NumberOfFilesProcessed

c     RealDum - a real*4 dummy variable, needed in ifix.
      real RealDum
c     SecondsSince1970InID - ID for NetCDF variable.
      integer SecondsSince1970InID
c     SSTCompressed - the compression flag for the input file. 
      logical SSTCompressed
c     SSTExists - returned from FileStatus, .true. if the file exists
      logical SSTExists
c     SSTTempFileName - is the temporary filename used for the
c     . decompressed version of the input SST file.
      character*319 SSTTempFileName
c     SSTFileName - the temporary name of the input file - uncompressed
      character*319 SSTFileName

c     t - damn, can't recall what I used this for, but I need it.
      real t
c     TempFileName - dummy filename, used when not expecting a 
c     . temporary filename back.
      character*319 TempFileName
c     ThinnedCompressed - the compression flag for the thinned file.
      logical ThinnedCompressed
c     ThinnedExists - .true. if the thinned file exists.
      logical ThinnedExists
c     ThinnedFileName - name of file with thinned contours. 
      character*319 ThinnedFileName

c     Function to generate UUID.
      character*36 UUID_Gen

c---------------------------------start here ---------------------------

      ProgName = BlankFill(:len(ProgName))
      ProgName = 'SIED_Main'

      if(Debug .eq. 1) print *, 'SIED #000'

c     Set FirstGoodFile to 1. Will change it to 0 after test on run 
c     number.
      
      FirstGoodFile = 1

c     Set LatLonFileNameSave to all blanks. The program will set it 
c     equal to the lat,lon file name for the first SST file that it
c     reads and then check to see if has changed after that. If it has,
c     it will read in the new lat,lon fields.

      LatLonFileNameSave = BlankFill(:len(LatLonFileNameSave))

c     Read in/set up input variables.

      call ReadInputs( ProgName, Start, CountToRead, GeoNameIn,
     1     YearStart, YearStartC, YearEnd, MonthStart, MonthEnd,
     2     InventoryFileName)
c$$$  include 'ReadInputs.f'
c$$$  include 'ReadInputs-NotAppQual.f'

      if(Debug .eq. 1) print *, 'SIED #100: Inut read in.'

c     Allocate space for dynamically allocated variables.

      allocate( MedSST(1:LenX,1:LenY), 
     *     stat=ierr)
      
      if (ierr .ne. 0) stop 'Allocation error for LenX, LenY arrays. Ex
     1iting.'

c     SIED requires that the length of the arrays on which it operates 
c     . are multiples of WinSize. Calculate the size of the new arrays
c     . here. Needed a real*4 number in ifix hence the RealDum variable.
      
      RealDum = LenX
      LenXA = (ifix((RealDum - 1) / float(WinSize)) + 1) * WinSize
      RealDum = LenY
      LenYA = (ifix((RealDum - 1) / float(WinSize)) + 1) * WinSize

      print *, 'SIED #110: New LenXA and LenYA: ', LenXA, LenYA
      write(UnitLog,*) 'SIED #110: New LenXA and LenYA: ', LenXA, LenYA

c     Allocate space for MedSSTA and lat, lon arrays. Need to allocate
c      the latter in the event that there is only one lat, lon file
c      for this archive, otherwise it would be necessary to read in
c      the array for every file. 

      allocate( MedSSTA(1:LenXA,1:LenYA), 
     1     Lat_Array(1:LenX,1:LenY),
     2     Lon_Array(1:LenX,1:LenY),
     *     stat=ierr)
      
      if (ierr .ne. 0) stop 'Allocation error for LenXA, LenYA arrays. 
     1Exiting.'

c     Read ArchiveInventory - the names of files to process. First,
c     allocate the arrays to be read.

      include 'AllocateInventoryVariables.f'

      call ReadInventory(InventoryFileName, InvSecondsSince1970, 
     1     InvNumClear, InvYear, InvMonth, InvDay, InvHour, 
     2     InvMinute, InvSecond, InvFileName, iFirst, iLast)

      write(UnitLog,*) 'SIED #120: Will process ArchiveInventory from ',
     1     iFirst, ' to ', iLast
      print *, 'SIED #120: Will process ArchiveInventory from ', 
     1     iFirst, ' to ', iLast

c___________________________MAIN LOOP _________________________________

c     Get the date and time so that we can determine the files/minute
c      processed

      call date_and_time(VALUES=DateTimeStart)

c     Loop over all images in the time range 

      r_n = '1'

c     Set pr_n (Previous Run Number) to 1. This will create a thinned 
c     filename xxx_thinned_1.nc. The program will then check for 
c     existence of this file. If it exists, then this is the second
c     pass of SIED. Assume only two passes.

      pr_n = '1'

      NumberOfFilesProcessed = 0
      do 1000 iImg=iFirst,iLast

         if(Debug .eq. 1) print *, 'SIED #130 iImg: ', iImg

c     Do not process this image if less than Thresh clear pixels in it.

         if(InvNumClear(iImg) .lt. NumClearThresh) go to 1000

         SSTFileName = BlankFill(:len(SSTFileName))
         SSTFileName = InvFileName(iImg)

         if(Debug .eq. 1) then
            print *, 'SIED #132. SSTFileName::', trim(SSTFileName), '::'
            print *, 'SIED #134.  BaseDir::', trim(BaseDir), '::' 
            print *, 'SIED #136.  SatName::', trim(SatName), '::'
         endif

         call ReplaceBaseDir(SSTFileName)

         if(Debug .eq. 1) print *, 'SIED #140. SSTFileName::', 
     1        trim(SSTFileName), '::'

c     Found the file, build the output file names ......................

         call GetDateCharStrings( 
     1        InvYear(iImg), InvMonth(iImg),  InvDay(iImg),  
     2        InvHour(iImg), Inv Minute(iImg),
     3        YearC, MonthC, DayC, HourC, MinuteC)

c     Find /Median/ and _Median in the filename. 

         TempString1 = BlankFill(:len(TempString1))
         TempString1 = 'Median/'

         TempString2 = BlankFill(:len(TempString2))
         TempString2 = '_Median.nc'

         MedianFileName = BlankFill(:len(MedianFileName))
         MedianFileName = GenerateFileName( SSTFileName, TempString1, 
     1        TempString2)

c     Check to see if the input file exists. 

         inquire( File=MedianFileName, Exist=MedianExists)

c     If file not found stop run, the input file should exist. The 
c     filename is in the inventory so if the file is not found there
c     is a problem.

         if(MedianExists .eqv. .false.) then
            write(UnitLog,*) 'SIED #160: *** Median file does not ',
     1           'exist for::', trim(MedianFileName), '::'
            print *, 'SIED #160: *** no Median file for::',
     1           trim(MedianFileName), '::'
            stop 'SIED #161: Stopping. This median file does not exist.'
         endif

c     Now check to make sure that LenX corresponds to what is in the
c     input file. If not equal, STOP, should not be the case.

         if(CheckDimensions(MedianFileName) .eqv. .false.) then
            write(UnitLog,*) 'SIED #170: Variable Dimensions do ',
     1           'not agree with LenX or LenY.'
            stop 'SIED #170: Dimensions do not agree with LenX & LenY'
         endif

c     Check to see if a thinned file exists for this file. If it does,
c     and the is the first file to be processed in this run, set r_n to
c     2, otherwise set it to 1. If it is not the first file to be
c     processed, write an error message if it did exist before, 
c     otherwise continue normally.

c     First, build the filename for the thinned contours.

         TempString1 = BlankFill(:len(TempString1))
         TempString1 = 'Thinned/'

         TempString2 = BlankFill(:len(TempString2))
         TempString2 = '_thinned_' // pr_n // '.nc'

         ThinnedFileName = GenerateFileName( SSTFileName, TempString1, 
     1        TempString2)

         if(debug .eq. 1) print *, 'SIED #180. Thinned file::', 
     1        trim(ThinnedFileName), '::'

         inquire( File=trim(ThinnedFileName), Exist=ThinnedExists)

         if(debug .eq. 1) print *, 'SIED *181. ThinnedExists:',
     1        ThinnedExists, ', FirstGoodFile: ', FirstGoodFile, 
     2        ', r_n:', r_n

         if(ThinnedExists .eqv. .true.) then
            if(FirstGoodFile .eq. 1) then
               r_n = '2'
            else
               if(r_n .eq. '1') then
                  write(UnitLog,*) 'SIED #190: A thinned file exists ',
     1                 'for::', trim(SSTFileName), '::, but one did ',
     2                 'not exist for the previous median file.'
                  print *, 'SIED #190: A thinned file exists ',
     1                 'for::', trim(SSTFileName), '::, but one did ',
     2                 'not exist for the previous median file.'
                  stop
               endif
            endif
         else
            if(FirstGoodFile .eq. 1) then
               r_n = '1'
               pr_n = '0'
            else
               if(r_n .eq. '2') then

c     Problem. Try sleeping for 15 seconds and then retry - may be that
c      the link to the disk with the thinned file was momentarily down.

                  call system( 'sleep 15' )
                  inquire( File=trim(ThinnedFileName), 
     1                 Exist=ThinnedExists)

c     If it still doesn't exist, write a message and stop.

                  if(ThinnedExists .eqv. .false.) then
                     write(UnitLog,*) 'SIED #200: No thinned file ',
     1                 'for::', trim(SSTFileName), '::, but one ',
     2                 'exists for the previous median file. Program ',
     3                 'was Testing for::', trim(ThinnedFileName), '::'
                     print *, 'SIED #200: No thinned file ',
     1                 'for::', trim(SSTFileName), '::, but one ',
     2                 'exists for the previous median file. Program ',
     3                 'was Testing for::', trim(ThinnedFileName), '::'
                     stop
                  endif

                  write(unitlog, *) 'SIED #205: Failed to find::',
     1                 trim(ThinnedFileName), ':: on first try. ',
     2                 'Paused 15 seconds and tried again - and found ',
     3                 'the file - whew, that was close.'
                  print *, 'SIED #205: Failed to find::',
     1                 trim(ThinnedFileName), ':: on first try. ',
     2                 'Paused 15 seconds and tried again - and found ',
     3                 'the file - whew, that was close.'
               endif
            endif
         endif

c     Set FirstGoodFile to 0 since use has been made of the fact that
c     this is the first good file.

         FirstGoodFile = 0
         
c     Build the .cnt filename and check for its existence. Skip to 
c     next time if it exists; it has already been created, would be 
c     stupit to recreate it.

         if(Debug .eq. 1) print *, 'SIED #210: r_n::', r_n, '::'
         TempString1 = BlankFill(:len(TempString1))
c$$$  TempString1 = 'Cnt' // r_n // '/'
         if(r_n .eq. '1') then
            TempString1 = 'Fronts_1st_Pass/'
         else
            TempString1 = 'Fronts_2nd_Pass/'
         endif
         if(Debug .eq. 1) print *, 'SIED #220: trim(TempString1)::', 
     1        trim(TempString1), '::'

         TempString2 = BlankFill(:len(TempString2))
c$$$  TempString2 = '.cnt' // r_n
         if(r_n .eq. '1') then
            TempString2 = '_fronts_1st_pass.nc'
         else
            TempString2 = '_fronts_2nd_pass.nc'
         endif

         FrontsFileName = BlankFill(:len(FrontsFileName))
         FrontsFileName = GenerateFileName( SSTFileName, TempString1, 
     1        TempString2)

         if(debug .eq. 1) print *, 'SIED #230: Cnt file::', 
     1        trim(FrontsFileName), '::'

         inquire( File=trim(FrontsFileName), Exist=FrontsExists)

         if(Debug .eq. 1) print *, 'SIED #240. FrontsExists: ', 
     1        FrontsExists

         if(FrontsExists .eqv. .true.) then
            write(UnitLog,*) 'SIED #250: Output ',
     1           'files exist - skip all processing for:',
     1           trim(FrontsFileName), '::'
            print *, 'SIED #250: Output files exist ',
     1           '- skip. CntFile::', trim(FrontsFileName), '::'
            go to 1000
         endif


c------------------------------SIED ------------------------------------

         if (debug .eq. 1) print *,'SIED #260: Before reading MedSST'

         write(UnitLog,*) 'SIED #270:SIEDing::',
     1        trim(FrontsFileName), '::' 

c     Approximately what percentage of the run is complete?

         PercentDone = 100.0 * float(iIMG - iFirst + 1) / 
     1        float(iLast - iFirst + 1)

         print *, PercentDone, iFirst, iLast, iIMG, 
     1        'SIED #270:SIEDing::', trim(FrontsFileName), '::'

c..............now apply sied

         if (debug .eq. 1) print *, 'SIED #280: Before call to sied'

c     Read in the median data.

         call ReadMedianFile( MedianFileName, ncMedID, MedID, 
     1        SecondsSince1970InID, MedSST, SecondsSince1970T)

c     Get the filename of the file with latitude and longitude arrays.

         LatLonFileName = BlankFill(:len(LatLonFileName))
         status = nf90_get_att( ncMedID, nf90_global,
     1        'LatLonFileName', LatLonFileName)
         if(status .ne. nf90_noerr) call handle_err(status)

         status = nf90_close(ncMedID)
         if(status .ne. nf90_noerr) call handle_err(status)

         LatLonFileName = trim(BaseDir) // trim(LatLonFileName)

c     Move input data to an SST array with dimensions multiples of 
c     . WinSize
         
         do 1550 i=1,LenXA
            do 1551 j=1,LenYA
               if( (i .le. LenX) .and. (j .le. LenY) .and.
     1              (MedSSTA(i,j) .ne. SSTFillValueIn) ) then
                  MedSSTA(i,j) = MedSST(i,j)
               else
                  MedSSTA(i,j) = FillValueInt2
               endif

 1551       continue
 1550    continue

         if(debug .eq. 1) print *, 'SIED #300.'

         call sing_edge( MedSSTA, FrontsFileName, MedianFileName,
     1        ThinnedFileName, SecondsSince1970InID, ProgName,
     2        Lat_Array, Lon_Array)

         if(debug .eq. 1) print *, 'SIED #310. Reopening ',
     1        trim(MedianFileName), ' to write out fronts image.'

c     Write a 0, no fronts, 1, fronts, image to the median_sst file. 
c     First move fronts field to an array the same size as the median
c     sst array.

         do 1560 i=1,LenX
            do 1561 j=1,LenY
               MedSST(i,j) = MedSSTA(i,j)
 1561       continue
 1560    continue


         status = nf90_open( MedianFileName, nf90_write, ncMedID)
         if (status .ne. nf90_noerr) call handle_err(status)

c     Write SIED data.

         if(debug .eq. 1) print *, 'SIED #320.' 

         status = nf90_inq_varid( ncMedID, 
     1        'cayula_cornillon_front_pixel', FrontsID)
         if(status .ne. nf90_noerr) call handle_err(status)

         status = nf90_put_var( ncMedID, FrontsID, MedSST)
         if(status .ne. nf90_noerr) call handle_err(status)

         status = nf90_close(ncMedID)
         if(status .ne. nf90_noerr) call handle_err(status)

         NumberOfFilesProcessed = NumberOfFilesProcessed + 1

         if(debug .eq. 1) print *, 'SIED #330. Have written fronts ',
     1        'image.'

c--------------------------ALL DONE PROCESSING -------------------------

c     Cleanup input file.

 1000 continue

      close(UnitLog)
      close(UnitInventory)

      if(debug .eq. 1) print *, 'SIED #999'

c     Now get the elapsed wall time - print out in the start and stop
c      times in the subroutine.

      call Duration( DateTimeStart, DurationInMinutes, 
     1     NumberOfFilesProcessed, FilesPerMinute)

      write(UnitLog, *) NumberOfFilesProcessed, ' files processed in ',
     1    DurationInMinutes, ' minutes ==> ', FilesPerMinute, 
     2     ' files/minute.'

      print *, NumberOfFilesProcessed, ' files processed in ',
     1    DurationInMinutes, ' minutes ==> ', FilesPerMinute, 
     2     ' files/minute.'

      stop
      end program SIED_Main

c***********************************************************************
      subroutine sing_edge( pict, FrontsFileName, MedianFileName,
     1     mt_file, SecondsSince1970InID, ProgName, Lat_Array, 
     2     Lon_Array)
c***********************************************************************
c     
      use netcdf

      implicit none
      
c     Parameter statements

      include 'ParameterStatements.f'

c     Variables

      character*319 mt_file

c     icoord - do loop parameter for 1st dimension of image.
      integer*2 icoord
c     index - the value separating two populations in the call to Markov
c     . to calculate the cohesion for the clear/cloudy array.
      integer*2 index
c     iWindow - do loop parameter for loop over SIED windows.
      integer iWindow

c     jcoord - do loop parameter for 2nd dimension of image.
      integer*2 jcoord
c     jWindow - do loop parameter for loop over SIED windows.
      integer jWindow

c     FrontsFileName - the output filename for fronts - uncompressed
      character*319 FrontsFileName

c     Lat_Array - latitude values for image. Read in once if it does
c     not change or for each file if it does change.
      real*4 Lat_Array(1:LenX,1:LenY)
c     Lon_Array - longitude values for image. Read in once if it does
c     not change or for each file if it does change.
      real*4 Lon_Array(1:LenX,1:LenY)

c     MedianFileName - the name of the output file for the 3x3 median
c     . filter.
      character*319 MedianFileName

c$$$c     MaxIWindow - minimum value of the 1st dimensin for SIED windows
c$$$      integer*2, allocatable :: MaxIWindow(:,:)
c$$$c     MaxJWindow - minimum value of the 2nd dimensin for SIED windows
c$$$      integer*2, allocatable :: MaxJWindow(:,:)
c$$$c     MinIWindow - minimum value of the 1st dimensin for SIED windows
c$$$      integer*2, allocatable :: MinIWindow(:,:)
c$$$c     MinJWindow - minimum value of the 2nd dimensin for SIED windows
c$$$      integer*2, allocatable :: MinJWindow(:,:)

c     nsd2 - the distance through which the SIED windows are stepped;
c     . typically WinSize/2.
      integer*2 nsd2
c     NumberOfFilesProcessed - the number of fields processed in this 
c     . run.
      integer NumberOfFilesProcessed

c     SecondsSince1970InID - ID for NetCDF variable.
      integer SecondsSince1970InID

c     WinDim1 - the number of SIED windows in the 1st dimension
      integer WinDim1
c     WinDim2 - the number of SIED windows in the 2nd dimension
      integer WinDim2
c     WinClearCohesion - the cohesion for clear areas in each 
c     . nsd2*2 x nsd2*2 (typically 32x32) pixel window.
      real*4, allocatable :: WinClearCohesion(:,:)

      real*4 cor(1:2)
      real*4 ratio(0:2)

      integer*4 titlength, outlength
      integer*4, allocatable ::  outpic(:,:)
      integer*2 pict(1:LenXA,1:LenYA)
      integer*2, allocatable ::  thresh(:,:)
      character*1 filtyp
      real*8 dumhdate
      integer*1 read_flag
      real*4, allocatable :: wpv(:,:,:)
      real*4, allocatable :: ppv(:,:)
      integer*2 i,j
      integer*2 med_size
      integer*4 DataRange, NAboveRange

c     Window Parameter Vector: WPV
      
c     WPV(0)=RATIO(0)	:global cohesion
c     WPV(1)=MAX(0)	:criterion function (theta)
c     WPV(2)=MAX(1)	:Mean of population A
c     WPV(3)=MAX(2)	:Mean of population B

c     Pixel Parameter vector: PPV

c     PPV(0)=WPV(0)        :global cohesion
c     PPV(1)=WPV(1)        :criterion function (theta)
c     PPV(2)=WPV(2)        :Mean of population A
c     PPV(3)=WPV(3)        :Mean of population B
c     PPV(4)=THRESH        :Temperature threshold
c     PPV(5)               :Number of windows where edge pixel detected
      
c     Flag was set to 2, but I set it to 3 to skip reading lat and long
c     PCC 12-Mar-2009

c     DataRange - maximum minus minimum value read in.
c     NAboveRange - number of input data values above Range. 

c     Compute the temperature of the main fronts (internally check the
c     validity of each front: see subroutines called by COMPUTE)
      
c     if (debug .eq. 1) 
c     1     call PrintArray2( inpict(istrt:iend,jstrt:jend), 
c     2     '--- inpict on input --- EOS')

c     Allocate space for dynamically allocated variables.

c---------------------------------start here ---------------------------

      if(Debug .eq. 1) print *, 'sing_edge #000'

      WinDim1 = LenXA / float(WinSize) * 2
      WinDim2 = LenYA / float(WinSize) * 2

      allocate( outpic(1:LenXA,1:LenYA),
     1     thresh(1:WinDim1,1:WinDim2),
     2     wpv(1:WinDim1,1:WinDim2,0:3),
     3     ppv(1:MaxEdge,0:5),
     4     stat=ierr)
      
      if (ierr .ne. 0) then
         print *, 'sing_edge #100: Allocation error for LenXA, LenYA ',
     1        'arrays. Exiting.'
         stop
      endif

c      if (debug .eq. 1) then
c         Msg50 = 'sing_edge #110: --- pict on input --- EOS'
c         call PrintArray2( pict(istrt:iend,jstrt:jend), Msg50)
c      endif

      call compute(pict,thresh,wpv)
      
c     Locate the main fronts and draw them in array OUTPIC.
      
c      if (debug .eq. 1) then
c         Msg50 = 'sing_edge #120: --- pict after compute --- EOS'
c         call PrintArray2( pict(istrt:iend,jstrt:jend), Msg50)
c      endif

      call locate(pict,outpic,thresh,wpv,ppv)

c     outpic is an image in which frontal pixels have non-zero values.
c     Non-frontal pixels are zero. The value of the pixel at a frontal
c     location is the front pixel number assigned to that location.
c     The value at the location of the first frontal pixel found is
c     1, of the second frontal pixel found is 2, etc. ppv contains
c     information about each of these frontal pixels - the mean
c     temperature of the two populations,...

c      if (debug .eq. 1) then
c         Msg50 = 'sing_edge #130: --- outpic after locate --- EOS'
c         call PrintArray4( outpic(istrt:iend,jstrt:jend), Msg50)
c      endif

c     It has been noticed that the algorithm does not follow all
c     contours completely (it gets the main contours only). To solve
c     this problem, the next subprogram (contour following algorithm)
c     has been added.
      
      call follow_main( FrontsFileName, MedianFileName, mt_file, outpic,
     1     pict, ppv, SecondsSince1970InID, ProgName, Lat_Array,
     2     Lon_Array)
      
c      if (debug .eq. 1) then
c         Msg50 = 'sing_edge #140: --- pict after follow_main --- EOS'
c         call PrintArray2( pict(istrt:iend,jstrt:jend), Msg50)
c
c         Msg50 = 'sing_edge #150: --- outpic after follow_main --- EOS'
c         call PrintArray4( outpic(istrt:iend,jstrt:jend), Msg50)
c      endif
      
c********In this section calculate the cohesion of the clear SST areas
c     on the nsd2*2 x nsd2*2 (typically 32x32) pixel windows.

      allocate( WinClearCohesion(1:WinDim1,1:WinDim2),
     4     stat=ierr)
      
cXXXXXXXXXBE VERY CAREFUL - USING PICT and THRESH they are repurposed.

c     pict has the SST field in it however it is no longer needed so,
c     . convert the field to 10 for good SST value, 5 otherwise. Markov
c     . checks to see if pict is less than 2 and skips the pixel if it
c     . is, hence the reason for setting cloudy pixels to 5.

      do 3100 j=1,LenYA 
         do 3110 i=1,LenXA 
            if(Pict(i,j) .eq. FillValueInt2) then
               pict(i,j) = 5
            else
               pict(i,j) = 10
            endif
 3110    continue
 3100 continue

      if(debug .eq. 1) print *, 'sing_edge #160: Computing cohesion.' 

c     Loop over SIED windows. First initialize variables. This is done
c     . here as opposed to in the next loops because the variables were
c     . allocated to be one element longer than they should have and
c     . changing this would mean modifications to a number of SIED
c     . subroutines that I'm not prepared to make at present. I think
c     . that the dimensions should be LenXA / float(WinSize) * 2 - 1. 
c     . Need to check this at some point.


      do 23200 iWindow=1,WinDim1
         do 23210 jWindow=1,WinDim2
            thresh(iWindow,jWindow) = 0
            WinClearCohesion(iWindow,jWindow) = FillValueInt1
23210    continue
23200 continue

      nsd2 = WinSize / 2

      do 23020 icoord=1,LenXA-nsd2,nsd2 

         iWindow = ifix(2 * float(icoord) / WinSize) + 1

         do 23022 jcoord=1,LenYA-nsd2,nsd2 

            jWindow = ifix(2 * float(jcoord) / WinSize) + 1

c     Now count the number of clear pixels for each nsd2*2 x nsd2*2 
c     . (typically 32x32) pixel window

            do 23024 i=icoord,icoord+nsd2*2-1
               do 23026 j=jcoord,jcoord+nsd2*2-1
                  if(pict(i,j) .eq. 10) Thresh(iWindow,jWindow) = 
     1                 Thresh(iWindow,jWindow) + 1
23026          continue
23024       continue

c     Now calculate the cohesion for the clear pixels in this window.
c     . If there are less than 100 clear pixels, set to the fill value.
c     . If all the pixels in this window are clear, then set to 100.
c     . Need to do this because Markov sets the cohesion to 0 if all
c     . of the pixels in the window belong to one population. 
c     . If more than 100 but less than all pixels clear multiply the
c     . cohesion calculated by Markov by 100; i.e., 1.0 ==> 100.

            if(Thresh(iWindow,jWindow) .eq. WinSize * WinSize) then
               WinClearCohesion(iWindow,jWindow) = 100
            elseif(Thresh(iWindow,jWindow) .gt. 100) then
               index = 7
               call markov( pict, icoord, jcoord, index, ratio, cor)
               WinClearCohesion(iWindow,jWindow) = ratio(0) * 100
            endif

23022    continue
23020 continue

c     Write out the number of clear pixels in each nsd2*2 x nsd2*2
c     . (typically 32x32) pixel window

      if(debug .eq. 1) print *, 'sing_edge #170: Writing number clear.' 

      status = nf90_put_var( dim2ID, WinNumClearID, Thresh)
      if(status .ne. nf90_noerr) call handle_err(status)

c     and write out the cohesion values

      if(debug .eq. 1) print *, 'sing_edge #170: Writing cohesion' 

      status = nf90_put_var( dim2ID, WinClearCohesionID, 
     1     WinClearCohesion)
      if(status .ne. nf90_noerr) call handle_err(status)

c     All done with file, close it now.

      status = nf90_close(dim2ID)
      if(status .ne. nf90_noerr) call handle_err(status)

      NumberOfFilesProcessed = NumberOfFilesProcessed + 1

      if(debug .eq. 1) print *, 'sing_edge #170: Have written ',
     1     'fronts image.'

c********The next section is for send the fronts field back to the
c     calling program so that it can be written out. Overwriting pict
c     one more time.

c     Superimpose the fronts from OUTPic on PICT.
      
      do 23014 j=1,LenYA 
         do 23016 i=1,LenXA 
            pict(i,j) = outpic(i,j)
23016    continue
23014 continue
      
      return
      end subroutine sing_edge 
 
c***********************************************************************
      subroutine compute(pict,thresh,wpv)
c***********************************************************************
c
c  This routine mainly calls 2 routines: HISTO (histogram and
c  unsupervised learning) and MARKOV (compactness of the 2 
c  populations found by HISTO).

c  The first pass of HISTO and MARKOV is executed to detect
c  the possible clouds: low correlation (absolute difference
c  between neighbors) or low cohesion and also high variance
c  (included in not normalized correlation coefficient).
 
      implicit none
 
c     Parameter statements

      include 'ParameterStatements.f'

c     General variables
 
c     icoord - do loop parameter for 1st dimension of image.
      integer*2 icoord

c     jcoord - do loop parameter for 2nd dimension of image.
      integer*2 jcoord

c     nsd2 - the distance through which the SIED windows are stepped;
c     . typically WinSize/2.
      integer*2 nsd2

      integer*2 pict(1:LenXA,1:LenYA)
      integer*2 thresh(1:(LenXA/WinSize)*2,1:(LenYA/WinSize)*2)
      integer*2 iwind,jwind
      integer*2 index
      real*4 max(0:2)
      real*4 cor(1:2)
      real*4 ratio(0:2)
      logical*1, allocatable :: cloud(:,:)
      logical*1, allocatable :: wcloud(:,:)
      integer*2 k
      real*4 wpv(1:(LenXA/WinSize)*2,1:(LenYA/WinSize)*2,0:3)
      integer*2 lun1
      integer*2 i,j
 
c PICT(1:SizOfImg,1:SizOfImg)  - integer image (0,255)
c thresh(1:NWindows,1:NWindows)    - temperatures of fronts
c NSD2               - NSize Divided by 2
c ICOORD,JCOORD      - coordinate of window corner
c IWIND,JWIND	     - window number
c INDEX              - temperature of window front
c MINSTEP            - minimum temperature step
c MAX(0:2)           - ratio of var and means
c COR(1:2)           - estimate of correlation
c RATIO(0:2)         - estimates of cohesion
c cloud(1:NWindows,1:NWindows)	- cloud detection
c wcloud(1:NWindows,1:NWindows)	- warm cloud detection
c K		- generic index
c WPV(1:NWindows,1:NWindows,0:3) - Parameter vector
c Range              - The upper limit to use in the histograms. PCC 
c    added this variable. The numbers were harded coded in as 255 in 
c    the .rat version. Note that this assumes the minimum is 0.
 
c  Window Parameter Vector: WPV
c  WPV(0)=RATIO(0)	:global cohesion
c  WPV(1)=MAX(0)	:criterion function (theta)
c  WPV(2)=MAX(1)	:Mean of population A
c  WPV(3)=MAX(2)	:Mean of population B
 
c---------------------------------start here ---------------------------

      allocate( cloud(1:(LenXA/WinSize)*2,1:(LenYA/WinSize)*2),
     1     wcloud(1:(LenXA/WinSize)*2,1:(LenYA/WinSize)*2),
     2     stat=ierr)
      
      if (ierr .ne. 0) then
         print *, 'Allocation error for LenXA, LenYA arrays. Exiting.'
         stop
      endif

      lun1 = 20
      nsd2 = WinSize / 2
 
c  After the possible clouds have been removed (this step was removed 
c  from program because it was not being used), the second pass of
c  HISTO and MARKOV output the temperature of valid fronts.
 
      iwind = 0
      
       if(Debug .eq. 1) then 
          print *, 'compute #100:: About to do array, size ',  nsd2
          print *, 'LenXA-nsd2', LenXA-nsd2, LenYA-nsd2, nsd2
       endif
      do 23020 icoord=1,LenXA-nsd2,nsd2 
         iwind = iwind+1
         jwind = 0
         do 23022 jcoord=1,LenYA-nsd2,nsd2 
            jwind = jwind+1
 
c  this routine returns the temperature of a possible front (INDEX)
c  as well as a level of certitude (MAX(0)).
 
            call histo(pict,icoord,jcoord,index,max)
c            if(jcoord.ge.424.and.jcoord.lt.496)then
c                  if(icoord.ge.1072.and.icoord.lt.1136)then
c                     print *, 'Loop Current region, index found was ', 
c     1                    index, ' and max is ', max(0), ' for ', icoord
c     2                    ,jcoord, ' and meanb - meana = ', max(2) - 
c     3                    max(1),' meana = ', max(1)
c                  endif
c               endif
c  A temperature difference of less than 3 digital counts between
c  2 populations because it is likely to be a result of the dicrete
c  nature of the data.
 
            if ((max(0) .gt. .6) .and. ((max(2)-max(1)) .ge. MinStep))
     +        then               
c  Using the temperature of the possible front, this routine
c  estimates the cohesion of the 2 populations (cold and warm).
c  The results are put in the array RATIO.
               call markov(pict,icoord,jcoord,index,ratio,cor)
 
c  If unsufficient cohesion is observed, the threshold is removed 
c  (Range+1).
 
               if ((ratio(1) .lt. .90) .or. (ratio(2) .lt. .90)) then
                  index = Range+1
               else
                  if (ratio(0) .lt. .92) then
                     index = Range+1
                  endif
               endif
            else
               index = Range+1
            endif
 
            thresh(iwind,jwind) = index
 
            wpv(iwind,jwind,0) = ratio(0)
            wpv(iwind,jwind,1) = max(0)
            wpv(iwind,jwind,2) = max(1)
            wpv(iwind,jwind,3) = max(2)
 
23022    continue
23020 continue
 
      return
      end
 
c***********************************************************************
      subroutine histo(pict,icoord,jcoord,index,max)
c***********************************************************************
c  Histogramming algorithm.
 
      implicit none
 
c     Parameter statements

      include 'ParameterStatements.f'

c     General variables

      integer*2 pict(1:LenXA,1:LenYA)
c      integer*2 picttemp(1:WinSize,1:WinSize)
      integer*2, allocatable :: picttemp(:,:)
      integer*2 icoord,jcoord
      integer*2 index
      real*4 max(0:2)
      integer*2 i,j, it, jt
      integer*4, allocatable :: count(:)
      integer*4, allocatable :: RealTemp(:)
      integer*4, allocatable :: na(:)
      integer*4, allocatable :: nb(:)
      real*4, allocatable :: meana(:)
      real*4, allocatable :: meanb(:)
      real*8 var
      real*4, allocatable :: suma(:)
      real*4, allocatable :: sumb(:)
      real*8, allocatable :: ssqa(:)
      real*8, allocatable :: ssqb(:)
      real*8, allocatable :: dif(:)
      integer*2, allocatable :: linearfit(:,:)
      integer*4 total
      real*4 sum
      real*8 ssq
      integer*2 pictmin
      logical*1, allocatable :: tcond(:)
      integer*4 s_pop

      logical, parameter :: PlaneDebug = .false.

c     WinBorder - the width of the border around the histogram window 
c      to include in the region for which a plane is fit. 
      integer WinBorder
 
c PICT(1:SizOfImg,1:SizOfImg)  - integer image (0,255)
c ICOORD,JCOORD      - coordinate of window corner
c INDEX              - temperature of window front
c MAX(0:2)           - ratio of var and means
c I,J                - all purpose indexes
c COUNT(0:255)       - histogram of window
c NA(0:255)          - sizes of population A and B
c NB(0:255)          - for different thresholds 
c MEANA(0:255)       - means of population A and B
c MEANB(0:255)       - for different thresholds 
c VAR                - variance in window
c SUMA(0:255)        - sum for population A
c SUMB(0:255)        - sum for population B
c SSQA(0:255)        - SSQ of population A
c SSQB(0:255)        - SSQ of population B
c DIF(0:255)         - separation between means
c TOTAL              - count accumulator
c SUM                - sum accumulator
c SSQ                - Sum of SQuare in window
c TCOND(0:255)       - check for 0s in histogram
c S_POP              - size of smallest population
c Range              - The upper limit to use in the histograms. PCC 
c    added this variable. The numbers were harded coded in as 255 in 
c    the .rat version.
 
c---------------------------------start here ---------------------------

      allocate( count(0:Range),
     2     na(0:Range),
     3     nb(0:Range),
     4     meana(0:Range),
     5     meanb(0:Range),
     6     suma(0:Range),
     7     sumb(0:Range),
     8     ssqa(0:Range),
     9     ssqb(0:Range),
     *     dif(0:Range),
     1     tcond(0:Range),
     2     RealTemp(0:Range),
     3     picttemp(1:WinSize,1:WinSize),
     4     linearfit(WinSize,WinSize),
     5     stat=ierr)
      
      if (ierr .ne. 0) then
         print *, 'Allocation error for LenXA, LenYA arrays. Exiting.'
         stop
      endif

      do 23030 i=0,2 
         max(i) = 0.0
23030 continue
 
      do 23032 i=0,Range 
         count(i) = 0
23032 continue

c     Remove plane fit to image if requested.
      if(RemovePlane) then
c     Code to fit bilinear surface to pict from icoord-8 to 
c     icoord+WinSize + 8 and same for jcoord.
         linearfit = 0.0
         call plane_fit(pict,linearfit,icoord,jcoord)
         
c     Write stuff out if this is a debug run for the plane.

         if(PlaneDebug .eqv. .true.) then
            WinBorder = (PlaneFitSize - WinSize) / 2
           open(8, FILE='/Volumes/JSFRONTS/Data.txt', ACCESS='APPEND'
     1        )
            if (icoord > WinBorder .and. 
     1          jcoord > WinBorder .and.jcoord<700) then
               do i = 1, PlaneFitSize
                  write(8,"(I8)") pict(icoord-WinBorder + 
     1                 (i-1),jcoord-WinBorder:jcoord+
     2                 PlaneFitSize-WinBorder)
               end do         
               write(8,"(I8)") -999            
               open(7, FILE='/Volumes/JSFRONTS/FitData.txt', ACCESS
     1              ='APPEND')
               do i = 1, WinSize
                  write(7,"(I8)") linearfit(i,1:WinSize)
               end do
               write(7,"(I8)") -999
               close(7)
               close(8)
            endif
         endif

c     Remove the plane from the data.
         pictmin = 0
         it = 0         
         do  i=icoord,icoord+WinSize-1
            it = it + 1
            jt = 0
            do j = jcoord,jcoord+WinSize-1 
               jt = jt + 1
               if(pict(i,j).ge.0) then
                  picttemp(it,jt) = pict(i,j) - linearfit(it,jt)
                  if(picttemp(it,jt) .lt. pictmin) then
                     pictmin = picttemp(it,jt)
                  endif
               else
                  if (it.gt.WinSize) print*, 'i = ', i, ' j = ', j,
     1                 '. Aborting.'
                  picttemp(it,jt) = SSTFillValueIn
               endif
            enddo
         enddo
         
         it = 0         
         do  i=icoord,icoord+WinSize-1
            it = it + 1
            jt = 0
            do j = jcoord,jcoord+WinSize-1 
               jt = jt + 1
               if(picttemp(it,jt) .ne. SSTFillValueIn)then
                  picttemp(it,jt) = picttemp(it,jt)+10+abs(pictmin)           
               endif
            enddo
         enddo

         i = icoord
         it = icoord+WinSize-1

         j = jcoord
         jt = jcoord+WinSize-1
         
         if (debug .eq. 1) then
            Msg50 = 'sing_edge #110: --- pict on input --- EOS'
            print*, 'index to print i = ', i, ' j = ',j
            call PrintArray2( pict(i+21:i+31,j+21:j+31), Msg50)
            Msg50 = 'sing_edge #110: --- linearfit --- EOS'
            call PrintArray2(linearfit(1:10,1:10), Msg50)
         endif
         
c     Then create a temporary field, picttemp that goes from 1,1 to 
c     winsize, winsize.
      else
         it = 0 
         do 23901  i=icoord,icoord+WinSize-1 
            it = it + 1
            jt = 0
            do 23902 j = jcoord,jcoord+WinSize-1 
               jt = jt + 1
               picttemp(it,jt) = pict(i,j)
23902       continue
23901    continue
      endif

c Now generate the historgam of the winsize by winsize region.
      
      it = 0
      RealTemp = 0
      do 23034 i=icoord,icoord+WinSize-1 
         it = it + 1
         jt = 0
         do 23036 j = jcoord,jcoord+WinSize-1 
            jt = jt + 1
            if (picttemp(it,jt) .gt. 2) then
               if(RemovePlane) then               
                  if(count(picttemp(it,jt)).eq.0)then
                     RealTemp(picttemp(it,jt)) = pict(i,j)
                  endif
               endif
               count(picttemp(it,jt)) = count(picttemp(it,jt))+1
            endif
23036    continue
23034 continue
 
c Computation of the Mean and variance of Population A. Population A 
c  isn't really defined at this point. What is actually being done is 
c  that the cumulative number of pixels from the coldest value possible
c  to the ith value, the sum of the corresponding SSTs of these pixels
c  and the sum of their squares is calculated. In the next section,
c  the computation for population B the same is done starting with the
c  the warmest temperature possible and moving to colder temperatures.
 
      sum = 0
      total = 0
      ssq = 0
      do 23040 i=0,Range 
         if (count(i) .ne. 0) then
            total = count(i)+total
            sum = count(i)*i+sum
            ssq = ssq+(count(i)*i*i)
         endif
         na(i) = total
         suma(i) = sum
         ssqa(i) = ssq
23040 continue
 
c     Computation of the Mean of Populations A and B.
 
      do 23044 i=0,Range 
         nb(i) = total-na(i)
         sumb(i) = sum-suma(i)
         ssqb(i) = ssq-ssqa(i)
         tcond(i) = ((na(i) .ne. 0) .and. (nb(i) .ne. 0))
         if (tcond(i)) then
            meana(i) = suma(i)/na(i)
            meanb(i) = sumb(i)/nb(i)
         endif
23044 continue
 
c Now, for each possible temperature value in the image, calculate the
c  the difference between the mean temperatures for all values 
c  temperature less than this value (in the 32x32 pixel square) and of
c  all temperatures warmer than this value squared times the product of
c  the number of pixels in each of the two regions and find the 
c  temperature at which this is a maximum. Save this temperature (well,
c  really the counts corresponding to this temperature) and the 
c  difference. dif(i) below is referred to as J_b(tau), equation 11, in
c  Cayula & Cornillon <<1992. Edge detection algorithm for SST images.
c  Journal Of Atmospheric And Oceanic Technology, 9(1), pp.67–80>> 
c  except for the square of the total number of clear pixels in the 
c  region.

      max(0) = 0
      index = Range+1
      do 23048 i = 0,Range 
         if (tcond(i)) then
            dif(i) = na(i)*nb(i)*((meana(i)-meanb(i))**2)
            if (max(0) .lt. dif(i)) then
               max(0) = dif(i)
               index = i
            endif
         endif
23048 continue
 
c  Computation of the variances (max(1) and max(2)) and the ratio of 
c  in-between to TOTAL variances from the sums of square of the 
c  population previously selected. An index value of range+1 indicates
c  that no maximum difference was found. There must be more than 100
c  clear pixels in this 32x32 pixel window AND the smallest of the two
c  populations must be bigger than 1/4 of the total number of clear 
c  pixels in the region to find a front.
 
      if ((total .gt. MinClear) .and. (index .ne. Range+1)) then
         s_pop = min(na(index),nb(index))
 
         if ((s_pop*4) .gt. total) then
            sum = sum/total
            var = ssq-((sum**2)*total)
 
            if (var .ne. 0) then
               max(0) = max(0)/(var*total)
               max(1) = meana(index)
               max(2) = meanb(index)
            endif
 
         else
            max(0) = 0
         endif
 
      else
         max(0) = 0
      endif
 
      if(max(0).gt.0 .and. RemovePlane)then
c     This part of the code is designed to put the proper values in for
c     max and index. It redoes the relevant parts of the previous loops and
c     should contain max and index pertinent to pict instead of picttmp
         sum = 0
         total = 0
         ssq = 0
         do i=0,Range 
            if (count(i) .ne. 0) then
               total = count(i)+total
               sum = count(i)*RealTemp(i)+sum
               ssq = ssq+(count(i)*RealTemp(i)*RealTemp(i))
            endif
            na(i) = total
            suma(i) = sum
            ssqa(i) = ssq
         enddo
         
         do i=0,Range 
            nb(i) = total-na(i)
            sumb(i) = sum-suma(i)
            ssqb(i) = ssq-ssqa(i)
            tcond(i) = ((na(i) .ne. 0) .and. (nb(i) .ne. 0))
            if (tcond(i)) then
               meana(i) = suma(i)/na(i)
               meanb(i) = sumb(i)/nb(i)
            endif
         enddo
                  
         max(0) = na(index)*nb(index)*((meana(index)-meanb(index))**2)
         sum = sum/total
         var = ssq-((sum**2)*total)
         max(0) = max(0)/(var*total)
         max(1) = meana(index)
         max(2) = meanb(index)
         
         index = RealTemp(index)
      endif

c max(0) = 0 means that there were not 100 total pixels, that the 
c  smallest of the two populations was less than 1/4 of the total or 
c  that no maximum difference was found. I don't think that the last
c  condition will ever happen.

c Looks to me like sum is never used after being set in the final if
c  statement above.

      return
      end
 
c***********************************************************************
      subroutine markov(pict,icoord,jcoord,index,ratio,cor)
c***********************************************************************
      implicit none
 
c     Parameter statements

      include 'ParameterStatements.f'

c     General variables

      integer*2 pict(1:LenXA,1:LenYA)
      logical*1, allocatable :: mark(:,:)
      integer*2 index
      integer*2 icoord,jcoord
      integer*2 i,j
      integer*4 count_eq(1:2)
      integer*4 total_a
      integer*4 total_b
      integer*4 count_a
      integer*4 count_b
      real*4 ratio(0:2)
      real*4 cor(1:2)
      real*4 dx(1:2)
      real*4 dy(1:2)
      real*4 adx(1:2)
      real*4 ady(1:2)
      integer*4 temp
      integer*4 k
      integer*2 jmax

c---------------------------------start here ---------------------------

      allocate( mark(1:LenXA,1:LenYA),
     2     stat=ierr)
      
      if (ierr .ne. 0) then
         print *, 'Allocation error for LenXA, LenYA arrays. Exiting.'
         stop
      endif

      do 23060 k = 1,2 
         dx(k) = 0
         adx(k) = 0
         dy(k) = 0
         ady(k) = 0
         count_eq(k) = 0
23060 continue
      total_a = 0
      total_b = 0
      do 23062 i = icoord,icoord+WinSize-1 
         do 23064 j = jcoord,min(jcoord+WinSize-1, LenYA) 
            mark(i,j) = .true.
23064    continue
23062 continue
      do 23066 i = icoord,icoord+WinSize-1 
         do 23068 j = jcoord,min(jcoord+WinSize-2, LenYA - 1) 
            if(pict(i,j) .gt. 2)then
               call vcompare(pict,mark,i,j,index,total_a,total_b,
     +           count_eq,dx,dy,adx,ady)
            endif
23068    continue
23066 continue
      count_a = count_eq(1)
      count_b = count_eq(2)
      jmax = jcoord+WinSize-1
      do 23072 i = icoord,min(icoord+WinSize-2, LenXA - 1) 
         do 23074 j = jcoord,jcoord+WinSize-1 
            if(pict(i,j) .gt. 2)then
               call hcompare(pict,mark,i,j,index,total_a,total_b,
     +           count_eq,dx,dy,adx,ady,jmax)
            endif
23074    continue
23072 continue
      do 23078 k = 1,2 
         if(count_eq(k) .ne. 0)then
            adx(k) = adx(k)-abs(dx(k))
            ady(k) = ady(k)-abs(dy(k))
            cor(k) = (adx(k)+ady(k))/count_eq(k)
         endif
23078 continue
      temp = total_a+total_b
      total_a = temp+count_a-count_b
      total_b = temp+count_b-count_a
      count_a = count_a+count_a
      count_b = count_b+count_b
      if((total_a .ne. 0) .and. (total_b .ne. 0))then
         ratio(1) = real(count_a)/real(total_a)
         ratio(2) = real(count_b)/real(total_b)
         ratio(0) = real(count_a+count_b)/real(total_a+total_b)
      else
         ratio(1) = 0
         ratio(2) = 0
         ratio(0) = 0
      endif
      return
      end
 
c***********************************************************************
      subroutine vcompare(pict,mark,i,j,index,total_a,total_b,
     +     count_eq,dx,dy,adx,ady)
c***********************************************************************
      implicit none
 
c     Parameter statements

      include 'ParameterStatements.f'

c     General variables

      integer*2 pict(1:LenXA,1:LenYA)
      logical*1 mark(1:LenXA,1:LenYA)
      integer*2 i,j
      integer*2 index
      integer*4 total_a
      integer*4 total_b
      integer*4 count_eq(1:2)
      real*4 dx(1:2)
      real*4 dy(1:2)
      real*4 adx(1:2)
      real*4 ady(1:2)
      integer*2 center
      integer*2 neighbor

      center = pict(i,j)
      neighbor = pict(i,j+1)
      if(neighbor .gt. 2)then
         if(center .le. index)then
            total_a = total_a+1
            if(neighbor .le. index)then
               count_eq(1) = count_eq(1)+1
               dy(1) = dy(1)+(center-neighbor)
               ady(1) = ady(1)+abs(center-neighbor)
            else
               mark(i,j) = .false.
            endif
         else
            total_b = total_b+1
            if(neighbor .gt. index)then
               count_eq(2) = count_eq(2)+1
               dy(2) = dy(2)+(center-neighbor)
               ady(2) = ady(2)+abs(center-neighbor)
            else
               mark(i,j) = .false.
            endif
         endif
      endif
      return
      end
 
c***********************************************************************
      subroutine hcompare(pict,mark,i,j,index,total_a,total_b,
     +     count_eq,dx,dy,adx,ady,jmax)
c***********************************************************************
      implicit none
 
c     Parameter statements

      include 'ParameterStatements.f'

c     General variables

      integer*2 pict(1:LenXA,1:LenYA)
      logical*1 mark(1:LenXA,1:LenYA)
      integer*2 i,j
      integer*2 index
      integer*4 total_a
      integer*4 total_b
      integer*4 count_eq(1:2)
      real*4 dx(1:2)
      real*4 dy(1:2)
      real*4 adx(1:2)
      real*4 ady(1:2)
      integer*2 center
      integer*2 neighbor
      integer*2 jmax

      center = pict(i,j)
      neighbor = pict(i+1,j)
      if(neighbor .gt. 2)then
         if(center .le. index)then
            if(neighbor .le. index)then
               count_eq(1) = count_eq(1)+1
               dx(1) = dx(1)+(center-neighbor)
               adx(1) = adx(1)+abs(center-neighbor)
            else
               if(mark(i,j) .and. mark(i+1,j) .and. (j .ne. jmax) )then
                  total_a = total_a+1
               endif
            endif
         else
            if(neighbor .gt. index)then
               count_eq(2) = count_eq(2)+1
               dx(2) = dx(2)+(center-neighbor)
               adx(2) = adx(2)+abs(center-neighbor)
            else
               if(mark(i,j) .and. mark(i+1,j) .and. (j .ne. jmax) )then
                  total_b = total_b+1
               endif
            endif
         endif
      endif
      return
      end
 
c***********************************************************************
      subroutine locate(pict,outpic,thresh,wpv,ppv) 
c***********************************************************************
c
c  This subroutine uses the separation value for each window (THRESH),
c  to draw the fronts in array OUTPIC.
 
      implicit none
 
c     Parameter statements

      include 'ParameterStatements.f'

c     General variables

      integer*2 pict(1:LenXA,1:LenYA)
      integer*4 outpic(1:LenXA,1:LenYA)
      integer*2 thresh(1:(LenXA/WinSize)*2,1:(LenYA/WinSize)*2)
      real*4 wpv(1:(LenXA/WinSize)*2,1:(LenYA/WinSize)*2,0:3)
      real*4 ppv(1:MaxEdge,0:5)
      integer*4 nep
      integer*4 onep
      integer*2 nsd2
      integer*2 icoord,jcoord
      integer*2 iwind,jwind
      integer*2 i,j
      logical*1 cond,cond1
      logical*1 cond2
 
c PICT(1:SizOfImg,1:SizOfImg)   - input image
c OUTPIC(1:SizOfImg,1:SizOfImg) - temporary output image; changed 
c   to Int*4 to allow for larger numbers of edge pixels in results
c THRESH(1:NWindows,1:NWindows)	- temperatures of fronts
c WPV(1:NWindows,1:NWindows,0:3)	- Window Parameter vector
c PPV(1:MaxEdge,0:5)    - Pixel Parameter vector
c NEP                 - Number of edge pixels; changed to Int*4 to 
c   allow for model results
c ONEP                - temporary NEP; changed to Int*4 to allow for 
c   model results
c NSD2                - NSize Divided by 2
c ICOORD,JCOORD       - top left corner of window
c IWIND,JWIND         - window coordinates
c I,J                 - indexes
c COND,COND1          - conditions used to show
c COND2               - edges in graphic output
c Range              - The upper limit to use in the histograms. PCC 
c    added this variable. The numbers were harded coded in as 255 in 
c    the .rat version.

c  Window Parameter Vector: WPV

c  WPV(0)=RATIO(0)	:global cohesion
c  WPV(1)=MAX(0)	:criterion function (theta)
c  WPV(2)=MAX(1)	:Mean of population A
c  WPV(3)=MAX(2)	:Mean of population B

c  Pixel Parameter vector: PPV

c  PPV(0)=WPV(0)        :global cohesion
c  PPV(1)=WPV(1)        :criterion function (theta)
c  PPV(2)=WPV(2)        :Mean of population A
c  PPV(3)=WPV(3)        :Mean of population B
c  PPV(4)=THRESH        :Temperature threshold
c  PPV(5)               :Number of windows where edge pixel detected
 
c---------------------------------start here ---------------------------

      do 23104 j=1,LenYA 
         do 23106 i=1,LenXA 
            outpic(i,j) = 0
23106    continue
23104 continue
 
      nep = 1
      nsd2 = WinSize / 2
      iwind = 0
 
      do 23108 icoord=1,LenXA-nsd2,nsd2 
         iwind = iwind+1
         jwind = 0
         do 23110 jcoord=1,LenYA-nsd2,nsd2 
            jwind = jwind+1
            if(thresh(iwind,jwind) .lt. Range+1)then
               do 23114 i = icoord,icoord+WinSize-1 
                  do 23116 j = jcoord,jcoord+WinSize-1 
                     cond = (pict(i,j) .gt. thresh(iwind,jwind))
 
c  Case when PICT(I,J) is a member of the warmer population.
 
                     if(cond)then
 
c  Check on the neighbor to the right.
 
                        if(i .lt. LenXA )then
                           cond1 = (pict(i+1,j) .gt. 
     +                       thresh(iwind,jwind))
                           cond1 = cond1 .or. (pict(i+1,j) .le. 2)
                        else
                           cond1 = .true.
                        endif
 
c  Check on neighbor above.
 
                        if(j .lt. LenYA )then
                           cond2 = (pict(i,j+1) .gt. 
     +                       thresh(iwind,jwind))
                           cond2 = cond2 .or. (pict(i,j+1) .le. 2)
                        else
                           cond2 = .true.
                        endif
 
c  Put edge pixel number in array OUTPic if front has been detected.
c  Specifically, if this pixel is in the warm population and the one to
c   the right or below is in the cold population, this is a candidate
c   front pixel. If this is the first time that this pixel has been
c   flagged as a front candidate, set outpic to the next front segment
c   number and populate ppv with the stats for this segment. If the 
c   pixel has already been flagged as a front pixel (in which case the 
c   value of outpic(i,j) will be the number of a previous front segment
c   found), increment the value of ppv(,5) for the previously found 
c   front sement, which was set to 1 the first time the pixel was 
c   flagged as a front candidate, by one to indicate that this point 
c   is also a front pixel on another segment and populate ppv for this 
c   front segment. A bit disconcerting is that the front segment counter
c   is not incremented after this. Not sure I understand completely 
c   (PCC - 7/3/13).
 
                        if( .not. (cond1 .and. cond2))then
                           if(outpic(i,j) .eq. 0)then
                              outpic(i,j) = nep
                              ppv(nep,0) = wpv(iwind,jwind,0)
                              ppv(nep,1) = wpv(iwind,jwind,1)
                              ppv(nep,2) = wpv(iwind,jwind,2)
                              ppv(nep,3) = wpv(iwind,jwind,3)
                              ppv(nep,4) = thresh(iwind,jwind)
                              ppv(nep,5) = 1
                              nep = nep + 1
                              if(nep .gt. MaxEdge) then
                                 write(UnitLog,*) 'Exeeded number of',
     1                                ' allowable pixels: ', MaxEdge
                                 print *, 'Exeeded number of',
     1                                ' allowable pixels: ', MaxEdge
                                 stop 'STOPPING'
                              endif
                           else
                              onep = outpic(i,j)
                              ppv(onep,5) = ppv(onep,5)+1
                              if(ppv(onep,0) .gt. 
     +                         wpv(iwind,jwind,0))then
                                 ppv(nep,0) = wpv(iwind,jwind,0)
                                 ppv(nep,1) = wpv(iwind,jwind,1)
                                 ppv(nep,2) = wpv(iwind,jwind,2)
                                 ppv(nep,3) = wpv(iwind,jwind,3)
                                 ppv(nep,4) = thresh(iwind,jwind)
                              endif
                           endif
                        endif
                     else
 
c  Case when PICT(I,J) is a member of the cold population.
 
                        if(pict(i,j) .gt. 2)then
 
c  Check on neighbor to the right.
 
                          if(i .lt. LenXA )then
                              cond1 = (pict(i+1,j) .le. 
     +                          thresh(iwind,jwind))
                           else
                              cond1 = .true.
                           endif
 
c  Check on the neighbor above.
 
                           if(j .lt. LenYA )then
                              cond2 = (pict(i,j+1) .le. 
     +                          thresh(iwind,jwind))
                           else
                              cond2 = .true.
                           endif
 
c  Put the value of the threshold associated with the front in array 
c  OUTPIC when a front has been detected.
 
c  Put edge pixel number in array OUTPic if front has been detected.
c   See description for the warm pixel above.
 
                           if( .not. (cond1 .and. cond2))then
                              if(outpic(i,j) .eq. 0)then
                                 outpic(i,j) = nep
                                 ppv(nep,0) = wpv(iwind,jwind,0)
                                 ppv(nep,1) = wpv(iwind,jwind,1)
                                 ppv(nep,2) = wpv(iwind,jwind,2)
                                 ppv(nep,3) = wpv(iwind,jwind,3)
                                 ppv(nep,4) = thresh(iwind,jwind)
                                 ppv(nep,5) = 1
                                 nep = nep + 1
                                 if(nep .gt. MaxEdge) then
                                    write(UnitLog,*) 'Exeeded number ',
     1                                   ' of allowable pixels: ', 
     2                                   MaxEdge
                                    print *, 'Exeeded number of',
     1                                   ' allowable pixels: ', MaxEdge
                                    stop 'STOPPING'
                                 endif
                              else
                                 onep = outpic(i,j)
                                 ppv(onep,5) = ppv(onep,5)+1
                                 if(ppv(onep,0) .gt. 
     +                            wpv(iwind,jwind,0))then
                                    ppv(nep,0) = wpv(iwind,jwind,0)
                                    ppv(nep,1) = wpv(iwind,jwind,1)
                                    ppv(nep,2) = wpv(iwind,jwind,2)
                                    ppv(nep,3) = wpv(iwind,jwind,3)
                                    ppv(nep,4) = thresh(iwind,jwind)
                                 endif
                              endif
                           endif
                        endif
                     endif
23116             continue
23114          continue
            endif
23110    continue
23108 continue
      return
      end
 
c***********************************************************************
      subroutine gradient(inpict,gradpic,gradv) 
c***********************************************************************
      implicit none
 
c     Parameter statements

      include 'ParameterStatements.f'

c     General variables

      integer*2 inpict(1:LenXA,1:LenYA)
      integer*2 gradv(1:2,1:LenXA,1:LenYA)
      real*4 gradpic(1:LenXA,1:LenYA)
      real*4 RangeFloat
      integer*2 i,j
      integer*2 ilo,iup
      integer*2 jlo,jup
      real*4 grad
 
c Range              - The upper limit to use in the histograms. PCC 
c    added this variable. The numbers were harded coded in as 255 in 
c    the .rat version. The addition of Range was done to accommodate 
c    the fact that in MSG SV images they use hundreths of a degree C 
c    rather than 1/8 with a max of 255.
 
 
      do 23170 j=1,LenYA 
         do 23172 i=1,LenXA 
            ilo = max(1,i-1)
            jlo = max(1,j-1)
            iup = min( LenXA ,i+1)
            jup = min( LenYA ,j+1)
            gradv(1,i,j) = inpict(iup,j)-inpict(ilo,j)
            gradv(2,i,j) = inpict(i,jup)-inpict(i,jlo)
            grad = (real(gradv(1,i,j))**2)+(real(gradv(2,i,j))**2)
c Becareful here. I changed 255.0 to Range. Range is an integer.
            RangeFloat = Range
            gradpic(i,j) = min(sqrt(grad),RangeFloat)
23172    continue
23170 continue
 
      return
      end
      
c***********************************************************************
      subroutine follow_main( FrontsFileName, MedianFileName, file4, 
     1     outpic, filpic, ppv, SecondsSince1970InID, ProgName, 
     2     Lat_Array, Lon_Array)
c***********************************************************************
c     This subroutine allows for an image containing up to 20,000 
c     contours. For extremely large images, or for large model images 
c     (> 2048x2048), there may be 70,000+ contours.  In this case, 
c     array SEGLENGTH must be redimensioned, CONTOUR_NUMBER and array 
c     FRNTPIC must be reassigned to Integer*4, and the same changes 
c     must be made in all subroutines called with those as arguments: 
c     FOLLOW_CONTOUR, CHECK_CONTOUR, FIND_EDGE, FIND_EXTEND. Look for 
c     similar changes to be made in PMERGE
      
      use netcdf

      Implicit none
      
c     Parameter statements

      include 'ParameterStatements.f'

c*******Functions

c     GenerateFileName - a function that will generate a filename for
c     . one of the output file types given the sudirectory name and
c     . the ending of the filename. These change for each program
c     . output.
      character*319 GenerateFileName

c******LatLon common

c     LatLonFileName - the name of the file containing the latitud and
c     . longitude arrays.
      character*319 LatLonFileName
c     LatLonFileNameSave - the saved LatLonFileName used to see if the
c     . the filename is new for this SST field.
      character*319 LatLonFileNameSave

      common /LatLon/ LatLonFileName, LatLonFileNameSave

c******General variables

      integer*4 outlength

c     aLat - latitude at front pixel location
      real*4 alat
c     aLon - longitude at front pixel location
      real*4 alon

c     ccl - length of current contour
      integer*2 ccl
c$$$  c     CFPixels - the distance from the front segment to which the 
c$$$  c     the outer region extends; +/- this many pixels.
c$$$  integer CFPixels
      integer, parameter :: CFP=8
c     CFSST - this vector contains the cross-front SST pixels. 
      integer*2 CFSST(2*CFP+1)
c     CFBGEastGrad - background eastward Sobel gradient (background is 
c     5 or more pixels from the front.
      integer*2 CFBGEastGrad
c     CFBGNorthGrad - background northward Sobel gradient (background is
c     5 or more pixels from the front.
      integer*2 CFBGNorthGrad
c     CFEastGrad - eastward Sobel gradient cross-front array
      integer*2 CFEastGrad(2*CFP+1)
c     CFNorthgrad - eastward Sobel gradient cross-front array
      integer*2 CFNorthgrad(2*CFP+1)
c     CFEastGradMax - the maximum eastward Sobel gradient within 1 
c     pixel of the front (normal to the local front).
      integer*2 CFEastGradMax
c     CFNorthGradMax - the maximum northward Sobel gradient within 1 
c     pixel of the front (normal to the local front).
      integer*2 CFNorthGradMax
c     Contour_Number - contour number (negative)
      integer*4 contour_number

c     d - separation in pixels along the front.
      integer*2 d
c     DeltaX - separation in the 1st dimension of front points along a 
c     segment.
      real DeltaX
c     DeltaY - separation in the 2nd dimension of front points along a 
c     segment.
      real DeltaY

c     EastGrad - eastward Sobel gradient array read in.
      real, allocatable :: EastGrad(:,:)

c     Factor1 - factor used to determine cross-front gradients.
      real Factor1
c     Factor2 - factor used to determine cross-front gradients.
      real Factor2
c     FactorX - factor used to determine cross-front position.
      real FactorX
c     FactorY - factor used to determine cross-front position.
      real FactorY
c     FileExists - file status for a file tested by FileStatus, .false.,
c     . if the file exists, .true., otherwise
      logical FileExists
c     FilPic(1:SizOfImg,1:SizOfImg) - median-filtrd img w/ clouds
      integer*2 filpic(1:LenXA,1:LenYA)
c     FrontsFileName - the output filename for fronts - uncompressed
      character*319 FrontsFileName

c     GeoSobelFileName - name of the file with the Sobel gradients in 
c     geo coordinates.
      character*319 GeoSobelFileName
c     GradPic(1:SizOfImg,1:SizOfImg) - gradient image
      real*4, allocatable :: gradpic(:,:)
c     GradV(1:2,1:SizOfImg,1:SizOfImg) - gradient vector
      integer*2, allocatable :: gradv(:,:,:)

c     JFGradXout - x component of the SST gradient
      integer*4 JFGradXout
c     JFGradYout - y component of the SST gradient
      integer*4 JFGradYout

c     i - loop index along the front segment
      integer*2 i
c     iFrontPixel - the number of frontal pixels.
      integer*4 iFrontPixel
c     iFP - front pixel counter when writing front pixels to output file
      integer*4 iFP
c     InnerPixels - the distance from the front segment to which the 
c     the inner region extends; +/- this many pixels.
      integer, parameter :: InnerPixels = 1

c     Lat_Array - latitude values for image. Read in once if it does
c     not change or for each file if it does change.
      real*4 Lat_Array(1:LenX,1:LenY)
c     Lon_Array - longitude values for image. Read in once if it does
c     not change or for each file if it does change.
      real*4 Lon_Array(1:LenX,1:LenY)

c     MaxLat - the maximum latitude for a segment.
      real MaxLat
c     MaxLon - the maximum longitude for a segment.
      real MaxLon
c     MaxX - the maximum x locations for a segment.
      integer*4 MaxX
c     MaxY - the maximum x locations for a segment.
      integer*4 MaxY
c     MedianFileName - the name of the output file for the 3x3 median
c     . filter.
      character*319 MedianFileName
c     MedID - ID for the median sst field.
      integer MedID
c     MinLat - the minumum latitude for a segment.
      real MinLat
c     MinLon - the minumum longitude for a segment.
      real MinLon
c     MinX - the minumum x locations for a segment.
      integer*4 MinX
c     MinY - the minumum x locations for a segment.
      integer*4 MinY
c     MissingValueInt2 - the missing value passed if the value read in
c     is integer*2.
      real*4 MissingValueInt2
c     MissingValueInt4 - the missing value passed if the value read in
c     is integer*4.
      real*4 MissingValueInt4
c     MissingValueReal - the missing value passed if the value read in
c     is real.
      real*4 MissingValueReal

c     ncID - dummy variable for netCDF ID in call to ReadMergedThinned.
c     Not used since the file is closed. Must be left over.
      integer ncID
c     ncVar1ID - dummy variable for netCDF variable ID in call to 
c     ReadMergedThinned. Not used since the file is closed. Must be 
c     left over.
      integer ncVar1ID
c     ncVar2ID - dummy variable for netCDF variable ID in call to 
c     ReadMergedThinned. Not used since the file is closed. Must be 
c     left over.
      integer ncVar2ID
c     nGoodA - number of good SST pixels in population A.
      integer nGoodA
c     nGoodB - number of good SST pixels in population B.
      integer nGoodB
c     nGoodBGEastGrad - number of good gradient values used for eastward
c     bacground gradient calculation.
      integer nGoodBGEastGrad
c     nGoodBGNorthGrad - number of good gradient values used for 
c     norhtward bacground gradient calculation.
      integer nGoodBGNorthGrad
c     NorthGrad - eastward Sobel gradient.
      real, allocatable :: NorthGrad(:,:)

c     OutPic(1:SizOfImg,1:SizOfImg) - image of edges; changed to 
      integer*4 outpic(1:LenXA,1:LenYA)
c     OutPicSave is a copy of outpic before the values in it are 
c     replaced with contour numbers.
      integer*4, allocatable :: OutPicSave(:,:)

c     ppv - Cayula-Cornillon front defining characteristics
c     ppv(0)=WPV(0)        :global cohesion
c     ppv(1)=WPV(1)        :criterion function (theta)
c     ppv(2)=WPV(2)        :Mean of population A
c     ppv(3)=WPV(3)        :Mean of population B
c     ppv(4)=THRESH        :Temperature threshold
      real*4 ppv(1:MaxEdge,0:5)

c     SecondsSince1970InID - ID for NetCDF variable.
      integer SecondsSince1970InID
c     SegArraySizeX - the size of the 1st dimension of the segment array
      integer SegArraySizeX
c     SegArraySizeY - the size of the 2nd dimension of the segment array
      integer SegArraySizeY
c     SegEnd - the last pixel, a fill value for the current segment
      integer*4 SegEnd
c     SegLength - the length of found frontal segments (a.k.a. contour)
      integer*2, allocatable :: SegLength(:)
c     SegNoTest - test segment number.
      integer SegNoTest
c     SegStart - first good pixel in the current segment
      integer*4 SegStart
c     SobelFileName - name of the file with the Sobel gradients.
      character*319 SobelFileName
c     TempFileName - a dummy name in this case for the call to 
c     . FileStatus.
      character*319 TempFileName
c     titlength - file name length
      integer*4 titlength

c     VariableName - the character string name of a variable to be read
c     by GetAttributes.
      character*50 VariableName
c     VariableType - string with the variable name.
      character*12 VariableType

c$$$  c     GradFillValueI - the fill value for the x Sobel gradient read in
c$$$  real GradFillValueI
c$$$  c     GradOffsetI - the offset for the x Sobel gradient read in.
c$$$  real GradOffsetI
c$$$  c     GradScaleFactorI - the scale factor for the x Sobel gradient 
c$$$  c     read in.
c$$$  real GradScaleFactorI
c     GradOffsetO - the offset for the x Sobel gradient to be written
c     out.
      real GradOffsetO
c     GradScaleFactorO - the scale factor for the x Sobel gradient 
c     to be written out.
      real GradScaleFactorO

c     xLocation - the x location of a frontal pixel, the value of the 
c     first dimension in the SST array.
      integer*4, allocatable :: xLocation(:)

c     yLocation - the y location of a frontal pixel, the value of the 
c     second dimension in the SST array.
      integer*4, allocatable :: yLocation(:)

c     ZenithAngleFileName - name of the file with the solar zenith 
c     angles at the pixel location.
      character*319 ZenithAngleFileName

      integer*4, allocatable :: frntpic(:,:)
      integer*4 j,k,m,n,il,jl
      integer*2, allocatable :: tseq1(:,:)
      integer*2, allocatable :: tseq(:,:)
      integer*4 odirlength
      integer*1 read_flag
      real*8 dumhdate
      character*319 file4
      integer*2, allocatable :: pedge(:,:)
c     common /matrx/pedge
c     logical*1 file_ex
      character filtyp
      integer*4 DataRange, NAboveRange

      integer ncThinnedID, ThinnedID

c     FILE1              - Input file name containing the image being 
c     processed.
c     filename
c     TITLENGTH	       - file name length
c     Int*4 to allow for increased - edge pixels in model results
c     ALON, ALAT	       			- pixel coordinates
c     FRNTPIC(1:SizOfImg,1:SizOfImg)    - front image
c     I,J,K,M,N,IL,JL        - indexes
c     and it will be discarded.
c     TSEQ1(0:1000,0:1)      - Temp Seq array vector
c     TSEQ(0:1000,0:1)       - Temp Seq array vector
c     EOS			 - end-of-string character
c     read_flag		 - indicates which values should be read in 
c     Read_Image
c     dumhdate		 - dummy value for date
c     FILE4,FRONTSFILENAME             - (file names)
c     PEDGE(1:SizOfImg,1:SizOfImg)    - persistent edges
c     FILE_EX                             - cld mask exists?
c     FILTYP		       	      - file type indicator
c     Range           - The upper limit to use in the histograms. PCC 
c     added this variable. The numbers were harded coded in as 255 in 
c     the .rat version.
c     
c     Pixel Parameter vector: PPV
c     
c     
c     DataRange - maximum minus minimum value read in.
c     NAboveRange - number of input data values above Range.
c     status - used in test for file existence.
c     UnitOut - Unit to which output messages will be written for this
c     . run.
c     

c---------------------------------start here ---------------------------

      if(Debug .eq. 1) print *, 'Follow_Main #000'

      allocate( OutPicSave(1:LenXA,1:LenYA),
     2     gradpic(1:LenXA,1:LenYA),
     3     gradv(1:2,1:LenXA,1:LenYA),
     4     frntpic(1:LenXA,1:LenYA),
     5     segLength(0:MaxNumOfSegments),
     6     tseq1(0:MaxNumOfSegments,0:1),
     7     tseq(0:MaxNumOfSegments,0:1),
     1     pedge(1:LenXA,1:LenYA),
     2     xLocation(1:MaxEdge),
     3     yLocation(1:MaxEdge),
     8     EastGrad(1:LenX,1:LenY),
     9     NorthGrad(1:LenX,1:LenY),
     1     stat=ierr)
      
      if (ierr .ne. 0) then
         print *, 'Follow_Main #100: Allocation error for LenXA, ',
     1        'LenYA arrays. Exiting.'
         stop
      endif

c     Gradient image later used to extend contours.
      
      call gradient(filpic,gradpic,gradv)
      
      if (debug .eq. 1) then
         Msg50 = '--- sst after gradient --- EOS'
         call PrintArray2( filpic(istrt:iend,jstrt:jend), Msg50)
         Msg50 = '--- gradv i after gradient --- EOS'
         call PrintArray2( gradv(1,istrt:iend,jstrt:jend), Msg50)
         Msg50 = '--- gradv j after gradient --- EOS'
         call PrintArray2( gradv(2,istrt:iend,jstrt:jend), Msg50)
      endif

c     Set the gradient value at this location to fill value if any
c     of the graidents within +/- 2 pixels of this pixel is a fill
c     value.

      do 23174 i=1,LenXA 
         do 23176 j=1,LenYA 

            OutPicSave(i,j) = OutPic(i,j)

c     Changed the range on the variables from i-2 to i+2 ==> i-1 to i+1

            do 23178 il = max(i-1,1), min(i+1, LenXA) 
               do 23180 jl = max(j-1,1), min(j+1, LenYA) 
                  if(filpic(il,jl) .eq. FillValueInt2) then
                     gradpic(i,j) = FillValueReal
                     gradv(1,i,j) = FillValueInt2
                     gradv(2,i,j) = FillValueInt2
                  endif
23180          continue
23178       continue
23176    continue
23174 continue
      
      if (debug .eq. 1) then
         Msg50 = '--- gradv i after setting to nan --- EOS'
         call PrintArray2( gradv(1,istrt:iend,jstrt:jend), Msg50)
         Msg50 = '--- gradv j after setting to nan --- EOS'
         call PrintArray2( gradv(2,istrt:iend,jstrt:jend), Msg50)
      endif

      read_flag = 3
      
c     Does the file for thinned persistent contours exist? If so, 
c     read the contours

      inquire( File=trim(file4), Exist=FileExists)

      if(FileExists .eqv. .true.) then

         if (debug .eq. 1) print *, 'Follow_Main #140: Reading::', 
     1        trim(file4), '::'

         call ReadMergedThinned( file4, ncID, ncVar1ID, ncVar2ID, 
     1        pedge, dumhdate, 2)

c     Need to close the MergedThinned file. It is left open for use
c     . in the program thinned.

         status = nf90_close(ncID)
         if(status .ne. nf90_noerr) call handle_err(status)

c     The following code is a bit odd; il and i are the same as are
c     jl and j. This is left over from earlier code when a 3x3 pixel
c     window around i,j was examined. 

         do 23190 i=1,LenXA 
            do 23192 j=1,LenYA 
               if((outpic(i,j) .eq. 0) .and. (pedge(i,j) .eq. 4))then
                  outpic(i,j) = -1
               endif
23192       continue
23190    continue
      endif
      
c     End of loop reading in and dealing with thinned contours.
      
      do 23196 i=1,LenXA 
         do 23198 j=1,LenYA 
            frntpic(i,j) = 0
23198    continue
23196 continue
      
c     outpic contains previously found edges. The edges are coded, such
c     that a value of 0 means no edge, a value greater than 1 (edge 
c     pixel number) means that a contour is there. When a pixel 
c     indicating an edge is found Follow_Contour is called to follow 
c     and extend the contour. (Not sure why -1 isn't mentioned as an
c     option. Oh well.
      
      contour_number = 0
      SegLength(contour_number) = 0
      iFrontPixel = 0
c$$$  iContourNumber = 0

      if (debug .eq. 1) print *, 'Follow_Main #142: Starting main loop.'

      do 23200 i=1,LenXA 
         do 23202 j=1,LenYA 
            
c     If outpic(i,j) is not equal to zero, then this is a front pixel.
c     if frntpic(i,j) is equal to 0 and, the pixel is a front pixel, 
c     i.e., outpic(i,j) not equal to zero, then the contour containing 
c     this pixel has not yet been defined. Do so now.

            if((outpic(i,j) .ne. 0) .and. (frntpic(i,j) .eq. 0)) then
               
c     After processing, the value of the thresholds is replaced by the 
c     contour number. To indicate that the contour has been followed, 
c     the corresponding pixel in the front image is set to value of 
c     the  contour number. 
               
               contour_number = contour_number + 1
               ccl = 1
               tseq1(ccl,0) = i
               tseq1(ccl,1) = j
               frntpic(i,j) = contour_number
               
c     Follow first branch of contour
               
               call follow_contour(outpic,filpic,gradpic,gradv,
     +              tseq1,contour_number,ccl,frntpic)
               
c     If the number of points on this segment exceeds 100,000, stop
c     something must be wrong.

               if(ccl .gt. MaxNumofSegments) then
                  write(UnitLog,*) 'Follow_Main #150: ccl too large: ',
     1                 ccl
                  print *, 'Follow_Main #150: ccl too large: ', ccl
                  stop
               endif

c     Resequence tseq array.
               
               do 23206 k = 1,ccl 
                  tseq(k,0) = tseq1((ccl+1)-k,0)
                  tseq(k,1) = tseq1((ccl+1)-k,1)
23206          continue
               
c     Follow second branch of contour (if (i,j) in middle of contour)
               
               if(ccl .ge. 5)then
                  call follow_contour(outpic,filpic,gradpic,
     +                 gradv,tseq,contour_number,ccl,frntpic)
               endif
               
               if(contour_number .eq. 99500) then
                  write(UnitLog,*) 'Follow_Main #160: Contour_Number: ',
     1                 contour_number, ' ****** WARNING: Approaching ', 
     2                 'Length array limit ******'
                  print *, 'Follow_Main #160: Contour_Number: ',
     1                 contour_number, ' ****** WARNING: Approaching ', 
     2                 'Length array limit ******'
               else
                  if(contour_number .eq. MaxNumOfSegments+1) then
                     write(UnitLog,*) 'Follow_Main #170: ',
     1                    'Contour_Number: ', Contour_Number,
     2                    '  ****** WARNING: Length array limit ',
     3                    ' surpassed ******'
                     print *, 'Follow_Main #170: ',
     1                    'Contour_Number: ', Contour_Number,
     2                    '  ****** WARNING: Length array limit ',
     3                    ' surpassed ******'
                  endif
               endif

               SegLength(contour_number) = ccl
               
c     Write out contour.
               
c     Write out the frontal point in lat, lon to the .dim file 
c     (device 8). Write out the frontal point in image coordinates 
c     to the .cnt file  (device 9).
               
               if(ccl .ge. minlength) then
                  
c     if (debug .eq. 1) print *, 'Follow_Main #172: ', 
c     1              ' Processing contour number: ', contour_number
                  
c     Start each contour segment with a break, nans in the .dim file 
c     for Matlab and 0s in the .cnt file.
                  
                  iFrontPixel = iFrontPixel + 1

                  xLocation(iFrontPixel) = FillValueInt4
                  yLocation(iFrontPixel) = FillValueInt4

c$$$  xLocation(iFrontPixel) = 99
c$$$  yLocation(iFrontPixel) = 99

c     If the number of points on this segment exceeds 100,000, stop
c     something must be wrong.

                  if(ccl .gt. MaxNumOfSegments) then
                     write(UnitLog,*) 'Follow_Main #180: 2nd ccl in ',
     1                    'Follow-Main too large : ', ccl
                     print *,'Follow_Main #180: 2nd ccl in ',
     1                    'Follow-Main too large : ', ccl
                     stop
                  endif

c     Now loop over contour points
                  
                  do 23216 k = ccl,1,-1 

c     For some reason m is one larger than LenX. I don't think that this
c     should happen but for now I'll just check for it and set it equal
c     to LenX if larger and to 1 if less than 0. Same for n and LenY.

                     m = tseq(k,0)
                     n = tseq(k,1)

                     if(m .gt. LenX) then
                        m = LenX
                        write(UnitLog, *) 'Follow_Main #190. m (', m,
     1                       ') larger than LenX (', LenX, '). m set ',
     2                       'to LenX.'
                        print *, 'Follow_Main #190. m (', m,
     1                       ') larger than LenX (', LenX, '). m set ',
     2                       'to LenX.'
                     endif

                     if(n .gt. LenY) then
                        n = LenY
                        write(UnitLog, *) 'Follow_Main #200. n (', n,
     1                       ') larger than LenY (', LenY, '). n set ',
     2                       'to LenY.'
                        print *, 'Follow_Main #200. n (', n,
     1                       ') larger than LenY (', LenY, '). n set ',
     2                       'to LenY.'
                     endif

                     if(m .lt. 1) then
                        m = 1
                        write(UnitLog, *) 'Follow_Main #210. m (', m,
     1                       ') less than 1. It is being set to 1.'
                        print *, 'Follow_Main #210. m (', m,
     1                       ') less than 1. It is being set to 1.'
                     endif

                     if(n .lt. 1) then
                        n = 1
                        write(UnitLog, *) 'Follow_Main #220. n (', n,
     1                       ') less than 1. It is being set to 1.'
                        print *, 'Follow_Main #220. n (', n,
     1                       ') less than 1. It is being set to 1.'
                     endif

c     Now save the x and y location of this frontal pixel to output
c     later.

                     iFrontPixel = iFrontPixel + 1

                     xLocation(iFrontPixel) = m
                     yLocation(iFrontPixel) = n

23216             continue
                  
               else
                  
c     If the number of points on this segment exceeds 100,000, stop
c     something must be wrong.

                  If(ccl .gt. MaxNumOfSegments) then
                     write(UnitLog,*) 'Follow_Main # 230: 3rd ccl in',
     1                    'follow-main too large : ', ccl
                     print *, 'Follow_Main # 230: 3rd ccl in',
     1                    'follow-main too large : ', ccl
                     stop
                  endif

                  do 23218 k = 1,ccl 
                     frntpic(tseq(k,0),tseq(k,1)) = 0
23218             continue
                  
                  SegLength(contour_number) = 0
                  contour_number = contour_number-1
               endif
               
c     End of front contour segment write
               
            endif
            
c     End of loop over x values
23202    continue
         
c     End of loop over y values
23200 continue
      
c$$$  if (debug .eq. 1) then
c$$$  Msg50 = '--- filpic in follow_main #240 --- EOS'
c$$$  call PrintArray2( filpic(istrt:iend,jstrt:jend), Msg50)
c$$$  
c$$$  Msg50 = '--- frntpic in follow_main #240 --- EOS'
c$$$  call PrintArray4( frntpic(istrt:iend,jstrt:jend), Msg50)
c$$$  endif
      
c     Increment the front pixel counter and create a final fill value
c     x and y pixel location. 
      
      iFrontPixel = iFrontPixel + 1

      xLocation(iFrontPixel) = FillValueInt4
      yLocation(iFrontPixel) = FillValueInt4

      if(Debug .eq. 1) print *, 'Follow_Main #240: ',
     1     iFrontPixel, ' Front pixels created for this image.'

c******Now write out front locations and front characteristics

      if (debug .eq. 1) then
         Msg50 = '--- outpicsave before WriteFrontData --- EOS'
         call PrintArray4( outpicsave(istrt:iend,jstrt:jend), Msg50)
      endif

      call WriteFrontData( FrontsFileName, MedianFileName, OutPicSave, 
     1     ppv, FrntPic, FilPic, iFrontPixel, xLocation, yLocation, 
     2     SegLength, ProgName, Lat_Array, Lon_Array, gradv)

      if (debug .eq. 1) then
         Msg50 = '--- outpicsave after WriteFrontData --- EOS'
         call PrintArray4( outpicsave(istrt:iend,jstrt:jend), Msg50)
      endif

c     Contours shorter than a certain value (defined by the user) are
c     deleted (or at least marked for deletion).
c     
c     do 23220 i=1,LenXA 
c     do 23222 j=1,LenYA 
c     contour_number = frntpic(i,j)
c     if(contour_number .ne. 0) then
c     if(SegLength(contour_number) .lt. minlength) then
c     if(Debug .eq. 1) print *, 'Follow_Main #250: ***** ',
c     1                 'Setting a contour value to zero. This means ',
c     2                 ' that the contour will not appear in the ',
c     3                 ' fronts field in the Median SST image. *****'
c     outpic(i,j) = 0
c     else
c     outpic(i,j) = Range
c     endif
c     else
c     outpic(i,j) = 0
c     endif
c     23222    continue
c     23220 continue

c     Close and compress output files. 

c     Done with the SST array, so load the fronts array into the SST
c     array location to write out to the median filtered image.

      do 23230 i=1,LenXA 
         do 23232 j=1,LenYA 
            if(frntpic(i,j) .gt. 0) then
               OutPic(i,j) = 1
            else
               OutPic(i,j) = 0
            endif
23232    continue
23230 continue

      if(debug .eq. 1) print *, 'Follow_Main #999'

      return
      end
 
c***********************************************************************
      subroutine follow_contour(outpic,filpic,gradpic,gradv,tseq,
     +     cont_number,SegLength,frntpic)
c***********************************************************************
      implicit none
 
c     Parameter statements

      include 'ParameterStatements.f'

c     General variables

      integer*4 outpic(1:LenXA,1:LenYA)
      integer*2 filpic(1:LenXA,1:LenYA)
      real*4 gradpic(1:LenXA,1:LenYA)
      integer*2 gradv(1:2,1:LenXA,1:LenYA)
      integer*4 frntpic(1:LenXA,1:LenYA)
      integer*2 ipres,jpres
      integer*2 inext,jnext
      integer*2 SegLength
      integer*4 cont_number
      logical*1 contour_end
      integer*2 count_new
      integer*2 mingrad
      real*4 ivecta,jvecta
      real*4 norma
      real*4 maxscal
      integer*2 zero_count
      integer*2 cont_count2
      integer*2 tseq(0:MaxNumOfSegments,0:1)
      real*4 ratio_grad

c---------------------------------start here ---------------------------

      mingrad = 2
      contour_end = .false.
      if(SegLength .gt. MaxNumOfSegments) then
         write(UnitLog,*) 'SegLength in follow-contour too',
     2        ' large : ', SegLength
         print *, 'SegLength in follow-contour too',
     2        ' large : ', SegLength
         stop
      endif
      ipres = tseq(SegLength,0)
      jpres = tseq(SegLength,1)
      count_new = 0
      cont_count2 = 0
      zero_count = 0
23228 if( .not. contour_end)then
         call check_cloud(filpic,ipres,jpres,zero_count,contour_end)
         call check_pcont(outpic,ipres,jpres,contour_end)
         call check_contour(frntpic,ipres,jpres,cont_number, 
     +     cont_count2,contour_end)
         call grad_stdy(gradpic,gradv,ipres,jpres,ratio_grad)
         if( .not. (contour_end))then
            contour_end = .true.
            ivecta = ipres-tseq(max(SegLength-15,1),0)
            jvecta = jpres-tseq(max(SegLength-15,1),1)
            norma = sqrt(ivecta**2+jvecta**2)
            if(norma .ne. 0)then
               ivecta = ivecta/norma
               jvecta = jvecta/norma
            endif
            if(SegLength .le. 2)then
               maxscal = -10
            else
               maxscal = 0
            endif
            call find_edge(frntpic,outpic,filpic,ipres,jpres,
     +        ivecta,jvecta, mingrad,maxscal,contour_end,inext,jnext)
            if( .not. contour_end)then
               count_new = 0
               if(ratio_grad .gt. 0.90)then
                  call find_extend(gradpic,gradv,frntpic, ipres,
     +              jpres,inext,jnext,ivecta,jvecta, contour_end,
     +              mingrad,maxscal)
                  contour_end = .false.
               endif
            endif
            if((SegLength .ge. 5) .and. (ratio_grad .gt. 0.90))then
               if((contour_end) .and. (count_new .le. 50))then
                  maxscal = 0.1
                  call find_extend(gradpic,gradv,frntpic, ipres,
     +              jpres,inext,jnext,ivecta,jvecta, contour_end,
     +              mingrad,maxscal)
                  if( .not. contour_end)then
                     count_new = count_new+1
                  endif
               endif
            endif
         endif
         if( .not. contour_end)then
            SegLength = SegLength+1

            if(SegLength .gt. MaxNumOfSegments) then
               write(UnitLog,*) '2nd SegLength in follow-contour too',
     2              ' large : ', SegLength
               print *, '2nd SegLength in follow-contour too large: ',
     2               SegLength
               stop
            endif

            tseq(SegLength,0) = inext
            tseq(SegLength,1) = jnext
            frntpic(inext,jnext) = cont_number
            ipres = inext
            jpres = jnext
         endif
         goto 23228
      endif
      return
      end
 
c***********************************************************************
      subroutine grad_stdy(gradpic,gradv,i,j,ratio_grad)
c***********************************************************************
      implicit none
 
c     Parameter statements

      include 'ParameterStatements.f'

c     General variables

      real*4 gradpic(1:LenXA,1:LenYA)
      integer*2 gradv(1:2,1:LenXA,1:LenYA)
      integer*2 i,j
      integer*2 il,jl
      real*4 sgv(1:2)
      real*4 sgm
      real*4 ratio_grad
      real*4 norm

      sgv(1) = 0
      sgv(2) = 0
      sgm = 0
      do 23248 il = max(1,i-1),min( LenXA,i+1) 
         do 23250 jl = max(1,j-1),min( LenYA,j+1) 
            sgv(1) = sgv(1)+gradv(1,il,jl)
            sgv(2) = sgv(2)+gradv(2,il,jl)
            sgm = sgm+gradpic(il,jl)
23250    continue
23248 continue
      norm = (sgv(1)*sgv(1))+(sgv(2)*sgv(2))
      if(sgm .ne. 0)then
         ratio_grad = sqrt(norm)/sgm
      else
         ratio_grad = 0
      endif
      return
      end
 
c***********************************************************************
      subroutine find_extend(gradpic,gradv,frntpic, ipres,jpres,
     +     inext,jnext,ivecta,jvecta, contour_end,mingrad,
     +     maxscal_fixed)
c***********************************************************************
      implicit none
 
c     Parameter statements

      include 'ParameterStatements.f'

c     General variables

      real*4 gradpic(1:LenXA,1:LenYA)
      integer*2 gradv(1:2,1:LenXA,1:LenYA)
      integer*4 frntpic(1:LenXA,1:LenYA)
      integer*2 ipres,jpres
      real*4 ivecta,jvecta
      real*4 ivectb,jvectb
      real*4 normb
      integer*2 i,j
      integer*2 inext,jnext
      logical*1 contour_end
      real*4 maxscal
      real*4 maxscal_fixed
      real*4 scalar
      real*4 grad_scal
      real*4 max_grad
      integer*2 mingrad

      maxscal_fixed = 0.1
      max_grad = 0.1
      do 23254 i = max(1,ipres-1),min( LenXA,ipres+1) 
         do 23256 j = max(1,jpres-1),min( LenYA,jpres+1) 
            if(frntpic(i,j) .eq. 0)then
               if(gradpic(i,j) .ge. mingrad)then
                  if((i .ne. ipres) .or. (j .ne. jpres))then
                     ivectb = i-ipres
                     jvectb = j-jpres
                     scalar = (ivecta*ivectb)+(jvecta*jvectb)
                     normb = (ivectb*ivectb)+(jvectb*jvectb)
                     scalar = scalar/sqrt(normb)
                     if(scalar .ge. maxscal_fixed)then
                        grad_scal = real(gradv(1,ipres,jpres))*
     +                    gradv(1,i,j)
                        grad_scal = grad_scal+ 
     +                    (real(gradv(2,ipres,jpres))*gradv(2,i,j))
                        if(grad_scal .gt. max_grad)then
                           max_grad = grad_scal
                           maxscal = scalar
                           inext = i
                           jnext = j
                           contour_end = .false.
                        else
                           if(grad_scal .eq. max_grad)then
                              if(scalar .gt. maxscal)then
                                 maxscal = scalar
                                 inext = i
                                 jnext = j
                                 contour_end = .false.
                              endif
                           endif
                        endif
                     endif
                  endif
               endif
            endif
23256    continue
23254 continue
      return
      end
 
c***********************************************************************
      subroutine find_edge(frntpic,outpic,filpic,ipres,jpres,ivecta,
     +     jvecta, mingrad,maxscal,contour_end,inext,jnext) 
c***********************************************************************
      implicit none
 
c     Parameter statements

      include 'ParameterStatements.f'

c     General variables

      integer*4 frntpic(1:LenXA,1:LenYA)
      integer*4 outpic(1:LenXA,1:LenYA)
      integer*2 filpic(1:LenXA,1:LenYA)
      integer*2 ipres,jpres
      integer*2 inext,jnext
      integer*2 i,j
      integer*2 differ
      integer*2 mindif
      logical*1 contour_end
      integer*2 mingrad
      real*4 ivecta,jvecta
      real*4 ivectb,jvectb
      real*4 normb
      real*4 maxscal
      real*4 scalar
 
c Range              - The upper limit to use in the histograms. PCC 
c    added this variable. The numbers were harded coded in as 255 in 
c    the .rat version.
 
      mindif = Range
      do 23272 i = max(1,ipres-1),min( LenXA,ipres+1) 
         do 23274 j = max(1,jpres-1),min( LenYA,jpres+1) 
            if((frntpic(i,j) .eq. 0) .and. (outpic(i,j) .ne. 0))then
               if((i .ne. ipres) .or. (j .ne. jpres))then
                  ivectb = i-ipres
                  jvectb = j-jpres
                  scalar = (ivecta*ivectb)+(jvecta*jvectb)
                  normb = (ivectb*ivectb)+(jvectb*jvectb)
                  scalar = scalar/sqrt(normb)
                  differ = filpic(i,j)-filpic(ipres,jpres)
                  differ = abs(differ)
                  if(maxscal .lt. scalar)then
                     mindif = differ
                     maxscal = scalar
                     inext = i
                     jnext = j
                     contour_end = .false.
                  else
                     if(maxscal .eq. scalar)then
                        differ = filpic(i,j)-filpic(ipres,jpres)
                        differ = abs(differ)
                        if(mindif .gt. differ)then
                           mindif = differ
                           inext = i
                           jnext = j
                        endif
                     endif
                  endif
               endif
            endif
23274    continue
23272 continue
      return
      end
 
c***********************************************************************
      subroutine check_cloud(filpic,ipres,jpres,zero_count,contour_end)
c***********************************************************************
      implicit none
 
c     Parameter statements

      include 'ParameterStatements.f'

c     General variables

      integer*2 filpic(1:LenXA,1:LenYA)
      integer*2 ipres,jpres
      integer*2 i,j
      logical*1 contour_end
      integer*2 zero_count

      zero_count = max(zero_count-1,0)
      do 23286 i = max(1,ipres-1),min( LenXA,ipres+1) 
         do 23288 j = max(1,jpres-1),min( LenYA,jpres+1) 
            if(filpic(i,j) .le. 1)then
               zero_count = zero_count+1
            endif
23288    continue
23286 continue
      if(zero_count .ge. 4)then
         contour_end = .true.
      endif
      return
      end
 
c***********************************************************************
      subroutine check_pcont(outpic,ipres,jpres,contour_end)
c***********************************************************************
      implicit none
 
c     Parameter statements

      include 'ParameterStatements.f'

c     General variables
 
      integer*4 outpic(1:LenXA,1:LenYA)
      integer*2 ipres,jpres
      integer*2 i,j
      logical*1 contour_end
      integer*2 pcount

      pcount = 0
      do 23294 i = max(1,ipres-1),min( LenXA,ipres+1) 
         do 23296 j = max(1,jpres-1),min( LenYA,jpres+1) 
            if(outpic(i,j) .eq. -1)then
               pcount = pcount+1
            endif
23296    continue
23294 continue
      if(pcount .ge. 1)then
         contour_end = .false.
      endif
      return
      end
 
c***********************************************************************
      subroutine check_contour(frntpic,ipres,jpres,cont_number, 
     +     cont_count2,contour_end)
c***********************************************************************
      implicit none
 
c     Parameter statements

      include 'ParameterStatements.f'

c     General variables

      integer*4 frntpic(1:LenXA,1:LenYA)
      integer*2 ipres,jpres
      integer*2 i,j
      integer*4 cont_number
      logical*1 contour_end
      integer*2 cont_count1subroutine
      integer*2 cont_count2

      cont_count1 = 0
      cont_count2 = max(cont_count2-1,0)
      do 23302 i = max(1,ipres-1),min( LenXA,ipres+1) 
         do 23304 j = max(1,jpres-1),min( LenYA,jpres+1) 
            if(frntpic(i,j) .ge. 1)then
               if(frntpic(i,j) .ne. cont_number)then
                  cont_count2 = cont_count2+1
               else
                  cont_count1 = cont_count1+1
               endif
            endif
23304    continue
23302 continue
      if((cont_count1 .ge. 5) .or. (cont_count2 .ge. 4))then
         contour_end = .true.
      endif
      return
      end
      
c**********************************************************************
      subroutine CreateNetCDFDimFile( FrontsFileName, MedianFileName,
     1     GeoSobelFileName, GradOffsetO, GradScaleFactorO,
     2     ZenithAngleFileName, ProgName)
c**********************************************************************
c     
c     This subroutine creates the .nc file for front pixel information.
c     
c     Written by Peter Cornillon, University of Rhode Island,
c     pcornillon@me.com 18 April 2009
c     
c**********Parameters
c     
c     Debug - 1 for all debug output, 2 for timing only, 0 for none.
c     UnitLog - the unit for the run log file.
c     
c**********Subroutine arguments
c     
c     FrontsFileName - the name of the file to create
c     status - return status after a netCDF call
c     ncMedID and ncSobelID - IDs for the input and output files.
c     MedID - ID for the sst variable will copy these attributes. 
c     SecondsSince1970ID - the ID of the number of seconds since  
c     . 00:00 1 Jan 1970 corresponding to the time of the sst array.
c     xGradID - ID for the x component of the Sobel gradient
c     yGradID - ID for the y component of the Sobel gradient
c     MagGradID - ID for the magnitude of the Sobel gradient
c     
c**********Other variables
c     
c     SecondsSince1970OutID - the ID of the number of seconds since  
c     . 0 1 Jan 1970 corresponding to the time of the output arrays.
c     SecondsSince1970 - the number of seconds since 00:00 of 1 Jan 
c     . 1970 corresponding to the time of the sst array. This value 
c     . will be read from the input file and writte to the output file.
c     
c     nxDimID and nyDimID - the IDs for the x and y dimension 
c     .variables.
c     DimsOut - the vector for the IDs of the dimensions.
c     
      use netcdf

      implicit none

c     Parameter statements

      include 'ParameterStatements.f'

c     Functions

      character*4 CommonSubsVersionNo

c     General variables

c     AttDescription - attribute description when needed.
      character*1000 AttDescription

c     chunksz_cf - chunksize array for cross-front records.
      integer chunksz_cf(2)
c     chunksz_records - chunksize array for front records.
      integer chunksz_records(1)
c     chunksz_segments - chunksize array for front segments.
      integer chunksz_segments(1)
c     chunksz_windows - chunksize array for SIED windows.
      integer chunksz_windows(2)
c     CloseFlag - .true. to close the file from which attributes are to
c     be read on exit, otherwise .false.
      logical CloseFlag

c     GeoSobelExists -  file status for GeoSobel, .true. if exists.
      logical GeoSobelExists
c     GeoSobelFileName - name of the file with the Sobel gradients in 
c     geo coordinates.
      character*319 GeoSobelFileName
c$$$c     GradFillValue - the fill value for the Sobel gradient magnitude.
c$$$      real GradFillValue
c     GradOffsetO - the offset to add to the output values when  
c     calculating the Sobel gradient.
      real GradOffsetO
c     GradScaleFactorO - the amount to multiply the output values by  
c     when calculating the Sobel gradient.
      real GradScaleFactorO

c     FrontsFileName - the output filename for fronts - uncompressed
      character*319 FrontsFileName

      character*67 Source

      character*150 OutputDimFileName

c     MedianFileName - the name of the output file for the 3x3 median
c     . filter.
      character*319 MedianFileName
c     MedID - ID for the median sst field.
      integer MedID
c     MissingValueInt2 - the missing value passed if the value read in
c     is integer*2.
      integer*2 MissingValueInt2
c     MissingValueInt4 - the missing value passed if the value read in
c     is integer*4.
      integer*4 MissingValueInt4
c     MissingValueReal - the missing value passed if the value read in
c     is real.
      real*4 MissingValueReal
c     MoreHistory - history for GeoSobel and Zenith Angle.
      character*400 MoreHistory

c     NCFDims - number of dimesions for cross-front variables
      integer NCFDims
c     ncMedID - NetCDF ID for the Median file.
      integer ncMedID
c     ncGeoSobelID - NetCDF ID for the GeoSobel file.
      integer ncGeoSobelID
c     ncSobelID - NetCDF ID for the GeoSobel file.
      integer ncSobelID
c     ncZenithAngleID - NetCDF ID for the GeoSobel file.
      integer ncZenithAngleID
c     NewTitle - the portion of the title for this file to prepend to
c     the Median file title.
      character*100 NewTitle

c     PlaneRemovalText - used for text describing removal of a plane
c      from the 48x48 pixel window centered on the window of interest
c      if a plane was removed. This is added to the end of the summary
c      written out to each netCDF SIED file.
      character*500 PlaneRemovalText 
c     PreviousProcessingHistory - history written to Fronts file in 
c     previous steps.
      character*2000 PreviousProcessingHistory
c     ProcessingHistory - augmented history to output to Fronts file.
      character*2000 ProcessingHistory
c     ProcessingProgram - the name of this program.
      character*100 ProcessingProgram

c     RecCFDims1 - array for the 1-dimension IDs for cross-front 
c     variables.
      integer RecCFDims1(1)
c     RecCFDims2 - array for the 2-dimension IDs for cross-front 
c     variables.
      integer RecCFDims2(2)
c     RecDims - length of dimensions (only one) for front pixels.
      integer RecDims(1)

c     SecondsSince1970InID - netCDF ID for time in the input median file
      integer SecondsSince1970InID
c     SecondsSince1970OutID - netCDF ID for time in the output fronts 
c     file.
      integer SecondsSince1970OutID
c     SegDims - length of front segment dimensions (only one).
      integer SegDims(1)
c     SobelExists -  file status for Sobel, .true. if exists.
      logical SobelExists
c     SobelFileName - name of the file with the Sobel gradients.
      character*319 SobelFileName
c$$$c     Summary - summary description of this file.
c$$$      character*4000 Summary

c     VariableName - the character string name of a variable to be read
c     by GetAttributes.
      character*50 VariableName
c     VariableType - string with the variable name.
      character*12 VariableType

c     WinDim1 - the number of SIED windows in the 1st dimension
      integer WinDim1
c     WinDim2 - the number of SIED windows in the 2nd dimension
      integer WinDim2
c     WinDimIID - netCDF ID for the 1st (i) dimension of the output
c     . cohesion and number clear fields.
      integer WinDimIID
c     WinDimJID - netCDF ID for the 2nd (j) dimension of the output
c     . cohesion and number clear fields.
      integer WinDimJID
c     WinDims - dimensions for netCDF file
      integer WinDims(2)

c     ZenithAngleExists -  file status for zenith angle.
      logical ZenithAngleExists
c     ZenithAngleFileName - name of the file with the solar zenith 
c     angles at the pixel location.
      character*319 ZenithAngleFileName
c     ZenithAngleText - used for the text in the summary for the file.
c      Changes if there is no zenith file.
      character*319 ZenithAngleText

      real*8 SecondsSince1970

      integer nxDimIDout, nyDimIDout
      integer DimsOut(2)

      real*4 dV(8)
      integer header, i
      character cr

!     Dim2 Header Record #1 Variables [Lines starting with #91]

      real*4 numFrontPixels
      real*4 sstA, sstB, sstMin
      real*4 sstSepValue

!     Dim2 Frontal Point Variables (Records starting with #10)

      real*4 latitude, longitude
      real*4 solarZenithAng

      character*150 FileNameIn, FileNameOut
      character*150 TempFile

c     Variables used to determine the current date and time

      integer Values(8)
      character*5 Zone
      character*10 Time
      character*8 Date
      character*19 DateTime

      include 'netcdf.inc'

c---------------------------------start here ---------------------------

      if(debug .eq. 1) print *, 'CreateNetCDFDimFile #000'

      if(debug .eq. 1) print *, 'CreateNetCDFDimFile #10: ',
     1     'FrontsFileName::', trim(FrontsFileName), '::'

c     Start by getting SST offset and scale factor attributes.

      VariableName = BlankFill(:len(VariableName))
      VariableName = 'median_' // trim(ParameterName_lc)

      VariableType = BlankFill(:len(VariableType))
      VariableType = 'integer*2'

c$$$      call GetAttributes( MedianFileName, VariableName, 
c$$$     1     VariableType, SSTOffset, SSTScaleFactor,
c$$$     2     MissingValueReal, MissingValueInt2, MissingValueInt4)
      call GetSSTAttributes(MedianFileName)

c     Now get the needed attributes for the zenith angle

      VariableName = BlankFill(:len(VariableName))
      VariableName = 'zenith_angle'

      VariableType = BlankFill(:len(VariableType))
      VariableType = 'real'

c   Test for the zenith angle file. If it does not exist, do not try
c     to open.

      inquire( File=trim(ZenithAngleFileName), Exist=ZenithAngleExists)

      if(ZenithAngleExists .eqv. .true.) then
         call GetZenithAngleAttributes(ZenithAngleFileName)
      endif

c     Create the Fronts file.

      chunksz_records(1) = 250
      chunksz_segments(1) = 25

      chunksz_cf(1) = 250
      chunksz_cf(2) = 17

      if(debug .eq. 1) print *, 'CreateNetCDFDimFile #110: ',
     1     'NetCDF Dim filename to be created::', 
     2     trim(FrontsFileName), '::'
      
      status = nf90_create( FrontsFileName, 
     1     OR(nf90_netcdf4, nf90_noclobber),  dim2ID)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Create the dimensions for the output front data.

      status = nf90_def_dim( dim2ID, 'segment', NF90_UNLIMITED, 
     1     SegDimID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_dim( dim2ID, 'record', NF90_UNLIMITED, 
     1     RecDimID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_dim( dim2ID, 'cross_front_record', 
     1     NF90_UNLIMITED, CFRecDimID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_dim( dim2ID, 'cross_front', 2*CFPixels+1, 
     1     CFSSTDimID)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Now for the dimensions for cohesion and number clear arrays.

      WinDim1 = LenXA / float(WinSize) * 2
      WinDim2 = LenYA / float(WinSize) * 2

      status = nf90_def_dim( dim2ID, 'SIED_window_i', WinDim1, 
     1     WinDimIID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_dim( dim2ID, 'SIED_window_j', WinDim2, 
     1     WinDimJID)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Populate dimension ID arrays. Front segments first:

      SegDims(1) = SegDimID

c     Now for front pixels.

      RecDims(1) = RecDimID

c     Cross-front arrays.

      RecCFDims1(1) = CFRecDimID

      RecCFDims2(1) = CFRecDimID
      RecCFDims2(2) = CFSSTDimID

c     SIED window arrays

      WinDims(1) = WinDimIID
      WinDims(2) = WinDimJID

c--------------------Define segments -----------------------------------

      if(Debug .eq. 1) print *, 'CreateNetCDFDimFile #120: '

c     Segment start - index for the start of this segment. The start 
c     location is the first good pixel value for the segment; i.e.,
c     skip the nan preceeding.

      status = nf90_def_var( dim2ID, 'segment_start', NF90_INT, 
     1     SegDims, SegStartID)
      if(status .ne. nf90_noerr) call handle_err(status)
      
      status = nf90_put_att( dim2ID, SegStartID, 'long_name',
     1     'first_pixel_of_front_segment' )

      status = nf90_def_var_chunking( dim2ID, SegStartID,
     1     NF90_CHUNKED, chunksz_segments)
      if(status .ne. nf90_noerr) call handle_err(status)

c     record length - number of good pixels in this front segment.

      status = nf90_def_var( dim2ID, 'segment_length', NF90_SHORT, 
     1     SegDims, SegLengthID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, SegLengthID, 'long_name',
     1     'number_of_pixels_in_front_segment' )

      status = nf90_def_var_chunking( dim2ID, SegLengthID,
     1     NF90_CHUNKED, chunksz_segments)
      if(status .ne. nf90_noerr) call handle_err(status)

c---  
c     Define min and max variables for latitude and longitude.
      if(SIED_Full) then  
         call DefineMinMaxLatLon(chunksz_segments, SegDims)
      endif
c     Define min and max variables for x and y location.

c$$$  call DefineMinMaxXY(chunksz_segments, SegDims)

c----------------------Define front pixels  ---------------------------

c     Latitude

      status = nf90_def_var( dim2ID, 'latitude', NF90_FLOAT, 
     1     RecDims, latRecID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_chunking( dim2ID, latRecID,
     1     NF90_CHUNKED, chunksz_records)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, latRecID, 'long_name', 'latitude')
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, latRecID, 'units', 'degrees')
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, latRecID, '_FillValue', 
     1     FillValueReal)
      if(status .ne. nf90_noerr) call handle_err(status)


c     Longitude
      
      status = nf90_def_var( dim2ID, 'longitude', NF90_FLOAT, 
     1     RecDims, lonRecID)
      if(status .ne. nf90_noerr) call handle_err(status)
      
      status = nf90_def_var_chunking( dim2ID, lonRecID,
     1     NF90_CHUNKED, chunksz_records)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, lonRecID, 'long_name', 'longitude')
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, lonRecID, 'units', 'degrees')
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, lonRecID, '_FillValue', 
     1     FillValueReal)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'CreateNetCDFDimFile #140: '

c     x location in the input array - 1st dimension.

      status = nf90_def_var( dim2ID, 'i', NF90_INT, 
     1     RecDims, xLocID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_chunking( dim2ID, xLocID,
     1     NF90_CHUNKED, chunksz_records)
      if(status .ne. nf90_noerr) call handle_err(status)

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = trim(iVarName) // 
     1     '_location_in_the_original_image'
      status = nf90_put_att( dim2ID, xLocID, 
     1     'long_name', AttDescription)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, xLocID, '_FillValue', 
     1     FillValueInt4)
      if(status .ne. nf90_noerr) call handle_err(status)

c     y location in the input array - 2nd dimension.

      status = nf90_def_var( dim2ID, 'j', NF90_INT, 
     1     RecDims, yLocID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_chunking( dim2ID, yLocID,
     1     NF90_CHUNKED, chunksz_records)
      if(status .ne. nf90_noerr) call handle_err(status)

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = trim(jVarName) // 
     1     '_location_in_the_original_image'
      status = nf90_put_att( dim2ID, yLocID, 
     1     'long_name', AttDescription)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, yLocID, '_FillValue', 
     1     FillValueInt4)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'CreateNetCDFDimFile #150: '

c     Cross-front SST.
      if(SIED_Full) then
         AttDescription = BlankFill(:len(AttDescription))
         AttDescription = 'cross_front_' // trim(ParameterName_lc)
         status = nf90_def_var( dim2ID, AttDescription, NF90_SHORT, 
     1        RecCFDims2, SSTcfID)
         if(status .ne. nf90_noerr) call handle_err(status)
         
         status = nf90_def_var_chunking( dim2ID, SSTcfID,
     1        NF90_CHUNKED, chunksz_cf)
         if(status .ne. nf90_noerr) call handle_err(status)
         
         status = nf90_put_att( dim2ID, SSTcfID, 
     1        'long_name','cross_front_sea_surface_temperature_profile')
         if(status .ne. nf90_noerr) call handle_err(status)
         
         status = nf90_put_att( dim2ID, SSTcfID, 'add_offset',  
     1        sstOffsetIn)
         if(status .ne. nf90_noerr) call handle_err(status)
         
         status = nf90_put_att( dim2ID, SSTcfID, 'scale_factor', 
     1        sstScaleFactorIn)
         if(status .ne. nf90_noerr) call handle_err(status)
         
         status = nf90_put_att( dim2ID, SSTcfID, '_FillValue', 
     1        FillValueInt2)
         if(status .ne. nf90_noerr) call handle_err(status)
         
         status = nf90_put_att( dim2ID, SSTcfID, 'units', 
     1        trim(ParameterUnits))
         if(status .ne. nf90_noerr) call handle_err(status)
      endif
      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = 'The program finds the approximate normal '
     1     // 'to the front. The normal is determined at each '
     2     // 'front location using the preceeding and following two '
     3     // 'pixels. (It uses the 5 end pixels for the 1st and 2nd '
     4     // 'pixels and the last and 2nd to last pixels on the '
     5     // 'segment.) It then locates the nearest pixel, 1, 2... 8 '
     6     // 'pixels from the front that are nearest this line. '
     7     // 'These pixel locations are available in the variables '
     8     // trim(iVarName) // '_pixel_location and ' // trim(jVarName)
     9     // '_pixel_location.'
      if(SIED_Full) then
         status = nf90_put_att( dim2ID, SSTcfID, 'Description', 
     1        trim(AttDescription))
         if(status .ne. nf90_noerr) call handle_err(status)
      endif
c     central difference gradient of the 1st dimension.
      if(SIED_Full) then
         AttDescription = BlankFill(:len(AttDescription))
         AttDescription = trim(iVarName) // '_gradient'
         status = nf90_def_var( dim2ID, AttDescription, NF90_SHORT, 
     1        RecDims, iGradID)
         if(status .ne. nf90_noerr) call handle_err(status)
         
         status = nf90_def_var_chunking( dim2ID, iGradID,
     1        NF90_CHUNKED, chunksz_records)
         if(status .ne. nf90_noerr) call handle_err(status)
         
         AttDescription = BlankFill(:len(AttDescription))
         AttDescription = trim(iVarName) // '_central_difference'
     1        // '_gradient'
         status = nf90_put_att( dim2ID, iGradID, 'long_name', 
     1        AttDescription)
         if(status .ne. nf90_noerr) call handle_err(status)
         
         status = nf90_put_att( dim2ID, iGradID, 'add_offset',  0)
         if(status .ne. nf90_noerr) call handle_err(status)
         
         status = nf90_put_att( dim2ID, iGradID, 'scale_factor', 
     1        sstScaleFactorIn)
         if(status .ne. nf90_noerr) call handle_err(status)
         
         AttDescription = BlankFill(:len(AttDescription))
         AttDescription = trim(ParameterUnits) // ' per (pixel*2)'
         status = nf90_put_att( dim2ID, iGradID, 'units',AttDescription)
         if(status .ne. nf90_noerr) call handle_err(status)
         
         status = nf90_put_att( dim2ID, iGradID, '_FillValue', 
     1        FillValueInt2)
         if(status .ne. nf90_noerr) call handle_err(status)
         
c     central difference gradient of the 2nd dimension.
         
         AttDescription = BlankFill(:len(AttDescription))
         AttDescription = trim(jVarName) // '_gradient'
         status = nf90_def_var( dim2ID, AttDescription, NF90_SHORT, 
     1        RecDims, jGradID)
         if(status .ne. nf90_noerr) call handle_err(status)
         
         status = nf90_def_var_chunking( dim2ID, jGradID,
     1        NF90_CHUNKED, chunksz_records)
         if(status .ne. nf90_noerr) call handle_err(status)
         
         AttDescription = BlankFill(:len(AttDescription))
         AttDescription = trim(jVarName) // '_central_difference'
     1        // '_gradient'
         status = nf90_put_att( dim2ID, jGradID, 'long_name', 
     1        AttDescription)
         if(status .ne. nf90_noerr) call handle_err(status)
         
         status = nf90_put_att( dim2ID, jGradID, 'add_offset', 0)
         if(status .ne. nf90_noerr) call handle_err(status)
         
         status = nf90_put_att( dim2ID, jGradID, 'scale_factor', 
     1        sstScaleFactorIn)
         if(status .ne. nf90_noerr) call handle_err(status)
         
         AttDescription = BlankFill(:len(AttDescription))
         AttDescription = trim(ParameterUnits) // ' per (pixel*2)'
         status = nf90_put_att( dim2ID, jGradID, 'units',AttDescription)
         
         status = nf90_put_att( dim2ID, jGradID, '_FillValue', 
     1        FillValueInt2)
         if(status .ne. nf90_noerr) call handle_err(status)
      endif
c     Define eastward and northward Sobel gradient cross-front 
c     parameters.

      call DefineSobelGradients(chunksz_records, chunksz_cf, RecCFDims1,
     1     RecCFDims2, GradOffsetO, GradScaleFactorO)

c     SST difference across the front. The mean temperature is determined
c     between 1 and 8 pixels from the front approximately normal to it 
c     and then differenced.
      
      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = trim(ParameterName_lc) // '_difference'
      status = nf90_def_var( dim2ID, AttDescription, NF90_SHORT, 
     1     RecCFDims1, SSTDiffID)
      if(status .ne. nf90_noerr) call handle_err(status)
      
      status = nf90_def_var_chunking( dim2ID, SSTDiffID,
     1     NF90_CHUNKED, chunksz_records)
      if(status .ne. nf90_noerr) call handle_err(status)

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = 'cross-front_' // trim(ParameterName_lc)
      status = nf90_put_att( dim2ID, SSTDiffID, 
     1     'long_name', AttDescription)
      if(status .ne. nf90_noerr) call handle_err(status)

c     The offset is 0 since this is the difference between two SST values
c     so the offset on the input SST is removed.

      status = nf90_put_att( dim2ID, SSTDiffID, 'add_offset',  
     1     0.0)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Scale factor is 1/10 that of the SST fields since this variable
c     is the difference between two averages. The averages are real and
c     the differences are small so the difference is multiplied by 10
c     before converting from real to integer.

      status = nf90_put_att( dim2ID, SSTDiffID, 'scale_factor', 
     1     sstScaleFactorIn/10)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, SSTDiffID, '_FillValue', 
     1     FillValueInt2)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, SSTDiffID, 'units', 
     1     trim(ParameterUnits))
      if(status .ne. nf90_noerr) call handle_err(status)

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = 'The cross-front '  // trim(ParameterName) 
     1     // ' difference is the mean '  // trim(ParameterName)
     2     // ' of the 8 pixels on one side of the front - along '
     3     // ' the line approximately normal to it - minus the '
     4     // 'mean ' // trim(ParameterName) // ' of the 8 pixels '
     5     // 'on the other side of the front.'
      status = nf90_put_att( dim2ID, SSTDiffID, 'Description', 
     1     trim(AttDescription))
      if(status .ne. nf90_noerr) call handle_err(status)

c     Mean SST for population A as determined by the Cayula-Cornillon 
c     algorithm.

      status = nf90_def_var( dim2ID, 'mean_population_a', NF90_FLOAT, 
     1     RecDims, SSTaID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_chunking( dim2ID, SSTaID,
     1     NF90_CHUNKED, chunksz_records)
      if(status .ne. nf90_noerr) call handle_err(status)

      AttDescription = 'mean_' // trim(ParameterName_lc) //
     1     '_of_SIED_''warm''_population'
      status = nf90_put_att( dim2ID, SSTaID,
     1     'long_name', AttDescription)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, SSTaID, 'add_offset',  
     1     sstOffsetIn)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, SSTaID, 'scale_factor', 
     1     sstScaleFactorIn)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, SSTaID, '_FillValue', 
     1     FillValueReal)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, SSTaID, 'units', 
     1     trim(ParameterUnits))
      if(status .ne. nf90_noerr) call handle_err(status)

c     Mean SST for population B as determined by the Cayula-Cornillon 
c     algorithm.

      status = nf90_def_var( dim2ID, 'mean_population_b', NF90_FLOAT, 
     1     RecDims, SSTbID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_chunking( dim2ID, SSTbID,
     1     NF90_CHUNKED, chunksz_records)
      if(status .ne. nf90_noerr) call handle_err(status)

      AttDescription = 'mean_' // trim(ParameterName_lc) //
     1     '_of_SIED_''cold''_population'
      status = nf90_put_att( dim2ID, SSTbID, 
     1     'long_name', AttDescription)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, SSTbID, 'add_offset',  
     1     sstOffsetIn)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, SSTbID, 'scale_factor', 
     1     sstScaleFactorIn)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, SSTbID, '_FillValue', 
     1     FillValueReal)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, SSTbID, 'units', 
     1     trim(ParameterUnits))
      if(status .ne. nf90_noerr) call handle_err(status)

c     Threshold between the two populations used to define the front.

      status = nf90_def_var( dim2ID, 'threshold', NF90_SHORT, 
     1     RecDims, SSTcID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_chunking( dim2ID, SSTcID,
     1     NF90_CHUNKED, chunksz_records)
      if(status .ne. nf90_noerr) call handle_err(status)

      AttDescription = trim(ParameterName_lc) //
     1     '_threshold_between_SIED_populations'
      status = nf90_put_att( dim2ID, SSTcID, 
     1     'long_name', AttDescription)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, SSTcID, 'add_offset',  
     1     sstOffsetIn)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, SSTcID, 'scale_factor', 
     1     sstScaleFactorIn)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, SSTcID, '_FillValue', 
     1     FillValueInt2)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, SSTcID, 'units', 
     1     trim(ParameterUnits))
      if(status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'CreateNetCDFDimFile #160: '

c     Solar zenith angle

      if(ZenithAngleExists .eqv. .true.) then

         status = nf90_def_var( dim2ID, 'solar_zenith_angle', 
     1        NF90_SHORT, RecDims,zenithAngRecID)
         if(status .ne. nf90_noerr) call handle_err(status)

         status = nf90_def_var_chunking( dim2ID, zenithAngRecID,
     1        NF90_CHUNKED, chunksz_records)
         if(status .ne. nf90_noerr) call handle_err(status)

c     create or copy attribues from input to outfile for solar zenith 
c     angle

         status = nf90_put_att( dim2ID, zenithAngRecID, 
     1        'long_name', 'Solar Zenith Angle')
         if(status .ne. nf90_noerr) call handle_err(status)

         status = nf90_put_att( dim2ID, zenithAngRecID, 'add_offset', 
     1        0.0)
         if(status .ne. nf90_noerr) call handle_err(status)

         status = nf90_put_att( dim2ID, zenithAngRecID, 'scale_factor', 
     1        1.0)
         if(status .ne. nf90_noerr) call handle_err(status)

         if(Debug .eq. 1) print *, 'CreateNetCDFDimFile #165: '

         status = nf90_put_att( dim2ID, zenithAngRecID, '_FillValue', 
     1        FillValueInt2)
         if(status .ne. nf90_noerr) call handle_err(status)

         if(Debug .eq. 1) print *, 'CreateNetCDFDimFile #160: '

         status = nf90_put_att( dim2ID, zenithAngRecID, 'units', 
     1        'degrees')
         if(status .ne. nf90_noerr) call handle_err(status)

         if(Debug .eq. 1) print *, 'CreateNetCDFDimFile #160: '
      endif

c     Finally, create the window arrays for number clear.

      if(Debug .eq. 1) print *, 'CreateNetCDFDimFile #162: WinDims: ',
     1     WinDims

      status = nf90_def_var( dim2ID, 'clear_pixels_in_SIED_windows', 
     1     NF90_SHORT, WinDims, WinNumClearID)
      if(status .ne. nf90_noerr) call handle_err(status)

      chunksz_windows(1) = WinDim1
      chunksz_windows(2) = WinDim2

      status = nf90_def_var_chunking( dim2ID, WinNumClearID,
     1     NF90_CHUNKED, chunksz_windows)
      if(status .ne. nf90_noerr) call handle_err(status)

      AttDescription = 'number_of_clear_pixels_in_'
     1     // trim(WinSizeChar) // 'x' // trim(WinSizeChar)  
     2     // '_pixel_windows'
      status = nf90_put_att( dim2ID, WinNumClearID, 
     1     'long_name', AttDescription)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, WinNumClearID, '_FillValue', 
     1     FillValueInt2)
      if(status .ne. nf90_noerr) call handle_err(status)

c     and the cohesion.

      status = nf90_def_var( dim2ID, 'cohesion_in_SIED_windows', 
     1     NF90_BYTE, WinDims, WinClearCohesionID)
      if(status .ne. nf90_noerr) call handle_err(status)

      AttDescription = 'cohesion_in_' // trim(WinSizeChar) 
     1     // 'x' // trim(WinSizeChar) // '_pixel_windows'
      status = nf90_put_att( dim2ID, WinClearCohesionID, 
     1     'long_name', AttDescription)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, WinClearCohesionID, 
     1     'add_offset', 0.0)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, WinClearCohesionID, 
     1     'scale_factor', 0.01)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, WinClearCohesionID, 
     1     '_FillValue',  FillValueInt1)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_chunking( dim2ID, WinClearCohesionID,
     1     NF90_CHUNKED, chunksz_windows)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'CreateNetCDFDimFile #168: '

c*****************Now create Global Attributes ************************

      if(Debug .eq. 1) print *, 'CreateNetCDFDimFile #170: '

c     create the time - seconds since 1970 and define its attributes

      call DefineSecondsSinceVar( dim2ID, SecondsSince1970OutID)

c     Generate the zenith angle text for the summary - blank if no 
c      zenith angle file. Needed the extra text to have the and in the
c      list of variables in the right place.

      ZenithAngleText = blankfill(:len(ZenithAngleText))
      if(ZenithAngleExists .eqv. .true.) then
         ZenithAngleText = 'the ' // trim(ParameterName) 
     1        // ' difference across the front ('
     2        // trim(ParameterName_lc) // '_difference); the mean '
     3        // 'of the histogram populations used to find the front '
     4        // '(mean_population_a and mean_population_b) and the '
     5        // 'associated ' // trim(ParameterName) 
     6        // ' threshold (threshold), and the solar zenith angle '
     7        // 'at the front (solar_zenith_angle). '
      else
         ZenithAngleText = 'the ' // trim(ParameterName) 
     1        // ' difference across the front ('
     2        // trim(ParameterName_lc) // '_difference), and the mean '
     3        // 'of the histogram populations used to find the front '
     4        // '(mean_population_a and mean_population_b) and the '
     5        // 'associated ' // trim(ParameterName) 
     6        // ' threshold (threshold). '
      endif

c     Now generate all of the global attributes

      NewTitle = BlankFill(:len(NewTitle))
      NewTitle = 'Cayula-Cornillon Fronts in'
      ProcessingProgram = trim(ProgName)

      Summary = BlankFill(1:len(Summary))
      minstep_str = BlankFill(1:len(minstep_str))
      minlength_str = BlankFill(1:len(minlength_str))
      minclear_str = BlankFill(1:len(minclear_str))

      write(minstep_str,'(i2)') MinStep
      write(minlength_str,'(i2)') MinLength
      write(minclear_str,'(i3)') MinClear
      
      Summary = 'This file consists of three sets of variables. One ' 
     1     // 'set corresponds to characteristics of front segments. ' 
     2     // 'A front segment is a set of contiguous front pixels ' 
     3     // 'found with the Cayula-Cornillon SIED algorithm. ' 
     4     // 'Front segments have a minimum length of '
     5     // trim(minlength_str) // ', and a minimum seperation '
     6     // 'of mean population temperatures of' // trim(minstep_str)
     7     // ' digital counts. At least ' // trim(minclear_str) 
     8     // ' of the  ' // trim(WinSizeChar) // 'x' 
     9     // trim(WinSizeChar) // ' window used by the algorithm had'
     *     // ' to be clear, and the smaller population had to be'
     1     // ' no less than 1/4 of the number of clear pixels. '
     4     // 'Variables in this set are, for each segment, the '
     5     // 'start location of the segment in the list of front '
     6     // 'pixels (segment_start), the length of the segment '
     7     // '(segment_length), and the minimum and maximum latitude '
     8     // 'and longitude (minimum_latitude, maximum_latitude, ' 
     9     // 'minimum_longitude and maximum_longitude). ' 
     1     // 'The second set corresponds to the variables ' 
     2     // 'associated with each front pixel. Variables in ' 
     3     // 'this group are: longitude (longitude), latitude '
     4     // '(latitude), ' // trim(iVarName) // ' (i) and '
     5     // trim(jVarName) // ' (j) location of the pixel; ' 
     6     // trim(ParameterName) // ' (cross_front_' 
     7     // trim(ParameterName_lc) // '), the eastward-component, ' 
     8     // '(eastward_gradient), the northward-component '
     9     // '(northward_gradient) and the magnitude '
     1     // '(gradient_magnitude) of the Sobel gradient, all in '
     2     // 'a cross-front coordinate system defined relative to '
     3     // 'the front (' // trim(iVarName) // '_pixel_location '
     4     // 'and ' // trim(jVarName) // '_pixel_location); the ' 
     5     // trim(iVarName) // ' and ' // trim(jVarName)  
     5     // ' components of center-difference gradients ('
     5     // trim(iVarName) // '_gradient and ' // trim(jVarName)
     5     // '_gradient), the Sobel gradient at the front '
     6     // '(in_front_eastward_gradient and '
     7     // 'in_front_northward_gradient) and the components ' 
     8     // 'near, but not in the front - the background gradients '
     9     // '(background_eastward_gradient and '
     1     // 'background_northward_gradient); the location of the ' 
     2     // 'maximum gradient in the vicinity of the front in the '
     3     // 'cross-front direction (maximum_gradient_location); '
     4     // trim(ZenithAngleText) 
     2     // ' The third set is for the cohesion of the clear '
     3     // 'pixels in each ' // trim(WinSizeChar) // 'x' 
     4     // trim(WinSizeChar) // ' window analyzed '
     5     // '(clear_cohesion_in_SIED_windows) and the number of '
     6     // 'clear pixels in each of these windows '
     7     // '(clear_pixels_in_SIED_windows).'

      PlaneRemovalText = BlankFill(1:len(PlaneRemovalText))
      if(RemovePlane) then
         PlaneRemovalText = ' Prior to performing the histogram '
     1        // 'analysis on the ' // trim(WinSizeChar) // 'x' 
     2        // trim(WinSizeChar) // ' window used by the algorithm, '
     3        // 'a plane, fit to the ' // trim(ParameterName)  
     4        // ' values on the ' // trim(PlaneFitSizeChar) // 'x' 
     5        // trim(PlaneFitSizeChar) // 'pixel region centered on '
     5        // 'the window of interest, was removed from data '
     6        // 'in the window.'
       endif
         
       Summary = trim(Summary) // trim(PlaneRemovalText)


c     Reopen the median file; will need some attributes from it.

      status = nf90_open( MedianFileName, nf90_nowrite, ncMedID)
      if (status .ne. nf90_noerr) call handle_err(status)

      call GenerateGlobals( ncMedID, dim2ID, NewTitle,
     1     ProcessingProgram, SIEDVersionNo, FrontsFileName)

c     Get seconds since 1970; will need it below.

      status = nf90_inq_varid( ncMedID, 'DateTime', 
     1     SecondsSince1970InID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_var( ncMedID, SecondsSince1970InID, 
     1     SecondsSince1970)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_close(ncMedID)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Now add the GeoSobel history and the Zenith angle history to
c     the history of this file if the Sobel gradients and zenith 
c     angle files exist.

c     Start with GeoSobel

      inquire( File=trim(GeoSobelFileName), Exist=GeoSobelExists)

      if(GeoSobelExists .eqv. .true.) then
         if(Debug .eq. 1) print *, 'CreateNetCDFDimFile #180: '

c     Get GeoSobel history

         status = nf90_open( GeoSobelFileName, nf90_write, ncGeoSobelID)
         if (status .ne. nf90_noerr) call handle_err(status)

         MoreHistory = BlankFill(1:len(MoreHistory))
         status = nf90_get_att( ncGeoSobelID, nf90_global, 'history', 
     1        MoreHistory)
         if(status .ne. nf90_noerr) call handle_err(status)

         status = nf90_close(ncGeoSobelID)
         if (status .ne. nf90_noerr) call handle_err(status)

c     Get history written into the Fronts file by GenerateGlobals

         PreviousProcessingHistory = 
     1        BlankFill(1:len(PreviousProcessingHistory))
         status = nf90_get_att( dim2ID, nf90_global, 'history', 
     1        PreviousProcessingHistory)
         if(status .ne. nf90_noerr) call handle_err(status)

c     Add GeoSobel history to history written out thus far.

         ProcessingHistory = BlankFill(1:len(ProcessingHistory))
         ProcessingHistory = trim(PreviousProcessingHistory) //
     1        ' [GeoSobel Processing History]' // trim(MoreHistory)

         if(debug .eq. 1) print *, 'CreateNetCDFDimFile #190: ',
     1        'Processing history::', trim(ProcessingHistory), '::'

         status = nf90_put_att( dim2ID, nf90_global, 
     1        'history', trim(ProcessingHistory))
         if(status .ne. nf90_noerr) call handle_err(status)

      endif

c     Now add zenith angle processing history.

      if(ZenithAngleExists .eqv. .true.) then
         if(Debug .eq. 1) print *, 'CreateNetCDFDimFile #200: '

c     Get ZenithAngle history

         status = nf90_open( ZenithAngleFileName, nf90_write, 
     1        ncZenithAngleID)
         if (status .ne. nf90_noerr) call handle_err(status)

         MoreHistory = BlankFill(1:len(MoreHistory))
         status = nf90_get_att( ncZenithAngleID, nf90_global, 'history',
     1        MoreHistory)
         if(status .ne. nf90_noerr) call handle_err(status)

         status = nf90_close(ncZenithAngleID)
         if (status .ne. nf90_noerr) call handle_err(status)

c     Get history written into the Fronts file by GenerateGlobals

         PreviousProcessingHistory = 
     1        BlankFill(1:len(PreviousProcessingHistory))
         status = nf90_get_att( dim2ID, nf90_global, 'history', 
     1        PreviousProcessingHistory)
         if(status .ne. nf90_noerr) call handle_err(status)

c     Add ZenithAngle history to history written out thus far.

         ProcessingHistory = BlankFill(1:len(ProcessingHistory))
         ProcessingHistory = trim(PreviousProcessingHistory) //
     1        ' [ZenithAngle Processing History]' // trim(MoreHistory)

         if(debug .eq. 1) print *, 'CreateNetCDFDimFile #210: ',
     1        'Processing history::', trim(ProcessingHistory), '::'

         status = nf90_put_att( dim2ID, nf90_global, 
     1        'history', ProcessingHistory)
         if(status .ne. nf90_noerr) call handle_err(status)

      endif

c     And the window size used to find fronts.

      status = nf90_put_att( dim2ID, nf90_global, 'window_size', 
     1     WinSize)
         if(status .ne. nf90_noerr) call handle_err(status)

c     All done defining the variables and attributes for output file.

      status = nf90_enddef(dim2ID)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(debug .eq. 1) print *, 'CreateNetCDFDimFile #220: ',
     1     'All attributes defined.'

C     Write seconds since to the output file.

      status = nf90_put_var( Dim2ID, SecondsSince1970OutID, 
     1     SecondsSince1970)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'CreateNetCDFDimFile #999: '

      end subroutine CreateNetCDFDimFile

c***********************************************************************
      subroutine DefineMinMaxLatLon(chunksz_segments, SegDims)
c***********************************************************************
c     This subroutine will define the min, max x and y variables.
      
      use netcdf

      Implicit none
      
c     Parameter statements

      include 'ParameterStatements.f'

c*******Functions

c*******Commons

c******General variables

      integer chunksz_segments(1)

      integer SegDims(1)

      include 'netcdf.inc'

c---------------------------------start here ---------------------------

      if(debug .eq. 1) print *, 'DefineMinMaxLatLon #000'

c     Minimum latitude of front pixels in segment.

      status = nf90_def_var( dim2ID, 'minimum_latitude', NF90_FLOAT,
     1     SegDims, LatMinID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_chunking( dim2ID, LatMinID,
     1    NF90_CHUNKED, chunksz_segments)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, LatMinID, 'units', 'degrees')
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, LatMinID, '_FillValue', 
     1     FillValueReal)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Maximum latitude of front pixels in segment.

      status = nf90_def_var( dim2ID, 'maximum_latitude', NF90_FLOAT,
     1     SegDims, LatMaxID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_chunking( dim2ID, LatMaxID,
     1    NF90_CHUNKED, chunksz_segments)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, LatMaxID, 'units', 'degrees')
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, LatMaxID, '_FillValue', 
     1     FillValueReal)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Minimum longitude of front pixels in segment.

      status = nf90_def_var( dim2ID, 'minimum_longitude', NF90_FLOAT,
     1     SegDims, LonMinID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_chunking( dim2ID, LonMinID,
     1    NF90_CHUNKED, chunksz_segments)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, LonMinID, 'units', 'degrees')
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, LonMinID, '_FillValue', 
     1     FillValueReal)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Maximum longitude of front pixels in segment.

      status = nf90_def_var( dim2ID, 'maximum_longitude', NF90_FLOAT, 
     1     SegDims, LonMaxID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_chunking( dim2ID, LonMaxID,
     1    NF90_CHUNKED, chunksz_segments)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, LonMaxID, 'units', 'degrees')
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, LonMaxID, '_FillValue', 
     1     FillValueReal)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(debug .eq. 1) print *, 'DefineMinMaxXY #999'

      return
      end subroutine DefineMinMaxLatLon

c***********************************************************************
      subroutine DefineSobelGradients(chunksz_records, chunksz_cf, 
     1     RecCFDims1, RecCFDims2, GradOffsetO, GradScaleFactorO)
c***********************************************************************
c     This subroutine will define the min, max x and y variables.
      
      use netcdf

      Implicit none
      
c     Parameter statements

      include 'ParameterStatements.f'

c*******Functions

c*******Commons

c******General variables

c     AttDescription - attribute description when needed.
      character*500 AttDescription

c     chunksz_cf - chunksize array for cross-front records.
      integer chunksz_cf(2)
c     chunksz_records - chunksize array for front records.
      integer chunksz_records(1)

      integer SegDims(1)

c     GradOffsetO - the offset to add to the input values when  
c      calculating the Sobel gradient magnitude.
      real GradOffsetO
c     GradScaleFactorO - the amount to multiply the input values by  
c      when calculating the Sobel gradient magnitude.
      real GradScaleFactorO

c     RecCFDims1 - array for the 1-dimension IDs for cross-front 
c     variables.
      integer RecCFDims1(1)
c     RecCFDims2 - array for the 2-dimension IDs for cross-front 
c     variables.
      integer RecCFDims2(2)

      include 'netcdf.inc'

c---------------------------------start here ---------------------------

      if(debug .eq. 1) print *, 'DefineSobelGradients #000'

c     Initialize the output offset and scale factor for the
c     Sobel gradient variables

       GradOffsetO = 0.0
       GradScaleFactorO = 0.0001

c     x dimension.

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = trim(iVarName) // '_pixel_location'
      status = nf90_def_var( dim2ID, AttDescription, 
     1     NF90_BYTE, RecCFDims2, CFmID)
      if(status .ne. nf90_noerr) call handle_err(status)

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = 'relative_' // trim(iVarName) // 
     1     '_pixel_location'
      status = nf90_put_att( dim2ID, CFmID, 
     1     'long_name', AttDescription)

      status = nf90_def_var_chunking( dim2ID, CFmID,
     1     NF90_CHUNKED, chunksz_cf)
      if(status .ne. nf90_noerr) call handle_err(status)


      status = nf90_put_att( dim2ID, CFmID, '_FillValue', 
     1     FillValueInt1)
      if(status .ne. nf90_noerr) call handle_err(status)

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = 'Added to the i variable, this variable '
     1     // 'gives the ' // trim(iVarName) // ' location of the '
     2     // 'cross-front pixel in the original image. This is '
     3     // 'necessary since the cross-front lines are '
     4     // 'approximately perpendicular to the front so finding '
     5     // 'their actual position in the original image is '
     6     // 'difficult at best.'
      status = nf90_put_att( dim2ID, CFmID, 'Description', 
     1     trim(AttDescription))

      if(status .ne. nf90_noerr) call handle_err(status)

c     y pixel location

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = trim(jVarName) // '_pixel_location'
      status = nf90_def_var( dim2ID, AttDescription, 
     1     NF90_BYTE, RecCFDims2, CFnID)
      if(status .ne. nf90_noerr) call handle_err(status)

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = 'relative_' // trim(jVarName) // 
     1     '_pixel_location'
      status = nf90_put_att( dim2ID, CFnID, 
     1     'long_name', AttDescription)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_chunking( dim2ID, CFnID,
     1     NF90_CHUNKED, chunksz_cf)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, CFnID, '_FillValue', 
     1     FillValueInt1)
      if(status .ne. nf90_noerr) call handle_err(status)

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = 'Added to the j variable, this variable '
     1     // 'gives the ' // trim(jVarName) // ' location of the '
     2     // 'cross-front pixel in the original image. This is '
     3     // 'necessary since the cross-front lines are '
     4     // 'approximately perpendicular to the front so finding '
     5     // 'their actual position in the original image is '
     6     // 'difficult at best.'
      status = nf90_put_att( dim2ID, CFnID, 'Description', 
     1     trim(AttDescription))
      if(status .ne. nf90_noerr) call handle_err(status)

      if(debug .eq. 1) print *, 'DefineSobelGradients #100'

c     Eastward component of the Sobel gradient magnitude

      status = nf90_def_var( dim2ID, 
     1      'eastward_gradient', 
     2      NF90_SHORT, RecCFDims2, CFEastGradID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_chunking( dim2ID, CFEastGradID,
     1     NF90_CHUNKED, chunksz_cf)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, CFEastGradID, 
     1     'long_name', 
     2     'eastward_component_of_Sobel_gradient')
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, CFEastGradID, 'add_offset',  
     1     GradOffsetO)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, CFEastGradID, 'scale_factor', 
     1     GradScaleFactorO)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, CFEastGradID, '_FillValue', 
     1     FillValueInt2)
      if(status .ne. nf90_noerr) call handle_err(status)

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = trim(ParameterUnits) // ' per kilometer'
      status = nf90_put_att( dim2ID, CFEastGradID, 'units', 
     1     AttDescription)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Northward component of the Sobel gradient magnitude

      status = nf90_def_var( dim2ID, 
     1     'northward_gradient', 
     1     NF90_SHORT, RecCFDims2, CFNorthGradID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_chunking( dim2ID, CFNorthGradID,
     1     NF90_CHUNKED, chunksz_cf)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, CFNorthGradID, 
     1     'long_name', 
     1     'northward_component_of_Sobel_gradient')
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, CFNorthGradID, 'add_offset',  
     1     GradOffsetO)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, CFNorthGradID, 'scale_factor', 
     1     GradScaleFactorO)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, CFNorthGradID, '_FillValue', 
     1     FillValueInt2)
      if(status .ne. nf90_noerr) call handle_err(status)

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = trim(ParameterUnits) // ' per kilometer'
      status = nf90_put_att( dim2ID, CFNorthGradID, 'units', 
     1     AttDescription)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'DefineSobelGradients #110: '

c     In-front gradient variable; this is the maximum Sobel gradient 
c      within +/- 1 pixel of the front, including the front location.

      status = nf90_def_var( dim2ID, 'in_front_eastward_gradient', 
     1     NF90_SHORT, RecCFDims1, CFEastGradMaxID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_chunking( dim2ID, CFEastGradMaxID,
     1     NF90_CHUNKED, chunksz_records)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, CFEastGradMaxID, 
     1     'long_name', 'in_front_eastward_Sobel_gradient')
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, CFEastGradMaxID, 'add_offset',  
     1     GradOffsetO)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, CFEastGradMaxID, 'scale_factor', 
     1     GradScaleFactorO)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, CFEastGradMaxID, '_FillValue', 
     1     FillValueInt2)
      if(status .ne. nf90_noerr) call handle_err(status)

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = trim(ParameterUnits) // ' per kilometer'
      status = nf90_put_att( dim2ID, CFEastGradMaxID, 'units', 
     1     AttDescription)
      if(status .ne. nf90_noerr) call handle_err(status)

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = 'Eastward Sobel gradient in the near-vicinity '
     1     // 'of the front. It is the maximum eastward gradient '
     2     // 'within +/- 3 pixels of the front along the line '
     3     // 'that is approximately normal to the front segment. '
     4     // 'Remember the SIED does not find the front based on '
     5     // 'gradient but on a threshold between two populations. '
     6     // 'These fronts are generally near a local maximum in the '
     7     // 'gradient, but not necessarily at the maximum, hence '
     8     // 'this value. '
      status = nf90_put_att( dim2ID, CFEastGradMaxID, 'Description', 
     1     trim(AttDescription))
      if(status .ne. nf90_noerr) call handle_err(status)

c     In-front gradient variable; this is the maximum Sobel gradient 
c     within +/- 1 pixel of the front, including the front location.

      status = nf90_def_var( dim2ID, 'in_front_northward_gradient', 
     1     NF90_SHORT, RecCFDims1, CFNorthGradMaxID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_chunking( dim2ID, CFNorthGradMaxID,
     1     NF90_CHUNKED, chunksz_records)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, CFNorthGradMaxID, 
     1     'long_name', 'in_front_northward_Sobel_gradient')
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, CFNorthGradMaxID, 'add_offset',  
     1     GradOffsetO)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, CFNorthGradMaxID, 'scale_factor', 
     1     GradScaleFactorO)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, CFNorthGradMaxID, '_FillValue', 
     1     FillValueInt2)
      if(status .ne. nf90_noerr) call handle_err(status)

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = trim(ParameterUnits) // ' per kilometer'
      status = nf90_put_att( dim2ID, CFNorthGradMaxID, 'units', 
     1     AttDescription)
      if(status .ne. nf90_noerr) call handle_err(status)

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = 'Northward Sobel gradient in the near-vicinity '
     1     // 'of the front. It is the maximum eastward gradient '
     2     // 'within +/- 3 pixels of the front along the line '
     3     // 'that is approximately normal to the front segment. '
     4     // 'Remember the SIED does not find the front based on '
     5     // 'gradient but on a threshold between two populations. '
     6     // 'These fronts are generally near a local maximum in the '
     7     // 'gradient, but not necessarily at the maximum, hence '
     8     // 'this value. '
      status = nf90_put_att( dim2ID, CFNorthGradMaxID, 'Description', 
     1     trim(AttDescription))
      if(status .ne. nf90_noerr) call handle_err(status)

c     Bacground gradient variables.

      status = nf90_def_var( dim2ID, 'background_eastward_gradient', 
     1     NF90_SHORT, RecCFDims1, CFBGEastGradID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_chunking( dim2ID, CFBGEastGradID,
     1     NF90_CHUNKED, chunksz_records)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, CFBGEastGradID, 
     1     'long_name', 'background_eastward_Sobel_gradient')
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, CFBGEastGradID, 'add_offset',  
     1     GradOffsetO)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, CFBGEastGradID, 'scale_factor', 
     1     GradScaleFactorO)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, CFBGEastGradID, '_FillValue', 
     1     FillValueInt2)
      if(status .ne. nf90_noerr) call handle_err(status)

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = trim(ParameterUnits) // ' per kilometer'
      status = nf90_put_att( dim2ID, CFBGEastGradID, 'units', 
     1     AttDescription)
      if(status .ne. nf90_noerr) call handle_err(status)

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = 'This variable is an approximation of the '
     1     // 'background eastward Sobel gradient in the vicinity '
     2     // 'of the front. It is the average of the eastward Sobel '
     3     // 'gradients between 5 and 8 pixels from the front along '
     4     // 'a line approximately normal to the front segment.'
      status = nf90_put_att( dim2ID, CFBGEastGradID, 'Description', 
     1     trim(AttDescription))
      if(status .ne. nf90_noerr) call handle_err(status)

c     Bacground gradient variable.

      status = nf90_def_var( dim2ID, 'background_northward_gradient', 
     1     NF90_SHORT, RecCFDims1, CFBGNorthGradID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_chunking( dim2ID, CFBGNorthGradID,
     1     NF90_CHUNKED, chunksz_records)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, CFBGNorthGradID, 
     1     'long_name', 'background_northward_Sobel_gradient')
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, CFBGNorthGradID, 'add_offset',  
     1     GradOffsetO)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, CFBGNorthGradID, 'scale_factor', 
     1     GradScaleFactorO)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, CFBGNorthGradID, '_FillValue', 
     1     FillValueInt2)
      if(status .ne. nf90_noerr) call handle_err(status)

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = trim(ParameterUnits) // ' per kilometer'
      status = nf90_put_att( dim2ID, CFBGNorthGradID, 'units', 
     1     AttDescription)
      if(status .ne. nf90_noerr) call handle_err(status)

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = 'This variable is an approximation of the '
     1     // 'background northward Sobel gradient in the vicinity '
     2     // 'of the front. It is the average of the northward Sobel '
     3     // 'gradients between 5 and 8 pixels from the front along '
     4     // 'a line approximately normal to the front segment.'
      status = nf90_put_att( dim2ID, CFBGNorthGradID, 'Description', 
     1     trim(AttDescription))
      if(status .ne. nf90_noerr) call handle_err(status)

c     Finally, the location of the maximum gradient magnitude along a
c      cross-frontal line and the corresponding cross-scan and 
c      along-scan pixel locations.

      status = nf90_def_var( dim2ID, 'maximum_gradient_location', 
     1     NF90_SHORT, RecCFDims1, I_MaxGradMagID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_chunking( dim2ID, I_MaxGradMagID,
     1     NF90_CHUNKED, chunksz_records)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, I_MaxGradMagID, 
     1     'long_name', 
     2     'cross_front_location_of_maximum_Sobel_gradient_magnitude')
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( dim2ID, I_MaxGradMagID, '_FillValue', 
     1     FillValueInt2)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(debug .eq. 1) print *, 'DefineSobelGradients #999'

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = 'The program applies a three-pixel '
     1     // 'moving average to the Sobel gradient magnitudes along '
     2     // 'a line approximately normal to the front segment. This '
     3     // 'variable is the location along this line of the '
     4     // 'maximum of this quantity relative to the front.'
      status = nf90_put_att( dim2ID, I_MaxGradMagID, 'Description', 
     1     trim(AttDescription))
      if(status .ne. nf90_noerr) call handle_err(status)

      return
      end subroutine DefineSobelGradients

c***********************************************************************
      subroutine WriteFrontData( FrontsFileName, MedianFileName, 
     1     OutPicSave, ppv, FrntPic, FilPic, NFrontPixel, 
     2     xLocation, yLocation, SegLength, ProgName, Lat_Array,
     3     Lon_Array, gradv)
c***********************************************************************
c     This subroutine writes out the front pixel data and calculates
c     segment statistics to determine the quality of the front data
c     point.
      
      use netcdf

      Implicit none
      
c     Parameter statements

      include 'ParameterStatements.f'

c*******Functions

c     GenerateFileName - a function that will generate a filename for
c     . one of the output file types given the sudirectory name and
c     . the ending of the filename. These change for each program
c     . output.
      character*319 GenerateFileName

c******LatLon common

c     LatLonFileName - the name of the file containing the latitud and
c     . longitude arrays.
      character*319 LatLonFileName
c     LatLonFileNameSave - the saved LatLonFileName used to see if the
c     . the filename is new for this SST field.
      character*319 LatLonFileNameSave

      common /LatLon/ LatLonFileName, LatLonFileNameSave

c******General variables

      integer*4 outlength

c     aLat - latitude at front pixel location
      real*4 alat
c     aLon - longitude at front pixel location
      real*4 alon


c     CFP - number of cross-frontal pixels on each side of front 
c     segment. Used to define vector lengths.
      integer*4, parameter :: CFP=8

c     CFBGEastGrad - background eastward Sobel gradient (background is 
c     5 or more pixels from the front.
      integer*2 CFBGEastGrad
c     CFBGNorthGrad - background northward Sobel gradient (background is
c     5 or more pixels from the front.
      integer*2 CFBGNorthGrad
c     CFE - a real version of CFEastGrad; used to calculate GradMag.
      real CFE
c     CFEastGrad - eastward Sobel gradient cross-front array
      integer*2 CFEastGrad(2*CFP+1)
c     CFEastGradMax - the maximum eastward Sobel gradient within 1 
c     pixel of the front (normal to the local front).
      integer*2 CFEastGradMax
c     CFmLoc - the i location in the original SST array of a pixel in
c     the cross-front array.
      integer*1 CFmLoc(2*CFP+1)
c     CFN - a real version of CFNorthGrad; used to calculate GradMag.
      real CFN
c     CFnLoc - the j location in the original SST array of a pixel in
c     the cross-front array.
      integer*1 CFnLoc(2*CFP+1)
c     CFNorthgrad - eastward Sobel gradient cross-front array
      integer*2 CFNorthgrad(2*CFP+1)
c     CFNorthGradMax - the maximum northward Sobel gradient within 1 
c     pixel of the front (normal to the local front).
      integer*2 CFNorthGradMax
c     CFSST - this vector contains the cross-front SST pixels. 
      integer*2 CFSST(2*CFP+1)

c     d - separation in pixels along the front.
      integer*4 d
c     DeltaX - separation in the 1st dimension of front points along a 
c     segment.
      real DeltaX
c     DeltaY - separation in the 2nd dimension of front points along a 
c     segment.
      real DeltaY

c     EastGrad - eastward Sobel gradient array read in.
      real, allocatable :: EastGrad(:,:)

c     Factor1 - factor used to determine cross-front gradients.
      real Factor1
c     Factor2 - factor used to determine cross-front gradients.
      real Factor2
c     FactorX - factor used to determine cross-front position.
      real FactorX
c     FactorY - factor used to determine cross-front position.
      real FactorY
c     FilPic(1:SizOfImg,1:SizOfImg) - median-filtrd img w/ clouds
      integer*2 filpic(1:LenXA,1:LenYA)
c     FrontsFileName - the output filename for fronts - uncompressed
      character*319 FrontsFileName
c     FrntPic - front image.
      integer*4 FrntPic(1:LenXA,1:LenYA)

c     GeoSobelExists -  file status for GeoSobel, .true. if exists.
      logical GeoSobelExists
c     GeoSobelFileName - name of the file with the Sobel gradients in 
c     geo coordinates.
      character*319 GeoSobelFileName
c$$$c     GradFillValueI - the fill value for the x Sobel gradient read in
c$$$      real GradFillValueI
c$$$c     GradOffsetI - the offset for the x Sobel gradient read in.
c$$$      real GradOffsetI
c     GradMag - the gradient magnitude as a function of cross-front 
c     location.
      real GradMag(2*CFP+1)
c$$$c     GradScaleFactorI - the scale factor for the x Sobel gradient 
c$$$c     read in.
c$$$      real GradScaleFactorI
c     GradOffsetO - the offset for the x Sobel gradient to be written
c     out.
      real GradOffsetO
c     GradScaleFactorO - the scale factor for the x Sobel gradient 
c     to be written out.
      real GradScaleFactorO
c     GradV - is the gradient vector as calculated by gradient. This is
c      a bit different than the Sobel calculation and is used in Pmerge.
c      Rather than change Pmerge, both this version and the Sobel 
c      version will be written out to the netCDF file.
      integer*2 gradv(1:2,1:LenXA,1:LenYA)

c     i - loop index along the front segment
      integer*4 i
c     I_MaxGradMag - the location of the maximum gradient magnitude 
c     along a cross-frontal line.
      integer*2 I_MaxGradMag
c     NFrontPixel - the number of frontal pixels.
      integer*4 NFrontPixel
c     iFP - front pixel counter when writing front pixels to output file
      integer*4 iFP
c     ip - do loop index, 1st dimesnsion
      integer*4 ip

c     jp - do loop index, 2nd dimesnsion
      integer*4 jp

c     kp - do loop index in the calculation of mean gradients.
      integer*4 kp

c     Lat_Array - latitude values for image. Read in once if it does
c     not change or for each file if it does change.
      real*4 Lat_Array(1:LenX,1:LenY)
c     LatLonFillValueReal - the value used for missing data. Be careful
c      here. The fill value in the netCDF file written out for this 
c      variable in integer, but the latitude and longitude are converted
c      to real in the call to ReadLatLon so the fill value is redefined
c      in that subroutine to be real. In order not to confuse it with 
c      the fill value read in, it is renamed as well. 
      real LatLonFillValueReal
c     Lon_Array - longitude values for image. Read in once if it does
c     not change or for each file if it does change.
      real*4 Lon_Array(1:LenX,1:LenY)
c     lp = counter in loop locating the maximum gradient magnitude on a
c     cross-front line.
      integer*4 lp

c     m - temporary container for xLocation.
      integer*4 m
c     MaxGradMag - the maximum gradient magnitude along a cross-frontal
c     line
      real MaxGradMag
c     MaxLat - the maximum latitude for a segment.
      real MaxLat
c     MaxLon - the maximum longitude for a segment.
      real MaxLon
c     c     MaxX - the maximum x locations for a segment.
c     integer*4 MaxX
c     c     MaxY - the maximum x locations for a segment.
c     integer*4 MaxY
c     MeanGradMag - the mean gradient magnitude for a 3 pixel cross-
c     front range.
      real MeanGradMag
c     MedianFileName - the name of the output file for the 3x3 median
c     . filter.
      character*319 MedianFileName
c     MedID - ID for the median sst field.
      integer*4 MedID
c     MinLat - the minumum latitude for a segment.
      real MinLat
c     MinLon - the minumum longitude for a segment.
      real MinLon
c     c     MinX - the minumum x locations for a segment.
c     integer*4 MinX
c     c     MinY - the minumum x locations for a segment.
c     integer*4 MinY
c     MissingValueInt2 - the missing value passed if the value read in
c     is integer*2.
      integer*2 MissingValueInt2
c     MissingValueInt4 - the missing value passed if the value read in
c     is integer*4.
      integer*4 MissingValueInt4
c     MissingValueReal - the missing value passed if the value read in
c     is real.
      real*4 MissingValueReal
c     mp1 - the starting 1st dimension point on the line normal to the 
c     segment.
      integer*4 mp1

c     n - temporary container for yLocation.
      integer*4 n
c     ncIDZA - zenith angle netCDF ID
      integer ncIDZA
c     nGoodA - number of good SST pixels in population A.
      integer*4 nGoodA
c     nGoodB - number of good SST pixels in population B.
      integer*4 nGoodB
c     nGoodBGEastGrad - number of good gradient values used for eastward
c     bacground gradient calculation.
      integer*4 nGoodBGEastGrad
c     nGoodBGNorthGrad - number of good gradient values used for 
c     norhtward bacground gradient calculation.
      integer*4 nGoodBGNorthGrad
c     NorthGrad - eastward Sobel gradient.
      real, allocatable :: NorthGrad(:,:)
c     np1 - the starting 2nd dimension point on the line normal to the 
c     segment.
      integer*4 np1

c     OutPicSave is a copy of outpic before the values in it are 
c     replaced with contour numbers.
      integer*4 OutPicSave(1:LenXA,1:LenYA)

c     ppv - Cayula-Cornillon front defining characteristics
c     ppv(0)=WPV(0)        :global cohesion
c     ppv(1)=WPV(1)        :criterion function (theta)
c     ppv(2)=WPV(2)        :Mean of population A
c     ppv(3)=WPV(3)        :Mean of population B
c     ppv(4)=THRESH        :Temperature threshold
      real*4 ppv(1:MaxEdge,0:5)

c     SegEnd - the last pixel, a fill value for the current segment
      integer*4 SegEnd
c     SegLength - the length of found frontal segments
      integer*2 SegLength(0:MaxNumOfSegments)
c     SegNoTest - test segment number. If negative, no printout. If a
c     good segment number, lots of printout for that segment number.
      integer*4, parameter :: SegNoTest = 1
c     SegStart - first good pixel in the current segment
      integer*4 SegStart
c     SobelFileName - name of the file with the Sobel gradients.
      character*319 SobelFileName
c     SSTMeanA - the mean temperature along the a line approximately 
c      normal to front from 1 to CF pixels away from the front.      
      real SSTMeanA
c     SSTMeanA_JF - the mean temperature of the lower peak in the SST 
c     histogram used to find a frontal pixel.
      real SSTMeanA_JF
c     SSTMeanB - the mean temperature along the a line approximately 
c      normal to front from -1 to -CF pixels away from the front.  
      real SSTMeanB
c     SSTMeanB_JF - the mean temperature of the upper peak in the SST 
c     histogram used to find a frontal pixel.
      real SSTMeanB_JF
c     SSTDiff - the mean temperature difference across the fronts:
c     SSTMeanA - SSTMeanB
      integer*2 SSTDiff
c     sstThresholdout - the SST separating the two populations in front
c     detection algoritm
      integer*2 sstThresholdout
c     SumOfMags - sum of gradient magnitudes over 3 cross-front pixels.
      real SumOfMags

c     VariableName - the character string name of a variable to be read
c     by GetAttributes.
      character*50 VariableName
c     VariableType - string with the variable name.
      character*12 VariableType

c     xLocation - the x location of a frontal pixel, the value of the 
c     first dimension in the SST array.
      integer*4 xLocation(1:MaxEdge)

c     yLocation - the y location of a frontal pixel, the value of the 
c     second dimension in the SST array.
      integer*4 yLocation(1:MaxEdge)

c     ZenithAngle - array of zenith angles at each pixel location.
      integer*2, allocatable :: ZenithAngle(:,:)
c     ZenithAngleFileName - name of the file with the solar zenith 
c     angles at the pixel location.
      character*319 ZenithAngleFileName
c     ZenithFileExists -  file status for zenith angle, .true. if exists
      logical ZenithFileExists

      integer*4 jpcc

c     FILE1              - Input file name containing the image being 
c     processed.
c     filename
c     Int*4 to allow for increased - edge pixels in model results
c     ALON, ALAT	       			- pixel coordinates
c     I,J,K,M,N,IL,JL        - indexes
c     and it will be discarded.
c     EOS			 - end-of-string character
c     read_flag		 - indicates which values should be read in 
c     Read_Image
c     dumhdate		 - dummy value for date
c     FILE4,FRONTSFILENAME             - (file names)
c     FILE_EX                             - cld mask exists?
c     FILTYP		       	      - file type indicator
c     Range           - The upper limit to use in the histograms. PCC 
c     added this variable. The numbers were harded coded in as 255 in 
c     the .rat version.
c     
c     Pixel Parameter vector: PPV
c     
c     
c     DataRange - maximum minus minimum value read in.
c     NAboveRange - number of input data values above Range.
c     status - used in test for file existence.
c     UnitOut - Unit to which output messages will be written for this
c     . run.
c     

c---------------------------------start here ---------------------------

      if(Debug .eq. 1) print *, 'WriteFrontData #000'

      allocate( EastGrad(1:LenX,1:LenY),
     1     NorthGrad(1:LenX,1:LenY),
     2     stat=ierr)
      
      if (ierr .ne. 0) then
         print *, 'WriteFrontData #100: Allocation error for LenXA, ',
     1        'LenYA arrays. Exiting.'
         stop
      endif

c**************Initialize the cross-front extent variable

      CFPixels = CFP

c**************Now write out front locations and front characteristics

c     Construct the file name for the geoSobel gradients and see if the
c     file exists. If it does not exist, will not process the cross-
c     front gradient stats.
      
      TempString1 = BlankFill(:len(TempString1))
      TempString1 = 'GeoSobel/'

      TempString2 = BlankFill(:len(TempString2))
      TempString2 = '_SobelEW.nc'

      GeoSobelFileName = GenerateFileName( MedianFileName, 
     1     TempString1, TempString2)

      inquire( File=trim(GeoSobelFileName), Exist=GeoSobelExists)

c     Now get the needed attributes for the Sobel gradient magnitude.

      if(GeoSobelExists .eqv. .true.) then

         allocate( EastGrad(1:LenX,1:LenY),
     1        NorthGrad(1:LenX,1:LenY),
     2        stat=ierr)

         VariableName = BlankFill(:len(VariableName))
         VariableName = 'northward_gradient'

         call ReadGeoSobelFile( GeoSobelFileName, VariableName, 
     1        NorthGrad)

         VariableName = BlankFill(:len(VariableName))
         VariableName = 'eastward_gradient'

         call ReadGeoSobelFile( GeoSobelFileName, VariableName, 
     1        EastGrad)

c     And get the GeoSobel gradient attributes.

         VariableType = BlankFill(:len(VariableType))
         VariableType = 'real'

c     Will use the eastward values for offset, scale factor and missing
c     value for both northward and eastward gradients - it would really
c     be odd if they were different.

         call GetGradAttributes(GeoSobelFileName)

         if(Debug .eq. 1) print *, 'WriteFrontData #110: ',
     1        'GradScaleFactorIn, GradScaleFactorO, GradOffsetIn, ',
     2        'GradOffsetO: ', GradScaleFactorIn, GradScaleFactorO, 
     3        GradOffsetIn, GradOffsetO
      endif

c     Next build the zenith angle file name and test for existence.

      TempString1 = BlankFill(:len(TempString1))
      TempString1 = 'ZenithAngle/'

      TempString2 = BlankFill(:len(TempString2))
      TempString2 = '_ZenithAngle.nc'

      ZenithAngleFileName = GenerateFileName( MedianFileName, 
     1     TempString1, TempString2)

      inquire( File=trim(ZenithAngleFileName), Exist=ZenithFileExists)

      if(debug .eq. 1) print *, 'WriteFrontData #120::',
     1     trim(ZenithAngleFileName), ':: exists ', ZenithFileExists

c     If the zenith angle file exists, open it and get the zenith anglt
c      data

      if(ZenithFileExists .eqv. .true.) then

         allocate( ZenithAngle(1:LenX,1:LenY),
     2        stat=ierr)

         call ReadZenithAngleFile( ZenithAngleFileName, ncIDZA, 
     1        ZenithAngle)

c     Get the zenith angle attributes.

         VariableName = BlankFill(:len(VariableName))
         VariableName = 'zenith_angle'

         VariableType = BlankFill(:len(VariableType))
         VariableType = 'real'

         call GetZenithAngleAttributes(ZenithAngleFileName)

         if(Debug .eq. 1) print *, 'WriteFrontData #125: ',
     1        'ZenithAngleOffsetIn, ZenithAngleScaleFactorIn, ',
     2        'ZenithAngleFillValueIn ', ZenithAngleOffsetIn, 
     3        ZenithAngleScaleFactorIn, ZenithAngleFillValueIn

      endif

c     Create and open the fronts output file.

      call CreateNetCDFDimFile( FrontsFileName, MedianFileName,
     1     GeoSobelFileName, GradOffsetO, GradScaleFactorO,
     2     ZenithAngleFileName, ProgName)


c     The file was opened in the call above and not closed. The netCDF 
c     ID for the file is in a common statement in parameterstatements.f

c     Get the lat, lon fields if the LatLonFileName has changed since
c     the last SST file processed.

      if(debug .eq. 1) then
         print *, 'WriteFrontData #130: LatLonFileName::',
     1        trim(LatLonFileName), '::'
         print *, '#131: LatLonFileNameSave::', 
     1        trim(LatLonFileNameSave), '::'
      endif
      
      if( LatLonFileNameSave(:len(trim(LatLonFileName))) .ne. 
     1     trim(LatLonFileName) ) then

         LatLonFileNameSave = BlankFill(:len(LatLonFileNameSave))
         LatLonFileNameSave = LatLonFileName

         if(Debug .eq. 1) then
            print *, 'WriteFrontData #140. Reading LatLon file.'
            print *, '#141: LatLonFileName::',
     1           trim(LatLonFileName),  '::'
            print *, '#142 LatLonFileNameSave::',
     1           trim(LatLonFileNameSave), '::'
         endif

         call ReadLatLon( LatLonFileName, Lat_Array, Lon_Array, 
     1        LatLonFillValueReal)

      endif

c-----------------------------------------------------------------------

c     Calculate factors used to determine cross-front gradient arrays.

      Factor1 = GradScaleFactorIn / GradScaleFactorO
      Factor2 = (GradOffsetIn - GradOffsetO) / GradScaleFactorO

c     Initialize counters and ranges before looping over front pixels.

      CFRecIndex = 0
      RecIndex = 0
      SegmentIndex = 0

      MinLat = 9999.99
      MaxLat = -9999.99
      MinLon = MinLat
      MaxLon = MaxLat

c     SegStart set to -1. It will be reset after the first complete
c      segment has been read in. In the mean time, it serves as a 
c      flag that we are still reading the first segment.

      SegStart = -1

      if(Debug .eq. 1) print *, 'WriteFrontData #145: NFrontPixel: ',
     1     NFrontPixel

      do 3000 iFP=1,NFrontPixel

         RecIndex = RecIndex + 1

         m = xLocation(iFP)
         n = yLocation(iFP)

         status = nf90_put_var( dim2ID, xLocID, xLocation(iFP),
     1        start=(/recIndex/))
         if(status .ne. nf90_noerr) call handle_err(status)

         status = nf90_put_var( dim2ID, yLocID, yLocation(iFP),
     1        start=(/recIndex/))
         if(status .ne. nf90_noerr) call handle_err(status)

c--- START if(xLocation(iFP) .eq. FillValueInt4) then
         if(xLocation(iFP) .eq. FillValueInt4) then

            SSTMeanA_JF = FillValueReal
            SSTMeanB_JF = FillValueReal
            sstThresholdout = FillValueInt2

            aLon = FillValueReal
            aLat = FillValueReal

c     Write out the center-difference gradients. Divide by 2 so that it
c     is ParameterUnits/pixel.
            if(SIED_Full) then
               status = nf90_put_var( dim2ID, iGradID, FillValueInt2,
     1              start=(/recIndex/))
               if(status .ne. nf90_noerr) call handle_err(status)
               
               status = nf90_put_var( dim2ID, jGradID, FillValueInt2,
     1              start=(/recIndex/))
               if(status .ne. nf90_noerr) call handle_err(status)
            endif
c     A segment ends here if this is not the first front record

c---  START if(iFP .gt. 1) then
            if(iFP .gt. 1) then

c     Note that the segment corresponding to SegmentIndex (below) is the
c     segment ending at iFP - 1.

               SegmentIndex = SegmentIndex + 1
               SegStart = iFP - SegLength(SegmentIndex)
               SegEnd = iFP - 1

c     Check to make sure that the front point before the start of this
c     segment is a fill value; i.e., that the segment has been lined
c     up properly with the data.

               if(xLocation(SegStart-1) .ne. FillValueInt4) then
                  print *, 'WriteFrontData #150: ************* ERROR ',
     1                 '************* Something wrong with this segment'
                  print *, '#151: SegStart: ', SegStart, 
     1                 ' Segment length: ', SegLength(SegmentIndex),
     2                 ' Current pixel number: ', iFP,
     3                 ' xLocation: ', xLocation(SegStart-1)
                  stop
               endif
               
c-----------------------------------------------------------------------
c     Everything looks OK with this segment, write out one line of
c     blank stuff for cross-front vectors and parameters determined 
c     from them. This line corresponds to the null segment separator 
c     in the front pixel list.

c     BE CAREFUL with the CFRecIndex, the cross-front arrays should 
c     correspond to the front variables which are written out with
c     RecIndex. The subroutine does check at the end of each segment
c     to make sure that CFRecIndex has been properly incremented for
c     this segment.

               CFRecIndex = CFRecIndex + 1

c     Generate a cross-front line of fill values for the gradient.

               do 250 ip=-CFPixels,CFPixels
                  jp = ip + CFPixels + 1
                  CFSST(jp) = 0
                  CFmLoc(jp) = FillValueInt1
                  CFnLoc(jp) = FillValueInt1
                  CFEastGrad(jp) = FillValueInt2
 250           continue
               if(SIED_Full) then
                  status = nf90_put_var( dim2ID, SSTcfID, CFSST, 
     1                 start=(/CFrecIndex, 1/), count=(/1, 17/))
                  if(status .ne. nf90_noerr) call handle_err(status)
                  
                  status = nf90_put_var(dim2ID, SSTDiffID,FillValueInt2, 
     1                 start=(/CFrecIndex/))
                  if(status .ne. nf90_noerr) call handle_err(status)
               endif
c     Now fillvalue cross-front and along front indices.
               if(SIED_Full) then
                  status = nf90_put_var( dim2ID, CFmID, CFmLoc, 
     1                 start=(/CFrecIndex, 1/), count=(/1, 17/))
                  if(status .ne. nf90_noerr) call handle_err(status)
                  
                  status = nf90_put_var( dim2ID, CFnID, CFnLoc, 
     1                 start=(/CFrecIndex, 1/), count=(/1, 17/))
                  if(status .ne. nf90_noerr) call handle_err(status)
               endif
c     Sobel gradients if an input GeoSobel file. Null line

               if(GeoSobelExists .eqv. .true.) then

c     Eastward gradient
                  
                  status = nf90_put_var( dim2ID, CFEastGradID, 
     1                 CFEastGrad, start=(/CFrecIndex, 1/), 
     2                 count=(/1, 17/))
                  if(status .ne. nf90_noerr) call handle_err(status)
                  
                  status = nf90_put_var( dim2ID, CFBGEastGradID, 
     1                 FillValueInt2, start=(/CFrecIndex/))
                  if(status .ne. nf90_noerr) call handle_err(status)
                  
                  status = nf90_put_var( dim2ID, CFEastGradMaxID, 
     1                 FillValueInt2, start=(/CFrecIndex/))
                  if(status .ne. nf90_noerr) call handle_err(status)
                  
c     Northward gradient - using CFEastGrad since it is all fill values.
                  
                  status = nf90_put_var( dim2ID, CFNorthGradID, 
     1                 CFEastGrad, start=(/CFrecIndex, 1/), 
     2                 count=(/1, 17/))
                  if(status .ne. nf90_noerr) call handle_err(status)
                  
                  status = nf90_put_var( dim2ID, CFBGNorthGradID, 
     1                    FillValueInt2, start=(/CFrecIndex/))
                  if(status .ne. nf90_noerr) call handle_err(status)
                  
                  status = nf90_put_var( dim2ID, CFNorthGradMaxID, 
     1                 FillValueInt2, start=(/CFrecIndex/))
                  if(status .ne. nf90_noerr) call handle_err(status)
                  
               endif
               
               if((Debug .eq. 1) .and. (SegmentIndex .eq. SegNoTest)) 
     1              then
                  print *, 'WriteFrontData #160:  i, CFRecIndex, ',
     1                 'CFSST: ', i, CFRecIndex, CFSST
                  print *, '#161: sstPopulation A: ', SSTMeanA,
     1                 ' sstPopulation B: ', SSTMeanB
               endif


c-----------------------------------------------------------------------
c     Now loop over points in this segment to get cross-front quantities
c-----------------------------------------------------------------------

               if( (Debug .eq. 1) .and. (SegmentIndex .eq. SegNoTest) )
     1              print *, 'WriteFrontData #165: SegmentIndex, ',
     2              'SegStart, SegEnd: ', SegmentIndex, SegStart, SegEnd

               do 200 i=SegStart,SegEnd

c     First get an approximate orientation of the front in this region.
                  
                  if(SegEnd-SegStart.lt.5)then
                     DeltaX = xLocation(SegEnd)-xLocation(SegStart)
                     DeltaY = yLocation(SegEnd)-yLocation(SegStart)
                  else
                  if(i .lt. SegStart+2) then
                     DeltaX = xLocation(SegStart+4) - 
     1                    xLocation(SegStart)
                     DeltaY = yLocation(SegStart+4) - 
     1                    yLocation(SegStart)
                  elseif(i .gt. SegEnd-2) then
                     DeltaX = xLocation(SegEnd) - xLocation(SegEnd-4)
                     DeltaY = yLocation(SegEnd) - yLocation(SegEnd-4)
                  else
                     DeltaX = xLocation(i+2) - xLocation(i-2)
                     DeltaY = yLocation(i+2) - yLocation(i-2)
                  endif
               endif

                  d = sqrt(DeltaX * DeltaX + DeltaY * DeltaY)

                  FactorX = DeltaY / d
                  FactorY = DeltaX / d

                  if((Debug .eq. 1) .and. 
     1                 (SegmentIndex .eq. SegNoTest)) then
                     print *, 'WriteFrontData #170: DeltaX, DeltaY, ',
     1                    'd: ',  DeltaX, DeltaY, d, 
     2                    ' Factor1, Factor2: ', Factor1, Factor2
                  endif

c**** Now get the cross-front temperature section, where cross-front
c     is normal to the approximate orientation of the front.

c     Zero various counters and accumulators first.

                  SSTMeanA = 0
                  SSTMeanB = 0
                  nGoodA = 0
                  nGoodB = 0

                  CFBGNorthGrad = 0
                  nGoodBGNorthGrad = 0
                  CFNorthGradMax = 0

                  CFBGEastGrad = 0
                  nGoodBGEastGrad = 0
                  CFEastGradMax = 0

                  do 210 ip=-CFPixels,CFPixels
                     
                     mp1 = min( LenX, max( 1, 
     1                    xLocation(i) - nint(ip * FactorX)))
                     np1 = min( LenY, max( 1, 
     1                    yLocation(i) + nint(ip * FactorY)))

c     Get the index for this point in the cross-front coordinate system.

                     jp = ip + CFPixels + 1

c     Store the locat of this pixel in the original image in the cross-
c     front coordinate system.

                     CFmLoc(jp) = mp1 - xLocation(i)
                     CFnLoc(jp) = np1 - yLocation(i)

c     Cross-front temperatures.

                     CFSST(jp) = filpic(mp1,np1)

                     if((Debug .eq. 1) .and. 
     1                    (SegmentIndex .eq. SegNoTest)) 
     2                    print *, 'WriteFrontData #180: i, ip, jp, ',
     3                    'mp1, np1, xlocation, ylocation, CFmLoc, ',
     4                    'CFnLoc, CFSST: ',
     4                    i, ip, jp, mp1, np1, xlocation(i), 
     5                    ylocation(i), CFmLoc(jp),  CFnLoc(jp), 
     6                    CFSST(jp)

c     Average the temperture on each side of the front.

                     if((ip .lt. 0) .and. (CFSST(jp) .gt. 0)) then
                        SSTMeanA = SSTMeanA + CFSST(jp)
                        nGoodA = nGoodA + 1
                     endif

                     if((ip .gt. 0) .and. (CFSST(jp) .gt. 0)) then
                        SSTMeanB = SSTMeanB + CFSST(jp)
                        nGoodB = nGoodB + 1
                     endif                     

c     If GeoSobel exists, calculate various quantities for the 
c     eastward and northward cross-front Sobel gradients. Note that
c     the scale factor is different for the gradients written out
c     by this subroutine and those read in from GeoSobel. This is
c     so that we can write them out as integers to save space.

                     if(GeoSobelExists .eqv. .true.) then

c     Eastward gradients.

                        if(EastGrad(mp1,np1) .eq. GradFillValueIn) then
                           CFEastGrad(jp) = FillValueInt2
                        else
                           CFEastGrad(jp) = 
     1                          EastGrad(mp1,np1) * Factor1 + Factor2

c     Calculate the mean background gradient, where background means
c     the gradient between 5 and CFPixels (inclusive) pixels of the
c     front along the cross-front line.

                           if(abs(ip) .ge. 5) then 
                              CFBGEastGrad = CFBGEastGrad + 
     1                             CFEastGrad(jp)
                              nGoodBGEastGrad = nGoodBGEastGrad + 1
                           endif

c     Calculate the gradient at the front. This is the maximum
c     gradient of the cross-front pixels within 3 (inclusive) 
c     pixels of the front.

                           if( (abs(ip) .le. 3) .and.
     1                          (abs(CFEastGrad(jp)) .gt. 
     2                          abs(CFEastGradMax)) ) CFEastGradMax = 
     3                          CFEastGrad(jp)
                        endif

c     if((Debug .eq. 1) .and. 
c     1                       (SegmentIndex .eq. SegNoTest)) print *,
c     2                       '#1  CFEastGrad,  nGoodBGEastGrad,  ',
c     3                       'CFBGEastGrad         : ', 
c     4                       CFEastGrad(jp), nGoodBGEastGrad, 
c     5                       CFBGEastGrad

c     Nrthward gradients.

                        if(Northgrad(mp1,np1) .eq. GradFillValueIn) then
                           CFNorthgrad(jp) = FillValueInt2
                           GradMag(jp) = FillValueReal
                        else
                           CFNorthgrad(jp) = 
     1                          Northgrad(mp1,np1) * Factor1 + Factor2

                           if(abs(ip) .ge. 5) then 
                              CFBGNorthGrad = CFBGNorthGrad + 
     1                             CFNorthGrad(jp)
                              nGoodBGNorthGrad = nGoodBGNorthGrad + 1
                           endif

                           if( (abs(ip) .le. 3) .and.
     1                          (abs(CFNorthGrad(jp)) .gt. 
     2                          abs(CFNorthGradMax)) ) CFNorthGradMax = 
     3                          CFNorthGrad(jp)

c     Have a good northward gradient at this point. If there is also
c     a good eastward gradient, then calculate gradient magnitude.

                           if(CFEastGrad(jp) .ne. FillValueInt2) then
                              
                              CFN = CFNorthGrad(jp)
                              CFE = CFEastGrad(jp)

                              GradMag(jp) = sqrt(CFN * CFN + CFE * CFE)

c     if((Debug .eq. 1) .and. 
c     1                             (SegmentIndex .eq. SegNoTest)) 
c     2                             print *, '#2 CFEastGrad, ',
c     3                             'CFNorthGrad GradMag: ', 
c     4                             CFEastGrad(jp),  CFNorthGrad(jp),
c     5                             GradMag(jp)
                           else
                              GradMag(jp) = FillValueReal

c     if((Debug .eq. 1) .and. 
c     1                             (SegmentIndex .eq. SegNoTest)) 
c     2                             print *, '#3 CFEastGrad, ',
c     3                             'CFNorthGrad GradMag: ', 
c     4                             CFEastGrad(jp),  CFNorthGrad(jp),
c     5                             GradMag(jp)
                           endif

                        endif

c     if((Debug .eq. 1) .and. 
c     1                       (SegmentIndex .eq. SegNoTest)) print *,
c     2                       '#4 CFNorthGrad, nGoodBGNorthGrad, ',
c     3                       'CFBGNorthGrad, GradMag: ', 
c     4                       CFNorthGrad(jp), nGoodBGNorthGrad, 
c     5                       CFBGNorthGrad, GradMag(jp)

                     endif
 210              continue

c     if((Debug .eq. 1) .and. 
c     1                 (SegmentIndex .eq. SegNoTest)) print *,
c     2                 '#10 GradMag: ', GradMag

                  if((Debug .eq. 1) .and. 
     1                 (SegmentIndex .eq. SegNoTest)) then
                     print *, '#10 CFmLoc: ', CFmLoc
                     print *, '#11 CFnLoc: ', CFnLoc
                  endif

c     Now get the location of the maximum gradient magnitude in the 
c     cross-frontal coordinate system. This is a quasi-front quality
c     control: If the location of the maximum gradient magnitude is far
c     from the front, then the contour following portion probably went
c     astray.

                  if(GeoSobelExists .eqv. .true.) then
                     MaxGradMag = -1
                     I_MaxGradMag = FillValueInt2
                     do 212 ip=-CFPixels+1,CFPixels-1
                        jp = ip + CFPixels + 1
                        lp = 0
                        SumOfMags = 0
                        do 214 kp=jp-1,jp+1
                           if( (CFNorthGrad(kp) .ne. FillValueInt2) 
     1                          .and. 
     2                          (CFEastGrad(kp) .ne. FillValueInt2) )
     3                          then
                              lp = lp + 1
                              SumOfMags = SumOfMags + GradMag(kp)

                              if((Debug .eq. 1) .and. 
     1                             (SegmentIndex .eq. SegNoTest)) 
     2                             print *, 'WriteFrontData #192: ip, ',
     3                             'jp, lp, kp, SumOfMags: ',  ip, jp, 
     4                             lp, kp, SumOfMags
                              
                           endif
 214                    continue

                        if(lp .ne. 0) then 
                           MeanGradMag = SumOfMags / lp

                           if(MeanGradMag .gt. MaxGradMag) then
                              MaxGradMag = MeanGradMag
                              I_MaxGradMag = ip
                           endif

                           if( (MeanGradMag .eq. MaxGradMag) .and.
     1                          (ip .lt. 0) ) then
                              MaxGradMag = MeanGradMag
                              I_MaxGradMag = ip
                           endif
                           
                        endif
 212                 continue                     

                     if((Debug .eq. 1) .and. 
     1                    (SegmentIndex .eq. SegNoTest)) print *,
     2                    'WriteFrontData #194 I_MaxGradMag: ', 
     3                    I_MaxGradMag

c     Write out the location of the maximum gradient magnitude.

                     status = nf90_put_var( dim2ID, I_MaxGradMagID, 
     1                    I_MaxGradMag, start=(/CFrecIndex/))
                     if(status .ne. nf90_noerr) call handle_err(status)

                  endif

c     Get the mean temperature on the two sides of the front and their
c     difference.

                  SSTMeanA = SSTMeanA / nGoodA
                  SSTMeanB = SSTMeanB / nGoodB
                  SSTDiff = (SSTMeanA - SSTMeanB) * 10

c     Now write out the data for this frontal point.

c     BE CAREFUL with the CFRecIndex, the cross-front arrays should 
c     correspond to the front variables which are written out with
c     RecIndex.

                  CFRecIndex = CFRecIndex + 1

                  if((Debug .eq. 1) .and. (SegmentIndex .eq. SegNoTest))
     1                 then
                     print *, 'WriteFrontData #200:  i, CFRecIndex, ',
     1                    'CFSST: ', i, CFRecIndex, CFSST
                     print *, '#201: sstPopulation A: ', SSTMeanA,
     1                    ' sstPopulation B: ', SSTMeanB
                     print *, '#202: CFBGEastGrad: ', CFBGEastGrad, 
     2                    ' CFBGNorthGrad: ', CFBGNorthGrad,
     3                    ' CFEastGradMax: ', CFEastGradMax,
     4                    ' CFNorthGradMax: ', CFNorthGradMax
                  endif

c     Write out the location in the original SST image of the cross-
c     front pixels pixels

                  if((Debug .eq. 1) .and. 
     1                 (SegmentIndex .eq. SegNoTest)) then
                     print *, '#12 CFmLoc: ', CFmLoc
                     print *, '#13 CFnLoc: ', CFnLoc
                  endif

                  status = nf90_put_var( dim2ID, CFmID, CFmLoc, 
     1                 start=(/CFrecIndex, 1/), count=(/1, 17/))
                  if(status .ne. nf90_noerr) call handle_err(status)

                  status = nf90_put_var( dim2ID, CFnID, CFnLoc, 
     1                 start=(/CFrecIndex, 1/), count=(/1, 17/))
                  if(status .ne. nf90_noerr) call handle_err(status)

                  if((Debug .eq. 1) .and. 
     1                 (SegmentIndex .eq. SegNoTest)) then
                     print *, '#14 CFmLoc: ', CFmLoc
                     print *, '#15 CFnLoc: ', CFnLoc
                  endif

c     Write out the cross-front SST and the delta SST
                  if(SIED_Full) then
                     status = nf90_put_var( dim2ID, SSTcfID, CFSST, 
     1                    start=(/CFrecIndex, 1/), count=(/1, 17/))
                     if(status .ne. nf90_noerr) call handle_err(status)
                  endif
                  status = nf90_put_var( dim2ID, SSTDiffID, 
     1                 SSTDiff, start=(/CFrecIndex/))
                  if(status .ne. nf90_noerr) call handle_err(status)

c     Write out the cross-front eastward and northward gradients and
c     the background gradients and in-front gradients - the maximum
c     gradient within +/- 1 pixel of the front position, normal to the
c     front.

                  if(GeoSobelExists .eqv. .true.) then
                     
c     First get the mean eastward and northward background gradients.

                     if(nGoodBGEastGrad .eq. 0) then
                        CFBGEastGrad = FillValueInt2
                     else
                        CFBGEastGrad = CFBGEastGrad / nGoodBGEastGrad
                     endif
                     
                     if(nGoodBGNorthGrad .eq. 0) then
                        CFBGNorthGrad = FillValueInt2
                     else
                        CFBGNorthGrad = CFBGNorthGrad / nGoodBGNorthGrad
                     endif

c     Now you can write out the friggin values.
                     status = nf90_put_var( dim2ID, CFEastGradID, 
     1                    CFEastGrad, start=(/CFrecIndex, 1/), 
     2                    count=(/1, 17/))
                     if(status .ne. nf90_noerr) call handle_err(status)
                     
                     status = nf90_put_var( dim2ID, CFBGEastGradID, 
     1                    CFBGEastGrad, start=(/CFrecIndex/))
                     if(status .ne. nf90_noerr) call handle_err(status)
                     
                     status = nf90_put_var( dim2ID, CFEastGradMaxID, 
     1                    CFEastGradMax, start=(/CFrecIndex/))
                     if(status .ne. nf90_noerr) call handle_err(status)
                     
c     and the northward gradient.
                     
                     status = nf90_put_var( dim2ID, CFNorthGradID, 
     1                    CFNorthGrad, start=(/CFrecIndex, 1/), 
     2                    count=(/1, 17/))
                     if(status .ne. nf90_noerr) call handle_err(status)

                     status = nf90_put_var( dim2ID, CFBGNorthGradID, 
     1                    CFBGNorthGrad, start=(/CFrecIndex/))
                     if(status .ne. nf90_noerr) call handle_err(status)

                     status = nf90_put_var( dim2ID, CFNorthGradMaxID, 
     1                    CFNorthGradMax, start=(/CFrecIndex/))
                     if(status .ne. nf90_noerr) call handle_err(status)
                  endif

c     End of loop over cross-front points.

 200           continue

c     Write out the starting location and length for this segment.

               status = nf90_put_var( dim2ID, SegStartID,
     1              SegStart, start=(/SegmentIndex/))
               if(status .ne. nf90_noerr) call handle_err(status)

               status = nf90_put_var( dim2ID, SegLengthID,
     1              SegLength(SegmentIndex), start=(/SegmentIndex/))
               if(status .ne. nf90_noerr) call handle_err(status)


c     Now for lat, lon min and max.

               status = nf90_put_var( dim2ID, LonMinID, MinLon, 
     1              start=(/SegmentIndex/))
               if(status .ne. nf90_noerr) call handle_err(status)

               status = nf90_put_var( dim2ID, LonMaxID, MaxLon, 
     1              start=(/SegmentIndex/))
               if(status .ne. nf90_noerr) call handle_err(status)

               status = nf90_put_var( dim2ID, LatMinID, MinLat, 
     1              start=(/SegmentIndex/))
               if(status .ne. nf90_noerr) call handle_err(status)

               status = nf90_put_var( dim2ID, LatMaxID, MaxLat, 
     1              start=(/SegmentIndex/))
               if(status .ne. nf90_noerr) call handle_err(status)

c     And set the min and max to zero for the next segment.

               MinLat = 9999.99
               MaxLat = -9999.99
               MinLon = MinLat
               MaxLon = MaxLat

c     Check to make sure that the same number of cross-front vectors
c     and front data points are still tracking. The number of cross-
c     records should be one less because nothing has been written for
c     the null record that resulted in this portion of the program.

               if(CFRecIndex .ne. RecIndex-1) then
                  print *, 'WriteFrontData #210: *********************',
     1                 ' The number of cross-front vectors does not ',
     2                 'agree with the number of front vectors. STOP.'
                  print *, '#211: CFRecIndex: ', CFRecIndex, 
     1                 ' RecIndex: ', RecIndex
                  stop
               endif

c--- END if(iFP .gt. 1) then
            endif

c-----------------------------------------------------------------------
c     End processing for this segment; next section to process a good 
c     front pixel.
c-----------------------------------------------------------------------

c--- ELSE if(xLocation(iFP) .eq. FillValueInt4) then
         else

c     Write out the center-difference gradients. Divide by 2 so that it
c     is ParameterUnits/pixel.

            status = nf90_put_var( dim2ID, iGradID, GradV( 1, m, n),
     1           start=(/recIndex/))
            if(status .ne. nf90_noerr) call handle_err(status)

            status = nf90_put_var( dim2ID, jGradID, GradV( 2, m, n),
     1           start=(/recIndex/))
            if(status .ne. nf90_noerr) call handle_err(status)

c     if(MinX .gt. m) MinX = m
c     if(MaxX .lt. m) MaxX = m
c     if(MinY .gt. n) MinY = n
c     if(MaxY .lt. n) MaxY = n

            if(OutPicSave(m,n) .gt. 0) then
               SSTMeanA_JF = ppv(OutPicSave(m,n),2)
               SSTMeanB_JF = ppv(OutPicSave(m,n),3)
               sstThresholdout = ppv(OutPicSave(m,n),4)
            else
               SSTMeanA_JF = FillValueReal
               SSTMeanB_JF = FillValueReal
               sstThresholdout = FillValueInt2
            endif

            if((Debug .eq. 1) .and. 
     1           (SegmentIndex .eq. SegNoTest)) 
     2           print *, 'WriteFrontData #220: iFP, m, n, ',
     3           'OutPicSave, SSTMeanA_JF, SSTMeanB_JF, ', 
     4           'sstThresholdout: ', iFP, m, n, OutPicSave(m,n), 
     5           SSTMeanA_JF, SSTMeanB_JF, sstThresholdout

c     Get the lat, lon location and the range for this segment.

            alon = Lon_Array(xLocation(iFP),yLocation(iFP))
            alat = Lat_Array(xLocation(iFP),yLocation(iFP))

            if(MinLat .gt. aLat) MinLat = aLat
            if(MaxLat .lt. aLat) MaxLat = aLat
            if(MinLon .gt. aLon) MinLon = aLon
            if(MaxLon .lt. aLon) MaxLon = aLon

c---  END if(xLocation(iFP) .eq. FillValueInt4) then
         endif

c     Write out the population temperature means from SIED.

         status = nf90_put_var( dim2ID, SSTaID, SSTMeanA_JF,
     1        start=(/recIndex/))
         if(status .ne. nf90_noerr) call handle_err(status)

         status = nf90_put_var( dim2ID, SSTbID, SSTMeanB_JF,
     1        start=(/recIndex/))
         if(status .ne. nf90_noerr) call handle_err(status)

         status = nf90_put_var( dim2ID, SSTcID, sstThresholdout,
     1        start=(/recIndex/))
         if(status .ne. nf90_noerr) call handle_err(status)

c    Write out the lat, lon varaibles.

         status = nf90_put_var( dim2ID, lonRecID, aLon,
     1        start=(/recIndex/))
         if(status .ne. nf90_noerr) call handle_err(status)

         status = nf90_put_var( dim2ID, latRecID, aLat,
     1        start=(/recIndex/))
         if(status .ne. nf90_noerr) call handle_err(status)

         if((Debug .eq. 1) .and. ((SegStart .eq. -1) .or.
     1        (SegmentIndex .eq. SegNoTest))) 
     2        print *, 'WriteFrontData #225: iFP, xlocation, ',
     3        'ylocation, latitude, longitude: ', iFP, xlocation(iFP), 
     4        ylocation(iFP), aLat, aLon, 'SSTMeanA_JF, SSTMeanB_JF, ', 
     5        'sstThresholdout: ', SSTMeanA_JF, SSTMeanB_JF, 
     6        sstThresholdout

c     Finally write out the zenith angle for this locations if one is
c     available.

         if(ZenithFileExists .eqv. .true.) then
            if( xlocation(iFP) .eq. FillValueInt4) then
               status = nf90_put_var( dim2ID, zenithAngRecID, 
     1              FillValueInt2, start=(/recIndex/))
            else
               status = nf90_put_var( dim2ID, zenithAngRecID, 
     1              ZenithAngle(m,n), start=(/recIndex/))
            endif
         endif
         if(status .ne. nf90_noerr) call handle_err(status)

 3000 continue

c     Write out one more blank line for cross-front data. This line 
c     corresponds to the final null segment separator in the front 
c     pixel list.

c     BE CAREFUL with the CFRecIndex, the cross-front arrays should 
c     correspond to the front variables which are written out with
c     RecIndex.

      CFRecIndex = CFRecIndex + 1

      do 252 ip=-CFPixels,CFPixels
         jp = ip + CFPixels + 1
         CFSST(jp) = 0
         CFmLoc(jp) = FillValueInt1
         CFnLoc(jp) = FillValueInt1
         CFEastGrad(jp) = FillValueInt2
 252  continue

      if((Debug .eq. 1) .and. (SegmentIndex .eq. SegNoTest)) then
         print *, 'WriteFrontData #230:  i, CFRecIndex, ',
     1        'CFSST: ', i, CFRecIndex, CFSST
         print *, '#221 sstPopulation A: ', SSTMeanA_JF,
     1        ' sstPopulation B: ', SSTMeanB_JF
      endif
      if(SIED_Full) then
         status = nf90_put_var( dim2ID, SSTcfID, CFSST, 
     1        start=(/CFrecIndex, 1/), count=(/1, 17/))
         if(status .ne. nf90_noerr) call handle_err(status)
      endif
      status = nf90_put_var( dim2ID, SSTDiffID, FillValueInt2, 
     1     start=(/CFrecIndex/))
      if(status .ne. nf90_noerr) call handle_err(status)

c     Now fillvalue cross-front and along front indices.

      status = nf90_put_var( dim2ID, CFmID, CFmLoc, 
     1     start=(/CFrecIndex, 1/), count=(/1, 17/))
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_var( dim2ID, CFnID, CFnLoc, 
     1     start=(/CFrecIndex, 1/), count=(/1, 17/))
      if(status .ne. nf90_noerr) call handle_err(status)

c     Sobel gradients if an input GeoSobel file.

      if(GeoSobelExists .eqv. .true.) then

c     Eastward gradient

         status = nf90_put_var( dim2ID, CFEastGradID, 
     1        CFEastGrad, start=(/CFrecIndex, 1/), 
     2        count=(/1, 17/))
         if(status .ne. nf90_noerr) call handle_err(status)

         status = nf90_put_var( dim2ID, CFBGEastGradID, 
     1        FillValueInt2, start=(/CFrecIndex/))
         if(status .ne. nf90_noerr) call handle_err(status)

         status = nf90_put_var( dim2ID, CFEastGradMaxID, 
     1        FillValueInt2, start=(/CFrecIndex/))
         if(status .ne. nf90_noerr) call handle_err(status)

c     Northward gradient - using CFEastGrad since it is all fill values.

         status = nf90_put_var( dim2ID, CFNorthGradID, 
     1        CFEastGrad, start=(/CFrecIndex, 1/), 
     2        count=(/1, 17/))
         if(status .ne. nf90_noerr) call handle_err(status)

         status = nf90_put_var( dim2ID, CFBGNorthGradID, 
     1        FillValueInt2, start=(/CFrecIndex/))
         if(status .ne. nf90_noerr) call handle_err(status)
         
         status = nf90_put_var( dim2ID, CFNorthGradMaxID, 
     1        FillValueInt2, start=(/CFrecIndex/))
         if(status .ne. nf90_noerr) call handle_err(status)
      endif
      
c     Don't close this file yet; still need to write the cohesion
c     . of clear pixels in each window and number of clear pixels in
c     . these windows. This is done in sing_edge. The netCDF ID for
c     . the output file is passed in a common defined in 
c     . ParameterStatements.

c$$$c     Close  output file.
c$$$
c$$$      status = nf90_close( dim2ID )
c$$$      if(status .ne. nf90_noerr) call handle_err(status)

c     Final check to make sure that the same number of cross-front
c     vectors were written as front data points.

      if(CFRecIndex .ne. RecIndex) then
         print *, 'WriteFrontData #240: *********************',
     1        ' The number of cross-front vectors does not ',
     2        'agree with the number of front vectors. STOP.'
         print *, '#231: CFRecIndex: ', CFRecIndex, 
     1        ' RecIndex: ', RecIndex
         stop
      endif
      
      if(debug .eq. 1) print *, 'WriteFrontData #999'

      return
      end subroutine WriteFrontData

c***********************************************************************
c***********************************************************************

      include 'CommonSubroutines-2.36.f'
