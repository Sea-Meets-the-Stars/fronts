c***********************************************************************
      program Pmerge_Main
c***********************************************************************

c     
c     This program merges the fronts of all images in the /sst/ 
c     directories of either /goes/ or /meteosat/ within a prescribed
c     amount of time or number of images of the current image. 
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
c     pcornillon@me.com 9 May 2009
c     
c     MODIFICATIONS
c     
c     3/20/11 PCC - Pmerge version number ==> 1.01
c     Changed test on ccount in accumul to avoid trying to go through 
c     do loops as do xxxx i=1,ccount and to avoid printing a quasi  
c     error message if 0 segments in a file that is being merged. 
c     This should not change the actual output merged file.
c     
c     4/4/11 PCC - All subroutines 
c     Changed all logical*1 to logical
c     Changed type of MergedTempFileName from logical to char*150
c     
c     5/12/11 PCC- Pmerge version number ==> 1.1
c     Changed to use netCDF4.
c     Added check following filestatus for median to see if 
c     compressed. If so, skip processing of this file. It should not
c     be for this version.
c     Changed cleanup to simply closing the open median file since no
c     compression for these files.
c     
c     6/1/11 - PCC - Added Msg50 in calls to PrintArray2 & ElapsedTime
c     
c     6/3/11 - PCC - 
c     Moved test for existence of output Pmerge file to preceed read of
c     Median data. This avoids the Median data being read even when it
c     will not be used.
c     Also added a print line to indicate when a file was being merged.
c     
c     6/6/11 - PCC - Changed length of SatName from 8 to 20 characters.
c     
c     6/7/11 - PCC - Major changes to simplify and correct handling of
c     strings. Removed all EndOfStrings. Added BlankFill before any 
c     definition of a string variable to force the end of the string 
c     to be all blanks.
c     
c     Version 1.10 ==> 1.11
c     
c     1/11/12 - PCC - Changed string lengths from 150 to 319 for file
c     names. Also added an explicit definition for uuid_gen.
c     
c     Version 1.11 ==> 2.00
c     
c     4/18/12 - PCC - removed SatID; it is no longer used. No impact.
c     
c     Version 2.00 ==> 2.01
c     
c     9/9/12 - PCC - Changed Common... to CommonSubroutines-2.09.f
c     Change printout of "Thinning..." line to trim filename.
c     
c     Version 2.01 ==> 2.02
c     
c     11/30/12 - PCC - Changed Common... to CommonSubroutines-2.12.f
c     
c     Version 2.02 ==> 3.00
c     
c     1/25/13 - PCC - CommonSubroutines-2.12 ==> 2.19.
c     . Changed reading of cnt files to Front files.
c     . Removed write to unitlog for debug statements. Cleaned up many 
c     .  of the print statements.
c     . Added code that gets starting and ending wall time and 
c     .  calculates the number of files processed/min. Also prints 
c     .  out the fraction of the processing completed.
c     1/28/13 -PCC - SatName moved to ArchiveDefinition common in 
c     .  ParameterStatements. This required removing SatName from 
c     .  subroutine calls and definitions for ReadInputs, CheckArchive 
c     .  and ReplaceBaseDir.
c     1/30/12 - PCC - Changed the directory and names for the input cnt
c     .  files, which are not fronts_xxx_pass files to correspond to the
c     .  output from SIED.
c     . Added code to calculate wSize, the number of pixels to use when 
c     .  matching fronts in adjacent images with gradients in the image 
c     .  for which the fronts are being merged.
c     2/1/13 - PCC - replaced code associated with checks for the 
c     .  existence of input files through calls to FileStatus and the
c     .  subsequent print outs with simple tests and printouts. Makes
c     .  the code a lot easier to read and was no longer needed since
c     .  none of the input files should be compressed anymore which
c     .  was the major purpose for FileStatus.
c     . Used the Aquamacs command to indent the entire program.
c     . Cleaned up debug statements.
c
c     Version 3.00 ==> 3.01
c     
c     2/2/13 -PCC - CommonSubroutines 2.19 ==> 2.21
c
c     Version 3.01 ==> 3.02
c     
c     3/15/13 -PCC - CommonSubroutines 2.21 ==> 2.23
c
c     Version 3.02 ==> 3.03
c     
c     4/26/13 -PCC - CommonSubroutines 2.23 ==> 2.29
c
c     Version 3.03 ==> 3.04
c     
c     4/26/13 -PCC - CommonSubroutines 2.29 ==> 2.35
c
c     2/11/14 -JPS - CommonSubroutines 2.35 ==> 2.36
c
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

c     Cnt1Compressed - the compression flag for the Cnt file. 
      logical Cnt1Compressed
c     Cnt1Exists - file status for Cnt, .true. if the file exists.
      logical Cnt1Exists
c     Cnt1FileName - the output filename for fronts - uncompressed
      character*319, allocatable :: Cnt1FileName(:)
c     Cnt2Compressed - the compression flag for the Cnt file. 
      logical Cnt2Compressed
c     Cnt2Exists - file status for Cnt, .true. if the file exists.
      logical Cnt2Exists
c     Cnt2FileName - the output filename for fronts - uncompressed
      character*319 Cnt2FileName
c     ConditionedSST - SST after subtracting Offset and truncating.
      integer*2, allocatable :: ConditionedSST(:,:)
c     Compressed - flag telling CleanUp that the input file was not
c     . compressed.
      logical Compressed
c     CPUTime - CPU time used by this program to date
      real CPUTime

c     Debug_2 - debug control. This debug allows printout for loops
c     . over all images in ArchiveInventory. Lots of output which is why
c     . it is separated from Debug.
      integer, parameter :: Debug_2=0
c     Decompress - flag telling FileStatus whether or not to decompress
c     . the file it is evaluating. Will do nothing if the file is not
c     . compressed.
      logical Decompress
c     DecompressInPlace - flag telling FileStatus to decompress the
c     . indicated file inplace, i.e., same name without the .gz. 
c     . Ignores this flag if Decompress is .false.
      logical DecompressInPlace
c     Dim2FileName - name of file with lat and lon located fronts. 
      character*319, allocatable :: Dim2FileName(:)
c     Dum0 - a dummy paramete used in the call to check archive.
      integer Dum0

c     FileNameToUse - is the filename to be use for the iput file. It's
c     . either the filename in the inventory or a temporary filename.
      character*319 FileNameToUse
c     FirstToProcess - true if the first image to be processed
      logical FirstToProcess

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
c     InvSecondsSince1970 - number of hours since 00:00 of 1 January 
c     . 1970 corresponding to the time of a given image. Read from
c     . the inventory
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
c     jImg - the loop parameter when generating the filename lists.
      integer jImg

c     LastChar1 - the location of the last character of a string. 
      integer LastChar1
c     LastChar2 - the last character of another string. 
      integer LastChar2
c     LastCPUTime - CPUTime the last time ElapsedTime was called.
      real LastCPUTime

c     MedianCompressed - the compression flag for the Median file. 
      logical MedianCompressed
c     MedianExists - file status for Median, .true. if the file exists
      logical MedianExists
c     MedianFileName - the name of the output file for the 3x3 median
c     . filter.
      character*319, allocatable :: MedianFileName(:)
c     MedianTempFileName - filename used for the decompressed 
c     . temporary input file.
      character*319 MedianTempFileName 
c     MedAlreadyRead - .true. if the input Median file has already been
c     . read; i.e., MedSST exists.
      logical MedAlreadyRead
c     MergedFronts - image of merged fronts
      integer*2, allocatable :: MergedFronts(:,:)      
c     MedID - ID for the median sst field.
      integer MedID
c     MedSST - 3x3 median filtered value of sst
      integer*2, allocatable :: MedSST(:,:)
c     MergedCompressed - the compression flag for the merged file. 
      logical MergedCompressed
c     MergedExists - file status for merged file.
      logical MergedExists
c     MergedFileName - name of file with images of merged contours.
      character*319, allocatable :: MergedFileName(:)
c     MergedString - the last part of a merged file name. Will be either
c     . _merged_1.nc or _merged_2.nc. The 1 or 2 is for the SIED pass
c     . number. (Actually, I don't think that it will ever be 2, but...
      character*12 MergedString
c     MergedTempFileName - filename for temporary file if decompressing
      character*319 MergedTempFileName
c     Message - message to be written out for call to ElapsedTime.
      character*30 Message

c     ncMedlID - NetCDF ID for Median input/outpout files.
      integer ncMedID
c     ncMergedID - NetCDF ID for merged file.
      integer ncMergedID
c     ncSSTID - NetCDFID for input SST file
      integer ncsstID
c     ncThinnedID - NetCDF ID for thinned file.
      integer ncThinnedID
c     NumberOfFilesProcessed - the number of fields processed in this 
c     . run.
      integer NumberOfFilesProcessed
c     NumberOfGoodImages - the number of images in this archive with 
c     . more than NumClearThresh clear pixels in the image
      integer NumberOfGoodImages

c     r_np1 - run number plus 1. Should be '2'.
      character*1 r_np1

c     SecondsSince1970ID - ID for NetCDF variable.
      integer SecondsSince1970ID
c     SSTCompressed - the compression flag for the input file. 
      logical SSTCompressed
c     SSTExists - returned from FileStatus, .true. if the file exists
      logical SSTExists
c     SSTTempFileName - is the temporary filename used for the
c     . decompressed version of the input SST file.
      character*319 SSTTempFileName
c     SSTFileName - the temporary name of the input file - uncompressed
      character*319, allocatable :: SSTFileName(:)
c     sstID - NetCDF ID for the SST variables
      integer sstID
c     SST - the sst of the input field.
      integer*2, allocatable :: sst(:,:)

c     t - damn, can't recall what I used this for, but I need it.
      real t
c     TempFileName - dummy filename, used when not expecting a 
c     . temporary filename back.
      character*319 TempFileName
c     TFID - NetCDF ID for thinned fronts
      integer TFID

c     wSize - Window sized to use when searching for a match between 
c     the front found in an adjacent image and the gradient contours 
c     in the image image being processed.
      integer*2 wSize

c     xPixelSpacing - The typical x pixel separation in integer km.
      integer*4 xPixelSpacing

c     yPixelSpacing - The typical y pixel separation in integer km.
      integer*4 yPixelSpacing

c     Function to generate UUID.
      character*36 UUID_Gen

c---------------------------------start here ---------------------------

      ProgName = BlankFill(:len(ProgName))
      ProgName = 'Pmerge_Main'

c     Read in/set up input variables.

      call ReadInputs( ProgName, Start, CountToRead, GeoNameIn,
     1     YearStart, YearStartC, YearEnd, MonthStart, MonthEnd,
     2     InventoryFileName)

      if(Debug .eq. 1) print *, 'Pmerge #000'

c     Although we could make more passes of SIED on the data, assume
c     that we will only make 2, so r_n is 1.

      r_n = '1'

c     Allocate space for dynamically allocated variables.

      allocate( Cnt1FileName(1:MaxNumberOfFiles),
     2     Dim2FileName(1:MaxNumberOfFiles),
     3     MedianFileName(1:MaxNumberOfFiles),
     4     MergedFileName(1:MaxNumberOfFiles),
     5     SSTFileName(1:MaxNumberOfFiles),
     6     ConditionedSST(1:LenX,1:LenY),
     7     MergedFronts(1:LenX, 1:LenY),
     8     MedSST(LenX,LenY), 
     9     sst(1:LenX,1:LenY),
     *     stat=ierr)
      
      if (ierr .ne. 0) stop 'Allocation error for LenX, LenY arrays.'

c     Read ArchiveInventory - the names of files to process. First,
c     allocate the arrays to be read.

      include 'AllocateInventoryVariables.f'

      call ReadInventory(InventoryFileName, InvSecondsSince1970, 
     1     InvNumClear, InvYear, InvMonth, InvDay, InvHour, 
     2     InvMinute, InvSecond, InvFileName, iFirst, iLast)

      write(unitlog,*) 'Pmerge #100: Will process ArchiveInventory ',
     1     'from ', iFirst, ' to ', iLast
      print *, 'Pmerge #100: Will process ArchiveInventory from ', 
     1     iFirst, ' to ', iLast

c-------------Generate the list of input and output filenames ---------

c     Need to do this here for pmerge since it uses images before and
c     after the current image.

      iFirst = 1
      iLast = 1
      jImg = 0
      do 2000 iImg=1,NumberOfImages

         if(Debug_2 .eq. 1) print *, 'PMerge #110. iImg: ', iImg

c     Do not include this image if less than Thresh clear pixels in it.

         if(InvNumClear(iImg) .lt. NumClearThresh) go to 2000
         
         jImg = jImg + 1

         if(Debug_2 .eq. 1) print *, 'PMerge #120. jImg: ', jImg

c     Copy inventory values to new location. If no images are skipped
c     the list isn't changed.

         InvYear(jImg) = InvYear(iImg)
         InvMonth(jImg) = InvMonth(iImg)
         InvDay(jImg) = InvDay(iImg)
         InvHour(jImg) = InvHour(iImg)
         InvMinute(jImg) = InvMinute(iImg)
         SSTFileName(jImg) = InvFileName(iImg)
         InvSecondsSince1970(jImg) = InvSecondsSince1970(iImg)

         if(InvSecondsSince1970(jImg) .le. 
     1        SecondsSinceStart) iFirst = jImg
         if(InvSecondsSince1970(jImg) .le. 
     1        SecondsSinceEnd) iLast = jImg

         call ReplaceBaseDir(SSTFileName(jImg))

c     Need to put SSTFileName in TempFileName for string handling below

         TempFileName = SSTFileName(jImg)

c     Check to see if the input file exists. If it doesn't stop.
         
         inquire( File=TempFileName, Exist=SSTExists)

         if(SSTExists .eqv. .false.) then
            write(unitlog, *) 'Pmerge #130::', trim(TempFileName), 
     1           ':: does not exist. Terminating run.'
            print *, 'Pmerge #130::', trim(TempFileName), 
     1           ':: does not exist. Terminating run.'
            stop
         endif
         
         call GetDateCharStrings( 
     1        InvYear(jImg), InvMonth(jImg),  InvDay(jImg),  
     2        InvHour(jImg), InvMinute(jImg),
     3        YearC, MonthC, DayC, HourC, MinuteC)

c     Construct the file name for the input median SST file.

         TempString1 = BlankFill(:len(TempString1))
         TempString1 = 'Median/'
         TempString2 = BlankFill(:len(TempString2))
         TempString2 = '_Median.nc'
         MedianFileName(jImg) = GenerateFileName( TempFileName, 
     1        TempString1, TempString2)

c     Construct the file name for the input front contour files.
         
         if(Debug_2 .eq. 1) print *, 'Pmerge #140: r_n::', r_n, '::'

         TempString1 = BlankFill(:len(TempString1))
         TempString2 = BlankFill(:len(TempString2))

         if(r_n .eq. '1') then
            TempString1 = 'Fronts_1st_Pass/'
            TempString2 = '_fronts_1st_pass.nc'
         elseif(r_n .eq. '2') then
            TempString1 = 'Fronts_2nd_Pass/'
            TempString2 = '_fronts_2nd_pass.nc'
         else
            write(UnitLog, *) 'Pmerge #150: r_n: ', r_n, ' but ',
     1           'should be either 1 or 2. Stopping.'
            print *, 'Pmerge #150: r_n: ', r_n, ' but should be ',
     1           'either 1 or 2. Stopping.'
         endif

         Cnt1FileName(jImg) = GenerateFileName( TempFileName, 
     1        TempString1, TempString2)

c     Construct file name for the merged images written by pmerge.

         TempString1 = BlankFill(:len(TempString1))
         TempString1 = 'Merged/'
         TempString2 = BlankFill(:len(TempString2))
         TempString2 = '_merged_' // r_n // '.nc'

         MergedFileName(jImg) = GenerateFileName( TempFileName, 
     1        TempString1, TempString2)

 2000 continue

c     Reset the counter for the number of images as some may have been
c     skipped because of too little data.

      NumberOfGoodImages = jImg

c-------------------------MAIN LOOP ------------------------------------

c     Get the date and time so that we can determine the files/minute
c     processed

      call date_and_time(VALUES=DateTimeStart)

c     Loop over all images with sufficient data in the time range 

      FirstToProcess = .true.

      NumberOfFilesProcessed = 0
      do 1000 iImg=iFirst,iLast

         if(debug .eq. 1) then
            print *, 'Pmerge #160: iImg: ', iImg,
     1           ' InvSecondsSince1970(iImg): ', 
     2           InvSecondsSince1970(iImg)
            print *, 'Pmerge #161. Median::', 
     1           trim(MedianFileName(iImg)), '::'
         endif

c     Check to see if the median file exists. If it doesn't skip to the
c     next median.
         
         inquire( File=MedianFileName(iImg), Exist=MedianExists)

         if(MedianExists .eqv. .false.) then
            write(unitlog,*) 'Pmerge #170: ERROR ***** Median file ',
     1           'does not exist::', trim(MedianFileName(iImg)), '::'
            print *,'Pmerge #170: ERROR ***** Median file ',
     1           'does not exist::', trim(MedianFileName(iImg)), '::'
            go to 1000
         endif

         if(debug .eq. 1) print *, 'Pmerge #180. Cnt1: ::', 
     1        trim(Cnt1FileName(iImg)), '::'

c     Check for the existence of the file with the first pass of fronts
c     in it - called Cnt1FileName for historical reasons. If it does
c     not exist, write error message and skip to next input file. 
c     (This should not be the case.)

         inquire( File=Cnt1FileName(iImg), Exist=Cnt1Exists)

         if(Cnt1Exists .eqv. .false.) then
            write(unitlog,*) 'Pmerge #190: ERROR ***** Contour file ',
     1           'does not exist for::', trim(Cnt1FileName(iImg)), '::'
            print *, 'Pmerge #190: ERROR ***** Contour file ',
     1           'does not exist for::', trim(Cnt1FileName(iImg)), '::'
            go to 1000
         endif

         if(debug .eq. 1) print *, 'Pmerge #200. Merged::', 
     1        trim(MergedFileName(iImg)), '::'

c...  Check for the existence of the file with merged field in it. 

         inquire( File=MergedFileName(iImg), Exist=MergedExists)

         if(MergedExists .eqv. .false.) then 

c---------------------Get Median SST field ----------------------------
c     
c     Read in the SST data. This is used for gradients. (I'm using 
c     the median filtered SST written out by either SIED or SobelPeaks.
c     JF used the origina SST I think; need to check to be sure though.)
c     Will need it regardless of which pass of SIED this is.

            call ReadMedianFile( MedianFileName(iImg), ncMedID, 
     1           MedID, SecondsSince1970ID, MedSST, 
     2           SecondsSince1970T)

c     The median file should still be open; get the lat, lon filename.

            status = nf90_get_att( ncMedID, nf90_global,
     1           'LatLonFileName', LatLonFileName)
            if(status .ne. nf90_noerr) call handle_err(status)

            LatLonFileName = trim(BaseDir) // trim(LatLonFileName)

c--------------------------PMERGE --------------------------------------
c     

c     OK, no merged file, but contour and median files exist so we can
c     process.

c     Approximately what percentage of the run is complete?

            PercentDone = 100.0 * float(iIMG - iFirst + 1) / 
     1           float(iLast - iFirst + 1)

            write(unitlog,*) PercentDone, iFirst, iLast, iIMG, 
     1           ' Pmerge #210 Processing::', trim(Cnt1FileName(iImg)), 
     2           '::'
            print *, PercentDone, iFirst, iLast, iIMG, 
     1           ' Pmerge #210 Processing::', trim(Cnt1FileName(iImg)), 
     2           '::'

            if (debug .eq. 1) then
               Msg50 = 'Pmerge #220. MedSST before call to pmerge EOS'
               call PrintArray2( MedSST(istrt:iend,jstrt:jend), Msg50)
            endif

c     A little strange, but I don't want to write over MedSST and
c     accumul returns the merged array in the same location as the
c     input array which should be MedSST, so copy MedSST to 
c     MergedFronts.

            do 3000 j=1,LenY
               do 3010 i=1,LenX
                  MergedFronts(i,j) = MedSST(i,j)
 3010          continue
 3000       continue

c     If this is the first file being processed, read the histogram of
c     of pixel separations from the GeoLoc file to determine the 
c     typical separation. This information is used to calculate wsize,
c     the number of pixels to use when matching fronts in adjacent 
c     images with gradients in the image for which the fronts are 
c     being merged. The distance is chosed to be about twice the 
c     distance that a front is likely to travel in the time span 
c     specified in HourWindow. However, if the data are spatially 
c     course, such that wsize is order 1, then set it to 3 to account 
c     for the jiggle in fronts found by SIED.

            if(FirstToProcess .eqv. .true.) then
               call ReadPixelSpacing( xPixelSpacing, yPixelSpacing)

c     Assume that the maximum speed of displacement of a front is that
c     associated with a Gulf Stream meander, about 5 km/day so define
c     the window size to be 10 km/(24 hours * pixel spacing in km)

               if(Debug .eq. 1) print *, 'Pmerge #225: HourWindow, ',
     1              'xPixelSpacing, yPixelSpacing, wSize: ', 
     1              HourWindow, xPixelSpacing, yPixelSpacing, wSize

               wSize = 10 * HourWindow / 
     1              ( 24 * min(xPixelSpacing, yPixelSpacing))

               if(wSize .lt. 3) wSize = 3

               if(wSize .gt. 8) then
                  write(UnitLog, *) 'PMerge #230: wSize=', wSize,
     1                 ' which seems very large. It is being set ',
     2                 'to 8 pixels. Must admit that I sort of ',
     3                 'pulled 8 out of my buttocks.'
                  print *, 'PMerge #230: wSize=', wSize,
     1                 ' which seems very large. It is being set ',
     2                 'to 8 pixels.'
               endif

               write(UnitLog, *) 'Pmerge #240: Window sized to use ',
     1              'when searching for a match between the front ',
     2              'found in an adjacent image and the gradient ',
     3              'contours in the image image being processed - ',
     4              'wSize: ', wSize
              print *, 'Pmerge #240: Window sized to use ',
     1              'when searching for a match between the front ',
     2              'found in an adjacent image and the gradient ',
     3              'contours in the image image being processed - ',
     4              'wSize: ', wSize
               
c     Set FirstToProcess flag to false so that it skips this on the
c     next pass. The assumption is that the spacing of pixels is 
c     similar for all passes in this archive. 

               FirstToProcess = .false.
            endif

            call accumul( iImg, InvSecondsSince1970,   
     1           NumberOfGoodImages, MergedFronts, Cnt1FileName, 
     2           MergedFileName, ncMergedID, ncMedID, MedID,
     3           SecondsSince1970ID, wSize)

            NumberOfFilesProcessed = NumberOfFilesProcessed + 1
            
            status = nf90_close(ncMedID)
            if(status .ne. nf90_noerr) call handle_err(status)

         else

c     Here if the output file already exists.

            write(unitlog,*) 'Pmerge #250: Skipping::', 
     1           trim(MergedFileName(iImg)), ':: It already exists.'
            print *, 'Pmerge #250: Skipping::', 
     1           trim(MergedFileName(iImg)), ':: It already exists.'

         endif
c     Endif for the existence of the merged file.

c---------------------------ALL DONE PROCESSING ------------------------

 1000 continue

      close(UnitLog)
      close(UnitInventory)

      if(Debug .eq. 1) print *, 'Pmerge #999: All done.'

c     Now get the elapsed wall time - print out in the start and stop
c     times in the subroutine.

      call Duration( DateTimeStart, DurationInMinutes, 
     1     NumberOfFilesProcessed, FilesPerMinute)

      write(UnitLog, *) NumberOfFilesProcessed, ' files processed in ',
     1     DurationInMinutes, ' minutes ==> ', FilesPerMinute, 
     2     ' files/minute.'

      print *, NumberOfFilesProcessed, ' files processed in ',
     1     DurationInMinutes, ' minutes ==> ', FilesPerMinute, 
     2     ' files/minute.'

      stop

      end program Pmerge_Main

c***********************************************************************
      subroutine accumul( iImg, file_dates, file_count, outpict, file3,
     1     file4, ncID, ncInID, sstID, SecondsSince1970ID, wSize)
c***********************************************************************

c     As in SIED, a very large number of contours in a single image  
c     will cause problems in this subroutine. Array CSTART would likely
c     need to have a larger capacity to run on model images>1024x1024.
c     Also, CCOUNT and CCN may need to be increased to Integer*4 to 
c     allow for large numbers of contours in larger model images (e.g. 
c     6144x6144). READ_CONTOUR, which accepts these values as passed 
c     arguments, would also need to be edited to accomodate a large 
c     model image. 
c     
c     The program looks for the variables year, yearday, hour, minute, 
c     and second. If time is not present, pmerge will base merging on 
c     the number of windows. There are two variables used to determine 
c     the number of images to use in the pmerge step, one is the number
c     of images +/- the image of interest and the second is the number 
c     of hours. pmerge loops over all images on the list that are 
c     within +/- ImageWindow of the image of interest and then tests 
c     the time of each image, i.e., +/- HourWindow. If no time
c     in the input files; i.e., year, yearday, hour, minute and second
c     is missing the time is returned as 0, so ImageWindow will define 
c     the number of images to use. If the time is present in the input 
c     file and ImageWindow is sufficiently large, then it will use 
c     HourWindow for the pmerge step. 
c     
c     21 May 2009 - PCC - changed integer*2 l to integer l. l is the 
c     loop parameter for over times and the number of elements in the
c     time array exceeds 32000.

      implicit none

c******Functions

c******Parameter statements

      include 'ParameterStatements.f'

c******General variables
      
      character*319 file3(1:MaxNumberOfFiles)
      character*319 file4(1:MaxNumberOfFiles)
      character*319 TempFileName
      integer file_count
      integer*2 outpict(1:LenX, 1:LenY)
      integer*2 titlength
      integer*2 i
      integer*4 m,j
      integer iImg
      integer l
      integer*2 tparx
      integer*2 tpary
      integer*2 ccont(1:4,1:MaxEdge)
      integer*4 cstart(1:MaxNumOfSegments)
      integer*2 ccount
      integer*2 ccn
      integer*4 dirlength
      integer*2 ri,rj
      logical out
      real*4 gradpic(1:LenX, 1:LenY)
      integer*2 gradv(1:2,1:LenX, 1:LenY)
      integer*2 it,jt
      integer*4 hdate1,hdate2
      real*8 difdate, DifDate2
      real*4 scal,sumg,maxsum,mag1,mag2
      real*4 sscal(1:MaxEdge)
      integer*2 stpar(1:3,1:MaxEdge)
      real*8 file_dates(1:MaxNumberOfFiles)
      character filtyp
      integer*1 read_flag
      real*4 dumlats(1:LenY),dumlons(1:LenX)
      real*8 dumhdate

      integer ncInID, sstID, SecondsSince1970ID

      logical file_ex, file_ex_gz
      logical test

c     Compress - .true. if the file is to be compressed, .false. 
c     - otherwise.
      logical Compress
c     Compressed - dummy variable for call to FileStatus, not used.
      logical Compressed

c     Debug_2 - debug control. This debug allows printout for loops
c     . over all images in ArchiveInventory. Lots of output which is why
c     . it is separated from Debug.
      integer, parameter :: Debug_2=0
c     Decompress - flag telling FileStatus to decompress contour file
      logical Decompress
c     DecompressInPlace - flag telling FileStatus to decomress contour
c     . file in place
      logical DecompressInPlace

c     FileExists - dummy variable needed for test of file existence.
      logical FileExists

c     GotOne - flag used for debug. .true. if already found one pass
c     . within HourWindow of the image of interest.
      logical GotOne
c     MergedOrThinned - 1 to read MergedFronts and 2 to read 
c     . ThinnedFronts
      integer*2 MergedOrThinned

c     ncID - a temporary variable used in call to cleanup. Will be set
c     . to -1
      integer ncID

c     wSize - Window sized to use when searching for a match between 
c     the front found in an adjacent image and the gradient contours 
c     in the image image being processed.
      integer*2 wSize

      integer MaxMerge, MinMerge
      integer*4 DataRange, NAboveRange

c     ImageWindow    # +/- this many images from the image of interest
c     will be considered when merging fronts.
c     HourWindow     # Of the 2*ImageWindows considered only those 
c     that are within HourWindow of the image of
c     interest will be used to develop merged fronts.
c     FILE3     # (edge file name)
c     FILE4     # (output file name)
c     TempFileName - dummy filename used in cleanup.
c     FILE_COUNT           # number of input files
c     OUTPICT(1: SizOfImg ,1: SizOfImg ) #integer image (0,255)
c     i		       # generic	
c     m,j		       # indexes	
c     K                    # current image #
c     L                    # neighbor image #
c     TPARX                # whole contour x-translat
c     TPARY                # whole contour y-translat
c     CCONT(1:4,1:MaxEdge) # contour coord and grad
c     CSTART(1:MaxNumOfSegments) # The input file being read consists 
c     . of a list of frontal pixels. cstart(j) is the index of 
c     the first frontal pixel on the jth segment. 
c     Changed to INT*4 and increased array capacity 
c     from 2000 to 10000. 
c     CCOUNT               # contour count
c     CCN                  # current contour number
c     DIRLENGTH	       # directory name length
c     RI,RJ
c     out
c     GRADPIC(1: SizOfImg ,1: SizOfImg )   # gradient image
c     GRADV(1:2,1: SizOfImg ,1: SizOfImg ) # gradient vector
c     IT,JT
c     hdate1,hdate2
c     difdate         # changed to real*8 to match data type of dates 
c     in file_dates
c     scal,sumg,maxsum,MAG1,mag2
c     sscal(1:MaxEdge)
c     stpar(1:3,1:MaxEdge)
c     wsize
c     file_dates(1:MaxNumberOfFiles) # start time of data; changed to 
c     . real*8 to match data type of date read from input file
c     FILTYP		# file type indicator
c     read_flag	        # indicates which values should be read from 
c     Read_Image
c     dumlats(1:SizOfImg),dumlons(1:SizOfImg) # dummy value for 
c     lats/lons
c     dumhdate			          # dummy value for date

c     Resolution km/pixel
c     changed by D.U. 1/17/97 to get a half window size of 10km
c     see Cayula and Cornillon ,1995... delta=10km
c     wsize=100/resolution

c     DataRange - maximum minus minimum value read in.
c     NAboveRange - number of input data values above Range. 

      if(debug .eq. 1) then
         print *, 'Accumul #000: iImg, file_dates, file_count: ', 
     1        iImg, file_dates(iImg),file_count
         print *, 'Accumul #100: file3::', trim(file3(iImg)), '::'
         print *, 'Accumul #101: file4::', trim(file4(iImg)), '::'

         Msg50 = 'Accumul #110: Before gradient --- EOS'
         call PrintArray2( outpict(istrt:iend,jstrt:jend), Msg50)
      endif

      call gradient(outpict,gradpic,gradv)

      if (debug .eq. 1) then
         Msg50 = 'Accumul #120: gradpic --- EOS'
         call PrintArrayReal( gradpic(istrt:iend,jstrt:jend), Msg50)
      endif

      do 23014 i=1, LenX 
         do 23016 j=1, LenY 
            outpict(i,j)=0
23016    continue
23014 continue

c     Looks like pmerge has two thresholds on the images that it uses 
c     to construct the merged contours. First, it only looks +/- 12 
c     images from the one of interest. Second it only used those images
c     that are within 62 hours. [12 and 62 were the settings in the 
c     program. I made them variables. Note that the image of interest 
c     is not used to build the merged fronts.

      if(debug .eq. 1) print *, 'Accumul #130: ImageWindow and ', 
     1     'file_dates: ', ImageWindow, file_dates(iImg)

      GotOne = .false.

      do 23018 l=max(1,iImg-ImageWindow),
     1     min(file_count,iImg+ImageWindow) 

         if(GotOne .eqv. .true.) go to 23018

         difdate=abs(file_dates(iImg)-file_dates(l))

         if(debug .eq. 1) print *, 'Accumul #140: Loop 23012: ',
     1        ' l, file_dates, difdate, HourWindow*3600: ', l, 
     2        file_dates(l), difdate, HourWindow*3600

c     This if selects the images that are within HourWindow of the
c     current image.

         if( (difdate .le. HourWindow*3600) .and.
     1        (difdate .ne. 0) ) then

c     Uncomment the following line to debug - to use only the first
c     . file in the list of files to merge with the current one.

c            GotOne = .true.

            if(debug .eq. 1) print *, 'Accumul #160: difdate, ',
     1           'HourWindow*3600: ', difdate, HourWindow*3600

c     Check to make sure that the .cnt file exists. If it does, read it,
c     if not, skip to the next file.

            inquire( File=file3(l), Exist=FileExists)

            if(FileExists .eqv. .false.) go to 23018

            call read_contour(ccont, cstart, file3(l), ccount)

            if(debug .eq. 1) then
               print *, 'Accumul #170: file3::', trim(file3(l)), '::'
               print *, 'Accumul #171: ccount: ', ccount
            endif

c     Get the hour difference between this cnt file and the next image
c     file on the list to process. This will be used to remove the
c     decompressed version if it will not be used again.

            DifDate2 = 0
            if(iImg+1 .le. file_count) then
               DifDate2 = file_dates(l) - file_dates(iImg+1)
            endif

            if(debug .eq. 1) print *, 'Accumul #180: DifDate2, ',
     1           ' file_dates(', iImg+1, ' ): ', DifDate2, 
     2           file_dates(iImg+1)

c     Fit contours to current image - check bounds just in case.

            if( (ccount .ge. 1) .and. (ccount .le. MaxNumOfSegments))
     1           then

               do 23022 ccn=1,ccount 
                  maxsum=0
                  tparx=0
                  tpary=0

                  if((cstart(ccn) .lt. 1) .or. 
     1                 (cstart(ccn) .gt. MaxEdge) .or.
     1                 (cstart(ccn+1) .lt. 1) .or.
     1                 (cstart(ccn+1) .gt. MaxEdge) ) then
                     write(unitlog,*) 'Accumul #190: ERROR ******* ',
     1                    'cnn, cstart: ', ccn, cstart(ccn), 
     2                    cstart(ccn+1)
                     print *, 'Accumul #190: ERROR ******* ',
     1                    'cnn, cstart: ', ccn, cstart(ccn), 
     2                    cstart(ccn+1)
                  endif

                  do 23024 m=cstart(ccn),cstart(ccn+1)-1 
                     stpar(1,m)=0
23024             continue
                  do 23026 it=-wsize,wsize 
                     do 23028 jt=-wsize,wsize 
                        sumg=0
                        do 23030 m=cstart(ccn),cstart(ccn+1)-1 
                           ri=ccont(1,m)+it
                           rj=ccont(2,m)+jt
                           sscal(m)=0
                           if((ri .ge. 1) .and. 
     1                          (ri .le. LenX )) then
                              if((rj .ge. 1) .and. 
     +                             (rj .le. LenY ))then
                                 scal = gradv(1,ri,rj) * 
     1                                ccont(3,m) + gradv(2,ri,rj)
     2                                * ccont(4,m)
                                 mag1 = ccont(3,m) * ccont(3,m) +
     +                                ccont(4,m) * ccont(4,m)
                                 mag2=max(gradpic(ri,rj)**2,2.0)
                                 if(scal .gt. 0)then
                                    if(mag1 .gt. mag2)then
                                       scal =scal / mag1
                                    else
                                       scal = scal / mag2
                                    endif
                                    sumg = sumg + scal
                                    sscal(m) = scal
                                 endif
                              endif
                           endif
23030                   continue
                        if(sumg .gt. maxsum)then
                           maxsum=sumg
                           tparx=it
                           tpary=jt
                        endif
                        do 23042 m=cstart(ccn),cstart(ccn+1)-2 
                           do 23044 j=m+1,
     1                          min(m+20,cstart(ccn+1)-1) 
                              sscal(m)=sscal(m)+sscal(j)
23044                      continue
23042                   continue
                        do 23046 m=cstart(ccn),cstart(ccn+1)-1 
                           if(sscal(m) .gt. stpar(1,m))then
                              stpar(1,m)=sscal(m)
                              stpar(2,m)=it
                              stpar(3,m)=jt
                           endif
23046                   continue
23028                continue
23026             continue

c     Place contour segment in image array.

                  do 23050 m=cstart(ccn),cstart(ccn+1)-1 
                     if(stpar(1,m) .gt. 10)then
                        do 23054 j=m,min(m+20,cstart(ccn+1)-1) 
                           ri=ccont(1,j)
                           rj=ccont(2,j)
                           ri=ri+stpar(2,m)
                           rj=rj+stpar(3,m)
                           if((ri .ge. 1) .and.
     1                          (ri .le. LenX )) then
                              if((rj .ge. 1) .and. 
     +                             (rj .le. LenY ))then
                                 outpict(ri,rj) = 4
                              endif
                           endif
23054                   continue
                     endif
23050             continue
23022          continue

            else
               if( (ccount .lt. 0) .or. (ccount .gt. MaxNumOfSegments)) 
     1              then
                  write(unitlog,*) 'Accumul #200: ********** ccount: ', 
     1                 ccount
                  print *, 'Accumul #200: ********** ccount: ', ccount
               endif
            endif
         endif

23018 continue

c     Get the max and min of the output field for yuks.

      if(Debug .eq. 1) then
         MaxMerge = -10000
         MinMerge = 10000
         do 9876 i=1,LenX
            do 9875 j=i,LenY
               if( outpict(i,j) .lt. MinMerge) then
                  MinMerge = outpict(i,j)
               endif
               if( outpict(i,j) .gt. MaxMerge) then
                  MaxMerge = outpict(i,j)
               endif
 9875       continue
 9876    continue

         print *, 'Accumul #210: Min and max merge', MinMerge, MaxMerge
         print *, 'Accumul #220: file_dates(iImg): ', file_dates(iImg)

         Msg50 = 'Accumul #221: Before WriteMerge --- EOS'
         call PrintArray2( outpict(istrt:iend,jstrt:jend), Msg50)
      endif

c     Now write out

      MergedOrThinned = 1
      call WriteMergedThinned( file4(iImg), file_dates(iImg), dumlats, 
     1     dumlons, outpict, MergedOrThinned, ncInID, sstID,
     2     SecondsSince1970ID)

      return
      end

c***********************************************************************
      subroutine read_contour( ccont, cstart, file, ccount)
c***********************************************************************
c
c  Read the .cnt file.

      use netcdf

      implicit none

c******Functions

c     HoursSince - returns the number of hours since the reference.
c     . Arguments are year, month, day, hour, minute and the 
c     . values for these.
      real*8 HoursSince

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

c     FrontFillValue - the fill value for front pixels. Each front
c      segment is separated from the next with a line of FrontFillValue
c      at the locations of the various parameters.
      integer*4 FrontFillValue

c     iFrontPixel - do loop parameter over front pixels
      integer iFrontPixel
c     iGrad - the vector for the central-difference gradients in i at 
c      the front pixel locations.
      integer*4, allocatable :: iGrad(:)
c     iLoc - the vector for the i location of front pixels.
      integer*4, allocatable :: iLoc(:)
c     iLocID - netCDF ID for iLoc
      integer iLocID

c     jGrad - the vector for the central-difference gradients in j  at 
c      the front pixel locations.
      integer*4, allocatable :: jGrad(:)
c     jLoc - the vector for the j location of front pixels.
      integer*4, allocatable :: jLoc(:)
c     jLocID - netCDF ID for jLoc
      integer jLocID

c     ncID - netCDF file ID.
      integer ncID
c     NumFrontPixels - the number of front pixels.
      integer NumFrontPixels

c     RecordDimID - netCDF ID for the number of front pixels.
      integer RecordDimID

      character*319 file
      integer*2 ccont(1:4,1:MaxEdge)
      integer*4 cstart(1:MaxNumOfSegments)
      integer*2 ccount
      integer*4 i,j
      real*4 ri,rj
      real*4 gx,gy
      real*4 inner
      real*4 gx1,gx2,gy1,gy2
      real*4 SST_A, SST_B, SST_C
      real*4 norm1,norm2

c     cstart - The input file being read consists of a list of frontal
c     pixels. cstart(j) is the index of the first frontal pixel on
c     the jth segment.

c     open (unit=11, file=file, status='old', readonly,
c     +     access='SEQUENTIAL')

      if(debug .eq. 1) print *, 'read_contour #000: Filename::', 
     1     trim(file), '::'

      status = nf90_open( file, nf90_nowrite, ncID)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Get the number of front pixels in this image and the number
c     of front segments.

      status = nf90_inq_dimid( ncID, "record", RecordDimID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_inquire_dimension( ncID, RecordDimID, 
     1     len = NumFrontPixels)
      if(status .ne. nf90_noerr) call handle_err(status)

c     allocate variables for this file

      allocate( iLoc(1:NumFrontPixels),
     1     jLoc(1:NumFrontPixels),
     2     iGrad(1:NumFrontPixels),
     3     jGrad(1:NumFrontPixels),
     4     stat=ierr)

      if (ierr .ne. 0) then
         write(UnitLog, *) 'read_contour #100: Allocation error for ',
     1        'LenXA: ', LenXA, ' and/or LenYA: ', LenYA, ' Exiting.'
         print *, 'read_contour #100: Allocation error for ',
     1        'LenXA: ', LenXA, ' and/or LenYA: ', LenYA, ' Exiting.'
         stop
      endif

      if(Debug .eq. 1) print *, 'read_contour #120: Variables ',
     1     'allocated'

c     Read the latitude first.

      status = nf90_inq_varid( ncID, 'i', iLocID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_var( ncID, iLocID, iLoc)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_att( ncID, iLocID, '_FillValue', 
     1     FrontFillValue)
      if(status .ne. nf90_noerr) call handle_err(status)
      
      status = nf90_inq_varid( ncID, 'j', jLocID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_var( ncID, jLocID, jLoc)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'read_contour #130: Read loc variables.'

      status = nf90_inq_varid( ncID, 'along-scan_gradient', iGradID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_var( ncID, iGradID, iGrad)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_inq_varid( ncID, 'cross-scan_gradient', jGradID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_var( ncID, jGradID, jGrad)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Close the input file

      status = nf90_close( ncID)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'read_contour #140: Data read.'

c     Loop over frontal pixels, breaking into contour segments.

      i=1
      j=0

      do 1 iFrontPixel=1,NumFrontPixels

c     1    read (11,*,end=2) ri, rj, gx, gy, SST_A, SST_B, SST_C

         ri = iLoc(iFrontPixel)
         rj = jLoc(iFrontPixel)
         gx = iGrad(iFrontPixel)
         gy = jGrad(iFrontPixel)

         if ((ri .eq. FrontFillValue) .and. (rj .eq. FrontFillValue)) 
     1        then
            j=j+1

            if (j .eq. MaxNumOfSegments-100) then
               write(unitlog,*) 'read_contour #150: j=', j,
     1              '  **Warning: Approaching CSTART array limit**'
               print *, 'read_contour #150: j=',
     1              j, '  **Warning: Approaching CSTART array limit**'
            endif
            if (j .eq. MaxNumOfSegments+1) then
               write(unitlog,*) 'read_contour #160: j=', j, 
     1              '  **Warning: CSTART array limit surpassed**'
               print *, 'read_contour #160: j=', j, 
     1              '  **Warning: CSTART array limit surpassed**'
            endif

            cstart(j)=i
         else
            ccont(1,i)=ri
            ccont(2,i)=rj
            ccont(3,i)=gx
            ccont(4,i)=gy
            i=i+1

c     It seems that this section examins contour segments that are 
c     longer than 15 pixels, gets the mean gradient in x and y 
c     between the 11th and 12th points back from the current one and 
c     the same for the 1st and 2nd points back from the current one 
c     and then checks to see if the normalized inner product of the 2 
c     is less than 1/2. If it is, it means that the segment has 
c     changed by more than 30 degrees. In this case, a new segment is 
c     started, j is incremented.

            if((i-cstart(j)) .gt. 15)then
               gx1 = ccont(3,i-11) + ccont(3,i-12)
               gx2 = ccont(3,i-1) + ccont(3,i-2)
               gy1 = ccont(4,i-11) + ccont(4,i-12)
               gy2 = ccont(4,i-1) + ccont(4,i-2)
               norm1 = sqrt((gx1*gx1) + (gy1*gy1))
               norm2 = sqrt((gx2*gx2) + (gy2*gy2))
               if((norm1*norm2) .ne. 0)then
                  inner = ((gx1*gx2) + (gy1*gy2)) / (norm1*norm2)
                  if(inner .le. .5)then
                     j=j+1
                     cstart(j)=i
                  endif
               endif
            endif
         endif

 1       continue

c     End of loop over frontal pixels in this file.

         ccount=j-1

         if(Debug .eq. 1) print *, 'read_contour #999: ', ccount, 
     1        ' segments defined.'

         return
         end

c***********************************************************************
      subroutine gradient(inpict,gradpic,gradv)
c***********************************************************************
      implicit none

c******Functions

c     HoursSince - returns the number of hours since the reference.
c     . Arguments are year, month, day, hour, minute and the 
c     . values for these.
      real*8 HoursSince

c******Parameter statements

      include 'ParameterStatements.f'

c******General variables

      integer*2 inpict(1:LenX,1:LenY)
      integer*2 gradv(1:2,1:LenX,1:LenY)
      real*4 gradpic(1:LenX,1:LenY)
      real*4 RangeFloat
      integer*2 i,j
      integer*2 ilo,iup
      integer*2 jlo,jup
      real*4 grad

c Range              - The upper limit to use in the histograms. PCC 
c    added this variable. The numbers were hard coded in as 255 in 
c    the .rat version. The addition of Range was done to accommodate 
c    the fact that in MSG SV images they use hundreths of a degree C 
c    rather than 1/8 with a max of 255.

      do 23170 i=1,LenX 
         do 23172 j=1,LenY 
            ilo = max(1,i-1)
            jlo = max(1,j-1)
            iup = min( LenX ,i+1)
            jup = min( LenY ,j+1)
            gradv(1,i,j) = inpict(iup,j)-inpict(ilo,j)
            gradv(2,i,j) = inpict(i,jup)-inpict(i,jlo)
            grad = (real(gradv(1,i,j))**2)+(real(gradv(2,i,j))**2)
c Be careful here. I changed 255.0 to Range. Range is an integer. 
            RangeFloat = Range
            gradpic(i,j) = min(sqrt(grad),RangeFloat)
23172    continue
23170 continue

      return
      end

c***********************************************************************
c***********************************************************************

      include 'CommonSubroutines-2.36.f'
