c***********************************************************************
      program GeoSobel_Main

c***********************************************************************
c     
c     This program will apply the Sobel operator to each image in the
c     /Median/ directories in BaseDir, then read in the lat,lon arrays
c     and convert from the Sobel gradient in Kelvin/pixel in the i,j
c     coordinate system to Kelvin/km in north-south, east-west.
c     
c     The program has originally written to process data from the MSG 
c     or the GOES archive at the Centre de Meteorologie Spatiale. To run
c     it on other archives will require some changes, but it has been
c     modified to be more general - really general I hope.
c     
c     Written by Peter Cornillon, University of Rhode Island,
c     pcornillon@me.com 18 April 2009
c     
c     CHANGE LOG
c     
c     PCC 4/28/09 - removed variables not used in Main program. Changed
c     year, month, day, hour, minute, numclear and hourssince1900 
c     variables read from the inventory to Invxxx. Also changed prog
c     to SobelPeaksSied
c     
c     PCC 12/11/10 - removed debug printout at line 318. Should not 
c     - affect operation in any way. Lines removed were to printout the
c     - Sobel filename.
c     
c     3/24/11-5/5/11 - PCC - 1.00 ==> 1.1
c     Added code to CreateSobelFile to allow for chunking output.
c     
c     4/4/11 PCC - All subroutines 
c     Changed all logical*1 to logical
c     Changed type of MergedTempFileName from logical to char*150
c     
c     5/11/11 - PCC - Main program
c     Added a check for compressed input file. It should not be, 
c     so stop if it is.
c     Replaced Cleanup with simple netCDF close commands. The old
c     version of the program dealt with compressed in/out files.
c     Added gradient magnitude to output. This will not add much to
c     the size of the file and with chunking will actually make for
c     faster input when one desires the gradient magnitude instead
c     of x and y.
c     
c     6/1/11 - PCC - Added Msg50 in calls to PrintArray2 and ElapsedTime
c     
c     6/6/11 - PCC - Changed length of SatName from 8 to 20 characters.
c     
c     6/7/11 - PCC - Major changes to simplify and correct handling of
c     strings. Removed all EndOfStrings. Added BlankFill before any 
c     definition of a string variable to force the end of the string
c     to be all blanks.
c     
c     6/20/11 - PCC - SobelSub - swapped gradx and grady. At least for
c     MSG, the first dimension, even though I call it x in the 
c     subroutine, is really for north south. Need to see if this is
c     true for other satellites.
c     
c     Changes from 1.1 to 1.11
c     
c     1/2/12 - PCC - Changed all occurrences of 150 to 319 when 
c     referring to a string length - to 319 because 150 is too short
c     and because 319 is pretty unique.
c     
c     Changes from 1.11 to 1.12
c     
c     1/2/12 - PCC - added the output file name to the call to 
c     GenerateGlobals to deal with the uuid_gen problem.
c     
c     Version 1.12 ==> 1.13
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
c     2/3/12 - DH - Wrapped compute debugging with new debug_max
c     flag to handle special case.  When using debug_max jstrt,jend,
c     istrt,iend set in ParameterStatements must be within array
c     bounds for input.
c     
c     Changes 1.14 to 2.00
c     
c     5/26/12 - PCC - Many changes, mainly added code to compute the
c     gradient in K/km eastward and northward. 
c     Changed the meaning of Debug_Max to be for debug printing out
c     arrays and writing out .nc files of intermediate variables. Be
c     very careful when turning this on, since it generates a lot of
c     output. Debug_Max is set in the progam, it is not a readin 
c     parameter.
c     
c     Changes 2.00 to 2.01
c     
c     7/21/12 - PCC - Added a check for debug equal to 1 in all parts
c     of code for debug_max equal to 1 except for the part that writes
c     out the sobel gradient components in satellite coordinates. This
c     allows for these to be written out without a real lot of other
c     debug stuff. Also set debug_max to 1.
c     
c     Changes 2.01 to 2.02
c     
c     7/29/12 - PCC - Added check for TMI - don't specify chunk size.
c     
c     Changes 2.02 to 3.00
c     
c     8/14/12 - PCC - Added code to deal with cases in which either
c     the x or y distances along the along-, across-track directions 
c     are zero. Also changed the calculation for east and north
c     gradients to use slopes, sines and cosines instead of gammas.
c     This is not as compact, but easier to follow. It also made
c     the code for the calculations when the along-track and/or the
c     across-track axes are parallel to east and/or north.
c
c     8/30/12 - PCC - Changed around some of the printout. More 
c      importantly, added the factor of 2 into the calculation of
c      the distance between pixels. Before it gave the distance
c      between adjacent pixels centered on the pixel of interest. The
c      modification divides this distance by 2 to represent the distance
c      between adjacent pixels.
c
c     Changes 3.00 to 3.01
c
c     9/9/12 - PCC - Changed Common... to CommonSubroutines-2.09.f
c
c     Changes 3.01 to 3.02
c
c     10/21/12 - PCC - Changed Common... to CommonSubroutines-2.11.f
c
c     Changes 3.02 to 3.03
c
c     10/23/12 - PCC - Changed so that it writes out a Sobel file in
c      addition to a GeoSobel file by default; i.e., not just when
c      Debug_Max is set to 1.
c
c     Changes 3.03 to 3.04
c
c     12/1/12 - PCC - Changed from 2.11 to 2.12 CommonSubroutines
c
c     Changes 3.04 to 3.05
c
c     12/1/12 - PCC - Changed from 2.12 to 2.13 CommonSubroutines
c
c     Changes 3.05 to 3.06
c
c     1/15/13 - PCC - Changed code that does the chunking. In the
c       previous version, the program checked to see if the input file
c       was AMSR-E or TMI and, if it was, it did not chunk. This was
c       because the width of the swath was much smaller than the y-
c       chunking parameter so the definition failed. In this version,
c       I replaced the standard y-chunking parameter with the minimum
c       of LenY and the standard y-chunking parameter and chunked.
c      Replaced GradFillValue with FillValueReal.
c      Fixed a problem that arises with the gradients when one of the
c       slopes is infinite. This error resulted in an infinite value
c       for the eastward and/or the northward gradient.
c      Added iFirst, iLast, IMG to the line that prints out the file 
c       being processed. This is so that we know exactly where in the
c       processing loop the job is. Also added a date time variable
c       to print out at the end of the job.
c      Changed the name of x and y gradients to along/cross scan
c       gradients depending on the value of AlongScanDimension.
c      Changed from 2.13 to 2.14 CommonSubroutines - new duration 
c       routine.
c      
c     Changes 3.06 to 3.07
c
c     1/19/13 - PCC - Added SSTFillValueIn in call to SobelSub. Added
c       SSTFillValueIn in call to ReadMedianFile.
c      Changed CommonSubroutines from 2.14 to 2.15
c     1/20/13 - PCC - CommonSubroutines ==> 2.16 added VariableName 
c      (also to parameter- statements and changed SSTFillValue to 
c      SSTFillValueIn. 
c     1/21/13 - PCC - Added code to make use of ParameterName and 
c       ParameterUnits in the generation of metadata for the netCDf 
c       output files.
c
c     Changes 3.07 to 3.08
c
c     1/23/13 - PCC - Changed the code to make use of iVarName and 
c       jVarName when generating metadata to eliminate the test
c       on AlongScanDimension. This test is now made in ReadInputs in
c       CommonSubroutines. These changes should not change the output
c       in any way.
c     CommonSubroutines 2.16 --> 2.17
c
c     Version 3.08 to 3.09
c
c     1/27/13 - PCC - Added a common for fill values, scale factors and 
c       offsets for gradient fields, lat/lon fields, sst fields and
c       zenith angle fields in ParameterStatements, changed calls to 
c       reading subroutines in CommonSubtroutines to accommodate the new
c       common statement and modified code in this programs to do so.
c      CommonSubroutines 2.17 --> 2.19
c      Added code to write the start and end times and the number of
c       files processed at the end of the main program to the log file.
c     1/28/13 -PCC - SatName moved to ArchiveDefinition common in 
c       ParameterStatements. This required removing SatName from 
c       subroutine calls and definitions for ReadInputs, CheckArchive 
c       and ReplaceBaseDir.
c      Cleaned up some of the debug statements.
c
c     Version 3.09 --> 3.10
c
c     2/5/13 -PCC - CommonStatements 2.19 ==> 2.21.
c     . Changed Thin to GeoSobel in printout statement for #133.
c
c     Version 3.10 --> 3.11
c
c     3/15/13 -JPS - CommonStatements 2.21 ==> 2.22.
c     made chunksizeX = min(LenY, chunkszLenY)
c
c     Version 3.11 --> 3.12
c
c     3/15/13 -PCC - CommonStatements 2.22 ==> 2.23.
c
c     Version 3.12 --> 3.13
c
c     4/10/13 -PCC - CommonStatements 2.23 ==> 2.26. This will update
c       LSTorGMT. Also fixed to read chlorophyll values.
c
c     Version 3.13 --> 3.14
c
c     4/10/13 -PCC - CommonStatements 2.26 ==> 2.28. This will update
c       LSTorGMT. Also fixed to read chlorophyll values.
c
c     Version 3.14 -->  3.15
c 
c     6/7/13 - DI - CommonStatements 2.29 ==> 2.30. This will update
c      the additional hycom archives.
c
c     7/22/13 - JPS - CommonSubroutines 2.30 ==> 2.31
c
c     Version 3.16 --> 3.17
c
c     7/24/13 - DI- Fixed Case 3 & 4 (which applies to axes being 
c      paralell to North and East) to create northward gradient field.
c      Previously, "SlopeA" in case 4 was "SlopeB" which resulted in a 
c      null sst_grad_north field. This was done using CommonSubroutines
c      2.30, as some bugs remain in 2.31
c
c     Version 3.17 --> 3.18
c
c     9/1/13 - PCC - CommonSubroutines 2.31 ==> 2.32
c
c     Version 3.18 --> 3.19
c
c     9/26/13 - PCC - CommonSubroutines 2.32 ==> 2.34
c
c     Version 3.19 --> 3.20
c
c     9/27/13 - PCC - CommonSubroutines 2.34 ==> 2.36
c
c.......................................................................
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
c     Be very careful with the order of dimensions in the input arrays.
c     If NetCDF they are sst(lat,lon), when read by Fortran they become
c     sst(lon,lat); i.e., the dimensions are reversed. For the
c     geostationary data from CMS, x (1:1980 for GOES) is Longitude 
c     and y (1:2431 for GOES) is Latitude.
c     
c     There are also two parameters called xLenMeteo and xLenGoes.
c     These are used to make sure that you are processing the correct
c     archive. They are currently set to 3712 and 1980. If you are
c     only using one archive of data, search for these parameters
c     and bypass the part of the program that uses them.
c     
c.......................................................................

      use netcdf

      implicit none

c******Functions

c     SecondsSince - returns the number of hours since the reference.
c     - Arguments are year, month, day, hour, minute and the 
c     - values for these.
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

c*******Functions
c     
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

c     CPUTime - CPU time used by this program to date
      real CPUTime

c     CosAlphaA - Cosine of the angle that the y-axis makes relative
c     to east.
      real*4, allocatable :: CosAlphaA(:,:)
c     CosAlphaB - Cosine of the angle that the x-axis makes relative
c     to east.
      real*4, allocatable :: CosAlphaB(:,:)

c     DebugFile - name of output netCDF file for Debug_Max runs.
      character*319 DebugFile
c     Dim1FileName - name of file with lat and lon located fronts. 
      character*319 Dim1FileName
c     DistX - separation of pixels in kms in the x-direction (first 
c     dimension.
      real*4, allocatable :: DistX(:,:)
c     DistY - separation of pixels in kms in the y-direction (second 
c     dimension.
      real*4, allocatable :: DistY(:,:)
c     Dum0 - a dummy paramete used in the call to check archive.
      integer Dum0

c     vError - used to check on component of velocity.
      real*4 vError

c     FileNameToUse - is the filename to be use for the iput file. It's
c     - either the filename in the inventory or a temporary filename.
      character*319 FileNameToUse

c     GeoSobelExists -  file status for GeoSobel, .true. if exists.
      logical GeoSobelExists
c     GeoSobelFileName - the name of the output file for the Sobel 
c     gradients after correction for satellite geometry and conversion
c     to K/km
      character*319 GeoSobelFileName

c     i - do loop parameter
      integer i
c     iImg - the main loop parameter - the index for the file on which
c     - we are working.
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
c     InvSecondsSince1970 - the number of seconds since 00:00 of 
c     - 1 January 1970 corresponding to the time of a given image. 
c     - Read from the inventory
      real*8, allocatable :: InvSecondsSince1970(:)
c     InvMinute - minute of the hour corresponding to an image in the 
c     - archive.
      integer, allocatable :: InvMinute(:)
c     InvMonth - month of the year corresponding to an image in the 
c     - archive.
      integer, allocatable :: InvMonth(:)
c     InvNumClear - Number of clear pixels in the image read from the 
c     - inventory. 
      integer, allocatable :: InvNumClear(:)
c     InvYear - Year corresponding to an image in the archive.
      integer, allocatable :: InvYear(:)
c     ix - do loop parameter
      integer ix

c     j - do loop parameter
      integer j
c     jy - do loop parameter
      integer jy

c     LastChar1 - the location of the last character of a string. 
      integer LastChar1
c     LastChar2 - the last character of another string. 
      integer LastChar2
c     LastCPUTime - CPUTime the last time ElapsedTime was called.
      real LastCPUTime

c     GradMag - the x-component of the Sobel gradient
      real*4, allocatable :: GradMag(:,:)
c     GradMagID - NetCDF ID for the xGrad variable.
      integer GradMagID
c     MedianCompressed - the compression flag for the Median file. 
      logical MedianCompressed
c     MedianExists - .true. if the median file existed before this run.
      logical MedianExists
c     MedianFileName - the name of the output file for the 3x3 median
c     - filter.
      character*319 MedianFileName
c     MedianTempFileName - filename used for the decompressed 
c     - temporary input file.
      character*319 MedianTempFileName 
c     MedSSTExists - .true. if MedSST exists; i.e., has either been
c     - read in in this program or created in this program.
      logical MedSSTExists
c     MedID - ID for the median sst field.
      integer MedID
c     MedSST - 3x3 median filtered value of sst
      integer*2, allocatable :: MedSST(:,:)
c     Message - message to be written out for call to ElapsedTime.
      character*30 Message

c     ncMedlID - NetCDF ID for Median input/outpout files.
      integer ncMedID
c     ncSobelID - NetCDF ID for Sobel input/output file
      integer ncSobelID
c     NumberOfFilesProcessed - the number of fields processed in this 
c     - run.
      integer NumberOfFilesProcessed

c     SecondsSince1970ID - ID for NetCDF variable.
      integer SecondsSince1970ID
c     SinAlphaA - Sin of the angle that dimension #1 makes relative
c     to east.
      real*4, allocatable :: SinAlphaB(:,:)
c     SinAlphaB - Sin of the angle that dimension #2 makes relative
c     to east.
      real*4, allocatable :: SinAlphaA(:,:)
c     SobelFileName - the name of the output file for the Sobel grads.
      character*319 SobelFileName
c     SlopeA - a coefficient used to convert from i,j to lat,lon
      real*4, allocatable :: SlopeA(:,:)
c     SlopeB - a coefficient used to convert from i,j to lat,lon
      real*4, allocatable :: SlopeB(:,:)
c     SobelTempFileName - filename used for the decompressed temporary 
c     - input file.
      character*319 SobelTempFileName
c     SobelCompressed - the compression flag for the Sobel file. 
      logical SobelCompressed
c     SobelExists -  file status for Sobel, .true. if the file exists.
      logical SobelExists
c     SobelFileOpen - .true. if the input Sobel file has been opened.
      logical SobelFileOpen
c     SSTFileName - the temporary name of the input file - uncompressed
      character*319 SSTFileName
c     sst_grad_east - eastward gradient in K/km
      real*4, allocatable :: sst_grad_east(:,:)
c     sst_grad_east - norhtward gradient in K/km
      real*4, allocatable :: sst_grad_north(:,:)

c     t - damn, can't recall what I used this for, but I need it.
      real t
c     TempFileName - dummy filename, used when not expecting a 
c     - temporary filename back.
      character*319 TempFileName
c     TempVA - a temporary variable used in calculating gradients.
      real*4 TempVA
c     TempVB - a temporary variable used in calculating gradients.
      real*4 TempVB

c     xGrad - the x-component of the Sobel gradient in K/pixel
      real*4, allocatable :: xGrad(:,:)
c     xGradID - NetCDF ID for the xGrad variable.
      integer xGradID

c     yGrad - the y-component of the Sobel gradient in K/pixel.
      real*4, allocatable :: yGrad(:,:)
c     yGradID - NetCDF ID for the yGrad variable.
      integer yGradID

c     Function to generate UUID.
      character*36 UUID_Gen
      
c---------------------------------start here ---------------------------

      ProgName = BlankFill(:len(ProgName))
      ProgName = 'GeoSobel_Main'

c     Set extended debugging 'off'

      debug_max = 1

c     Set LatLonFileNameSave to all blanks. The program will set it 
c     equal to the lat,lon file name for the first SST file that it
c     reads and then check to see if has changed after that. If it has,
c     it will read in the new lat,lon fields.

      LatLonFileNameSave = BlankFill(:len(LatLonFileNameSave))

c     Read in/set up input variables.

      call ReadInputs( ProgName, Start, CountToRead, GeoNameIn,
     1     YearStart, YearStartC, YearEnd, MonthStart, MonthEnd,
     2     InventoryFileName)

      if(Debug .eq. 1) print *, 'Sobel #100'

c     Allocate space for dynamically allocated variables.

      allocate( MedSST(LenX,LenY),
     1     xGrad(LenX,LenY), 
     2     yGrad(LenX,LenY), 
     3     sst_grad_east(LenX,LenY), 
     4     sst_grad_north(LenX,LenY), 
     5     GradMag(LenX,LenY),
     6     DistX(LenX,LenY),
     7     DistY(LenX,LenY),
     8     SlopeA(LenX, LenY),
     9     SlopeB(LenX, LenY),
     1     SinAlphaA(LenX, LenY),
     2     CosAlphaA(LenX, LenY),
     3     SinAlphaB(LenX, LenY),
     4     CosAlphaB(LenX, LenY),
     5     stat=ierr)
      
      if (ierr .ne. 0) then
         print *, 'Sobel # 102. Allocation error for LenX, LenY ',
     1        'arrays. Exiting.'
         stop
      endif

c     Read ArchiveInventory - the names of files to process. First,
c     allocate the arrays to be read.

      include 'AllocateInventoryVariables.f'

      call ReadInventory(InventoryFileName, InvSecondsSince1970, 
     1     InvNumClear, InvYear, InvMonth, InvDay, InvHour, 
     2     InvMinute, InvSecond, InvFileName, iFirst, iLast)

      write(UnitLog,*) 'Will process ArchiveInventory from ', 
     1     iFirst, ' to ', iLast
      print *, 'GeoSobel #120: Will process ArchiveInventory from ', 
     1     iFirst, ' to ', iLast

c___________________________MAIN LOOP _________________________________

c     Get the date and time so that we can determine the files/minute
c      processed

      call date_and_time(VALUES=DateTimeStart)

c     Loop over all images in the time range 

      NumberOfFilesProcessed = 0
      do 1000 iImg=iFirst,iLast

         if(Debug .eq. 1) print *, 'Sobel #200: iImg: ', iImg

         MedSSTExists = .false.
         SobelFileOpen = .false.

c     Do not process this image if less than Thresh clear pixels in it.

         if(InvNumClear(iImg) .lt. NumClearThresh) go to 1000

         SSTFileName = InvFileName(iImg)

         call ReplaceBaseDir(SSTFileName)
         
c     Found the file, build the output file names ......................

         call GetDateCharStrings( 
     1        InvYear(iImg), InvMonth(iImg),  InvDay(iImg),  
     2        InvHour(iImg), Inv Minute(iImg),
     3        YearC, MonthC, DayC, HourC, MinuteC)

c     Generate Median and Sobel filenames.

         TempString1 = BlankFill(:len(TempString1))
         TempString1 = 'Median/'
         TempString2 = BlankFill(:len(TempString2))
         TempString2 = '_Median.nc'
         MedianFileName = GenerateFileName( SSTFileName, TempString1, 
     1        TempString2)

         TempString1 = BlankFill(:len(TempString1))
         TempString1 = 'Sobel/'
         TempString2 = BlankFill(:len(TempString2))
         TempString2 = '_Sobel.nc'
         SobelFileName = GenerateFileName( SSTFileName, TempString1, 
     1        TempString2)


         TempString1 = BlankFill(:len(TempString1))
         TempString1 = 'GeoSobel/'
         TempString2 = BlankFill(:len(TempString2))
         TempString2 = '_SobelEW.nc'
         GeoSobelFileName = GenerateFileName( SSTFileName, TempString1, 
     1        TempString2)


c     Check for existence of the median file.

         if(debug .eq. 1) print *,'Sobel #210: Median filename::', 
     1        trim(MedianFileName), '::'
         call FileStatus( MedianFileName, MedianExists, 
     1        MedianCompressed, YearStartC, TempFileName,
     2        .false., .false.)

c     The median file should not be compressed for netCDF4 chunked; 
c     it is compressed internally - no .gz

         if(MedianCompressed .eqv. .true.) then
            write(UnitLog,*) 'Input file compressed. Should not be.'
            stop 'Input file compressed should not be.'
         endif

         if(debug .eq. 1) print *, 'Sobel #220: Sobel filename::', 
     1        trim(SobelFileName), '::'
         call FileStatus( SobelFileName, SobelExists, SobelCompressed,
     1        YearStartC, TempFileName, .false., .false.)

         if(debug .eq. 1) print *, 'Sobel #225: GeoSobel filename::', 
     1        trim(GeoSobelFileName), '::'
         call FileStatus( GeoSobelFileName, GeoSobelExists, 
     1        SobelCompressed, YearStartC, TempFileName, .false., 
     2        .false.)

c     If Sobel already exists, skip to next input file.

         if( (SobelExists .eqv. .true.) .or. 
     1        (GeoSobelExists .eqv. .true.) ) then

            write(UnitLog,*) 'Sobel #230: output files exist - skip ',
     1           'processing for::', trim(GeoSobelFileName), '::'
            print *, 'Sobel #230: Sobel output files exist - skip.',
     1           GeoSobelFileName
            go to 1000

         else
            if(MedianExists .eqv. .false.) then
               write(UnitLog,*) 'Sobel #235: Median file does not ',
     1              'exist. It should. STOP::', trim(MedianFileName), 
     2              '::'
               print *, 'Sobel #240: Median file does not exist. ',
     1              'It should. STOP::', trim(MedianFileName), '::'
               stop
            else
               if ((debug_max .eq. 1) .and. (Debug .eq. 1)) then
                  Msg50 = 'Sobel #240: Before doing the Sobel EOS'
                  call PrintArray2( MedSST(istrt:iend,jstrt:jend),
     2                 Msg50)
                  LastCPUTime = -99.0
                  Msg50 = 'none EOS'
                  call ElapsedTime( LastCPUTime, CPUTime, Msg50)
               endif

c     Approximately what percentage of the run is complete?

               PercentDone = 100.0 * float(iIMG - iFirst + 1) / 
     1              float(iLast - iFirst + 1)

               write(UnitLog,*) 'Sobel #245 ',
     1              'Processing::', trim(GeoSobelFileName), '::'
               print *, PercentDone, iFirst, iLast, iIMG, 
     1              ' Sobel #250 Processing::', trim(GeoSobelFileName), 
     2              '::'

c     Read the input SST array. May need to decompress before reading.
               
               call FileStatus(MedianFileName, MedianExists, 
     1              MedianCompressed, YearStartC, MedianTempFileName, 
     2              .true., .false.)

               if(MedianCompressed .eqv. .true.) then
                  write(UnitLog,*) 'Sobel #248: Input file should ',
     1                 'not be compressed'
                  stop 'Sobel #248: Input file should not be compressed 
     1.'
               else
                  call ReadMedianFile( MedianFileName, ncMedID, 
     1                 MedID, SecondsSince1970ID, MedSST, 
     2                 SecondsSince1970T)
               endif

c     Do Sobel

               call SobelSub( MedSST, xGrad, yGrad)

c     Create Sobel NetCDF output file and write data to it 
               if (SobelFileFlag) then
                  call CreateSobelFile( SobelFileName, 
     1                 ncMedID, MedID, SecondsSince1970ID, ncSobelID, 
     2                 xGradID, yGradID, GradMagID)
                  
                  if(debug .eq. 1) print *, 'Sobel #251: Sobel ',
     1                 'filename: ', SobelFileName
                  
                  status =  nf90_put_var( ncSobelID, xGradID, xGrad)
                  if(status .ne.  nf90_noerr) call handle_err(status)
                  
                  if(debug .eq. 1) print *, 'Sobel #252: Sobel ',
     1                 'filename: ', SobelFileName
                  
                  status =  nf90_put_var( ncSobelID, yGradID, yGrad)
                  if(status .ne.  nf90_noerr) call handle_err(status)
                  
                  if(debug .eq. 1) print *, 'Sobel #253: Sobel ',
     1                 'filename: ', SobelFileName
                  
                  if(debug .eq. 1) print *, 'Sobel #254: Sobel ',
     1                 'filename: ', SobelFileName
                  
                  status = nf90_close(ncSobelID)
                  if(status .ne. nf90_noerr) call handle_err(status)
               endif
c     Now correct for geometry

               if(Debug .eq. 1) print *, 'Sobel #260: Do geoSobel ',
     1              'correction. ncMedID', ncMedID
               
c     I used to do this in a subroutine, but that didn't work so hot.
c     The problem was that I lost the coefficients used to calculate
c     the geo Sobel gradients because they were contained in 
c     GetGeoGradients. I could have passed them back to the main
c     program, but I'm not sure what the memory implications are for
c     this, so I decided to simply move the code from 
c     GetGeoGradients here.

c$$$  call GetGeoGradients( ncMedID, ncSobelID, xGrad, , 
c$$$  1              yGrad, sst_grad_east, sst_grad_north, GradMag)             
               if(debug .eq. 1) print *, 'Sobel #1000: All debug ',
     1              'statements over 1000 are for the section doing ',
     2              'the geographic correction.'

c     Get the filename of the file with latitude and longitude arrays,
c     will need it later. The median file should still be open.

               LatLonFileName = BlankFill(:len(LatLonFileName))
               status = nf90_get_att( ncMedID, nf90_global,
     1              'LatLonFileName', LatLonFileName)
               if(status .ne. nf90_noerr) call handle_err(status)

               LatLonFileName = trim(BaseDir) // trim(LatLonFileName)

c     Get the lat, lon fields if the LatLonFileName has changed since
c     the last SST file processed.

               if(debug .eq. 1) then
                  print *, 'Sobel #1262: LatLonFileName::',
     1                 trim(LatLonFileName), '::'
                  print *, 'Sobel #1263: LatLonFileNameSave::',
     1                 trim(LatLonFileNameSave), '::'
               endif
               
               if( LatLonFileNameSave(:len(trim(LatLonFileName))) .ne. 
     1              trim(LatLonFileName) ) then

                  if(debug .eq. 1) print *, 'Sobel #1265. Reading ',
     1                 'LatLon file.'

                  LatLonFileNameSave = 
     1                 BlankFill(:len(LatLonFileNameSave))
                  LatLonFileNameSave = LatLonFileName

                  call GenerateCoefficientsForGeoSobel( DistX, DistY, 
     1                 SlopeA, SlopeB, SinAlphaA, CosAlphaA, 
     2                 SinAlphaB, CosAlphaB)

c     The following lines write out temporary files with the coefficient
c     fields in them.

                  if( (Debug_Max .eq. 1) .and. (Debug .eq. 1) ) then
                     DebugFile = BlankFill(:len(DebugFile))
                     DebugFile = trim(BaseDir) // 'TmpDir/DistX.nc'
                     call QuickWrite(DebugFile, ncMedID, DistX)

                     DebugFile = BlankFill(:len(DebugFile))
                     DebugFile = trim(BaseDir) // 'TmpDir/DistY.nc'
                     call QuickWrite(DebugFile, ncMedID, DistY)

                     DebugFile = BlankFill(:len(DebugFile))
                     DebugFile = trim(BaseDir) // 
     1                    'TmpDir/SlopeA.nc'
                     call QuickWrite( DebugFile, ncMedID, 
     1                    SlopeA)

                     DebugFile = BlankFill(:len(DebugFile))
                     DebugFile = trim(BaseDir) // 
     1                    'TmpDir/SlopeB.nc'
                     call QuickWrite( DebugFile, ncMedID, 
     1                    SlopeB)

                     DebugFile = BlankFill(:len(DebugFile))
                     DebugFile = trim(BaseDir) // 
     1                    'TmpDir/SinAlphaA.nc'
                     call QuickWrite( DebugFile, ncMedID, 
     1                    SinAlphaA)

                     DebugFile = BlankFill(:len(DebugFile))
                     DebugFile = trim(BaseDir) // 
     1                    'TmpDir/CosAlphaA.nc'
                     call QuickWrite(DebugFile, ncMedID, 
     1                    CosAlphaA)

                     DebugFile = BlankFill(:len(DebugFile))
                     DebugFile = trim(BaseDir) // 
     1                    'TmpDir/SinAlphaB.nc'
                     call QuickWrite( DebugFile, ncMedID, 
     1                    SinAlphaB)

                     DebugFile = BlankFill(:len(DebugFile))
                     DebugFile = trim(BaseDir) // 
     1                    'TmpDir/CosAlphaB.nc'
                     call QuickWrite(DebugFile, ncMedID, 
     1                    CosAlphaB)
                  endif

                  if(Debug .eq. 1) print *, 'Sobel #1269. GeoSobel ',
     1                 'coefficients generated.'

               endif

c     Convert pixel gradients to gradients per kilometer.
               
               if(Debug .eq. 1) print *, 'Sobel #1270. Generate ',
     1              'K/km gradients in north-east coordinate system.'

               do 210 jy=1,LenY
                  do 220 ix=1,LenX
                     
                     if( (abs(DistX(ix,jy)) .gt. 0.001) .and. 
     1                    (xGrad(ix,jy) .ne. FillValueReal) ) then
                        
                        xGrad(ix,jy) = xGrad(ix,jy) / DistX(ix,jy)
                        yGrad(ix,jy) = yGrad(ix,jy) / DistY(ix,jy)

c     If neither axis is parallel to either east or north --------------

c$$$                        if(  (SlopeA(ix,jy) .ne. 0.0) .and. 
c$$$     1                       (SlopeB(ix,jy) .ne. 0.0) .and. 
c$$$     2                       (abs(SlopeA(ix,jy)) .ne. Infinity) .and. 
c$$$     3                       (abs(SlopeB(ix,jy)) .ne. Infinity) )
c$$$     4                       then

                        if(  (SlopeA(ix,jy) .ne. 0.0) .and. 
     1                       (SlopeB(ix,jy) .ne. 0.0) ) then
                           
                           TempVA = SinAlphaA(ix,jy) + CosAlphaA(ix,jy)
     1                          / SlopeA(ix,jy)

                           TempVB = SinAlphaB(ix,jy) + CosAlphaB(ix,jy)
     1                          / SlopeB(ix,jy)

                           sst_grad_east(ix,jy) = 
     1                          (TempVB * yGrad(ix,jy)-
     1                          TempVA * xGrad(ix,jy)) / 
     3                          (1.0 / SlopeB(ix,jy) - 
     4                          1.0 / SlopeA(ix,jy) )

                           sst_grad_north(ix,jy) = -1.0 / SlopeB(ix,jy) 
     1                          * sst_grad_east(ix,jy) +
     1                          TempVB * yGrad(ix,jy)
                        endif

c     CASE 1 AND 2: Dimension #1 (A, x) is parallel to east ------------

                        if(SlopeA(ix,jy) .eq. 0.0) then
                           
                           sst_grad_east(ix,jy) = xGrad(ix,jy) * 
     1                          CosAlphaA(ix,jy)

                           TempVB = SinAlphaB(ix,jy) + CosAlphaB(ix,jy)
     1                          / SlopeB(ix,jy)

c     CASE 1: Dimensiont #2 not parallel to north

                           if(SlopeB(ix,jy) .ne. Infinity) then
                              sst_grad_north(ix,jy) = 
     1                             -sst_grad_east(ix,jy) / 
     2                             SlopeB(ix,jy) + TempVB * yGrad(ix,jy)
                           endif

c     CASE 2: Dimension #2 parallel to north.

                           if(SlopeB(ix,jy) .eq. Infinity) then
                              sst_grad_north(ix,jy) = SinAlphaB(ix,jy) *
     1                             yGrad(ix,jy)
                           endif

                        endif

c     CASE 3 AND 4: Dimension #2 (A, x) is parallel to east ------------

                        if(SlopeB(ix,jy) .eq. 0.0) then

                           sst_grad_east(ix,jy) = yGrad(ix,jy) *
     1                          CosAlphaB(ix,jy)

                           TempVA = SinAlphaA(ix,jy) + CosAlphaA(ix,jy)
     1                          / SlopeA(ix,jy)

c     CASE 3: Dimensiont #1 not parallel to north

                           if(SlopeA(ix,jy) .ne. Infinity) then
                              sst_grad_north(ix,jy) = 
     1                             -sst_grad_east(ix,jy) / 
     2                             SlopeA(ix,jy) + TempVA * xGrad(ix,jy)
                           endif

c     CASE 2: Dimension #1 parallel to north.

                           if(SlopeA(ix,jy) .eq. Infinity) then
                              sst_grad_north(ix,jy) = SinAlphaA(ix,jy) *
     1                             xGrad(ix,jy)
                           endif
                        endif

c$$$                        if(Debug .eq. 1) print *, 
c$$$     1                       'Sobel #1271: ix, jy, ',
c$$$     2                       ' sst_grad_east, sst_grad_north, ',
c$$$     3                       ' vError, xGrad, yGrad, SinAlphaA, ',
c$$$     4                       ' CosAlphaA, SinAlphaB, CosAlphaB: ',
c$$$     5                       ix, jy, sst_grad_east(ix,jy),
c$$$     6                       sst_grad_north(ix,jy), vError, 
c$$$     7                       xGrad(ix,jy), yGrad(ix,jy), 
c$$$     8                       SinAlphaA(ix,jy), CosAlphaA(ix,jy), 
c$$$     9                       SinAlphaB(ix,jy), CosAlphaB(ix,jy)

c     Now calculate the gradient magnitude.

                        GradMag(ix,jy) = sqrt(
     1                       sst_grad_east(ix,jy) * 
     2                       sst_grad_east(ix,jy) + 
     3                       sst_grad_north(ix,jy) * 
     4                       sst_grad_north(ix,jy) )
                     else
                        xGrad(ix,jy) = FillValueReal
                        yGrad(ix,jy) = FillValueReal
                        sst_grad_east(ix,jy) = FillValueReal
                        sst_grad_north(ix,jy) = FillValueReal
                        GradMag(ix,jy) = FillValueReal
                     endif

c$$$                     if((abs(ix-1666) .lt. 3) .and. 
c$$$     1                    (abs(jy-230) .lt. 3) ) then
c$$$                        print *, 'Sobel #800: ix, jy, DistX, DistY, ',
c$$$     1                       'SlopeA, SlopeB, xGrad, yGrad, TempVA, ',
c$$$     2                       'TempVB: ',
c$$$     3                       ix, jy, DistX(ix,jy), DistY(ix,jy), 
c$$$     4                       SlopeA(ix,jy), SlopeB(ix,jy), xGrad(ix,jy),
c$$$     5                       yGrad(ix,jy), TempVA, TempVB
c$$$                        print *, 'Sobel #801: sst_grad_east, ',
c$$$     1                       ' sst_grad_north, CosAlphaA, SinAlphaA, ',
c$$$     2                       'CosAlphaB, SinAlphaB: ',
c$$$     3                       sst_grad_east(ix,jy),sst_grad_north(ix,jy),
c$$$     4                       CosAlphaA(ix,jy), SinAlphaA(ix,jy), 
c$$$     5                       CosAlphaB(ix,jy), SinAlphaB(ix,jy)
c$$$                     endif

 220              continue
 210           continue

c     Now print out a lot of debug stuff and write some debug files.

               if( (Debug_Max .eq. 1)  .and. (Debug .eq. 1) ) then

                  print*, 'iStrt, iEnd, jStrt, jEnd: ', iStrt, iEnd, 
     1                 jStrt, jEnd
                  Msg50 = '--- xGrad --- EOS'
                  call PrintArrayReal( 
     1                 xGrad(istrt:iend,jstrt:jend), Msg50)
                  Msg50 = '--- yGrad --- EOS'
                  call PrintArrayReal( 
     1                 yGrad(istrt:iend,jstrt:jend), Msg50)
                  Msg50 = '--- sst_grad_east --- EOS'
                  call PrintArrayReal( 
     1                 sst_grad_east(istrt:iend,jstrt:jend), Msg50)
                  Msg50 = '--- sst_grad_north --- EOS'
                  call PrintArrayReal( 
     1                 sst_grad_north(istrt:iend,jstrt:jend), Msg50)

                  DebugFile = BlankFill(:len(DebugFile))
                  DebugFile = trim(BaseDir) // 'TmpDir/xGrad.nc'
                  call QuickWrite( DebugFile, ncMedID, xGrad)

                  DebugFile = BlankFill(:len(DebugFile))
                  DebugFile = trim(BaseDir) // 'TmpDir/yGrad.nc'
                  call QuickWrite( DebugFile, ncMedID, yGrad)

                  DebugFile = BlankFill(:len(DebugFile))
                  DebugFile = trim(BaseDir) // 'TmpDir/sst_grad_east.nc'
                  call QuickWrite( DebugFile, ncMedID, 
     1                 sst_grad_east)
               endif

               if(Debug .eq. 1) print *, 'Sobel #270. Generate ',
     1              'K/km gradients in north-east coordinate system.'

c     Create Sobel NetCDF output file, write data to it and cleanup.

               call CreateGeoSobelFile( GeoSobelFileName, 
     1              ncMedID, MedID, SecondsSince1970ID, ncSobelID, 
     2              xGradID, yGradID, GradMagID)

               if(debug .eq. 1) print *, 'Sobel #251: Sobel filename: ',
     1              SobelFileName

               status =  nf90_put_var( ncSobelID, xGradID, 
     1              sst_grad_east)
               if(status .ne.  nf90_noerr) call handle_err(status)
               
               if(debug .eq. 1) print *, 'Sobel #252: wrote grad_east'

               status =  nf90_put_var( ncSobelID, yGradID, 
     1              sst_grad_north)
               if(status .ne.  nf90_noerr) call handle_err(status)

               if(debug .eq. 1) print *, 'Sobel #253: wrote grad_north'

               status =  nf90_put_var( ncSobelID, GradMagID, GradMag)
               if(status .ne.  nf90_noerr) call handle_err(status)

               if(debug .eq. 1) print *, 'Sobel #254: wrote grad_mag'

c     Done with GeoSobel file; close it.

               status = nf90_close(ncSobelID)
               if(status .ne. nf90_noerr) call handle_err(status)

c     All done with this median file; close i.t

               status = nf90_close(ncMedID)
               if(status .ne. nf90_noerr) call handle_err(status)

c     Increment the number of files processed.

               NumberOfFilesProcessed = NumberOfFilesProcessed + 1

            endif
         endif
         

c--------------------------ALL DONE PROCESSING ------------------------

 1000 continue

      close(UnitLog)
      close(UnitInventory)

      if(Debug .eq. 1) print *, 'GeoSobel #999: All done.'

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
      end program GeoSobel_Main

c**********************************************************************
      subroutine SobelSub( SST, xGrad, yGrad)
c**********************************************************************
c     
c     This subroutine will calculate the x and y Sobel gradients. All 
c     - values are normalized to the units of the SST field per pixel.
c     
c     Written by Peter Cornillon, University of Rhode Island,
c     pcornillon@me.com 18 April 2009
c     
c**********Functions 
c     
c     etime - elapsed time

c**********Parameter statements
c     
c     Debug - 1 for all debug output, 2 for timing only, 0 for none.
c     UnitLog - unit number for log file of this run.
c     FillValue

c**********General Variables.
c     
c     InArray - array on which to apply the Sobel operator
c     xGrad - the x-component of the Sobel gradient
c     yGrad - the y-component of the Sobel gradient
c     GradMag - the magnitude of the Sobel gradient
c     ix, jy - do loop parameters
c     MinSST - used to test for missing value
c     MaxSST - used to test for missing value
c     

c*************System timing variables
c     
c     ElapsedTime - time since the process started
c     LastElapsedTime - place to save the last measured elapsed time.
c     TimeArgs - after call to etime, the first TimeArgs is the user
c     . portion of the elapsed time and the second one is the system
c     . portion.
c     LastTimeArgs
c     
      use netcdf

      implicit none

c     Parameter statements

      include 'ParameterStatements.f'
      
c     Functions

      real etime

c******General variables

      integer*2 SST(1:LenX,1:LenY)
      real*4 xGrad(1:LenX,1:LenY)
      real*4 yGrad(1:LenX,1:LenY)
c$$$  real GradMag(1:LenX,1:LenY)

      integer MinSST, MaxSST

      integer ix, iy

      real Norm, xGradTemp, yGradTemp

      common /SobelCommon/ Norm

c     System timing variables.

      real ElapsedTime, LastElapsedTime, TimeArgs(2), LastTimeArgs(2)

c---------------------------------start here --------------------------

c     First set boarder pixels to fill value, can't generate gradients
c     for them.

      if(debug .eq. 1) print *, 'SobelSub #100' 

c     Note that the scaling here is based on the scale factor for the
c      SST input field - ignore the input and output scale factors in 
c      the run control file. These are used by AppQual only and do not
c      apply here. The scale factor to use here is the one that is in
c      the median SST file; i.e., sstScaleFactorIn!

      Norm = sstScaleFactorIn / 8

      do 100 iy=1,LenY
         xGrad(1,iy) = FillValueReal
         yGrad(1,iy) = FillValueReal

         xGrad(LenX,iy) = FillValueReal
         yGrad(LenX,iy) = FillValueReal
 100  continue

      do 110 ix=1,LenX
         xGrad(ix,1) = FillValueReal
         yGrad(ix,1) = FillValueReal

         xGrad(ix,LenY) = FillValueReal
         yGrad(ix,LenY) = FillValueReal
 110  continue

      if(debug .eq. 1) print *, 'SobelSub #110' 

c     Now Calculate the gradients. Divide x and y gradients by 8 to
c     normalize to K/piyel

      if(debug .eq. 1) LastElapsedTime = etime(TimeArgs)

      do 200 iy=2,LenY-1
         do 210 ix=2,LenX-1

c     Get the min SSTs for this 3x3 region to test for missing values,
c     SSTFillValueIn in this case. If yes, skip calculation of Sobel. 

            MinSST = min(SST(ix-1,iy-1), SST(ix,iy-1), SST(ix+1,iy-1),
     1           SST(ix-1,iy),   SST(ix,iy),   SST(ix+1,iy),
     2           SST(ix-1,iy+1), SST(ix,iy+1), SST(ix+1,iy+1))

            if(MinSST .eq. SSTFillValueIn) then
               xGrad(ix,iy) = FillValueReal
               yGrad(ix,iy) = FillValueReal

            else
c     
c     Be careful here. The first dimension is the northsouth direction
c     for MSG. Need to make sure that that is the case for other
c     satellites.

               yGrad(ix,iy) = float(
     1              SST(ix+1,iy+1) + 2*SST(ix,iy+1) + SST(ix-1,iy+1) - 
     2              SST(ix+1,iy-1) - 2*SST(ix,iy-1) - SST(ix-1,iy-1) ) 
     3              * Norm
               xGrad(ix,iy) = float(
     1              SST(ix+1,iy+1) + 2*SST(ix+1,iy) + SST(ix+1,iy-1) - 
     2              SST(ix-1,iy+1) - 2*SST(ix-1,iy) - SST(ix-1,iy-1) )
     3              * Norm

            endif

 210     continue

 200  continue

      if(debug .eq. 1) print *, 'SobelSub #200' 

      end subroutine SobelSub

c**********************************************************************
      subroutine GenerateCoefficientsForGeoSobel( DistX, DistY, 
     1     SlopeA, SlopeB, SinAlphaA, CosAlphaA, SinAlphaB, CosAlphaB)
c**********************************************************************
c     
c     This subroutine will generate the distance between pixels in the 
c     i and the j directions of the input lat, lon array. It will also
c     calculate the coefficients required to convert from a separation
c     in pixel space to one in latitude and longitude.
c     
c     Written by Peter Cornillon, University of Rhode Island,
c     pcornillon@me.com 21 May 2011.
c     
c**********Functions 
c     
c     etime - elapsed time

c**********Parameter statements
c     
c     Debug - 1 for all debug output, 2 for timing only, 0 for none.
c     UnitLog - unit number for log file of this run.
c     FillValue
      
      use netcdf

      implicit none

c     Parameter statements

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

c     CosAlphaA - Cosine of the angle that the y-axis makes relative
c     to east.
      real*4 CosAlphaA(1:LenX,1:LenY)
c     CosAlphaB - Cosine of the angle that the x-axis makes relative
c     to east.
      real*4 CosAlphaB(1:LenX,1:LenY)

c     Deg2Rad - conversion factor from degrees to radians: pi/180
      real*4, parameter :: Deg2Rad=0.01745329
c     DistX - the distance between pixels along the "x" dimension; i.e.,
c     the first dimension.
      real*4 DistX(1:LenX,1:LenY)
c     DistX - the distance between pixels along the "y" dimension; i.e.,
c     the second dimension.
      real*4 DistY(1:LenX,1:LenY)

c     Factor - km/degree used to calculate distances in lat and lon
      real*4, parameter :: Factor=111.0

c     ix - do loop parameter
      integer ix

c     jy - do loop parameter
      integer jy

c     lat - the latitude array of pixel locations read from the latlon 
c     file.
      real*4, allocatable :: lat(:,:)
c     LatD - temporary variable for the latitudinal distance between 
c     points in the grid.
c$$$      real*4, allocatable :: LatD(:,:)
      real*4 LatD
c     LatLonFillValueReal - the value used for missing data. Be careful
c      here. The fill value in the netCDF file written out for this 
c      variable in integer, but the latitude and longitude are converted
c      to real in the call to ReadLatLon so the fill value is redefined
c      in that subroutine to be real. In order not to confuse it with 
c      the fill value read in, it is renamed as well. 
      real LatLonFillValueReal
c     lon - the longitude array of pixel locations read from the latlon 
c     file.
      real*4, allocatable :: lon(:,:)
c     LonD - temporary variable for the longitudinal distance between 
c     points in the grid.
c$$$      real*4, allocatable :: LonD(:,:)
      real*4 LonD

c     ncID - NetCDF ID for the input data set
      integer ncID

c     SinAlphaA - Sin of the angle that the y-axis makes relative
c     to east.
      real*4 SinAlphaA(1:LenX,1:LenY)
c     SinAlphaB - Sin of the angle that the x-axis makes relative
c     to east.
      real*4 SinAlphaB(1:LenX,1:LenY)
c     SlopeB - The slope of the line connecting pixels in 1st dimension
c     relative to east.
      real*4 SlopeB(1:LenX,1:LenY)
c     SlopeA - The slope of the line connecting pixels in 2nd dimension
c     relative to east.
      real*4 SlopeA(1:LenX,1:LenY)

c---------------------------------start here --------------------------

      if(debug .eq. 1) print *, 'GenerateCoefficientsForSobel #000' 

c$$$      allocate( lat(LenX,LenY),
c$$$     1     lon(LenX,LenY),
c$$$     2     LatD(LenX,LenY),
c$$$     3     LonD(LenX,LenY),
c$$$     4     stat=ierr)

      allocate( lat(LenX,LenY),
     1     lon(LenX,LenY),
     2     stat=ierr)

c     First, get latitude and longitude arrays.

      call ReadLatLon( LatLonFileName, Lat, Lon, LatLonFillValueReal)

      if(debug .eq. 1) print *, 'GenerateCoefficientsForSobel #100. ',
     1     'Latitude and longitude variables allocated and read.'

c     Get the separation between pixels in the "x" and "y" directions 
c     and calculate coefficients needed for the swath to geo conversion.
c     Set DistX and DistY to zero for border values and missing values
c     of latitude and/or longitude.

      do 210 jy=1,LenY
         do 220 ix=1,LenX

c     Make sure that there is a good lat and a good lon value.

            if( (Lat(ix,jy) .ne. LatLonFillValueReal) .and. 
     1           (Lon(ix,jy) .ne. LatLonFillValueReal) .and.
     2           (ix .gt. 1) .and. (ix .lt. LenX) .and.
     3           (jy .gt. 1) .and. (jy .lt. LenY) ) then

c     Do the "x" direction first. Need to divide by two since the 
c      distances being calculated here are the distances between 
c      pixels on the opposite side of the pixel of interest; i.e., twice
c      the distance between adjacent pixels.

c$$$               LatD(ix,jy) = (Lat(ix+1,jy) - Lat(ix-1,jy)) * Factor / 
c$$$     1              2.
c$$$               LonD(ix,jy) = (Lon(ix+1,jy) - Lon(ix-1,jy)) * Factor / 
c$$$     1              2.0
c$$$               LonD(ix,jy) = LonD(ix,jy) * cos(Lat(ix,jy) * Deg2Rad)
c$$$
c$$$               DistX(ix,jy) = sqrt(LatD(ix,jy) * LatD(ix,jy) + 
c$$$     1              LonD(ix,jy) * LonD(ix,jy))
c$$$
c$$$               SinAlphaA(ix,jy) = LatD(ix,jy) / DistX(ix,jy)
c$$$               CosAlphaA(ix,jy) = LonD(ix,jy) / DistX(ix,jy)

               LatD = (Lat(ix+1,jy) - Lat(ix-1,jy)) * Factor / 2.0
               LonD = (Lon(ix+1,jy) - Lon(ix-1,jy)) * Factor / 2.0
               LonD = LonD * cos(Lat(ix,jy) * Deg2Rad)

               DistX(ix,jy) = sqrt(LatD * LatD + LonD * LonD)

               SinAlphaA(ix,jy) = LatD / DistX(ix,jy)
               CosAlphaA(ix,jy) = LonD / DistX(ix,jy)

c     If LatD = 0, SlopeA = 0; if LonD = 0, SlopeA = infinity, 
c     otherwise SlopeA = LatD / LonD. Both LatD and LonD can not
c     equal zero.

c$$$               if(LatD(ix,jy) * LonD(ix,jy) .ne. 0.0) then
c$$$                  SlopeA(ix,jy) = LatD(ix,jy) / LonD(ix,jy)
c$$$               else
c$$$                  if(LatD(ix,jy) .eq. 0.0) then
               if(LatD * LonD .ne. 0.0) then
                  SlopeA(ix,jy) = LatD / LonD
               else
                  if(LatD .eq. 0.0) then
                     SlopeA(ix,jy) = 0.0
                  else
                     SlopeA(ix,jy) = Infinity
                  endif
               endif

               if( (Debug_Max .eq. 1)  .and. (Debug .eq. 1) ) then
                  if( (jy .ge. jstrt) .and. (jy .le. jend) .and.
     1                 (ix .ge. istrt) .and. (ix .le. iend) ) then
                     print *, 'ix, jy, LatD, LonD, DistX, SlopeA, ',
     1                    'SinAlphaA, CosAlpha: ', ix, jy, 
     2                    LatD, LonD, DistX(ix,jy), 
     3                    SlopeA(ix,jy), SinAlphaA(ix,jy), 
     4                    CosAlphaA(ix,jy)
c$$$     2                    LatD(ix,jy), LonD(ix,jy), DistX(ix,jy), 
c$$$     3                    SlopeA(ix,jy), SinAlphaA(ix,jy), 
c$$$     4                    CosAlphaA(ix,jy)
                  endif
               endif

c     And now the "y" direction.

c$$$               LatD(ix,jy) = (Lat(ix,jy+1) - Lat(ix,jy-1)) * Factor /
c$$$     1              2.0              
c$$$               LonD(ix,jy) = (Lon(ix,jy+1) - Lon(ix,jy-1)) * Factor /
c$$$     1              2.0              
c$$$               LonD(ix,jy) = LonD(ix,jy) * cos(Lat(ix,jy) * Deg2Rad)
c$$$
c$$$               DistY(ix,jy) = sqrt(LatD(ix,jy) * LatD(ix,jy) + 
c$$$     1              LonD(ix,jy) * LonD(ix,jy))
c$$$
c$$$               SinAlphaB(ix,jy) = LatD(ix,jy) / DistY(ix,jy)
c$$$               CosAlphaB(ix,jy) = LonD(ix,jy) / DistY(ix,jy)

               LatD = (Lat(ix,jy+1) - Lat(ix,jy-1)) * Factor / 2.0
               LonD = (Lon(ix,jy+1) - Lon(ix,jy-1)) * Factor / 2.0
               LonD = LonD * cos(Lat(ix,jy) * Deg2Rad)

               DistY(ix,jy) = sqrt(LatD * LatD + LonD * LonD)

               SinAlphaB(ix,jy) = LatD / DistY(ix,jy)
               CosAlphaB(ix,jy) = LonD / DistY(ix,jy)

c     If LatD = 0, SlopeB = 0; if LonD = 0, SlopeB = infinity, 
c     otherwise SlopeB = LatD / LonD. Both LatD and LonD can not
c     equal zero.

c$$$               if(LatD(ix,jy) * LonD(ix,jy) .ne. 0.0) then
c$$$                  SlopeB(ix,jy) = LatD(ix,jy) / LonD(ix,jy)
c$$$               else
c$$$                  if(LatD(ix,jy) .eq. 0.0) then
                if(LatD * LonD .ne. 0.0) then
                  SlopeB(ix,jy) = LatD / LonD
               else
                  if(LatD .eq. 0.0) then
                     SlopeB(ix,jy) = 0.0
                  else
                     SlopeB(ix,jy) = Infinity
                  endif
               endif

               if( (Debug_Max .eq. 1) .and. (Debug .eq. 1) ) then
                  if( (jy .ge. jstrt) .and. (jy .le. jend) .and.
     1                 (ix .ge. istrt) .and. (ix .le. iend) ) then
                     print *, 'LatD, LonD, DistY, SlopeB, ',
     1                    'SinAlphaB, CosAlphaB: ', 
     2                    LatD, LonD, DistY(ix,jy), 
     3                    SlopeB(ix,jy), SinAlphaB(ix,jy), 
     4                    CosAlphaB(ix,jy)
c$$$     2                    LatD(ix,jy), LonD(ix,jy), DistY(ix,jy), 
c$$$     3                    SlopeB(ix,jy), SinAlphaB(ix,jy), 
c$$$     4                    CosAlphaB(ix,jy)
                  endif
               endif

            else
               DistX(ix,jy) = 0.0
               DistY(ix,jy) = 0.0
            endif

 220     continue
 210  continue

c     Write out more debug_max stuff.

      if ((Debug_Max .eq. 1) .and. (Debug .eq. 1) ) then
         print*, 'iStrt, iEnd, jStrt, jEnd: ', iStrt, iEnd, jStrt, jEnd
         Msg50 = '--- longitude --- EOS'
         call PrintArrayReal( Lon(istrt:iend,jstrt:jend), Msg50)
         Msg50 = '--- latitude --- EOS'
         call PrintArrayReal( Lat(istrt:iend,jstrt:jend), Msg50)
c$$$         Msg50 = '--- LonD --- EOS'
c$$$         call PrintArrayReal( LonD(istrt:iend,jstrt:jend), Msg50)
c$$$         Msg50 = '--- LatD --- EOS'
c$$$         call PrintArrayReal( LatD(istrt:iend,jstrt:jend), Msg50)
         Msg50 = '--- DistX --- EOS'
         call PrintArrayReal( DistX(istrt:iend,jstrt:jend), Msg50)
         Msg50 = '--- DistY --- EOS'
         call PrintArrayReal( DistY(istrt:iend,jstrt:jend), Msg50)
         Msg50 = '--- SlopeA --- EOS'
         call PrintArrayReal( SlopeA(istrt:iend,jstrt:jend), Msg50)
         Msg50 = '--- SlopeB --- EOS'
         call PrintArrayReal( SlopeB(istrt:iend,jstrt:jend), Msg50)
         Msg50 = '--- SinAlphaA --- EOS'
         call PrintArrayReal( SinAlphaA(istrt:iend,jstrt:jend), Msg50)
         Msg50 = '--- CosAlphaA --- EOS'
         call PrintArrayReal( CosAlphaA(istrt:iend,jstrt:jend), Msg50)
         Msg50 = '--- SinAlphaB --- EOS'
         call PrintArrayReal( SinAlphaB(istrt:iend,jstrt:jend), Msg50)
         Msg50 = '--- CosAlphaB --- EOS'
         call PrintArrayReal( CosAlphaB(istrt:iend,jstrt:jend), Msg50)
      endif

      if(debug .eq. 1) print *, 'GenerateCoefficientsForSobel #999' 

      end subroutine GenerateCoefficientsForGeoSobel

c**********************************************************************
      subroutine CreateSobelFile( FileName, ncInID, sstID, 
     1     SecondsSince1970InID, ncSobelID, xGradID, yGradID, GradMagID)
c**********************************************************************
c     
c     This subroutine creates the .nc file for the Sobel gradient
c     fields.
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
c     FileName - the name of the file to create
c     status - return status after a netCDF call
c     ncInID and ncSobelID - IDs for the input and output files.
c     sstID - ID for the sst variable will copy these attributes. 
c     SecondsSince1970InID - the ID of the number of seconds since  
c     . 00:00 1 Jan 1970 corresponding to the time of the sst array.
c     xGradID - ID for the x component of the Sobel gradient
c     yGradID - ID for the y component of the Sobel gradient
c     GradMagID - ID for the magnitude of the Sobel gradient
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

c******General variables

c     AttDescription - attribute description when needed.
      character*1000 AttDescription

      character*319 FileName
      character*100 NewTitle
      character*100 ProcessingProgram
      character*67 Source

c      character*3000 Summary

      integer ncInID, ncSobelID

      integer sstID, SecondsSince1970InID
      integer xGradID, yGradID, GradMagID, SecondsSince1970OutID
      integer LenXinID, LenYinID

      real*8 SecondsSince1970

      integer nxDimIDout, nyDimIDout
      integer DimsOut(2)

      real NewFactor

      real Norm

      common /SobelCommon/ Norm

c     Variables used to determine the current date and time

      integer Values(8)
      character*5 Zone
      character*10 Time
      character*8 Date

c     include 'netcdf.inc'

c---------------------------------start here --------------------------

      if(debug .eq. 1) print *, 'CreateSobelFile #000'

c     Check that the dimensions of the SST array agree with those
c     - of the Sobel array.

      status =  nf90_inq_dimid( ncInID, 'nx', LenXinID)
      if(status .ne.  nf90_noerr) call handle_err(status)
      
      status = nf90_inquire_dimension( ncInID, LenXinID, len=LenXin)
      if(status .ne. nf90_noerr) call handle_err(status)

      status =  nf90_inq_dimid( ncInID, 'ny', LenYinID)
      if(status .ne.  nf90_noerr) call handle_err(status)
      
      status = nf90_inquire_dimension( ncInID, LenYinID, len=LenYin)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, ' LenXin, LenYin: ', LenXin, LenYin

      if( (LenXin .ne. LenX) .or. (LenYin .ne. LenY) )  stop 
     1     'CreateSobelFile #100: SST dimensions do not match Median dim
     2ensions. STOP'

c************Now start creating the Sobel file ***********************

c     Create the output file.

      if(debug .eq. 1) print *, 'CreateSobelFile #110: ',
     1     'Sobel filename: ', FileName
      
      status =  nf90_create( FileName, OR(nf90_netcdf4, nf90_noclobber),
     1     ncSobelID)
      if(status .ne.  nf90_noerr) call handle_err(status)

c     Create the dimensions for the output fields.

      status =  nf90_def_dim( ncSobelID, 'nx', LenX, nxDimIDout)
      if(status .ne.  nf90_noerr) call handle_err(status)

      status =  nf90_def_dim( ncSobelID, 'ny', LenY, nyDimIDout)
      if(status .ne.  nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'CreateSobelFile #120: '

c     Create xGrad, yGrad and GradMag fields. x first.

      DimsOut(1) = nxDimIDout
      DimsOut(2) = nyDimIDout

      chunksize(1) = min(LenX, chunkszLenX)
      chunksize(2) = min(LenY, chunkszLenY)

      status = nf90_def_var( ncSobelID, 'x_gradient', nf90_real, 
     1     DimsOut, xGradID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_chunking( ncSobelID, xGradID, 0, 
     1     chunksize)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_deflate( ncSobelID, xGradID, 0, 1, 4)
      if(status .ne. nf90_noerr) call handle_err(status)

c     y

      status = nf90_def_var( ncSobelID, 'y_gradient', nf90_real, 
     1     DimsOut, yGradID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_chunking( ncSobelID, yGradID, 0, 
     1     chunksize)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_deflate( ncSobelID, yGradID, 0, 1, 4)
      if(status .ne. nf90_noerr) call handle_err(status)

c     magnitude

      status = nf90_def_var( ncSobelID, 'gradient_magnitude', 
     1     nf90_real, DimsOut, GradMagID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_chunking( ncSobelID, GradMagID, 0, 
     1     chunksize)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_deflate( ncSobelID, GradMagID, 0, 1, 4)
      if(status .ne. nf90_noerr) call handle_err(status)

c     create or copy attribues from input to outfile for x-gradient

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = trim(iVarName) // '_Sobel_gradient'
      status =  nf90_put_att( ncSobelID, xGradID, 
     1     'long_name', AttDescription)
      if(status .ne.  nf90_noerr) call handle_err(status)

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = trim(ParameterUnits) // '_per_pixel'
      status =  nf90_put_att( ncSobelID, xGradID, 
     1     'units', AttDescription)
      if(status .ne.  nf90_noerr) call handle_err(status)

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = 'grid_x_derivative_of_' // trim(ParameterName)
      status =  nf90_put_att( ncSobelID, xGradID, 
     1     'standard_name',  AttDescription)
      if(status .ne.  nf90_noerr) call handle_err(status)

      NewFactor = 1.0

      status =  nf90_put_att( ncSobelID, xGradID, 'add_offset', 0)
      if(status .ne.  nf90_noerr) call handle_err(status)

      status =  nf90_put_att( ncSobelID, xGradID,
     1     'scale_factor', NewFactor)
      if(status .ne.  nf90_noerr) call handle_err(status)

      status =  nf90_put_att( ncSobelID, xGradID,
     1     '_FillValue', FillValueReal)
      if(status .ne.  nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'CreateSobelFile #130: '

c     create or copy attribues from input to outfile for y-gradient

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = trim(jVarName) // '_Sobel_gradient'
      status =  nf90_put_att( ncSobelID, yGradID, 
     1     'long_name', AttDescription)
      if(status .ne.  nf90_noerr) call handle_err(status)

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = trim(ParameterUnits) // '_per_pixel'
      status =  nf90_put_att( ncSobelID, yGradID, 
     1     'units', AttDescription)
      if(status .ne.  nf90_noerr) call handle_err(status)

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = 'grid_y_derivative_of_' // trim(ParameterName)
      status =  nf90_put_att( ncSobelID, yGradID, 
     1     'standard_name', AttDescription)
      if(status .ne.  nf90_noerr) call handle_err(status)

      status =  nf90_put_att( ncSobelID, yGradID, 'add_offset', 0)
      if(status .ne.  nf90_noerr) call handle_err(status)

      status =  nf90_put_att( ncSobelID, yGradID,
     1     'scale_factor', NewFactor)
      if(status .ne.  nf90_noerr) call handle_err(status)

      status =  nf90_put_att( ncSobelID, yGradID,
     1     '_FillValue', FillValueReal)
      if(status .ne.  nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'CreateSobelFile #140: '

c     create or copy attribues from input to outfile for gradient 
c     magnitude

      status =  nf90_put_att( ncSobelID, GradMagID, 
     1     'long_name', 'Sobel_gradient_magnitude')
      if(status .ne.  nf90_noerr) call handle_err(status)

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = trim(ParameterUnits) // '_per_pixel'
      status =  nf90_put_att( ncSobelID, GradMagID, 
     1     'units', AttDescription)
      if(status .ne.  nf90_noerr) call handle_err(status)

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = 'magnitude_of_derivative_of_' // 
     1     trim(ParameterName)
      status =  nf90_put_att( ncSobelID, GradMagID, 
     1     'standard_name', AttDescription)
      if(status .ne.  nf90_noerr) call handle_err(status)

      status =  nf90_put_att( ncSobelID, GradMagID, 'add_offset', 0)
      if(status .ne.  nf90_noerr) call handle_err(status)

      status =  nf90_put_att( ncSobelID, GradMagID,
     1     'scale_factor', NewFactor)
      if(status .ne.  nf90_noerr) call handle_err(status)

      status =  nf90_put_att( ncSobelID, GradMagID,
     1     '_FillValue',  nf90_fill_real)
      if(status .ne.  nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'CreateSobelFile #150: '

c     create the time - seconds since 1970 and define its attributes

      call DefineSecondsSinceVar( ncSobelID, SecondsSince1970OutID)

c     Now generate all of the global attributes

      NewTitle = BlankFill(:len(NewTitle))
      NewTitle = 'x, y and magnitude Sobel gradient of'
      ProcessingProgram = BlankFill(:len(ProcessingProgram))
      ProcessingProgram = 'GeoSobel_Main'

      Summary = BlankFill(1:len(Summary))
      Summary = 'The fields in this file were generated by applying ' //
     1     ' a Sobel operator to the input file in both the i and j ' //
     2     ' directions. The resulting gradients are digital counts ' //
     3     ' per pixel in the respective directions.'

      call GenerateGlobals( ncInID, ncSobelID, NewTitle,
     1      ProcessingProgram, GeoSobelVersionNo, FileName)

      if(Debug .eq. 1) print *, 'CreateSobelFile #160: '

c     All done defining the variables and attributes for output file.

      status =  nf90_enddef(ncSobelID)
      if(status .ne.  nf90_noerr) call handle_err(status)

c     Get the time from the SST file and write it to the Median file. 

      if(Debug .eq. 1) print *, 'CreateSobelFile #170: '

      status =  nf90_get_var( ncInID, SecondsSince1970InID, 
     1     SecondsSince1970)
      if(status .ne.  nf90_noerr) call handle_err(status)

      status =  nf90_put_var( ncSobelID, SecondsSince1970OutID, 
     1     SecondsSince1970)
      if(status .ne.  nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'CreateSobelFile #999: '

      end subroutine CreateSobelFile
      
c**********************************************************************
      subroutine CreateGeoSobelFile( FileName, ncInID, sstID, 
     1     SecondsSince1970InID, ncSobelID, xGradID, yGradID, 
     2     GradMagID)
c**********************************************************************
c     
c     This subroutine creates the .nc file for the Sobel gradient
c     K/km fields in lat, lon coordinates.
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
c     FileName - the name of the file to create
c     status - return status after a netCDF call
c     ncInID and ncSobelID - IDs for the input and output files.
c     sstID - ID for the sst variable will copy these attributes. 
c     SecondsSince1970InID - the ID of the number of seconds since  
c     . 00:00 1 Jan 1970 corresponding to the time of the sst array.
c     xGradID - ID for the x component of the Sobel gradient
c     yGradID - ID for the y component of the Sobel gradient
c     GradMagID - ID for the magnitude of the Sobel gradient
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

c******General variables

c     AttDescription - attribute description when needed.
      character*1000 AttDescription

      character*319 FileName
      character*100 NewTitle
      character*100 ProcessingProgram
      character*67 Source

c      character*3000 Summary

      integer ncInID, ncSobelID

      integer sstID, SecondsSince1970InID
      integer xGradID, yGradID, GradMagID, SecondsSince1970OutID
      integer LenXinID, LenYinID

      real*8 SecondsSince1970

      integer nxDimIDout, nyDimIDout
      integer DimsOut(2)

      real NewFactor

      real Norm

      common /SobelCommon/ Norm

c     Variables used to determine the current date and time

      integer Values(8)
      character*5 Zone
      character*10 Time
      character*8 Date

c     include 'netcdf.inc'

c---------------------------------start here --------------------------

      if(debug .eq. 1) print *, 'CreateGeoSobelFile #000'

c     Check that the dimensions of the SST array agree with those
c     - of the Sobel array.

      if(Debug .eq. 1) print *, 'CreateGeoSobelFile #090: ncInID',
     1     ncInID

      status =  nf90_inq_dimid( ncInID, 'nx', LenXinID)
      if(status .ne.  nf90_noerr) call handle_err(status)
      
      if(Debug .eq. 1) print *, 'CreateGeoSobelFile #91: Read LenXinID'

      status = nf90_inquire_dimension( ncInID, LenXinID, len=LenXin)
      if(status .ne. nf90_noerr) call handle_err(status)

      status =  nf90_inq_dimid( ncInID, 'ny', LenYinID)
      if(status .ne.  nf90_noerr) call handle_err(status)
      
      status = nf90_inquire_dimension( ncInID, LenYinID, len=LenYin)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, ' LenXin, LenYin: ', LenXin, LenYin

      if( (LenXin .ne. LenX) .or. (LenYin .ne. LenY) )  stop 
     1     'CreateGeoSobelFile #100: SST dimensions do not match Median 
     2dimensions. STOP'

c************Now start creating the Median file ***********************

c     Create the output file.

      if(debug .eq. 1) print *, 'CreateGeoSobelFile #110: ',
     1     'Sobel filename::', trim(FileName), '::'
      
      status =  nf90_create( FileName, OR(nf90_netcdf4, nf90_noclobber),
     1     ncSobelID)
      if(status .ne.  nf90_noerr) call handle_err(status)

c     Create the dimensions for the output fields.

      status =  nf90_def_dim( ncSobelID, 'nx', LenX, nxDimIDout)
      if(status .ne.  nf90_noerr) call handle_err(status)

      status =  nf90_def_dim( ncSobelID, 'ny', LenY, nyDimIDout)
      if(status .ne.  nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'CreateGeoSobelFile #120: '

c     Create xGrad, yGrad and GradMag fields. x first.

      DimsOut(1) = nxDimIDout
      DimsOut(2) = nyDimIDout

      chunksize(1) = min(LenX, chunkszLenX)
      chunksize(2) = min(LenY, chunkszLenY)

      status = nf90_def_var( ncSobelID, 'eastward_gradient', 
     1     nf90_real, DimsOut, xGradID)
      if(status .ne. nf90_noerr) call handle_err(status)

         status = nf90_def_var_chunking( ncSobelID, xGradID, 0, 
     1        chunksize)
         if(status .ne. nf90_noerr) call handle_err(status)

         status = nf90_def_var_deflate( ncSobelID, xGradID, 0, 1, 4)
         if(status .ne. nf90_noerr) call handle_err(status)

c     y

      status = nf90_def_var( ncSobelID, 'northward_gradient', 
     1     nf90_real, DimsOut, yGradID)
      if(status .ne. nf90_noerr) call handle_err(status)

         status = nf90_def_var_chunking( ncSobelID, yGradID, 0, 
     1        chunksize)
         if(status .ne. nf90_noerr) call handle_err(status)

         status = nf90_def_var_deflate( ncSobelID, yGradID, 0, 1, 4)
         if(status .ne. nf90_noerr) call handle_err(status)

c     magnitude

      status = nf90_def_var( ncSobelID, 'gradient_magnitude', 
     1     nf90_real, DimsOut, GradMagID)
      if(status .ne. nf90_noerr) call handle_err(status)

         status = nf90_def_var_chunking( ncSobelID, GradMagID, 0, 
     1        chunksize)
         if(status .ne. nf90_noerr) call handle_err(status)

         status = nf90_def_var_deflate( ncSobelID, GradMagID, 0, 1, 4)
         if(status .ne. nf90_noerr) call handle_err(status)

c     create or copy attribues from input to outfile for x-gradient

      status =  nf90_put_att( ncSobelID, xGradID, 
     1     'long_name', 'Sobel_eastward_gradient')
      if(status .ne.  nf90_noerr) call handle_err(status)

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = trim(ParameterUnits) // '_per_km'
      status =  nf90_put_att( ncSobelID, xGradID, 
     1     'units', AttDescription)
      if(status .ne.  nf90_noerr) call handle_err(status)

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = 'eastward_derivative_of_' // 
     1     trim(ParameterName)
      status =  nf90_put_att( ncSobelID, xGradID, 
     1     'standard_name', AttDescription)
      if(status .ne.  nf90_noerr) call handle_err(status)

      status =  nf90_put_att( ncSobelID, xGradID, 'add_offset', 0)
      if(status .ne.  nf90_noerr) call handle_err(status)

      status =  nf90_put_att( ncSobelID, xGradID, 'scale_factor', 1.0)
      if(status .ne.  nf90_noerr) call handle_err(status)

      status =  nf90_put_att( ncSobelID, xGradID,
     1     '_FillValue',  FillValueReal)
      if(status .ne.  nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'CreateGeoSobelFile #130: '

c     create or copy attribues from input to outfile for y-gradient

      status =  nf90_put_att( ncSobelID, yGradID, 
     1     'long_name', 'Sobel_northward_gradient')
      if(status .ne.  nf90_noerr) call handle_err(status)

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = trim(ParameterUnits) // '_per_km'
      status =  nf90_put_att( ncSobelID, yGradID, 
     1     'units', AttDescription)
      if(status .ne.  nf90_noerr) call handle_err(status)

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = 'northward_derivative_of_' // 
     1     trim(ParameterName)
      status =  nf90_put_att( ncSobelID, yGradID, 
     1     'standard_name', AttDescription)
      if(status .ne.  nf90_noerr) call handle_err(status)

      status =  nf90_put_att( ncSobelID, yGradID, 'add_offset', 0)
      if(status .ne.  nf90_noerr) call handle_err(status)

      status =  nf90_put_att( ncSobelID, yGradID, 'scale_factor', 1.0)
      if(status .ne.  nf90_noerr) call handle_err(status)

      status =  nf90_put_att( ncSobelID, yGradID,
     1     '_FillValue', FillValueReal)
      if(status .ne.  nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'CreateGeoSobelFile #140: '

c     create or copy attribues from input to outfile for gradient 
c     magnitude

      status =  nf90_put_att( ncSobelID, GradMagID, 
     1     'long_name', 'Sobel_gradient_magnitude')
      if(status .ne.  nf90_noerr) call handle_err(status)

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = trim(ParameterUnits) // '_per_km'
      status =  nf90_put_att( ncSobelID, GradMagID, 
     1     'units', AttDescription)
      if(status .ne.  nf90_noerr) call handle_err(status)

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = 'magnitude_of_derivative_of_' // 
     1     trim(ParameterName)
      status =  nf90_put_att( ncSobelID, GradMagID, 
     1     'standard_name', AttDescription)
      if(status .ne.  nf90_noerr) call handle_err(status)

      status =  nf90_put_att( ncSobelID, GradMagID, 'add_offset', 0)
      if(status .ne.  nf90_noerr) call handle_err(status)

      status =  nf90_put_att( ncSobelID, GradMagID, 'scale_factor', 1.0)
      if(status .ne.  nf90_noerr) call handle_err(status)

      status =  nf90_put_att( ncSobelID, GradMagID,
     1     '_FillValue',  FillValueReal)

      if(status .ne.  nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'CreateGeoSobelFile #150: '

c     create the time - seconds since 1970 and define its attributes

      call DefineSecondsSinceVar( ncSobelID, SecondsSince1970OutID)

c     Now generate all of the global attributes

      NewTitle = BlankFill(:len(NewTitle))
      NewTitle = 'Eastward, Northward and Magnitude of the Sobel ' //
     1     'Gradient for'

      ProcessingProgram = BlankFill(:len(ProcessingProgram))
      ProcessingProgram = 'GeoSobel_Main'

      Summary = BlankFill(1:len(Summary))
      Summary = 'The fields in this file were generated by applying '
     1     // 'a Sobel operator to the input file in both the i, '
     2     // trim(iVarName) // ', and j, ' // trim(jVarName) 
     3     // ', directions and then correcting for the orientation '
     4     // 'of the axes and the separation of pixels to obtain '
     5     // 'the gradient in ' // trim(ParameterUnits) 
     6     // '/km eastward and northward.'

      call GenerateGlobals( ncInID, ncSobelID, NewTitle,
     1      ProcessingProgram, GeoSobelVersionNo, FileName)

      if(Debug .eq. 1) print *, 'CreateMedianFile #160: '

c     All done defining the variables and attributes for output file.

      status =  nf90_enddef(ncSobelID)
      if(status .ne.  nf90_noerr) call handle_err(status)

c     Get the time from the SST file and write it to the Median file. 

      status =  nf90_get_var( ncInID, SecondsSince1970InID, 
     1     SecondsSince1970)
      if(status .ne.  nf90_noerr) call handle_err(status)

      status =  nf90_put_var( ncSobelID, SecondsSince1970OutID, 
     1     SecondsSince1970)
      if(status .ne.  nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'CreateGeoSobelFile #999: '

      end subroutine CreateGeoSobelFile
      
c**********************************************************************
      subroutine QuickWrite( FileName, ncInID, VarOut)
c**********************************************************************
c     
c     This subroutine creates the .nc file for the Sobel gradient
c     K/km fields in lat, lon coordinates.
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
c     FileName - the name of the file to create
c     status - return status after a netCDF call
c     ncInID and ncSobelID - IDs for the input and output files.
c     sstID - ID for the sst variable will copy these attributes. 
c     SecondsSince1970InID - the ID of the number of seconds since  
c     . 00:00 1 Jan 1970 corresponding to the time of the sst array.
c     xGradID - ID for the x component of the Sobel gradient
c     yGradID - ID for the y component of the Sobel gradient
c     GradMagID - ID for the magnitude of the Sobel gradient
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

c$$$c******FillValues common
c$$$
c$$$c     FillValueReal - fill value for the gradients.
c$$$      real*4 GradFillValue
c$$$
c$$$      common/FillValues/ GradFillValue

c******General variables

      character*319 FileName

      integer ncInID, ncIDout

      integer VarID, ncVarOutID
      integer LenXinID, LenYinID

      integer nxDimIDout, nyDimIDout
      integer DimsOut(2)

      real*4 VarOut(LenX,LenY)

c---------------------------------start here --------------------------

c     Check that the dimensions of the SST array agree with those
c     - of the Sobel array.

      if(Debug .eq. 1) print *, 'QuickWrite #000: ncInID',
     1     ncInID

      status =  nf90_inq_dimid( ncInID, 'nx', LenXinID)
      if(status .ne.  nf90_noerr) call handle_err(status)
      
      if(Debug .eq. 1) print *, 'QuickWrite #90: Read LenXinID'

      status = nf90_inquire_dimension( ncInID, LenXinID, len=LenXin)
      if(status .ne. nf90_noerr) call handle_err(status)

      status =  nf90_inq_dimid( ncInID, 'ny', LenYinID)
      if(status .ne.  nf90_noerr) call handle_err(status)
      
      status = nf90_inquire_dimension( ncInID, LenYinID, len=LenYin)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, ' LenXin, LenYin: ', LenXin, LenYin

c************Now start creating the Median file ***********************

c     Create the output file.

      if(debug .eq. 1) print *, 'QuickWrite #110: Output filename::', 
     1     trim(FileName), '::'
      
      status =  nf90_create( FileName, OR(nf90_netcdf4, nf90_noclobber),
     1     ncIDout)
      if(status .ne.  nf90_noerr) call handle_err(status)

c     Create the dimensions for the output fields.

      status =  nf90_def_dim( ncIDout, 'nx', LenX, nxDimIDout)
      if(status .ne.  nf90_noerr) call handle_err(status)

      status =  nf90_def_dim( ncIDout, 'ny', LenY, nyDimIDout)
      if(status .ne.  nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'QuickWrite #120: '

c     Create output fields. 

      DimsOut(1) = nxDimIDout
      DimsOut(2) = nyDimIDout

      chunksize(1) = min(LenX, chunkszLenX)
      chunksize(2) = min(LenY, chunkszLenY)

      status = nf90_def_var( ncIDout, 'temp_var', nf90_real, DimsOut,
     1     ncVarOutID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_chunking( ncIDout, ncVarOutID, 0, chunksize)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_deflate( ncIDout, ncVarOutID, 0, 1, 4)
      if(status .ne. nf90_noerr) call handle_err(status)

c     create or copy attribues from input to outfile for y-gradient

      status =  nf90_put_att( ncIDout, ncVarOutID, 
     1     'long_name', 'Sobel_northward_gradient')
      if(status .ne.  nf90_noerr) call handle_err(status)

      status =  nf90_put_att( ncIDout, ncVarOutID, 'scale_factor', 1.0)
      if(status .ne.  nf90_noerr) call handle_err(status)

      status =  nf90_put_att( ncIDout, ncVarOutID,
     1     '_FillValue', FillValueReal)
      if(status .ne.  nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'QuickWrite #140: '


c     All done defining the variables and attributes for output file.

      status =  nf90_enddef(ncIDout)
      if(status .ne.  nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'QuickWrite #999: '

c     Now output the variable

      status =  nf90_put_var( ncIDout, ncVarOutID, VarOut)
      if(status .ne.  nf90_noerr) call handle_err(status)

c     And close the output file.

      status = nf90_close(ncIDout)
      if(status .ne. nf90_noerr) call handle_err(status)

      end subroutine QuickWrite
      
c**********************************************************************
c**********************************************************************

      include 'CommonSubroutines-2.36.f'
