c***********************************************************************
      program Thin_Main
c***********************************************************************

c     
c     This program thins the merged fronts of all images in the /sst/ 
c     directories of either /goes/ or /meteosat/ within a prescribed
c     amount of time or number of images of the current image.
c     SobelPeaksSied.f
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
c     4/4/11 PCC - Thin version number ==> 1.01
c     Changed all logical*1 to logical
c     Added TempString3 in calls to PrintArray2 and ElapsedTime.
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
c     6/1/11 - PCC - Added Msg50 in calls to PrintArray2 and ElapsedTime
c     
c     6/6/11 - PCC - Changed length of SatName from 8 to 20 characters.
c     
c     6/7/11 - PCC - Major changes to simplify and correct handling of
c     strings. Removed all EndOfStrings. Added BlankFill before any 
c     definition of a string variable to force the end of the string 
c     to be all blanks.
c     
c     Version 1.01 ==> 1.02
c     
c     1/11/12 - PCC - Changed string lengths from 150 to 319 for file
c     names. Also added an explicit definition for uuid_gen.
c     
c     Version 1.02 ==> 2.00
c     
c     4/18/12 - PCC - removed SatID; it is no longer used. No impact.
c     
c     5/21/12 - PCC - removed a lot of commented out lines. No change in
c     version number.
c     
c     Version 2.00 ==> 2.01
c     
c     9/9/12 - PCC - Changed Common... to CommonSubroutines-2.09.f
c     Change printout of "Thinning..." line to trim filename.
c     
c     Version 2.01 ==> 2.02
c     
c     9/9/12 - PCC - Changed Common... to CommonSubroutines-2.12.f
c     
c     Version 2.02 ==> 2.03
c     
c     1/28/13 -PCC - SatName moved to ArchiveDefinition common in 
c     ParameterStatements. This required removing SatName from 
c     subroutine calls and definitions for ReadInputs, CheckArchive 
c     and ReplaceBaseDir.
c     2/1/13 - PCC - CommonSubroutines 2.12 ==> 2.19.
c
c     Version 2.03 ==> 2.04
c     
c     2/2/13 -PCC - Added code to get wall clock at start, print out
c     .  fraction processed and print out timing information at the end.
c
c     Version 2.04 ==> 2.05
c     
c     2/2/13 -PCC - CommonSubroutines 2.20 ==> 2.21
c
c     Version 2.05 ==> 2.06
c     
c     3/16/13 -PCC - CommonSubroutines 2.21 ==> 2.23
c
c     Version 2.06 ==> 2.07
c     
c     4/26/13 -PCC - CommonSubroutines 2.23 ==> 2.29
c
c     Version 2.07 ==> 2.08
c     
c     1/27/14 -PCC - CommonSubroutines 2.29 ==> 2.35
c
c     2/11/14 -JPS - CommonSubroutines 2.35 ==> 2.36
c
c......................................................................

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

c******General variables

c     ArchiveID - is 1 for meteosat and 2 for goes
      integer ArchiveID

c     ConditionedSST - SST after subtracting Offset and truncating.
      integer*2, allocatable :: ConditionedSST(:,:)
c     Compressed - flag telling CleanUp that the input file was not
c     . compressed.
      logical Compressed
c     CPUTime - CPU time used by this program to date
      real CPUTime

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
c     MergedID - NetCDF ID for merged variable.
      integer MergedID
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
c     ThinnedCompressed - the compression flag for the thinned file. 
      logical ThinnedCompressed
c     ThinnedExists - file status for thinned file.
      logical ThinnedExists
c     ThinnedFileName - name of file with image of thinned contours. 
      character*319, allocatable :: ThinnedFileName(:)
c     ThinnedTempFileName - filename for temp file if decompressing
      character*319 ThinnedTempFileName

c     Function to generate UUID.
      character*36 UUID_Gen

c---------------------------------start here ---------------------------

      ProgName = BlankFill(:len(ProgName))
      ProgName = 'Thin_Main'

c     Read in/set up input variables.

      call ReadInputs( ProgName, Start, CountToRead, GeoNameIn,
     1     YearStart, YearStartC, YearEnd, MonthStart, MonthEnd,
     2     InventoryFileName)

      if(Debug .eq. 1) print *, 'Thin_Main #100:'

c     Although we could make more passes of SIED on the data, assume
c     that we will only make 2, so r_n is 1.

      r_n = '1'

c     Allocate space for dynamically allocated variables.

      allocate( ThinnedFileName(1:MaxNumberOfFiles),
     2     Dim2FileName(1:MaxNumberOfFiles),
     3     MedianFileName(1:MaxNumberOfFiles),
     4     MergedFileName(1:MaxNumberOfFiles),
     5     SSTFileName(1:MaxNumberOfFiles),
     6     ConditionedSST(1:LenX,1:LenY),
     7     MergedFronts(1:LenX, 1:LenY),
     8     MedSST(LenX,LenY), 
     9     sst(1:LenX,1:LenY),
     *     stat=ierr)
      
      if (ierr .ne. 0) stop 'Thin_Main #110: Allocation error for LenX, 
     1LenY arrays.'

c     Read ArchiveInventory - the names of files to process. First,
c     allocate the arrays to be read.

      include 'AllocateInventoryVariables.f'

      call ReadInventory(InventoryFileName, InvSecondsSince1970, 
     1     InvNumClear, InvYear, InvMonth, InvDay, InvHour, 
     2     InvMinute, InvSecond, InvFileName, iFirst, iLast)

      write(UnitLog,*) 'Thin_Main #120: Will process ArchiveInventory ',
     1     'from ', iFirst, ' to ', iLast
      print *, 'Thin_Main #120: Will process ArchiveInventory from ', 
     1     iFirst, ' to ', iLast

c------------Generate the list of input and output filenames -----------

c     Need to do this here for pmerge since it uses images before and
c     after the current image.

      iFirst = 1
      iLast = 1
      jImg = 0
      do 2000 iImg=1,NumberOfImages

         if(Debug .eq. 1) print *, 'Thin_Main #130: iImg: ', iImg

c     Do not include this image if less than Thresh clear pixels in it.

         if(InvNumClear(iImg) .lt. NumClearThresh) go to 2000
         
         jImg = jImg + 1

         if(Debug .eq. 1) print *, 'Thin_Main #140: jImg: ', jImg

c     Copy inventory values to new location. If no images are skipped
c     the list isn't changed.

         InvYear(jImg) = InvYear(iImg)
         InvMonth(jImg) = InvMonth(iImg)
         InvDay(jImg) = InvDay(iImg)
         InvHour(jImg) = InvHour(iImg)
         InvMinute(jImg) = InvMinute(iImg)
         MedianFileName(jImg) = InvFileName(iImg)
         InvSecondsSince1970(jImg) = InvSecondsSince1970(iImg)

         if(InvSecondsSince1970(jImg) .le. SecondsSinceStart) 
     1        iFirst = jImg
         if(InvSecondsSince1970(jImg) .le. SecondsSinceEnd) 
     1        iLast = jImg
         
         if(Debug .eq. 1) then
            print *, 'Thin_Main #150: SatName::', trim(SatName), '::'
            print *, 'Thin_Main #151: MedianFileName::', 
     1           trim(MedianFileName(jImg)), 
     1           '::'
         endif

         call ReplaceBaseDir(MedianFileName(jImg))

c     Need to put MedianFileName in TempFileName for string handling 
c     below.

         TempFileName = MedianFileName(jImg)

c     Check to see if the input file exists. If it doesn't stop.
         
         inquire( File=TempFileName, Exist=MedianExists)

         if(MedianExists .eqv. .false.) then
            write(unitlog, *) 'Thin_Main #160::', trim(TempFileName), 
     1           ':: does not exist. Terminating run.'
            print *, 'Thin_Main #160::', trim(TempFileName), 
     1           ':: does not exist. Terminating run.'
            stop
         endif
         
         call GetDateCharStrings( 
     1        InvYear(jImg), InvMonth(jImg),  InvDay(jImg),  
     2        InvHour(jImg), InvMinute(jImg),
     3        YearC, MonthC, DayC, HourC, MinuteC)

         Loc1 = index( TempFileName, '/Median/') 
         if(Loc1 .eq. 0) then
            write(UnitLog, *) 'Thin_Main #170: Problem with filename. ',
     1           'No /Median/'
            stop 'Thin_Main #170: Problem with filename. No /Median/.'
         endif

         Loc2 = index(TempFileName, '_Median.')
         if(Loc1 .eq. 0) then
            write(UnitLog, *) 'Thin_Main #180: Problem with filename. ',
     1           'No _Median.'
            stop 'Thin_Main #180: Problem with filename. No _Median.'
         endif

c     Construct file name for the merged images written by pmerge and 
c     thin.

         MergedFileName(jImg) = BlankFill(:len(MergedFileName(jImg)))
         MergedFileName(jImg) =  TempFileName(:Loc1) // 'Merged/' // 
     1        TempFileName(Loc1+8:Loc2-1) // '_merged_' // r_n // 
     2        '.nc'

         if(Debug .eq. 1) print *, 'Thin_Main #190: Built ',
     1        'MergedFileName(jImg)::', trim(MergedFileName(jImg)), '::'

c     Construct file name for the thinned images written by pmerge and 
c     thin.

         ThinnedFileName(jImg) = BlankFill(:len(ThinnedFileName(jImg)))
         ThinnedFileName(jImg) =  TempFileName(:Loc1) // 'Thinned/' // 
     1        TempFileName(Loc1+8:Loc2-1) // '_thinned_' // r_n // 
     2        '.nc'

         if(Debug .eq. 1) print *, 'Thin_Main #200: Built ',
     1        'ThinnedFileName(jImg): ', ThinnedFileName(jImg)

 2000 continue

c     Reset the counter for the number of images as some may have been
c     skipped because of too little data.

      NumberOfImages = jImg

c-------------------------MAIN LOOP ------------------------------------

c     Get the date and time so that we can determine the files/minute
c     processed

      call date_and_time(VALUES=DateTimeStart)

c     Loop over all images with sufficient data in the time range 

      NumberOfFilesProcessed = 0
      do 1000 iImg=iFirst,iLast

         if(Debug .eq. 1) then
            print *, 'Thin_Main #210: iImg: ', iImg
            write(UnitLog,*), iImg, InvSecondsSince1970(iImg)
            print *, iImg, InvSecondsSince1970(iImg)
         endif

c...  Check for the existence of the file with median SST field in it. 

         if(debug .eq. 1) print *, 'Thin_Main #220: Median::', 
     1        trim(MedianFileName(iImg)), '::'

         if(MedianExists .eqv. .false.) stop 
     1        'Thin_Main #221: Median file does not exist stopping.'

c...  Check for the existence of the file with merged field in it. 

         if(debug .eq. 1) print *, 'Thin_Main #230: Merged::', 
     1        trim(MergedFileName(iImg)), '::'

         inquire( File=MergedFileName(iImg), Exist=MergedExists)

         if(MergedExists .eqv. .false.) then
            write(UnitLog,*) 'Thin_Main #240: ERROR ***** Merged ',
     1           'file does not exist for::', 
     2           trim(MergedFileName(iImg)), '::'
            print *, 'Thin_Main #240: ERROR ***** no merged file for::',
     1           trim(MergedFileName(iImg)), '::'
            go to 1000
         endif

c...  Check for the existence of the file with thinned field in it. 

         if(debug .eq. 1) print *, 'Thin_Main #250: Thinned::', 
     1        trim(ThinnedFileName(iImg)), '::'

         inquire( File=ThinnedFileName(iImg), Exist=ThinnedExists)

         if(ThinnedExists .eqv. .false.) then

c     Approximately what percentage of the run is complete?

            PercentDone = 100.0 * float(iIMG - iFirst + 1) / 
     1           float(iLast - iFirst + 1)

            write(unitlog,*) PercentDone, iFirst, iLast, iIMG, 
     1           'Thin_Main #210 Processing::', 
     2           trim(ThinnedFileName(iImg)), '::'
            print *, PercentDone, iFirst, iLast, iIMG, 
     1           ' Thin_Main #210 Processing::', 
     2           trim(ThinnedFileName(iImg)), '::'

c-----------------OK, no thinned fields so THIN this one --------------
            
            if (debug .eq. 1) then
               Msg50 = '--- Thin_Main #270: Before MedSST --- EOS'
               call PrintArray2( MedSST(istrt:iend,jstrt:jend), Msg50)
            endif

c     Read in the Median field.

            call ReadMedianFile( MedianFileName(iImg), ncMedID, 
     1           MedID, SecondsSince1970ID, MedSST, 
     2           SecondsSince1970T)

            status = nf90_close(ncMedID)
            if(status .ne. nf90_noerr) call handle_err(status)

            if(debug .eq. 1) print *, 'Thin_Main #300: '

c     Read in the merged field.

            call ReadMergedThinned( MergedFileName(iImg), 
     1           ncMergedID, MergedID, SecondsSince1970ID, 
     2           MergedFronts, SecondsSince1970T, 1)

c     OK, you can thin now.

            if (debug .eq. 1) then
               Msg50 = '--- Thin_Main #320: Before call to thin --- EOS'
               call PrintArray2( MedSST(istrt:iend,jstrt:jend), Msg50)
            endif

            call thin( ThinnedFileName(iImg), InvSecondsSince1970(iImg),
     1           MedSST, MergedFronts, ncMergedID, MergedID, 
     2           SecondsSince1970ID)

            NumberOfFilesProcessed = NumberOfFilesProcessed + 1

            status = nf90_close(ncMergedID)
            if(status .ne. nf90_noerr) call handle_err(status)

         endif
c------------------------ALL DONE PROCESSING ---------------------------

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

      end program Thin_Main

c**********************************************************************
      subroutine thin( MTFileName, hdate, MedSST, MergedFronts,
     1     ncMergedID, MergedID, SecondsSince1970ID)
c**********************************************************************
c     
c     MTFILENAME    - (input/output file name)

c     MEDSST(1: SizOfImg ,1: SizOfImg ) - med-fltrd image (0,255)
c     MergedFronts(1: SizOfImg ,1: SizOfImg )   - original edge img
c     Thinnedfronts(1: SizOfImg ,1: SizOfImg )   - thinned edge image
c     
c     I,J                 - generic indices
c     K                   - current image - 
c     
c     DIFTEMP             - difference of temperature
c     MAXDIF              - maximum DIFTEMP
c     IMAX,JMAX           - MAXDIF coordinates
c     COUNT               - distance from last edge
c     win                 - median filter window
c     dumhdate	      - dummy value for date passed from Read_Image
c     dumlats(1:SizOfImg),dumlons(SizOfImg)	- dummy values for 
c     lats/lons from Read_Image
c     
c     DataRange - maximum minus minimum value read in.
c     NAboveRange - number of input data values above Range. 
c     UnitOut - the unit number to which ProcInfo will be written
c
c     21-May-2009 - PCC - The upper limits of the 23032 and 23034 loops
c     were backward; LenY for the first dimension and LenX for the 
c     second dimension. I reversed them.

      implicit none

c******Functions

c     HoursSince - returns the number of hours since the reference.
c     . Arguments are year, month, day, hour, minute and the 
c     . values for these.
      real*8 HoursSince

c******Parameter statements

      include 'ParameterStatements.f'

c******General variables
      
      character*319 MTFileName

c     Variables used to test for thinned fronts in input file.

      integer ncMTID, status2
      integer frID, frDims(2)

      integer*2 MedSST(1:LenX, 1:LenY)
c$$$      integer*2 MedSST1(1:LenX, 1:LenY)
      integer*2 MergedFronts(1:LenX, 1:LenY)
      integer*2, allocatable ::  ThinnedFronts(:,:)
      integer*2 i,j
      integer*2 k
      integer*2 diftemp
      integer*2 maxdif
      integer*2 imax,jmax
      integer*2 count
      integer*2 win
      character filtyp
      real*8 hdate
      real*4, allocatable :: dumlats(:), dumlons(:)
      integer*4 DataRange, NAboveRange
c     MergedOrThinned - 1 to read MergedFronts and 2 to read 
c     . ThinnedFronts
      integer*2 MergedOrThinned

      integer ncMergedID, MergedID, SecondsSince1970ID

      include 'netcdf.inc'

      allocate( ThinnedFronts(1:LenX,1:LenY), 
     1     dumlats(1:LenY), 
     2     dumlons(1:LenX),
     3     stat=ierr)
      
      if (ierr .ne. 0) stop 'thin #100: Allocation error for LenX, LenY 
     1arrays.'

      win=5

      if(debug .eq. 1) write(6,*) MTFileName
      if(debug .eq. 1) write(UnitLog,*) MTFileName

c$$$c     Median filter the SST image.
c$$$  
c$$$      call median(MedSST1,MedSST,win)
  
      if (debug .eq. 1) then
         Msg50 = '--- thin #330: SST after median in thin --- EOS'
         call PrintArray2( MedSST(istrt:iend,jstrt:jend), Msg50)
      endif

c     Set output thinned front data to 0.

      do 23014 i=1, LenX 
         do 23016 j=1, LenY
            ThinnedFronts(i,j)=0
23016    continue
23014 continue

      diftemp = 32767

      do 23018 i=1,LenX 
         maxdif=2
         do 23020 j=2,LenY-1
            if(MergedFronts(i,j) .eq. 4)then
               count=0
               if(min(MedSST(i,j-1),MedSST(i,j+1)) .gt. 8)then
                  diftemp=abs(MedSST(i,j+1)-MedSST(i,j-1))
               endif
               if(maxdif .lt. diftemp)then
                  maxdif=diftemp
                  imax=i
                  jmax=j
               endif
            else
               if(maxdif .ne. 2)then
                  if(count .ge. 2)then
                     ThinnedFronts(imax,jmax)=4
                     maxdif=2
                  endif
                  count=count+1
               endif
            endif
23020    continue
23018 continue

      do 23032 j=1,LenY
         maxdif=2
         do 23034 i=2,LenX-1
            if(MergedFronts(i,j) .eq. 4)then
               count=0
               if(min(MedSST(i-1,j),MedSST(i+1,j)) .gt. 8)then
                  diftemp=abs(MedSST(i+1,j)-MedSST(i-1,j))
               endif
               if(maxdif .lt. diftemp)then
                  maxdif=diftemp
                  imax=i
                  jmax=j
               endif
            else
               if(maxdif .ne. 2)then
                  if(count .ge. 2)then
                     ThinnedFronts(imax,jmax)=4
                     maxdif=2
                  endif
                  count=count+1
               endif
            endif
23034    continue
23032 continue

      MergedOrThinned = 2
      call WriteMergedThinned( MTFileName, hdate, dumlats, 
     1     dumlons, ThinnedFronts, MergedOrThinned, ncMergedID, 
     2     MergedID, SecondsSince1970ID)

      return
      end

c***********************************************************************
c***********************************************************************

      include 'CommonSubroutines-2.36.f'
