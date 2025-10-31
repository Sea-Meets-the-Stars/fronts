c***********************************************************************
      program AppQualMed_Main
c***********************************************************************
c     
c     This program reads the list of files in each of the input monthly 
c     directory, checks to see if the corresponding output file exists,
c     processes it if it does not and skips it if it does. After 
c     processing all files the program writes merges the temporary
c     archiveinventory file with any other existing ones and writes out
c     ArchiveInventory.
c     
c     BE VERY CARFUL with the order of dimensions in the input arrays.
c     If NetCDF they are sst(i,j), when read by Fortran they become
c     sst(j,i); i.e., the dimensions are reversed. 
c
c     Version 3.00
c     
c     8/14/13 - PCC - This program was derived from 
c       AppQualMed_AMSRE_Main-1.60.f. There are many changes from 1.60 
c       to generalize. It is the first version of the generalized 
c       version of AppQual. It is given a version 3 because the datasets
c       are referred to by the version of AppQual and for some of the
c       archives there is already a version 2.something.
c
c     Version 3.01
c
c     1/7/14 - PCC - Changed the name of the AMSR-E archive from 
c       amsre to l2_amsre_v7 to allow for other amsre archives.
c      Changed the names of amsre subroutines to correspond to v7_l2. 
c     1/13/14 - PCC - CommonSubroutines 2.32 ==> 2.33
c     
c     3.01 ==> 3.02
c
c     1/26/14 - PCC - CommonSubroutines 2.33 ==> 2.34
c      Modified title of geolocation file to use SatName instead of 
c       being hardwired to amsre or trim.
c      The way the program creates output file names was changed to 
c       accommodate the day/night RSS fields in one input file. 
c     
c     3.02 ==> 3.03
c
c     1/27/14 - PCC - Redefined the chunking parameter in both x and y
c       as the minimum of LexX(Y) and ChunkSzLenX(Y). This only affected
c       WriteMergedThinned.
c      CommonSubroutines 2.34 ==> 2.35
c
c     3.03 ==> 3.04
c
c     12/16/14 - JPS - Added AVHRR L2P gac options for AppQualMed
c       MinSST and MaxSST were flipped. Fixed.
c
c     1/13/15 - JPS - Added lines for avhrr_hrpt runs and ecco2_4km 
c      runs in archive specific sections
c
c     3.04 ==> 3.05
c
c     5/27/16 - PCC - Added CMS package
c      Changed how the program handles filenames for geolocation files.
c       This change should have no impact on the actual names.
c
c     Version 3.05 ==> 3.06
c
c     6/2/16 - PCC - updated CMS subroutines to version 1.01 in which I
c      fixed read for longitude. Should not impact processing of SST.
c
c     Version 3.06 ==> 3.07
c
c     6/15/16 - PCC -Not sure why, but, when building the output 
c      filename the prgram searched for .gz if the file was  
c      compressed. This resulted in the filename having an extra .nc
c      in them. Fixed now. I hope that it does not cause other problems.
c
c**********Functions     
c     
c     SecondsSince - returns the number of seconds since the reference.
c     ...Arguments are year, month, day, hour, minute and the 
c     ...values for these.
c     
c**********General Variables.
c     
c     BaseDir - the directory in which the subdirectories for the input
c     ...images and the subdirectories for the output images exist.
c     DirectoryIn - the directory with the input files.
c     DirectoryOut - the directory into which the output files will go.
c     
c     ImageName - the image name no directory. This is read from a 
c     directory listing.
c     FileNameIn - the name of the input file - in /Original/
c     FileNameOut - the name of the output file - in /SST
c     FileNameOut_gz - the name of the output file if compressed 
c     TempFile - filename used for the decompressed temporary file.
c     
c     Numbrs - ascii representation of the ten digits
c     
c     Year, Month, Day, Hour - year, month, day and hour working on
c     YearT, MonthT, DayT, HourT - save as above except read from the
c     ...inventory. Used to test to see if the inventory is correct.
c     YearDay - day in the given year.
c     Days - day in month for the month and year working on
c     YearStart - first year in archive - 2001 for goes, 2003 for 
c     ...meteosat
c     
c     iSat - a do loop index
c     
c     exis -  .true. if file exists when inquiring
c     exis_gz -  .true. if gzipped version of file exists 
c     Compressed - .true. if the _mt_ file is compressed
c     eos - end of string
c     SpaceC - a blank space.
c     
c*************Inventory Variables
c     
c     NumClear - Number of clear pixels in the image.
c     NumClearT - same as above except read from the inventory. Used to
c     ...test to see if the inventory is correct.
c     NumberOfFilesTotal - the number of fields in the inventory
c     
c*************Variables to build file name.
c     
c     YearC, MonthC, DayC, HourC, MinuteC - more dummy variables
c     r_n - run number. Usually 1.
c     c for the last 4.
c     SatName - either 'goes' or 'meteosat'
c     
c$$$  c     ArchiveID - is 1 for meteosat and 2 for goes
c     
c*************netCDF Variables - the first four are in a parameter 
c     ...statement
c     
c     LenXin, LenYin - the x, y dimensions of the input image
c     LenX, LenY - the x, y dimensions of the output image
c     ...the input and output images may be of different size since
c     ...sied requires that the image dimensions be a multiple of 32.
c     xStart, xEnd, yStart, yEnd - the beginning and end of portions
c     ...of the sst array to write out - for meteosat this is the 
c     ...entire, but for goes the left and bottom are truncated. 
c     
c     ncInID, ncOutID - netCDF IDs for the input and output files
c     nxDimIDin, nyDimIDin, nxDimIDout and nyDimIDout  - dimension IDs
c     sstDimsOut - vectors for dimension IDs
c     sstIDin, sstQualIDin, sstIDout - variable IDs
c     geosIDin, geosIDout - IDs for geos
c     HoursSinceID - ID for hours since 
c     
c     Dummy - a dummy variable for the definition of a scalar.
c     
c     HoursSince1900 - the number of hours since midnight on 31 
c     ...December 1899 or the first second of 1 January 1900
c     HoursSince1900T - the number of hours since 1900 to this image 
c     ...read from the inventory
c     geos - the metadata defining the projection.
c     sstIn, sstQualIn, sstOut - the sst input and output fields.

      use netcdf

      implicit none

c     Functions

      real secnds
      real*8 SecondsSince
      character*20 Num2StrInt, Num2StrFloat, Num2StrG

c******Parameter statements

      include 'ParameterStatements.f'

c     General variables

      character*319 DirectoryIn, DirectoryOut
      character*319 ImageName, FileNameIn, FileNameOut, FileNameOut_gz
      character*319 TempFileName
      character*319 ListOfFiles

      character*319 ListOfDailyFiles

      character*1 Numbrs(0:9)

      character*(10) DateC

c     ClearIDout - the netCDF ID for the clear pixel field.
      integer ClearIDout

c     DayOrNightIDout - the netCDF ID for the day or night pixel field.
      integer DayOrNightIDout

c     GeoExist - true if the output file exists, false otherwise.
      logical GeoExist

c     FrontsIDout - the netCDF ID for the front pixel field.
      integer FrontsIDout

c     iDayNight - 1 for day time fields, 2 for nighttime fields. This 
c      variable is only used for RSS data.
      integer iDayNight

c     iMonth - Do loop parameter for days of month.
      integer iMonth
c     ix - Do loo parameter for x-dimension.
      integer ix

c     jy - Do loo parameter for y-dimension.
      integer jy

c     LatIn - input array for latitude
      real*4, allocatable :: LatIn(:,:)
c     LonIn - input array for latitude
      real*4, allocatable :: LonIn(:,:)

c     MonthEndTemp - a temporary variable for the month at the end of 
c     the range to process. 
      integer MonthEndTemp
c     MonthStartTemp - a temporary variable for the month at the start  
c     of the range to process. 
      integer MonthStartTemp

      integer NumberOfFilesM

c     RefTime - reference time, seconds since 1981 -- AMSR only
      integer RefTime

c     SecondsSinceID - netCDF ID for SecondsSince
      integer SecondsSinceID
c     SS1981 - Seconds for 00:00 00 on 1/1/1981, used to get the hours,
c     minutes and seconds in the current day for the reference time.
      real*8 SS1981
c     sstIDout - netCDF ID for SST
      integer sstIDout
c     sstOut - SST written out.
      integer*2, allocatable :: sstOut(:,:) 

c     TimeID - id for reference time, time since 1981 -- AMSR only
      integer TimeID

c     YearDayStart - the year day for the first day of each month.
      integer YearDayStart(13)

      integer Year, Month, Day, Hour, Minute, Second
      integer Days, YearDay

      integer Dum0

      integer YearT, MonthT, DayT, HourT, MinuteT, SecondT
      integer YearDayT

      real*8 SecondsSince1970
      real*8, allocatable :: Times(:)

      integer iSat

      character*8 TimeC
      real t

      logical exis, exis_gz
      logical Compressed

c     Inventory variables

      integer NumClear, NumClearT
      integer NumberOfFilesTotal

c     netCDF variables

      integer ncInID, ncOutID
      integer LatIDout, LonIDout

      integer Dummy

c     character geos

c     Define starting and ending indices for the input array. This
c     program uses netCDF subsetting commands to pull out the part
c     of the array to use. For GOES, it starts at (32,29) (y,x) and
c     goes to the end of the rows and columns. For meteosat it does
c     the whole array, starts at (1,1). Both use a stride of (1,1)
      
c     Data statements

      data Numbrs/ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'/
      data stride/ 1, 1/

c---------------------------------start here ---------------------------

      ProgName = BlankFill(:len(ProgName))
      ProgName = 'AppQualMed_Main-' // 
     1     AppQualMed_VersionNo

      if(Debug .eq. 1) print *, 'AppQual #000:'

c     Read in/set up input variables.

      call ReadInputs( ProgName, Start, CountToRead, GeoNameIn, 
     1     YearStart, YearStartC, YearEnd, MonthStart, MonthEnd,
     2     InventoryFileName)

      if(Debug .eq. 1) print *, 'AppQual #100: YearStart:', YearStart

c     Allocate space for dynamically allocated variables.

      allocate( LatIn(LenX,LenY),
     1     LonIn(LenX,LenY),
     2     sstOut(LenX,LenY), 
     3     Times(MaxNumberOfFiles),
     *     stat=ierr)
      
      if (ierr .ne. 0) then
         write(UnitLog,*) 'AppQual #110: Allocation error, exiting.'
         stop 'AppQual #110: Allocation error, exiting.'
      endif
      
c     Check to make sure that there are enough pixels in the line or
c     lines in the file to read.

      if(  (Start(1)+CountToRead(1)-1 .gt. LenXin) .or.
     1     (Start(2)+CountToRead(2)-1 .gt. LenYin) ) then
         write(UnitLog,*) 'AppQual #120: Stopping - starting pixel or ',
     1        'line and number of pixels or lines to read exceeds ', 
     2        'either LenXin or LenYin.'
         stop 'AppQual #120: rows or columns to read exceeds LenXin/Yin'
      endif

c     Open various output files - use InputFileName temporarily.

      InputFileName = BlankFill(:len(InputFileName))
      InputFileName = 'Temp.Data'
      call OpenOutputFiles(InputFileName)

c-----------------------------------------------------------------------
c     Main loop over all possible images
c-----------------------------------------------------------------------

c     Get the date and time so that we can determine the files/minute
c     processed

      call date_and_time(VALUES=DateTimeStart)

      NumberOfFilesTotal = 0

      if(Debug .eq. 1) print *, 'AppQual #130: YearStart:', YearStart,
     1     'YearEnd: ', YearEnd

      do 1010 year=YearStart,YearEnd
c***********************************************************************
c     Get the ending day for each month. Will use this later to see if
c     the file being considered is in the selected time range. 
c     YearDayStart for month 13 is one count more than the number of 
c     days in the current year.

         YearDayStart(1) = 1
         do iMonth=1,12
            call calend( Year, iMonth, days)
            YearDayStart(iMonth+1) = YearDayStart(iMonth) + days
         enddo

c     Now get start and end months for this year. These vary depending 
c     on whether it is the first year in the period, a year in the 
c     middle or the last year.
c***********************************************************************
         if(year .eq. YearStart) then
            MonthStartTemp = MonthStart
         else
            MonthStartTemp = 1
         endif

         if(year .eq. YearEnd) then
            MonthEndTemp = MonthEnd
         else
            MonthEndTemp = 12
         endif

         do 1011 month=MonthStartTemp,MonthEndTemp
            
            NumberOfFilesM = 0

c     Get string representation for year and month.

            DummyCharacter = Num2StrInt( year)
            YearC = DummyCharacter(1:4)

            DummyCharacter = Num2StrInt( month)
            if(month .lt. 10) then
               MonthC = '0' // DummyCharacter(1:1)
            else
               MonthC = DummyCharacter(1:2)
            endif

            if(Debug .eq. 1) print *, 'AppQual #140: YearC:', YearC
            
c     Generate list of input files for this year/month

            DirectoryIn = BlankFill(:len(DirectoryIn))
            DirectoryIn = trim(BaseDir) // 'Original/' // YearC // 
     1           '/' // MonthC // '/'

            call GenerateFileList( DirectoryIn, ListOfDailyFiles)

            if(Debug .eq. 1) print *, 'AppQual #150: File list made.'

c     Open the temporary file containing the list of input files just
c     written.

            open (unit=UnitTempIn, file=ListOfDailyFiles, status='old',
     1           access='SEQUENTIAL')

            if(Debug .eq. 1) print *, 'AppQual #160: Files opened.'

c     Loop over file names in the temporary list of filenames.

            do 1020 iFiles=1,MaxNumberOfFiles

               if(Debug .eq. 1) print *, 'AppQual #170: iFiles:', 
     1              iFiles

               read(UnitTempIn,*, end=1100) ImageName

               if(debug .eq. 1) print *, 'AppQual #180: ImageName::', 
     1              trim(ImageName), '::'

c     Add the input directory to the ImageName to generate the input 
c     filename.

               FileNameIn = BlankFill(:len(FileNameIn))
               FileNameIn = trim(DirectoryIn) // trim(ImageName)

               if(debug .eq. 1) then
                  print *, 'App_Qual #190: ', 
     1                 ' DirectoryIn::', trim(DirectoryIn), '::' 
                  print *, 'App_Qual #191: ', 
     2                 ' ImageName::', trim(ImageName), '::'
                  print *, 'App_Qual #192'': ', 
     2                 ' FileNameIn::', trim(FileNameIn), '::'
               endif

c     Check to see if the input file exists and whether or not it is 
c     compressed. Do not decompress at this time.

               call FileStatus( FileNameIn, exis, Compressed, 
     1              YearC, TempFileName, .false., .false.)

               if(exis .eqv. .false.) then
                  write( UnitLog,*) 'AppQual #200: Input file does ',
     1                 'not exist. Stop.'
                  print *, 'AppQual #200: Input file does not exist. ',
     1                 'FileNameIn::', trim(FileNameIn), '::'
                  stop
               endif

c     Increment files processed counters.

               NumberOfFilesM = NumberOfFilesM + 1

c     Set iDayNight to 0. This will be used as a flag to determine if
c     day or night needs to be added to the name of the output field. 
c     For l3 AMSR-E fields this will be determined from BaseDir in 
c     AMSRE_v7_l3_GetDateTime. For other archives it might determined 
c     differently. If not changed; i.e., =0, means don't worry about it.

               iDayNight = 0

c%%%%%%%%%%%%%%%%%%%%%%% Start Archive specific %%%%%%%%%%%%%%%%%%%%%%%%

c     Get the year, month, day,... for this pass from the filename. This
c     information is used to determine if this pass is in the requested
c     interval. Since the structure of the filenames changes from 
c     satellite to satellite, we need one of these for each satellite.

               select case (SatName)

c********************* AMSR-E Level 2 Version 7 ************************

               case ('l2_amsre_v7')
                  call AMSRE_v7_l2_GetDateTime( ImageName, Year, Month,
     1                 YearT, MonthT, Day, Hour, Minute, Second)

c********************* AVHRR level 2 gac Version 1 ************************

               case ('avhrr_gac')
                  call AVHRR_gac_l2p_v1_GetDateTime( ImageName, 
     1                 Year, Month,YearT, MonthT, Day, Hour, 
     2                 Minute, Second)


c********************* AVHRR level 2 hrpt ************************

               case ('avhrr_hrpt')
                  call AVHRR_hrpt_l2_GetDateTime( ImageName, 
     1                 Year, Month,YearT, MonthT, Day, Hour, 
     2                 Minute, Second)



c********************* ECCO2_4km ************************

               case ('ecco2_4km')
                  call ECCO2_4km_GetDateTime( ImageName, 
     1                 Year, Month,YearT, MonthT, Day, Hour, 
     2                 Minute, Second)


c********************* AMSR-E Level 3 Version 7 ************************

               case ('l3_amsre_v7')

c     Also determine if this is a day or night image for special RSS
c     read for.

                  call AMSRE_v7_l3_GetDateTime( ImageName, Year, Month,
     1                 YearT, MonthT, Day, Hour, Minute, Second,
     2                 MonthC, YearC, iDayNight)

c********************* Data from CMS on sabbatical *********************

               case ('msg','goes')
                  call CMS_GetDateTime( ImageName, Year, Month,
     1                 YearT, MonthT, Day, Hour, Minute, Second,
     2                 MonthC, YearC, iDayNight)

c******************************* default *******************************

               case default
                  print *, 'AppQual #210. ERROR No subroutine to',
     1                 ' get year, month, day... from filename. ',
     2                 'ABORTING.'
                  stop

               end select

c%%%%%%%%%%%%%%%%%%%%%%%% End Archive specific %%%%%%%%%%%%%%%%%%%%%%%%

               if(Debug .eq. 1) print *, 'AppQual #220: Year, Month, ',
     1              'Day, Hour, Minute: ', Year, Month, Day, Hour, 
     2              Minute, 'YearT, MonthT: ', YearT, MonthT, 
     3              ' iDayNight: ', iDayNight

c     Get seconds since the start of 1970.

               SecondsSince1970 = SecondsSince( 
     1              YearT, MonthT, Day, Hour, Minute, Second,
     1              1970, 1,     1,   0,    0,      0)    

               if(Debug .eq. 1) print *, 'AppQual #225: SecondsSince: ',
     1              SecondsSince1970

c     Is this time in the specified time range:

               if(Debug .eq. 1) print *, 'AppQual #230 ',
     1              'SecondsSince1970, Start Seconds, End Seconds:', 
     1              SecondsSince1970, SecondsSinceStart, SecondsSinceEnd

               if( (SecondsSince1970 .lt. SecondsSinceStart) .or. 
     1              (SecondsSince1970 .gt. SecondsSinceEnd) ) go to 1020

c     Now build the output (SST) filename and check for its existence. 
c      If iDayNight = 0; standard name.
c      If iDayNight = 1; add 'day_' before 'Median' in filename.
c      If iDayNight = 2; add 'night_' before 'Median' in filename.
c     iDayNight may be changed in the archive specific lines above, but
c      in general it will not be changed.

c               if(Compressed .eqv. .false.) 
c     1              Loc1 = index( ImageName, '.nc') 
c               if(Compressed .eqv. .true.) 
c     1              Loc1 = index( ImageName, '.gz') 
               Loc1 = index( ImageName, '.nc') 

               FileNameOut = BlankFill(:len(FileNameOut))
               if(iDayNight .eq. 0) then
                  FileNameOut = trim(BaseDir) // 'Median/' // YearC // 
     1              '/' // Monthc // '/' // ImageName(1:Loc1-1) //
     2              '_Median.nc'
               elseif(iDayNight .eq. 1) then
                     FileNameOut = trim(BaseDir) // 'Median/' // YearC 
     1                    // '/' // Monthc // '/' // ImageName(1:Loc1-1)
     2                    // '_day_Median.nc'
               elseif(iDayNight .eq. 2) then
                     FileNameOut = trim(BaseDir) // 'Median/' // YearC 
     1                    // '/' // Monthc // '/' // ImageName(1:Loc1-1)
     2                    // '_night_Median.nc'
               endif

               if(debug .eq. 1) then
                  print *, 'App_Qual #240: ',
     1                 ' Loc1: ', Loc1, ' ImageName::', 
     2                 trim(ImageName), '::'
                  print *, 'App_Qual #241:  BaseDir::', trim(BaseDir), 
     1                 '::' 
                  print *, 'App_Qual #242:  FileNameOut::', 
     1                 trim(FileNameOut), '::'
               endif

               inquire( File = trim(FileNameOut), Exist=exis)

c     If exists, write message and read the next filename on the list.

               if(exis .eqv. .true.) then
                  write(UnitLog,*) 'AppQual #250 Skippping::', 
     1                 trim(FileNameOut), '::' 
                  print *, 'AppQual #250 Skipping ::', 
     1                 trim(FileNameOut), '::'
                  go to 1020
               endif

c-----------------------------------------------------------------------
c     Output SST file doesn't exist so we will create and populate it.

               NumberOfFilesTotal = NumberOfFilesTotal + 1

               write(UnitLog,*) 'AppQual #260: FileNameout::', 
     1              trim(FileNameOut), ':: and SatName ::',
     2              trim(SatName), '::'
               print *, 'AppQual #260: FileNameOut)::', 
     1              trim(FileNameOut), ':: and SatName ::',
     2              trim(SatName), '::'

c%%%%%%%%%%%%%%%%%%%%%%% Start Archive specific %%%%%%%%%%%%%%%%%%%%%%%%

c     Build the output geolocation filename. GeoName is written out to
c     the SST file. GeoNameFull is used to create the geospatial file.
c     This is done here so that the geolocation file can be created and
c     filled when the SST data is written - saves a select case 
c     statement later. This one is simpler than most. There are
c     basically two cases here, geoloc files by inut file or one for the
c     archive, hence the archive specific nature.

               select case (SatName(1:2))

c     For geolocation files by input file.
                  
               case ('l2')
                  GeoName = BlankFill(:len(GeoName))
                  GeoName = 'GeoLoc/' // YearC // '/' // Monthc //  
     1                 '/' // ImageName(1:Loc1-1) // '_geolocation.nc'

               case ('av')
                  GeoName = BlankFill(:len(GeoName))
                  GeoName = 'GeoLoc/' // YearC // '/' // Monthc //  
     1                 '/' // ImageName

c     For geolocation files by archive.
                  
               case ('l3','ms','go','ec')
                  GeoName = BlankFill(:len(GeoName))
                  GeoName = 'GeoLoc/' // trim(LocFileNamePrefix) // 
     1                 '_geolocation.nc'
                                    
               case default
                  print *, 'AppQual #260. ERROR No case for::',
     1                 trim(SatName), ':: ABORTING.'
                  stop

               end select

c%%%%%%%%%%%%%%%%%%%%%%%% End Archive specific %%%%%%%%%%%%%%%%%%%%%%%%

               GeoNameFull = BlankFill(:len(GeoNameFull))
               GeoNameFull = trim(BaseDir) // trim(GeoName)

               if(Debug .eq. 1) print *, 'AppQual #270: GeoName::', 
     1              trim(GeoName), '::'
               
c     Next decompress the input file if it is compressed.

               if(debug .eq. 1) print *, 'AppQual #275: ',
     1              'Compressed: ', Compressed, ' trim(FileNameIn)::', 
     2              trim(FileNameIn), '::'

c *********** might be able to replace most of the following with a call
c     to FileStatus.

               if(Compressed .eqv. .true.) then

c     First, generate a temporary file name for the decompressed file.
c     Use the original image name to make the temporary name unique.

                  TempFileName = BlankFill(:len(TempFileName))
                  TempFileName = trim(BaseDir) // 'TmpDir/' // 
     1                 'tmpfile_' // trim(ImageName) // '_tmpfile' 

c     Now generate the decompress command and issue it.

                  Command = BlankFill(:len(Command))
                  Command = 'gzip -dc ' // trim(FileNameIn) // 
     1                 ' > ' // trim(TempFileName)

                  if(Debug .eq. 1) print *, 'AppQual #280: ',
     1                 'trim(Command)::', trim(Command), '::'
                  
                  call system(trim(Command))

               else
                  TempFileName = BlankFill(:len(TempFileName))
                  TempFileName = FileNameIn
               endif

c%%%%%%%%%%%%%%%%%%%%%%% Start Archive specific %%%%%%%%%%%%%%%%%%%%%%%%

c     Get the output SST data and create the output SST file. NOTE THAT
c     SATNAME MUST BE LOWER CASE.

               select case (SatName)

c********************* AMSR-E Level 2 Version 7 ************************

               case ('l2_amsre_v7')

                  call AMSRE_v7_l2_GetSSTOutData(TempFileName, sstOut, 
     1                 SecondsSince1970)

                  call AMSRE_v7_l2_CreateOutputSSTFile( FileNameOut, 
     1                 ncOutID, ProgName, sstIDout, ClearIDout, 
     2                 FrontsIDout, DayOrNightIDout, SecondsSinceID, 
     3                 Start, YearC, MonthC, DayC)

                  inquire( File=trim(GeoNameFull), Exist=GeoExist)

                  if(GeoExist .eqv. .false.) then
                     call AMSRE_v7_l2_CreateOutputSSTFile( FileNameOut, 
     1                 ncOutID, ProgName, sstIDout, ClearIDout, 
     2                 FrontsIDout, DayOrNightIDout, SecondsSinceID, 
     3                 Start, YearC, MonthC, DayC)
                  endif

c********************* AVHRR Level 2 GAC Version 1 *********************

               case ('avhrr_gac')

                  call AVHRR_gac_l2p_v1_GetSSTOutData(TempFileName, 
     1                 sstOut, SecondsSince1970)

                  call AVHRR_gac_l2p_v1_CreateOutputSSTFile( 
     1                 FileNameOut,ncOutID, ProgName, sstIDout, 
     2                 ClearIDout,FrontsIDout, DayOrNightIDout, 
     3                 SecondsSinceID, Start, YearC, MonthC, DayC)

                  inquire( File=trim(GeoNameFull), Exist=GeoExist)

                  if(GeoExist .eqv. .false.) then
                     call AVHRR_gac_l2p_v1_GetGeoData(FileNameIn, LatIn,
     1                    LonIn, SecondsSince1970, FileNameOut, 
     2                    NumClear,YearC, MonthC, Day,Hour, Minute, 
     3                    Second, GeoNameFull)
                  endif

c********************* AVHRR Level 2 hrpt *********************

               case ('avhrr_hrpt')

                  call AVHRR_hrpt_l2_GetSSTOutData(TempFileName, 
     1                 sstOut, SecondsSince1970)

                  call AVHRR_hrpt_l2_CreateOutputSSTFile( 
     1                 FileNameOut,ncOutID, ProgName, sstIDout, 
     2                 ClearIDout,FrontsIDout, DayOrNightIDout, 
     3                 SecondsSinceID, Start, YearC, MonthC, DayC)

                  inquire( File=trim(GeoNameFull), Exist=GeoExist)

                  if(GeoExist .eqv. .false.) then
                     call AVHRR_hrpt_l2_GetGeoData(FileNameIn, LatIn,
     1                    LonIn, SecondsSince1970, FileNameOut, 
     2                    NumClear,YearC, MonthC, Day,Hour, Minute, 
     3                    Second, GeoNameFull)
                  endif

c********************* ECCO2_4km ************************

               case ('ecco2_4km')

                  call ECCO2_4km_GetSSTOutData(TempFileName, 
     1                 sstOut, SecondsSince1970)

                  call ECCO2_4km_CreateOutputSSTFile( FileNameOut, 
     1                 ncOutID, ProgName, sstIDout, ClearIDout, 
     2                 FrontsIDout, DayOrNightIDout, SecondsSinceID, 
     3                 Start, YearC, MonthC, DayC)

                  if(debug .eq. 1) print *, 'AppQual #288: ',
     1              'Created output SST file.'

                  inquire( File=trim(GeoNameFull), Exist=GeoExist)

                  if(GeoExist .eqv. .false.) then
                     call ECCO2_4km_GetGeoData(FileNameIn, LatIn,
     1                    LonIn, SecondsSince1970, FileNameOut, 
     2                    NumClear,YearC, MonthC, Day,Hour, Minute, 
     3                    Second, GeoNameFull)
                  endif

                  if(debug .eq. 1) print *, 'AppQual #289: ',
     1              'Got geolocation data.'

c********************* AMSR-E Level 3 Version 7 ************************

               case ('l3_amsre_v7')
                  
                  call AMSRE_v7_l3_GetSSTData( iDayNight, TempFileName, 
     1                 sstOut)

                  call AMSRE_v7_l3_CreateOutputSSTFile( FileNameOut, 
     1                 ncOutID, ProgName, sstIDout, ClearIDout, 
     2                 FrontsIDout, DayOrNightIDout, SecondsSinceID, 
     3                 Start, YearC, MonthC, DayC)

                  if(debug .eq. 1) print *, 'AppQual #290: ',
     1              'Created output SST file.'

                  inquire( File=trim(GeoNameFull), Exist=GeoExist)

                  if(GeoExist .eqv. .false.) then
                     call AMSRE_v7_l3_GetGeoData( TempFileName, LatIn, 
     1                    LonIn)
                  endif

                  if(debug .eq. 1) print *, 'AppQual #291: ',
     1              'Got geolocation data.'

c******************************** CMS **********************************

               case ('msg','goes')

                  if(debug .eq. 1) print *, 'AppQual #292: ',
     1                 'Start, CountToRead: ', Start, CountToRead,
     2                 ' LenX, LenY: ', LenX, LenY 

                  call CMS_GetSSTData( TempFileName, Start, CountToRead,
     1                 Stride, sstOut)

                  call CMS_CreateOutputSSTFile( FileNameOut, 
     1                 ncOutID, ProgName, sstIDout, ClearIDout, 
     2                 FrontsIDout, DayOrNightIDout, SecondsSinceID, 
     3                 Start, YearC, MonthC, DayC)

                  if(debug .eq. 1) print *, 'AppQual #293: ',
     1              'Created output SST file.'

                  inquire( File=trim(GeoNameFull), Exist=GeoExist)

                  if(GeoExist .eqv. .false.) then
                     call CMS_GetGeoData( Start, CountToRead, Stride,
     1                    LatIn, LonIn)
                  endif

                  if(debug .eq. 1) print *, 'AppQual #294: ',
     1              'Got geolocation data.'

c******************************* default *******************************

               case default
                  print *, 'AppQual #299. ERROR No subroutine to ',
     1                 'get the output SST data and create the ',
     2                 'the output SST file. STOPPING.'
                  stop

               end select

c%%%%%%%%%%%%%%%%%%%%%%%% End Archive specific %%%%%%%%%%%%%%%%%%%%%%%%

c     Remove the temporary input file if one was created because of
c     a compressed input file.

               if (Compressed .eqv. .true.) then
                  Command = BlankFill(:len(Command))
                  Command = 'rm ' // trim(TempFileName)
                  call system(Command)
               endif

c     Create and write the geolocation file if it does not already exist

               inquire( File=trim(GeoNameFull), Exist=GeoExist)

               if(Debug .eq. 1) print *, 'AppQual #300: Writing ',
     1              'geolocation file.'

               if(GeoExist .eqv. .false.) then
                  call WriteGEOData( LatIn, LonIn,  GeoNameFull, 
     1                 ProgName)
               endif

               if(Debug .eq. 1) print *, 'AppQual #310: Geolocation ',
     1              'file written.'

c     Write the sst field.

               status = nf90_put_var( ncOutID, sstIDout, SSTOut)
               if(status .ne. nf90_noerr) call handle_err(status)

c     And the reference time for this field.

               status = nf90_put_var( ncOutID, SecondsSinceID, 
     1              SecondsSince1970)
               if(status .ne. nf90_noerr) call handle_err(status)

               if(Debug .eq. 1) print *, 'AppQual #320: SST and time ',
     1              'written.'

c_______________________________________________________________________

c     Next clear pixels. The front field will not be written here, 
c     but later by SIED. Start by generating the cloud free field. 
c     Use the same array as the sst field to save space. sstOut is
c     not used again.

               NumClear = 0
               do 1023 jy=1,LenY
                  do 1024 ix=1,LenX
                     if(sstOut(ix,jy) .eq. FillValueInt2) then
                        sstOut(ix,jy) = 0
                     else
                        sstOut(ix,jy) = 1
                        NumClear = NumClear + 1
                     endif

 1024             continue
 1023          continue

               status = nf90_put_var( ncOutID, ClearIDout, sstOut)
               if(status .ne. nf90_noerr) call handle_err(status)

               if(Debug .eq. 1) print *, 'AppQual #330: Clear field ',
     1              'written.'

c     Now close the output SST file.

               status = nf90_close(ncOutID)
               if(status .ne. nf90_noerr) call handle_err(status)

c     Now write out inventory information

               write(UnitInventory,
     1              fmt='(F14.2,I11,6I5,A1,A319)', err=9876) 
     2              SecondsSince1970, NumClear, Year, Month, Day,
     3              Hour, Minute, Second, SpaceC, FileNameOut

               Times(NumberOfFilesTotal) = SecondsSince1970

c     End of directory listing loop

 1020       continue

            stop 'AppQual #330: # of records in the inventory exceeds 
     1MaxNumberOfFiles.'

c     Here when end of directory listing for this month.

 1100       continue

            close(UnitTempIn)

            write(UnitLog,*) 'AppQual #340: Year', Year, 'Month',  
     1           Month, ' # of files',  NumberOfFilesM
            print *, 'AppQual #340: Year', Year, 'Month', Month, 
     1           ' # of files',  NumberOfFilesM

c     End of month loop

 1011    continue

c     End of year loop

 1010 continue

      if(Debug .eq. 1) print *, 'AppQual #350: Finished loops.'

      close(UnitLog)
      close(UnitInventory)

      call MergeArchiveInventories

      if(Debug .eq. 1) print *, 'AppQual #360: Archives merged.'

c     Now get the elapsed wall time - print out in the start and stop
c     times in the subroutine.

      call Duration( DateTimeStart, DurationInMinutes, 
     1     NumberOfFilesTotal, FilesPerMinute)

      if(debug .eq. 1) print *, 'AppQual #999'

      print *, NumberOfFilesTotal, ' files processed in ',
     1     DurationInMinutes, ' minutes ==> ', FilesPerMinute, 
     2     ' files/minute.'

      stop

c     Error statements for writing to the archive inventory file. 

 9876 continue
      write(UnitLog,*) 'AppQual #810: Error writing the Times ',
     1     'archive file.'
      stop 'AppQual #810: Error writing the Times archive file.'

      end program AppQualMed_Main

c***********************************************************************
c     Subroutines in this package:
c      subroutine CreateOutputGeoFile( ncInID, Title, ProgName,
c     1     ncOutID, GeoNameFull, LatIDout, LonIDout, LatLonOutScale, 
c     2     Start, resXbinCtr, resXbinCnt, resYbinCtr, resYbinCnt,
c     4     resXbinCtrIDout, resXbinCntIDout,
c     5     resYbinCtrIDout, resYbinCntIDout)
c
c      subroutine WriteGEOData( LatIn, LonIn, GeoNameFull)
c
c      subroutine CreateOutputSSTFile( FileNameOut, ncOutID, ProgName, 
c     1     nxDimIDout, nyDimIDout, sstDimsOut, sstIDout, ClearIDout, 
c     2     FrontsIDout, DayOrNightIDout, SecondsSinceID, Title, 
c     3     SummaryPart1, creator_name, creator_url, creator_email,
c     4     license, contributor_name, contributor_role,publisher_name, 
c     5     publisher_url, publisher_institution, cdm_data_type)
c
c      subroutine ConditionInput( sstIn, sstOut)
c
c**********************************************************************
      subroutine CreateOutputGeoFile( ncInID, Title, ProgName,
     1     ncOutID, GeoNameFull, LatIDout, LonIDout, LatLonOutScale, 
     2     Start, resXbinCtr, resXbinCnt, resYbinCtr, resYbinCnt,
     4     resXbinCtrIDout, resXbinCntIDout,
     5     resYbinCtrIDout, resYbinCntIDout)
c**********************************************************************
c     
c     This subroutine will generate the lat, lon output file.
c     
      use netcdf

      implicit none

c******Parameter statements

      include 'ParameterStatements.f'

c     Functions

      character*4 CommonSubsVersionNo

c     Variables
      character*200 Title
      character*200 Source
      character*64 date_created
      character*256 history

c     netCDF variables

c     LatLonOutScale - scale factor for the output latitude and 
c      longitude values.
      real LatLonOutScale

      integer ncInID, ncOutID
      integer nxDimIDout, nyDimIDout
      integer LatIDout, LonIDout
      integer LatLonDimsOut(2)

      integer resXbinCtr(1:2000), resYbinCtr(1:2000)
      integer resXbinCnt(1:2000), resYbinCnt(1:2000)

      integer resXbinCtrIDout, resXbinCntIDout
      integer resYbinCtrIDout, resYbinCntIDout
      integer resXDimIDout, resYDimIDout

      character*50 TempVar

      integer Values(8)
      character*5 Zone
      character*10 Time
      character*8 Date
      character*19 ProcessingDateTime

      character*128 XPCtrAttribute
      character*512 XPCntAttribute
      character*128 YPCtrAttribute
      character*512 YPCntAttribute

c     Function to generate UUID.
      character*36 UUID_Gen

c---------------------------------start here --------------------------

c***********************************************************************

c     Start by creating the output file.
      
      if(Debug .eq. 1) print *, 'CreateGeoOut #100: GeoNameFull', 
     1     GeoNameFull

      status = nf90_create( GeoNameFull, 
     1     OR(nf90_netcdf4, nf90_noclobber), ncOutID)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Create the LatOut and LonOut fields in the new data set, 
c     dimensions first.

      status = nf90_def_dim( ncOutID, 'nx', LenX, nxDimIDout)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_dim( ncOutID, 'ny', LenY, nyDimIDout)
      if(status .ne. nf90_noerr) call handle_err(status)

      LatLonDimsOut(1) = nxDimIDout
      LatLonDimsOut(2) = nyDimIDout

      chunksize(1) = min( LenX, chunkszLenX)
      chunksize(2) = min( LenY, chunkszLenY)

      status = nf90_def_var( ncOutID, 'latitude', nf90_int, 
     1     LatLonDimsOut, LatIDout)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_chunking( ncOutID, LatIDout, 0, 
     1     chunksize)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_deflate( ncOutID, LatIDout, 0, 1, 4)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var( ncOutID, 'longitude', nf90_int, 
     1     LatLonDimsOut, LonIDout)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_chunking( ncOutID, LonIDout, 0, 
     1     chunksize)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_deflate( ncOutID, LonIDout, 0, 1, 4)
      if(status .ne. nf90_noerr) call handle_err(status)

c     write attribues from to  output file for latitude.

      if(debug .eq. 1) print *, 'CreateGeoOut: #110'

      status = nf90_put_att( ncOutID, LatIDout, 'long_name', 'latitude')
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, LatIDout,
     1     'units', 'degrees_north')
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, LatIDout, 
     1     'add_offset', 0.0)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, LatIDout, 'scale_factor', 
     1     1.0/LatLonOutScale)
      if(status .ne. nf90_noerr) call handle_err(status)

       status = nf90_put_att( ncOutID, LatIDout, 'valid_min',
     1     GeospatialLatMin)
      if(status .ne. nf90_noerr) call handle_err(status)

       status = nf90_put_att( ncOutID, LatIDout, 'valid_max', 
     1     GeospatialLatMax)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, LatIDout, '_FillValue', 
     1     FillValueInt4)

      if(status .ne. nf90_noerr) call handle_err(status)

c     write  attribues from to output file for longitude

      if(debug .eq. 1) print *, 'CreateGeoOut: #120'

      status = nf90_put_att( ncOutID, LonIDout,
     1     'long_name', 'longitude')
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, LonIDout,
     1     'units', 'degrees_east')
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, LonIDout, 'add_offset', 0.0)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, LonIDout, 'scale_factor', 
     1     1.0/LatLonOutScale)
      if(status .ne. nf90_noerr) call handle_err(status)

       status = nf90_put_att( ncOutID, LonIDout, 'valid_min', 
     1     GeospatialLonMin)
      if(status .ne. nf90_noerr) call handle_err(status)

       status = nf90_put_att( ncOutID, LonIDout, 'valid_max',
     1     GeospatialLonMax)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, LonIDout, '_FillValue', 
     1     FillValueInt4)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Now enter the grid cell resolution information.

      if(debug .eq. 1) print *, 'CreateGeoOut: #130'

      status = nf90_def_dim( ncOutID, 'resx', nf90_unlimited,
     1     resXDimIDout)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var( ncOutID, 'XPixelSeparation', nf90_int,
     1     resXDimIDout, resXbinCtrIDout)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(debug .eq. 1) print *, 'CreateGeoOut: #140'

      XPCtrAttribute = BlankFill(:len(XPCtrAttribute))
      XPCtrAttribute = 'The center of 1 km bins used to'
     1     // ' characterize the separation of pixels for this'
     2     // ' data set in the x-direction.'
      status = nf90_put_att( ncOutID, resXbinCtrIDout, 
     1    'Description', trim(XPCtrAttribute))
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, resXbinCtrIDout, 'units',
     1     'kilometers')
      if(status .ne. nf90_noerr) call handle_err(status)
c
      status = nf90_def_var( ncOutID, 'XPixelCount', nf90_int,
     1     resXDimIDout, resXbinCntIDout)
      if(status .ne. nf90_noerr) call handle_err(status)

      XPCntAttribute = BlankFill(:len(XPCntAttribute))
      XPCntAttribute = 'The number of pixels falling in the'
     1     // ' corresponding XPixelSeparation bin. --'
     2     // ' The distance in kilometers for every adjacent'
     3     // ' pair of pixels in the x-direction is calculated'
     4     // ' from the lat,lon arrays and then rounded to the'
     5     // ' integer kilometer.  The XPixelSeparation bin'
     6     // ' corresponding to this separation is then'
     7     // ' incremented by one.'
      status = nf90_put_att( ncOutID, resXbinCntIDout, 'Description', 
     1    trim(XPCntAttribute))
      if(status .ne. nf90_noerr) call handle_err(status)

      if(debug .eq. 1) print *, 'CreateGeoOut: #150'

      status = nf90_def_dim( ncOutID, 'resy', nf90_unlimited,
     1     resYDimIDout)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var( ncOutID, 'YPixelSeparation', nf90_int,
     1     resYDimIDout, resYbinCtrIDout)
      if(status .ne. nf90_noerr) call handle_err(status)

      YPCtrAttribute = BlankFill(:len(XPCtrAttribute))
      YPCtrAttribute = 'The center of 1 km bins used to'
     1     // ' characterize the separation of pixels for this'
     2     // ' data set in the y-direction.'
      status = nf90_put_att( ncOutID, resYbinCtrIDout, 'Description',
     1    trim(YPCtrAttribute))
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, resYbinCtrIDout, 'units',
     1     'kilometers')
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var( ncOutID, 'YPixelCount', nf90_int,
     1     resYDimIDout, resYbinCntIDout)
      if(status .ne. nf90_noerr) call handle_err(status)

      YPCntAttribute = BlankFill(:len(YPCntAttribute))
      YPCntAttribute = 'The number of pixels falling in the'
     1     // ' corresponding YPixelSeparation bin. --'
     2     // ' The distance in kilometers for every adjacent'
     3     // ' pair of pixels in the y-direction is calculated'
     4     // ' from the lat,lon arrays and then rounded to the'
     5     // ' integer kilometer.  The YPixelSeparation bin'
     6     // ' corresponding to this separation is then'
     7     // ' incremented by one.'
      status = nf90_put_att( ncOutID, resYbinCntIDout, 
     1    'Description', trim(YPCntAttribute))
      if(status .ne. nf90_noerr) call handle_err(status)

      if(debug .eq. 1) print *, 'CreateGeoOut: #160'

c     Now enter the global attributes. 

      status = nf90_put_att( ncOutID, nf90_global, 
     1     'Conventions', 'CF-1.4')
      if(status .ne. nf90_noerr) call handle_err(status)

c     Title/institution/source

      status = nf90_put_att( ncOutID, nf90_global, 
     1     'Title', trim(Title))
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, nf90_global,
     1     'Institution', trim(Institution))
      if(status .ne. nf90_noerr) call handle_err(status)
     
      uuid = BlankFill(:len(uuid))
c      uuid = 'e98c8bbf-8e63-44c1-a701-4eb437f81c02'
      uuid = UUID_Gen(GeoNameFull)
      status = nf90_put_att( ncOutID, nf90_global, 
     1     'uuid', trim(uuid))
      if (status .ne. nf90_noerr) call handle_err(status)

c     Next processing date and time

      Zone = '+0000'
      call date_and_time( Date, Time, Zone, Values)

      date_created = BlankFill(:len(date_created))
c$$$      date_created = '20110831 105334.1962'
      date_created = trim(Date // ' ' // Time)
      status = nf90_put_att( ncOutID, nf90_global,
     1  'date_created', trim(date_created))
      if (status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) then
         print *, 'CreateGeo #170. date_created::', trim(date_created), 
     1        '::'
         print *, 'CreateGeo #171. CommonAppQualSubsVersionNo::', 
     1        trim(CommonAppQualSubsVersionNo), '::'
         print *, 'CreateGeo #172. CommonSubsVersionNo::', 
     1        trim(CommonSubsVersionNo()), '::'
         print *, 'CreateGeo #173. UUID::', trim(uuid), '::'
      endif

      history = BlankFill(:len(history))
      history = '{' // trim(date_created) // ' ' // trim(ProgName) //
     2     '; ' // trim(SatName) // 'Subroutines version ' // 
     3     trim(CommonAppQualSubsVersionNo) // 
     4     '; CommonSubroutines version ' // 
     5     trim(CommonSubsVersionNo()) // ' : ' //
     6     trim(uuid) // '}'

      status = nf90_put_att( ncOutID, nf90_global,
     1  'history', trim(history))
      if (status .ne. nf90_noerr) call handle_err(status)

      if(debug .eq. 1) print *, 'CreateGeoOut: #180'

c     Next processing date and time

      Zone = '+0000'
      call date_and_time( Date, Time, Zone, Values)

      ProcessingDateTime = BlankFill(:len(ProcessingDateTime))
      ProcessingDateTime = Date // ' ' // Time
      status = nf90_put_att( ncOutID, nf90_global, 
     1     'Processing_date_and_time', trim(ProcessingDateTime))
      if(status .ne. nf90_noerr) call handle_err(status)

c     Information about where we started copying data from the original
c     . array to the sst array which will be used in all subsequent
c     . processing.

      status = nf90_put_att( ncOutID, nf90_global, 
     1     'First_line_in_image', Start(2))
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, nf90_global, 
     1     'First_pixel_in_line', Start(1))
      if(status .ne. nf90_noerr) call handle_err(status)

c     All done defining the variables and attributes for output file.

      status = nf90_enddef(ncOutID)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(debug .eq. 1) print *, 'CreateGeoOut: #999'

      end subroutine CreateOutputGeoFile

c***********************************************************************
      subroutine WriteGEOData( LatIn, LonIn, GeoNameFull, ProgName)
c***********************************************************************
c     
c     This subroutine will read in the latitude and longitude values
c     from a file. The input arrays should match the size of the input
c     SST arrays. It will then write these out in a new file.

c******Subroutine arugments
c     
c     sstIn - the input array.
c     sstOut - the output - conditioned - array.

c******General Variables
c     
c     ix, iy - do loop parameters
c     
c     NumberTooSmall - number of pixels that fall below Offset, 
c     . excluding FillValues.
c     NumberTooLarge - the number of pixels that are above Range plus
c     . Offset, excluding FillValues.
c     
c     MinSST - the min value in the input array excluding FillValues
c     MaxSST - the max value in the input array excluding FillValues
c     
      
      use netcdf

      implicit none

c     Parameter statements

      include 'ParameterStatements.f'
      
c     Common statements

c     RsstFillValue - missng value in input arrays.
      integer*2 RsstFillValue
      real*4 RlatFillValue
      real*4 RlonFillValue
      common /MissingValues/ RlatFillValue, RlonFillValue, RsstFillValue

c     General variables

c     LatIDout - netCDF identifier for the output latitude variable.
      integer LatID
c     LatLonOutScale - scale factor for the output latitude and 
c     longitude values.
      real LatLonOutScale
c     LatOut - latitude at each pixel locaion,  written out.
      integer*4, allocatable :: LatOut(:,:)
c     LonIDout - netCDF identifier for the output longitude variable.
      integer LonID
c     LonOut - longitude at each pixel locaion, written out.
      integer*4, allocatable ::LonOut(:,:)

c     Title - the title to use for the output geo location file.
      character*200 Title

      character*319 FileNameIn

      real*4 LatIn(1:LenX,1:LenY), LonIn(1:LenX,1:LenY)

      integer ncInID, ncOutID, LatIDin, LonIDin, LatIDout, LonIDout
      integer nxDimID, nyDimID


      integer nxN, nyN

      integer resXbinCtrIDout, resXbinCntIDout
      integer resYbinCtrIDout, resYbinCntIDout

      real LatScaleFactor, LatOffset, LatFillValue
      real LonScaleFactor, LonOffset, LonFillValue

      integer*4 ix, jy
      
      real resMin, resMax
      real dX, dY
      real distance
      integer*4 idx
      integer outrange
      integer rdx

      integer resBin(1:2000)

      integer resXbinCtr(1:2000)
      integer resXbinCnt(1:2000)
      integer numXbins

      integer resYbinCtr(1:2000)
      integer resYbinCnt(1:2000)
      integer numYbins

c----------------------------------start here --------------------------

c***********************************************************************
c**** Get the lat and lon from some form of input.
c     
c**** This section will change from archive-to-archive.

      allocate( LatOut(LenX,LenY),
     1     LonOut(LenX,LenY),
     *     stat=ierr)
      
      if (ierr .ne. 0) then
         write(UnitLog,*) 'WriteGeoData #000: Allocation error, stop.'
         stop 'WriteGeoData #000: Allocation error, stop.'
      endif

c**** Calcuate stuff from the lat, lon fields.
c     
c**** This section should not change from archive-to-archive.

c     Set the output lat,lon scale factor.

      LatLonOutScale = 10000

c     Input lat and lon scale and offset factors

      LatScaleFactor = 1.0
      LatOffset = 0.0

      LonScaleFactor = 1.0
      LonOffset = 0.0

c     Find the min and max lat and lon values. These will be used in the
c     global metadata written to the GeoLoc file. Also load LatIn and 
c     LonIn into LatOut and LonOut with the correct missing value flag.

      GeospatialLatMin = 999.0 * LatLonOutScale
      GeospatialLatMax = -999.0 * LatLonOutScale
      GeospatialLonMin = 999.0 * LatLonOutScale
      GeospatialLonMax = -999.0 * LatLonOutScale

      if(Debug .eq. 1)  print *, 'LatScaleFactor, LatOffset, ',
     1     ' LatLonOutScale: ', LatScaleFactor, LatOffset, 
     2     LatLonOutScale

      do 1021 jy=1,LenY
         do 1022 ix=1,LenX

            if(LatIn(ix,jy) .eq. FillValueReal) then
               LatOut(ix,jy) = FillValueInt4
               LonOut(ix,jy) = FillValueInt4
            else

               LatOut(ix,jy) = (LatIn(ix,jy)  * LatScaleFactor +
     1              LatOffset) * LatLonOutScale
               LonOut(ix,jy) = (LonIn(ix,jy) * LonScaleFactor +
     1              LonOffset) * LatLonOutScale

               if(LatOut(ix,jy) .lt. GeospatialLatMin)
     1              GeospatialLatMin = LatOut(ix,jy)
               if(LatOut(ix,jy) .gt. GeospatialLatMax)
     1              GeospatialLatMax = LatOut(ix,jy)
               if(LonOut(ix,jy) .lt. GeospatialLonMin)
     1              GeospatialLonMin = LonOut(ix,jy)
               if(LonOut(ix,jy) .gt. GeospatialLonMax)
     1              GeospatialLonMax = LonOut(ix,jy)

            endif

 1022    continue
 1021 continue

      GeospatialLatMin = GeospatialLatMin / LatLonOutScale
      GeospatialLatMax = GeospatialLatMax / LatLonOutScale
      GeospatialLonMin = GeospatialLonMin / LatLonOutScale
      GeospatialLonMax = GeospatialLonMax / LatLonOutScale

      write( UnitLog,*) 'WriteGeoData #125 LatMin: ', GeospatialLatMin
      write( UnitLog,*) 'LatMax: ', GeospatialLatMax
      write( UnitLog,*) 'LonMin: ', GeospatialLonMin
      write( UnitLog,*) 'LonMax: ', GeospatialLonMax
      print *, 'WriteGeoData #125 LatMin: ', GeospatialLatMin
      print *, 'LatMax: ', GeospatialLatMax
      print *, 'LonMin: ', GeospatialLonMin
      print *, 'LonMax: ', GeospatialLonMax

c     Now get a histogram of pixel separations in x and y. These 
c     histograms will also be written to GeoLoc.

      do 1026 jy=1,2000
         resXbinCtr(jy) = 0
         resXbinCnt(jy) = 0
         resBin(jy) = 0
 1026 continue

      resMin = 999.0
      resMax = -999.0
      outrange = 0

      do 2023 jy=1,LenY
         do 2024 ix=1,LenX-1

            if((LatOut(ix,jy) .ne. FillValueInt4) .and. 
     1           (LatOut(ix,jy) .ne. FillValueInt4) .and. 
     1           (LonOut(ix+1,jy) .ne. FillValueInt4) .and. 
     1           (LonOut(ix+1,jy) .ne. FillValueInt4) ) then 

               dX = (LatOut(ix,jy)-LatOut(ix+1,jy)) * 111.0 / 
     1              LatLonOutScale
               dY = (LonOut(ix,jy)-LonOut(ix+1,jy)) * 111.0 / 
     1              LatLonOutScale

               dY = dY * cos((LatOut(ix,jy) / LatLonOutScale) * 
     1              3.14159 / 180)

               distance = sqrt(dX**2 + dY**2)

               if(distance .gt. resMax)
     1              resMax = distance
               if(distance .lt. resMin)
     1              resMin = distance

               rdx = floor(distance + 0.5)

               if((rdx .lt. 1) .or. (rdx .gt. 2000)) then

c$$$                  if(debug .eq. 1) then 
c$$$                     print *, 'WriteGeoData #130: distance:',
c$$$     1                    distance, ':', rdx
c$$$                     print *, '#131: ix, jy, dX: ', ix, jy, dX
c$$$                     print *, '#133: LatOut(ix,jy), LatOut(ix+1,jy): ',
c$$$     1                    LatOut(ix,jy), LatOut(ix+1,jy)
c$$$                     print *, '#134: dY: ', dY
c$$$                     print *, '#135: LonOut(ix,jy), LonOut(ix+1,jy): ',
c$$$     1                    LonOut(ix,jy), LonOut(ix+1,jy)
c$$$                  endif

                  outrange = outrange + 1
               else
                  resBin(rdx) = resBin(rdx) + 1
               endif
            endif

 2024    continue
 2023 continue

      if(debug .eq. 1)then
         print *, 'WriteGeoData #140: resMin: ', resMin
         print *, 'WriteGeoData #141: resMax: ', resMax
         print *, 'WriteGeoData #142: out-of-range:', outrange
      endif

      numXbins = 0
      idx = 1

      do 2025 jy=1,2000

         if(resBin(jy) .ne. 0) then
            resXbinCtr(idx) = jy
            resXbinCnt(idx) = resBin(jy)
            numXbins = numXbins + 1

c$$$  if(debug .eq. 1) print *, 'WriteGeoData #165: ',
c$$$  1           'idx, resXbinCtr(idx), resXbinCnt(idx), numXbins:',
c$$$  2           idx, resXbinCtr(idx), resXbinCnt(idx), numXbins

            idx = idx + 1
         endif

 2025 continue

      resMin = 999.0
      resMax = -999.0
      outrange = 0

      do 3029 jy=1,2000
         resYbinCtr(jy) = 0
         resYbinCnt(jy) = 0
         resBin(jy) = 0
 3029 continue

      do 3026 jy=1,LenY-1
         do 3027 ix=1,LenX

            if(  (LatOut(ix,jy) .ne. FillValueInt4) .and. 
     1           (LatOut(ix,jy+1) .ne. FillValueInt4) .and. 
     1           (LonOut(ix,jy) .ne. FillValueInt4) .and. 
     1           (LonOut(ix,jy+1) .ne. FillValueInt4) ) then 

               dX = (LatOut(ix,jy)-LatOut(ix,jy+1)) * 111.0 / 
     1              LatLonOutScale
               dY = (LonOut(ix,jy)-LonOut(ix,jy+1)) * 111.0 / 
     1              LatLonOutScale
               dY = dY * cos((LatOut(ix,jy) / LatLonOutScale) * 
     1              3.14159 / 180)

               distance = sqrt(dX**2 + dY**2)

               if(distance .gt. resMax)
     1              resMax = distance
               if(distance .lt. resMin)
     1              resMin = distance

               rdx = floor(distance + 0.5)

               if((rdx .lt. 1) .or. (rdx .gt. 2000))then
c$$$                  if(debug .eq. 1) print *, 'WriteGeoData #150: ',
c$$$     1                 'distance:',distance, ':', rdx
                  outrange = outrange + 1
               else
                  resBin(rdx) = resBin(rdx) + 1
               endif
            endif

 3027    continue
 3026 continue

      if(debug .eq. 1) then
         print *, 'WriteGeoData #160: resMin: ', resMin
         print *, 'WriteGeoData #161: resMax: ', resMax
         print *, 'WriteGeoData #162: out-of-range:', outrange
      endif

      numYbins = 0
      idx = 1

      do 3028 jy=1,2000

         if(resBin(jy) .ne. 0) then
            resYbinCtr(idx) = jy
            resYbinCnt(idx) = resBin(jy)
            numYbins = numYbins + 1
            idx = idx + 1
         endif

 3028 continue
      
c     Generate the title. Which satellite first, AMSR-E or TMI?


      Loc1 = index( SatName, 'amsr')
      Title  = 'Latitude,longitude file corresponding to  ' //
     1     'conditioned SST fields for ' // trim(SatName) // ' data.'

c     Create output file and put in metadata

      call CreateOutputGeoFile( ncInID, Title, ProgName,
     1     ncOutID, GeoNameFull, LatIDout, LonIDout, LatLonOutScale, 
     2     Start, resXbinCtr, resXbinCnt, resYbinCtr, resYbinCnt,
     4     resXbinCtrIDout, resXbinCntIDout,
     5     resYbinCtrIDout, resYbinCntIDout)

c     And write the variables out.

      if(debug .eq. 1) print *, 'WriteGeoData #170'

      status = nf90_put_var( ncOutID, LatIDout, LatOut)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_var( ncOutID, LonIDout, LonOut)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_var( ncOutID, resXbinCtrIDout,
     1     resXbinCtr(1:numXbins))
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_var( ncOutID, resXbinCntIDout,
     1     resXbinCnt(1:numXbins))
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_var( ncOutID, resYbinCtrIDout,
     1     resYbinCtr(1:numYbins))
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_var( ncOutID, resYbinCntIDout,
     1     resYbinCnt(1:numYbins))
      if(status .ne. nf90_noerr) call handle_err(status)

c     Now clean up.

      if(debug .eq. 1) print *, 'WriteGeoData #180'

c     status = nf90_close(ncInID)
c     if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_close(ncOutID)
      if(status .ne. nf90_noerr) call handle_err(status)
      if(debug .eq. 1) print *, 'WriteGeoData #190'

      end subroutine WriteGEOData

c**********************************************************************
      subroutine CreateOutputSSTFile( FileNameOut, ncOutID, ProgName, 
     1     nxDimIDout, nyDimIDout, sstDimsOut, sstIDout, ClearIDout, 
     2     FrontsIDout, DayOrNightIDout, SecondsSinceID, Title, Source, 
     3     creator_name, creator_url, creator_email,
     4     license, contributor_name, contributor_role, publisher_name, 
     5     publisher_url, publisher_institution, cdm_data_type)
c**********************************************************************
c     
c     This subroutine will generate the output file.
c     
      use netcdf

      implicit none

c******Parameter statements

      include 'ParameterStatements.f'

c     Functions

      character*20 Num2StrInt, Num2StrFloat, Num2StrG

c     AttDescription - attribute description when needed.
      character*1000 AttDescription

      character*36 UUID_Gen

      character*4 CommonSubsVersionNo
      character*200 Title
      character*67 Source
      character*218 Comments

c     Filenames

      character*319 FileNameOut
      character*319 Geospatial_Metadata_File

c     netCDF variables

      integer LastChar, Dummy

      integer ncOutID, nxDimIDin, nyDimIDin, nxDimIDout, nyDimIDout
      integer sstIDout, SecondsSinceID
      integer ClearIDout, FrontsIDout, DayOrNightIDout

      integer sstDimsOut(2) 

      integer Values(8)
      character*5 Zone
      character*10 Time
      character*8 Date
      character*19 ProcessingDateTime
c$$$      character*3000 Summary, SummaryPart1, SummaryPart2, SummaryPart3
      
      character*20 OffsetChar, ScaleInputChar, RangeChar

      character*16 Conventions
      character*64 Meta_Conventions
      character*64 date_created
      character*256 history
      character*64 standard_name_vocab
      character*64 creator_name
      character*64 creator_email
      character*256 creator_url
      character*1024 license
      character*512 contributor_name
      character*512 contributor_role
      character*256 publisher_name
      character*256 publisher_url
      character*256 publisher_institution
      character*32 cdm_data_type

      real*4 geospatial_lat_res
      character*32 geospatial_lat_units

      real*4 geospatial_lon_res
      character*32 geospatial_lon_units

      character*32 geospatial_vert_units
      character*32 geospatial_vert_pos
      real*4 geospatial_vert_min
      real*4 geospatial_vert_max
      real*4 geospatial_vert_res

      character*32 time_coverage_start
      real*4 time_coverage_duration

      integer i

c     Flags - 0 or 1, used for fronts, day or night and clear or cloudy.
      integer*2 Flags(2)

      logical exis

c---------------------------------start here --------------------------

      if(Debug .eq. 1) print *, 'CreateMedian: #100: FileNameOut::',
     1     trim(FileNameOut), '::'

c     Start by creating the output file.
      
      status = nf90_create( FileNameOut, 
     1     OR(nf90_netcdf4, nf90_noclobber), ncOutID)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Create the sstOut field in the new data set, dimensions first.

      if(Debug .eq. 1) print *, 'CreateMedian: #110'

      status = nf90_def_dim( ncOutID, 'nx', LenX, nxDimIDout)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_dim( ncOutID, 'ny', LenY, nyDimIDout)
      if(status .ne. nf90_noerr) call handle_err(status)

      sstDimsOut(1) = nxDimIDout
      sstDimsOut(2) = nyDimIDout

      if(Debug .eq. 1) print *, 'CreateMedian: #112. sstDimeOut ',
     1     sstDimsOut

      chunksize(1) = min( LenX, chunkszLenX)
      chunksize(2) = min( LenY, chunkszLenY)

      if(Debug .eq. 1) print *, 'CreateMedian: #120'

c     Now create the entry for the median SST field.

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = 'median_' // trim(ParameterName_lc)
      status = nf90_def_var( ncOutID, AttDescription, nf90_short, 
     1     sstDimsOut, sstIDout)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_chunking( ncOutID, sstIDout, 0, 
     1     chunksize)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_deflate( ncOutID, sstIDout, 0, 1, 4)
      if(status .ne. nf90_noerr) call handle_err(status)

c     create the attribues sst

      if(Debug .eq. 1) print *, 'CreateMedian: #130'

      status = nf90_put_att( ncOutID, sstIDout, 'long_name', 
     1     trim(ParameterName_lc))
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, sstIDout, 'units', 
     1     trim(ParameterUnits))
      if(status .ne. nf90_noerr) call handle_err(status)

      AttDescription = BlankFill(:len(AttDescription))
      AttDescription = '3x3_median_of_' // trim(ParameterName_lc)
      status = nf90_put_att( ncOutID, sstIDout,
     1     'standard_name', AttDescription)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, sstIDout, 
     1     'add_offset',  OutputOffset)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, sstIDout, 
     1     'scale_factor', OutputScaleFactor)
      if(status .ne. nf90_noerr) call handle_err(status)

       status = nf90_put_att( ncOutID, sstIDout, 'valid_min', 0)
      if(status .ne. nf90_noerr) call handle_err(status)

       status = nf90_put_att( ncOutID, sstIDout, 'valid_max', Range)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, sstIDout, 
     1     '_FillValue', FillValueInt2)
      if(status .ne. nf90_noerr) call handle_err(status)

c***********************************************************************
c     Next create the entry for the clear field.

      status = nf90_def_var( ncOutID, 'cloud_free_pixels', nf90_short, 
     1     sstDimsOut, ClearIDout)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_chunking( ncOutID, ClearIDout, 0, 
     1     chunksize)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_deflate( ncOutID, ClearIDout, 0, 1, 4)
      if(status .ne. nf90_noerr) call handle_err(status)

c     create the attributes for the cloud free pixels

      if(Debug .eq. 1) print *, 'CreateMedian: #133'

      status = nf90_put_att( ncOutID, ClearIDout, 'long_name', 
     1     'cloud_free_pixels')
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, ClearIDout, 'valid_min', 0)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, ClearIDout, 'valid_max', 1)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, ClearIDout, 
     1     '_FillValue', FillValueInt2)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, ClearIDout,
     1     'flag_meanings', 'cloudy clear')
      if(status .ne. nf90_noerr) call handle_err(status)

      Flags(1) = 0
      Flags(2) = 1
      status = nf90_put_att( ncOutID, ClearIDout,
     1     'flag_values', Flags)
      if(status .ne. nf90_noerr) call handle_err(status)

c***********************************************************************
c     Next create the entry for the fronts field.

      status = nf90_def_var( ncOutID, 'cayula_cornillon_front_pixel', 
     1     nf90_short, sstDimsOut, FrontsIDout)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_chunking( ncOutID, FrontsIDout, 0, 
     1     chunksize)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_deflate( ncOutID, FrontsIDout, 0, 1, 4)
      if(status .ne. nf90_noerr) call handle_err(status)

c     create the attribues for the Cayula-Cornillon front pixels.

      if(Debug .eq. 1) print *, 'CreateMedian: #137'

      status = nf90_put_att( ncOutID, FrontsIDout, 'long_name', 
     1     'cayula_cornillon_front_pixel')
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, FrontsIDout, 'valid_min', 0)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, FrontsIDout, 'valid_max', 1)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, FrontsIDout, 
     1     '_FillValue', FillValueInt2)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, FrontsIDout,
     1     'flag_meanings', 'no_front front')
      if(status .ne. nf90_noerr) call handle_err(status)

      Flags(1) = 0
      Flags(2) = 1
      status = nf90_put_att( ncOutID, FrontsIDout,
     1     'flag_values', Flags)
      if(status .ne. nf90_noerr) call handle_err(status)

c***********************************************************************
c     Next create the entry for the day or night field.

      status = nf90_def_var( ncOutID, 'day_or_night_pixel', 
     1     nf90_short, sstDimsOut, DayOrNightIDout)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_chunking( ncOutID, DayOrNightIDout, 0, 
     1     chunksize)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_var_deflate( ncOutID, DayOrNightIDout, 0, 1, 4)
      if(status .ne. nf90_noerr) call handle_err(status)

c     create the attribues for the day or night pixels.

      if(Debug .eq. 1) print *, 'CreateMedian: #138'

      status = nf90_put_att( ncOutID, DayOrNightIDout, 'long_name', 
     1     'day_or_night_pixel')
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, DayOrNightIDout, 'valid_min', 0)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, DayOrNightIDout, 'valid_max', 1)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, DayOrNightIDout, 
     1     '_FillValue', FillValueInt2)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, DayOrNightIDout,
     1     'flag_meanings', 'Day Night')
      if(status .ne. nf90_noerr) call handle_err(status)

      Flags(1) = 0
      Flags(2) = 1
      status = nf90_put_att( ncOutID, DayOrNightIDout,
     1     'flag_values', Flags)
      if(status .ne. nf90_noerr) call handle_err(status)

c***********************************************************************
c     create the time - seconds since 1970 and define its attributes

      if(Debug .eq. 1) print *, 'CreateMedian: #140'

      call DefineSecondsSinceVar( ncOutID, SecondsSinceID)

c***********************************************************************
c**** Now for the global attributes ************************************

c     Will use the date and time created in a few places so get this
c      info now.

      Zone = '+0000'
      call date_and_time( Date, Time, Zone, Values)

      date_created = BlankFill(:len(date_created))
c$$$      date_created = '20110831 105334.1962'
      date_created = trim(Date // ' ' // Time)

      status = nf90_put_att( ncOutID, nf90_global, 'title', Title)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Now build the rest of the Summary statement for this file.
c     Start with Part2 which describes how the output digitigal counts
c     are calculated. Get character renditions of variables to print out

      Summary = BlankFill(1:len(Summary))
      SummaryPart2 = BlankFill(1:len(SummaryPart2))
      SummaryPart3 = BlankFill(1:len(SummaryPart3))

      OffsetChar = BlankFill(1:len(OffsetChar))
      OffsetChar = Num2StrFloat(InputOffset2OutputOffset)

      if(Debug .eq. 1) print *, 'CreateMedian: #151. OffsetChar: ',
     1     OffsetChar, ' InputOffset2OutputOffset: ',
     2     InputOffset2OutputOffset

      ScaleInputChar = BlankFill(1:len(ScaleInputChar))
      ScaleInputChar = Num2StrFloat(ScaleInput)
      RangeChar = BlankFill(1:len(RangeChar))
      RangeChar = Num2StrInt(Range)
      SSTFillValueChar = BlankFill(1:len(SSTFillValueChar))
c     Num2StrInt requires integer input. SSTFillValue is integer*2, so
c     set it equal to a dummy variable before calling num2string.
      Dummy = SSTFillValueIn
      SSTFillValueChar = Num2StrInt(Dummy)

      if(InputOffset2OutputOffset .gt. 0) then
         SummaryPart2 = ' ' // trim(ScaleInputChar) // ' + ' //
     1     trim(OffsetChar) // '); Counts_Out < 0 or > ' //
     2     trim(RangeChar) // ' set to ' // trim(sstFillValueChar)
      else
         SummaryPart2 = ' ' // trim(ScaleInputChar) // ' ' //
     1     trim(OffsetChar) // '); Counts_Out < 0 or > ' //
     2     trim(RangeChar) // ' set to ' // trim(sstFillValueChar)
      endif

c     And now for Part3.

      SummaryPart3 = '. The first dimension in the array (i) is ' 
     1     // trim(iVarName) // '. The second dimension in the array '
     2     // '(j) is ' // trim(jVarName) // '. The conditioned data '
     3     // 'were then median filtered with a 3x3 median filter. This'
     4     // ' file also contains a cloud/clear field, 0 for cloud '
     5     // 'contaminated and 1 for clear; a Cayula-Cornillon '
     6     // 'fronts field, 0 for no front, 1 for a front pixel, '
     7     // 'and; a day/night field based on the solar zenith '
     8     // 'angle at the pixel, 1 if this angle is less than '
     9     // '90 degrees, day, and 0 otherwise, night.'

c     Finally put the three parts together.

      Summary = trim(SummaryPart1) // trim(SummaryPart2) // 
     1     trim(SummaryPart3)

      status = nf90_put_att( ncOutID, nf90_global, 'summary', Summary)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, nf90_global,
     1     'Conventions', 'CF-1.5')
      if(status .ne. nf90_noerr) call handle_err(status)

      standard_name_vocab = BlankFill(:len(standard_name_vocab))
      standard_name_vocab = 'CF-1.5'
      status = nf90_put_att( ncOutID, nf90_global,
     1  'standard_name_vocabulary', trim(standard_name_vocab))
      if (status .ne. nf90_noerr) call handle_err(status)

      Meta_Conventions = BlankFill(:len(Meta_Conventions))
      Meta_Conventions = 'Unidata Dataset Discovery v1.0'
      status = nf90_put_att( ncOutID, nf90_global,
     1  'Metadata_Conventions', trim(Meta_Conventions))
      if (status .ne. nf90_noerr) call handle_err(status)

      uuid = BlankFill(:len(uuid))
c     e.g., uuid = 'e98c8bbf-8e63-44c1-a701-4eb437f81c02'
      uuid = UUID_Gen(FileNameOut)

      status = nf90_put_att( ncOutID, nf90_global,
     1  'uuid', trim(uuid))
      if (status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'CreateMedian: #141. Wrote batch 1.'

c     Next processing date and time

      status = nf90_put_att( ncOutID, nf90_global,
     1  'date_created', trim(date_created))
      if (status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, nf90_global, 
     1      'source', trim(Source))
      if(status .ne. nf90_noerr) call handle_err(status)

c     Generate the history information.

      history = BlankFill(:len(history))
      history = '{' // trim(date_created) // ' ' // trim(ProgName) //
     2     '; ' // trim(SatName) // 'Subroutines version ' // 
     3     trim(CommonAppQualSubsVersionNo) // 
     4     '; CommonSubroutines version ' // 
     5     trim(CommonSubsVersionNo()) // ' : ' //
     6     trim(uuid) // '}'

      status = nf90_put_att( ncOutID, nf90_global,
     1  'history', trim(history))
      if (status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, nf90_global,
     1  'creator_name', trim(creator_name))
      if (status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, nf90_global,
     1  'creator_url', trim(creator_url))
      if (status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, nf90_global,
     1  'creator_email', trim(creator_email))
      if (status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, nf90_global,
     1  'institution', trim(institution))
      if (status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, nf90_global,
     1  'project', trim(project))
      if (status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, nf90_global,
     1  'acknowledgement', trim(acknowledgement))
      if (status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, nf90_global,
     1  'license', trim(license))
      if (status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, nf90_global,
     1  'contributor_name', trim(contributor_name))
      if (status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, nf90_global,
     1  'contributor_role', trim(contributor_role))
      if (status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, nf90_global,
     1  'publisher_name', trim(publisher_name))
      if (status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, nf90_global,
     1  'publisher_url', trim(publisher_url))
      if (status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, nf90_global,
     1  'publisher_institution', trim(publisher_institution))
      if (status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, nf90_global,
     1  'cdm_data_type', trim(cdm_data_type))
      if (status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'CreateMedian: #142. Wrote batch 2.'

c     The name of the file containing the lat and lon arrays

      LastChar = len(trim(GeoName))
      status = nf90_put_att( ncOutID, nf90_global,
     1     'LatLonFileName', trim(GeoName))
      if(status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'CreateMedian: #143. Wrote GeoName.'

c****** Now add the geospatial metadata. These metadata are created  ***
c****** in AppQualMed main. They are written to each median file and ***
c****** to the geoloc file.                                          ***

c      geospatial_lat_min = MinLat
      status = nf90_put_att( ncOutID, nf90_global,
     1  'geospatial_lat_min', GeospatialLatMin)
      if (status .ne. nf90_noerr) call handle_err(status)

c      geospatial_lat_max = MaxLat
      status = nf90_put_att( ncOutID, nf90_global,
     1  'geospatial_lat_max', GeospatialLatMax)
      if (status .ne. nf90_noerr) call handle_err(status)

c      geospatial_lon_min = MinLon
      status = nf90_put_att( ncOutID, nf90_global,
     1  'geospatial_lon_min', GeospatialLonMin)
      if (status .ne. nf90_noerr) call handle_err(status)

c      geospatial_lon_max = MaxLon
      status = nf90_put_att( ncOutID, nf90_global,
     1  'geospatial_lon_max', GeospatialLonMax)
      if (status .ne. nf90_noerr) call handle_err(status)

      geospatial_vert_min = 0.
      status = nf90_put_att( ncOutID, nf90_global,
     1  'geospatial_vertical_min', geospatial_vert_min)
      if (status .ne. nf90_noerr) call handle_err(status)

      geospatial_vert_max = 0.
      status = nf90_put_att( ncOutID, nf90_global,
     1  'geospatial_vertical_max', geospatial_vert_max)
      if (status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, nf90_global,
     1  'geospatial_vertical_units', 'm')
      if (status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'CreateMedian: #144. Wrote Geo data.'

c     Information about where we started copying data from the original
c     . array to the sst array which will be used in all subsequent
c     . processing.

      status = nf90_put_att( ncOutID, nf90_global, 
     1     'First_line_in_image', Start(2))
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, nf90_global, 
     1     'First_pixel_in_line', Start(1))
      if(status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'CreateMedian: #146. Finished ',
     1     'attributes.'

c     All done defining the variables and attributes for output file.

      status = nf90_enddef(ncOutID)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'CreateMedian: #147. end def.'

      return

 991  continue
      write( UnitLog, *) 'CreateMedianOutputFile #1000. Error opening ',
     1     'Geophysical_Metadata_File to read'
      print *, 'CreateMedianOutputFile #1000. Error opening ',
     1     'Geophysical_Metadata_File to read'
      stop

 992  continue
      write( UnitLog, *) 'CreateMedianOutputFile #1010. Error opening ',
     1     'Geophysical_Metadata_File to write'
      print *, 'CreateMedianOutputFile #1010. Error opening ',
     1     'Geophysical_Metadata_File to write'
      stop

      end subroutine CreateOutputSSTFile

c***********************************************************************
      subroutine ConditionInput( sstIn, sstOut)
c***********************************************************************
c     
c     This subroutine will condition the input. Specifically, it will
c     subtract Offset from all values and then set values less than 
c     zero and larger than Range to 0. 
c     
c     Will also divide the input by 10 as agreed with Pierre for CMS
c     SSTs in centidegrees.
c     
c     For example, if the lowest acceptable value is -100 and the 
c     largest acceptable value is 1400, the Offset would be -100 and
c     the Range 1500.

c******Subroutine arugments
c     
c     sstIn - the input array.
c     sstOut - the output - conditioned - array.

c******General Variables
c     
c     ix, iy - do loop parameters
c     
c     NumberTooSmall - number of pixels that fall below Offset, 
c     . excluding FillValues.
c     NumberTooLarge - the number of pixels that are above Range plus
c     . Offset, excluding FillValues.
c     
c     MinSST - the min value in the input array excluding FillValues
c     MaxSST - the max value in the input array excluding FillValues
      
      implicit none

c     Parameter statements

      include 'ParameterStatements.f'

c     Common statements

c     RsstFillValue - missng value in input arrays. 
      integer*2 FillValueIn
      real*4 RlatFillValue
      real*4 RlonFillValue
      common /MissingValues/ RlatFillValue, RlonFillValue, FillValueIn

c     General variables

      real*4 sstIn(1:LenX,1:LenY,1)
      integer*2 sstOut(1:LenX,1:LenY)

      integer*2 MinSST, MaxSST
      integer*2 NumberTooSmall, NumberTooLarge

      integer*2 ix, iy
      
c----------------------------------start here --------------------------

c     Convert from integer to double. This was done to allow use of
c     Convert input counts to output counts for good data. Get and
c     printout the max and min input counts while you're at it. Also,
c     set values of output counts less than 0 or greater than Range to
c     sstFillValue. 

      NumberTooSmall = 0
      NumberTooLarge = 0

      MinSST = -32768
      MaxSST = 32767

      if(debug .eq. 1) print *, 'ConditionInput #000: ScaleInput:',
     1     ScaleInput, ' InputOffset2OutputOffset: ', 
     2     InputOffset2OutputOffset

      do 100 iy=1,LenY
         do 110 ix=1,LenX
            if(sstIn(ix,iy,1) .eq. FillValueReal) then
               sstOut(ix,iy) = FillValueInt2
            else

c     Get the min and max SST values in this aray excluding FillValue.

               if(sstIn(ix,iy,1) .lt. MinSST) MinSST = sstIn(ix,iy,1)
               if(sstIn(ix,iy,1) .gt. MaxSST) MaxSST = sstIn(ix,iy,1)

c     Get counts out from counts in.

               sstOut(ix,iy) = nint( sstIn(ix,iy,1)  * 
     1              ScaleInput + InputOffset2OutputOffset)

c     Set values < 0 and > range to FillValueInt2.

               if(sstOut(ix,iy) .le. 0) then
                  NumberTooSmall = NumberTooSmall + 1
                  sstOut(ix,iy) = FillValueInt2
               endif
               
               if(sstOut(ix,iy) .gt. Range) then
                  NumberTooLarge = NumberTooLarge + 1
                  sstOut(ix,iy) = FillValueInt2
               endif

            endif
 110     continue
 100  continue

      if(Debug .eq. 1) then
         write(UnitLog,*) 'The min SST: ', MinSST, 'The max SST', MaxSST
         print *, 'ConditionInput #100: The min SST: ', MinSST, 
     1        'The max SST', MaxSST
      endif

      if(NumberTooSmall .gt. 0) then
         print *, 'ConditionInput #110: ', NumberTooSmall, 
     1        ' values below ', OutputOffset,
     2        ' which correspond to 0 counts.'
         write(UnitLog,*) NumberTooSmall, ' values below ', 
     1        OutputOffset, ' which correspond to 0 counts.'
      endif

      if(NumberTooLarge .gt. 0) then
         print *, 'ConditionInput #111: ', NumberTooLarge, 
     1        ' values above ', MaxSSTOut,
     2        ' which correspond to ', Range, ' counts.'
         write(UnitLog,*) NumberTooLarge, ' values above ', MaxSSTOut,
     1        ' which correspond to ', Range, ' counts.'
      endif

      return

      end subroutine ConditionInput

c**********************************************************************
c**********************************************************************

      include 'CommonSubroutines-2.36.f'
      include 'AppQual_AMSRE_v7_l2_Subroutines-1.01.f'
      include 'AppQual_AVHRR_gac_L2P_v1_Subroutines-1.01.f'
      include 'AppQual_AMSRE_v7_l3_Subroutines-1.00.f'
      include 'AppQual_AVHRR_hrpt_l2_Subroutines-1.00.f'
      include 'AppQual_ECCO2_4km_Subroutines-1.00.f'
      include 'AppQual_CMS_Subroutines-1.01.f'

