c***********************************************************************
c     Subroutines and functions in this package:
c     
c     function CommonAppQualSubsVersionNo()
c     The above function was replaced by code in the subroutine below.
c
c     subroutine AMSRE_v7_l3_GetDateTime(ImageName, Year, Month, 
c     1     YearT, MonthT, Day, Hour, Minute, Second)
c     
c     subroutine AMSRE_v7_l3_GetSSTOutData( FileNameIn, ConditionedSST, 
c     1     SecondsSince1970)
c     
c     subroutine AMSRE_v7_l3_CreateOutputSSTFile( FileNameOut, ncOutID, 
c     1     sstIDout, ClearIDout, FrontsIDout, DayOrNightIDout,
c     2     SecondsSinceID, Start, YearC, MonthC, DayC)
c     
c      subroutine AMSRE_v7_l3_GetGeoData( FileNameIn, LatIn, LonIn, 
c     1     SecondsSince1970, FileNameOut, NumClear, YearC, MonthC, Day 
c     2     Hour, Minute, Second, GeoNameFull)
c     
c***********************************************************************
c     Modifications 
c     
c     1.00
c     
c     1/7/14 - PCC - Created from AppQual_AMSRE_v7_l2_Subroutines.
c     
c***********************************************************************
      subroutine AMSRE_v7_l3_GetDateTime( ImageName, Year, Month, 
     1     YearT, MonthT, Day, Hour, Minute, Second,
     2     MonthC, YearC, iDayNight)
c***********************************************************************
c     This subroutine gets the year, month, day, hour, minute and
c     seconds for the image specified by the image name. These times
c     are approximate and are used to determine if this image is in
c     in the time range specified. They should be considered accurate
c     the nearest hour. The precise time will be determined from 
c     metadata in the input file if available.
c
c     The subroutine will also determine if this is a day or night image
c     
      implicit none

c     Functions

      real secnds
      real*8 SecondsSince

c******Parameter statements

      include 'ParameterStatements.f'

c     General variables

c     Day - Day of the satellite pass.
      integer Day 

c     Hour - Hour of the satellite pass.
      integer Hour 

c     iDayNight - 1 for day time fields, 2 for nighttime fields. This 
c      variable is only used for RSS data.
      integer iDayNight
c     ImageName - the filename for the input image.
      character*319 ImageName
      
c     Minute - Minute of the satellite pass.
      integer Minute 

c     Month - This is the month in the loop over months.
      integer Month
c     MonthT - Month of the satellite pass.
      integer MonthT

c     Second - Second of the satellite pass.
      integer Second

c     Year - This is the year in the loop over years.
      integer Year
c     YearT - Year of the satellite pass.
      integer YearT

c---------------------------------start here --------------------------- 
      if(Debug .eq. 1) print *, 'GetTimes #000: ImageName::',
     1     trim(ImageName), '::'

c       The next line sets the version of the subroutines in this file.
c       It needs to be incremented whenever this file is modified.

      CommonAppQualSubsVersionNo = '1.00'

c     Get year, month, day. 

      read( ImageName(7:10), '(i4)') YearT
      read( ImageName(11:12), '(i2)') MonthT
      read( ImageName(13:14), '(i2)') Day

      if(Debug .eq. 1) print *, 'GetTimes #100: YearT, ',
     1     'MonthT, Day: ', YearT, MonthT, Day

c     Make sure that the time in the satellite name corresponds to the
c     year and month being read.

      if((Year .ne. YearT) .or. (Month .ne. MonthT)) then
         write(UnitLog, *) 'GetTimes #120. Problem with times. ',
     1        'Year and month in filename: ', YearT, MonthT,
     2        ' does not match that in loops:', Year, Month
         print *, 'GetTimes #120. Problem with times. ',
     1        'Year and month in filename: ', YearT, MonthT,
     2        ' does not match that in loops:', Year, Month
         stop
      endif

c     Set the flag that says that the times written out are GMT.

      LSTorGMT = 1

c     Get seconds since the start of 1970. Will get a more accurate 
c     time later when it reads from the file, but have to decompress
c     and do all that, so skip for now.

      Hour = 0
      Minute = 0
      Second = 0

      if(Debug .eq. 1) print *, 'GetTimes #110'

c     Set the flag for day or night. This will be determined from the
c     directory name in BaseDir which should have either day or night
c     in it.

      if(index( BaseDir, 'day') .gt. 0) then
         iDayNight = 1;
         if(Debug .eq. 1) print *, 'GetTimes #120: ',
     1        'Reading in daytime data for ', trim(ImageName)

      elseif(index( BaseDir, 'night') .gt. 0) then
         iDayNight = 2;
         if(Debug .eq. 1) print *, 'GetTimes #130: ',
     2        'Reading in nighttime data for ', trim(ImageName)

      else
            print *, 'GetTimes #140: Neither day nor night ',
     1           'in BaseDir; need it for l3 AMSRE data. ',
     2           'BaseDir::',  trim(BaseDir), '::'
            stop '************ ERROR ***************'
      endif

      if(Debug .eq. 1) print *, 'GetTimes #999'

      end subroutine AMSRE_v7_l3_GetDateTime

c***********************************************************************
      subroutine AMSRE_v7_l3_GetSSTData( iDayNight, FileNameIn, sstOut)
c***********************************************************************
c     
c     This subroutine will get the input SST data for AMSR-E. 
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
c     
      
      use netcdf

      implicit none

c     Parameter statements

      include 'ParameterStatements.f'
      
c     General variables

c     FileNameIn - input filename for SST data.
      character*319 FileNameIn

c     ia - Do loop index for input fields.
      integer ia
c     iDayNight - 1 for day time fields, 2 for nighttime fields.
      integer iDayNight

c     ConditionedSST - SST after subtracting Offset and truncating.
      integer*2, allocatable :: ConditionedSST(:,:)

c     sstIn - SST read in.
      real*4, allocatable :: sstIn(:,:)
c     sstOut - SST written out.
      integer*2 sstOut(LenX, LenY) 

c     TempSST - variable used to read SST, allocated for each file.
      character(1), allocatable :: TempSST(:,:,:)

      integer*2 MinSST, MaxSST, NumberTooSmall, NumberTooLarge

      integer SecondsSinceID

      integer*4 ix, jy

c----------------------------------start here --------------------------
c     

      if(Debug .eq. 1) print *, 'AMSRE_GetSSTData #000: ',
     1     'trim(FileNameIn)::', trim(FileNameIn), '::'

c     Allocate input and output variables.

      allocate( sstIn(LenX,LenY),
     1     TempSST(LenX,LenY,7),
     2     ConditionedSST(LenX,LenY),
     *     stat=ierr)
      
      if (ierr .ne. 0) then
         write(UnitLog,*) 'AMSRE_GetSSTData #130: Allocation error, ',
     1        'exiting.'
         stop 'AMSRE_GetSSTData #131: Allocation error, exiting.'
      endif

c     Open the input file.

      open(UnitInputSST, FILE=FileNameIn, STATUS='OLD', RECL=7257600,
     1 ACCESS='DIRECT', FORM='UNFORMATTED')

c     Now read the data - read the day or night field depending on 
c     the value set in the calling program.

      READ( UnitInputSST, rec=iDayNight) TempSST

      if(Debug .eq. 1) print *, 'AMSRE_GetSSTData #140: Input file',
     1     'trim(FileNameIn)::', trim(FileNameIn), ':: opened and read.'

c     Convert from byte to integer*2. This is to give the input variable
c      the type expected in the calling program.

      sstIn(:,:) = int(ICHAR(TempSST(:,:,2)),2)

c     Set values greater than 250 to missing value. 

      where(sstIn(:,:) > 250)
         sstIn(:,:) = FillValueReal
      endwhere

c     Now condition SST data. ******************************************

      call ConditionInput( sstIn, ConditionedSST)
      
c     Next Median filter the conditioned data.

      call median( ConditionedSST, SSTOut, 3)

      if(Debug .eq. 1) print *, 'AMSRE_GetSSTData #150: ',
     1     'Median SST generated. Closing input file.'

c     Now clean up.

      close(UnitInputSST)

      end subroutine AMSRE_v7_l3_GetSSTData

c**********************************************************************
      subroutine AMSRE_v7_l3_CreateOutputSSTFile( FileNameOut, ncOutID, 
     1     ProgName, sstIDout, ClearIDout, FrontsIDout, DayOrNightIDout,
     2     SecondsSinceID, Start, YearC, MonthC, DayC)
c**********************************************************************
c     
c     This subroutine will generate the output file.
c     
      use netcdf

      implicit none

c******Parameter statements

      include 'ParameterStatements.f'

c     Functions

c     AttDescription - attribute description when needed.
      character*1000 AttDescription

c     SatName_uc - the upper case version of the satellite name.
      character*20 SatName_uc

      character*200 Title
      character*67 Source
      character*218 Comments

c     Filenames

      character*319 FileNameOut

c     netCDF variables

      integer Dummy

      integer ncOutID, nxDimIDin, nyDimIDin, nxDimIDout, nyDimIDout
      integer sstIDout, SecondsSinceID
      integer ClearIDout, FrontsIDout, DayOrNightIDout

      integer sstDimsOut(2) 

      integer Values(8)
      character*5 Zone
      character*10 Time
      character*8 Date
      character*19 ProcessingDateTime
c      character*3000 SummaryPart1
      
      character*20 OffsetChar, ScaleInputChar, RangeChar

      character*16 Conventions
      character*64 Meta_Conventions
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

      character*32 time_coverage_start
      real*4 time_coverage_duration

      integer i

      logical exis

c---------------------------------start here --------------------------

c***********************************************************************
c**** Define lobal attributes particular to this archive, but don't 
c     write them out yet. Will do this in the section below that does 
c     not change from archive-to-archive.
c     
c**** This section will change from archive-to-archive.

      if(Debug .eq. 1) print *, 'CreateMedian: #000. FileNameOut::', 
     1     trim(FileNameOut)

c     AMSR-E or TMI

      SatName_uc = trim(SatName)
      call upper_case(SatName_uc)

      Loc1 = index( SatName, 'amsr')
      
      Title = BlankFill(:len(Title))
      Title  = 'Conditioned ' // trim(SatName_uc) // ' '
     1     // trim(ParameterName) // ' fields from Remote Sensing '
     2     // 'Systems Inc.'

      if(Debug .eq. 1) print *, 'CreateMedian: #100: SatName::', 
     1     trim(SatName), '::, SatName_uc::', trim(SatName_uc),
     2     '::, Title::', trim(Title)

c     Build the summary for this data set in 3 parts. Three parts are
c     needed because of formatting required for the variables in the 
c     middle part.

      SummaryPart1 = BlankFill(1:len(SummaryPart1))
      SummaryPart1 = 'The field in this file was generated from the '
     1     // trim(SatName_uc) // ' ' // trim(ParameterName) 
     2     // ' field obtained from Remote Sensing Systems, Inc. '
     3     // trim(ParameterName) // ' values were '
     8     // 'scaled as follows: Counts_out = nint( Counts_in * '

      Source = BlankFill(:len(Source))
      Source = 'Satellite observation' 

      creator_name = BlankFill(:len(creator_name))
      creator_name = 'Peter Cornillon'

      creator_url = BlankFill(:len(creator_url))
      creator_url = 'http://www.sstfronts.org'

      creator_email = BlankFill(:len(creator_email))
      creator_email = 'pcornillon@me.com'

      institution = BlankFill(:len(institution))
      institution = 'University of Rhode Island - ' //
     1     'Graduate School of Oceanography'

      project = BlankFill(:len(project))
      project = 'SST Fronts'

      acknowledgement = BlankFill(:len(acknowledgement))
      acknowledgement = 'The Graduate School of Oceanography ' //
     1     'generated this data file with support from the NASA ' //
     2     'MEaSUREs program (Grant #NNX09AD81G), from NSF ' //
     3     '(OCE-1060397) and from the State of Rhode Island and ' //
     4     'Providence Plantations. Please acknowledge the use of ' //
     5     'these data with the following: {These data, were ' //
     6     'produced by the University of Rhode Island with ' //
     7     'sponsorship from NASA, NSF, and the State of Rhode ' //
     8     'Island and Providence Plantations from SST fields ' //
     9     'produced by Remote Sensing Systems and sponsored ' //
     1     'by the NASA Earth Science MEaSUREs DISCOVER Project ' //
     2     'and the NASA AMSR-E Science Team. The AMSR-E SST ' //
     3     'fields are available at www.remss.com.} See the license ' //
     5     'attribute for information regarding use and ' //
     6     're-distribution of these data.'

      license = BlankFill(:len(license))
      license = 'The fields associated with the AMSR-E data are '
     1     // 'publicly available from Remote Sensing Systems, Inc at: '
     2     // 'ftp.discover-earth.org/sst/misst/' // trim(SatName) 
     3     // '_v07. Please acknowledge use of these data with the '
     4     // 'text given in the acknowledgement attribute.'


      contributor_name = BlankFill(:len(contributor_name))
      contributor_name = 'Chelle Gentemann at Remote Sensing ' //
     1     'Systems, Inc: gentemann@remss.com'

      contributor_role = BlankFill(:len(contributor_role))
      if(Loc1 .eq. 1) then
         contributor_role = 'Generated SST fields from the AMSR-E ' //
     1        'data stream.'
      else
         contributor_role = 'Generated SST fields from the TMI ' //
     1        'data stream.'
      endif

      publisher_name = BlankFill(:len(publisher_name))
      publisher_name = 'SST Fronts'

      publisher_url = BlankFill(:len(publisher_url))
      publisher_url = 'http://www.sstfronts.org'

      publisher_institution = BlankFill(:len(publisher_institution))
      publisher_institution = 'University of Rhode Island - ' //
     1     'Graduate School of Oceanography'

      cdm_data_type = BlankFill(:len(cdm_data_type))
      cdm_data_type = 'Grid'

      ProgVersionNo = AppQual_l3_AMSRE_v7_VersionNo

      call CreateOutputSSTFile( FileNameOut, ncOutID, ProgName, 
     1     nxDimIDout, nyDimIDout, sstDimsOut, sstIDout, ClearIDout, 
     2     FrontsIDout, DayOrNightIDout, SecondsSinceID, Title, Source, 
     3     creator_name, creator_url, creator_email,
     4     license, contributor_name, contributor_role, publisher_name, 
     5     publisher_url, publisher_institution, cdm_data_type)

      return

      end subroutine AMSRE_v7_l3_CreateOutputSSTFile

c***********************************************************************
      subroutine AMSRE_v7_l3_GetGeoData( FileNameIn, LatIn, LonIn)
c***********************************************************************
c     
c     This subroutine will generate lat and lon vectors and then build
c     the lat, lon arrays for these data.

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
      
c     General variables

      character*319 FileNameIn, TempFile

c     LatIDin - netCDF identifier for the latitude variable.
      integer LatIDin
c     LatIn - latitude at each pixel locaion, read in.
      real*4 LatIn(LenX,LenY)
c     LatInVec - latitudes for the array.
      real, allocatable :: LatInVec(:)
c     LonIDin - netCDF identifier for the longitude variable.
      integer LonIDin
c     LonIn - longitude at each pixel locaion, read in.
      real*4 LonIn(LenX,LenY)
c     LonInVec - longitudes for the array.
      real, allocatable :: LonInVec(:)

      integer ncInID, ncOutID, LatIDout, LonIDout

c     nxDimID - netCDF identifier for the x dimension variable
      integer nxDimID
c     nxN - length of the x dimension
      integer nxN
c     nyDimID - netCDF identifier for the y dimension variable
      integer nyDimID
c     nyN - length of the y dimension
      integer nyN

      real LatScaleFactor, LatOffset, LatFillValue
      real LonScaleFactor, LonOffset, LonFillValue

      integer*2 ix, jy

      logical Compressed
      
c     include 'netcdf.inc'
      
c----------------------------------start here -------------------------

c***********************************************************************
c**** Get the lat and lon from some form of input.
c
c**** This section will change from archive-to-archive.

c     Allocate arrays first.

      allocate(LatInVec(LenY),
     1     LonInVec(LenX),
     2     stat=ierr)

c     The scale factor and offset is not set in the input Pathfinder
c      files, so set it here.

      LatScaleFactor = 1.0
      LonScaleFactor = 1.0

      LatOffset = 0.0
      LonOffset = 0.0

c     Generate lat and lon vectors

      if(debug .eq. 1) print *, 'GetGeoData #101:'

      do 8010 jy=1,LenY
c         LatInVec(jy) = (jy - LenY / 2) * 0.25 - 0.125
         LatInVec(jy) = (jy - 1) * 0.25 - 89.875
 8010 continue

      do 8020 ix=1,LenX
c         LonInVec(ix) = (ix - LenX / 2) * 0.25 - 0.125
         LonInVec(ix) = (ix - 1) * 0.25 + 0.125
 8020 continue

c     Now load the lat and lon 2-d arrays from the lat, lon vectors.

      do 8021 jy=1,LenY
         do 8022 ix=1,LenX
            LonIn(ix,jy) = LonInVec(ix)
 8022    continue
 8021 continue

      do 8023 ix=1,LenX
         do 8024 jy=1,LenY
            LatIn(ix,jy) = LatInVec(jy)
 8024    continue
 8023 continue

      if(debug .eq. 1) print *, 'GetGeoData: #999'

      end subroutine AMSRE_v7_l3_GetGeoData
