c***********************************************************************
c     Subroutines and functions in this package:
c     
c     function CommonAppQualSubsVersionNo()
c     The above function was replaced by code in the subroutine below.
c
c     subroutine CMS_GetDateTime(ImageName, Year, Month, 
c     1     YearT, MonthT, Day, Hour, Minute, Second)
c     
c     subroutine CMS_GetSSTOutData( FileNameIn, ConditionedSST, 
c     1     SecondsSince1970)
c     
c     subroutine CMS_CreateOutputSSTFile( FileNameOut, ncOutID, 
c     1     sstIDout, ClearIDout, FrontsIDout, DayOrNightIDout,
c     2     SecondsSinceID, Start, YearC, MonthC, DayC)
c     
c      subroutine CMS_GetGeoData( FileNameIn, LatIn, LonIn, 
c     1     SecondsSince1970, FileNameOut, NumClear, YearC, MonthC, Day 
c     2     Hour, Minute, Second, GeoNameFull)
c     
c***********************************************************************
c     Modifications 
c     
c     1.00
c     
c     5/27/16 - PCC - Created from AppQual_CMS_Subroutines.
c
c     Version 1.00 ==> 1.01
c
c     6/2/16 - PCC - fixed read for longitude. Should not impact 
c      processing of SST.
c
c     
c***********************************************************************
      subroutine CMS_GetDateTime( ImageName, Year, Month, 
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

      CommonAppQualSubsVersionNo = '1.01'

c     Get year, month, day.
c   sst_meteosat08_20050303_1500.nc
c   sst_goes12_20050201_1100.nc
c   123456789-123456789-123456789-

c     Found a file. Get year, month, day, hour and minute. Assign 
c      seconds to 0.

      Loc1 = index(ImageName, 'goes')
      if(Loc1 .gt. 1) then
         read( ImageName(12:15), '(i4)') YearT
         read( ImageName(16:17), '(i2)') MonthT
         read( ImageName(18:19), '(i2)') Day
         read( ImageName(21:22), '(i2)') Hour
         read( ImageName(23:24), '(i2)') Minute
      else
         read( ImageName(16:19), '(i4)') YearT
         read( ImageName(20:21), '(i2)') MonthT
         read( ImageName(22:23), '(i2)') Day
         read( ImageName(25:26), '(i2)') Hour
         read( ImageName(27:28), '(i2)') Minute
      endif
               
      Second = 0

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

c     Set the flag for day or night to 0. Since the images span large
c     intervals in time, whether a given pixel is day or night will be
c     determined from the images with solar zenith angle.

      iDayNight = 0;

      if(Debug .eq. 1) print *, 'GetTimes #999'

      end subroutine CMS_GetDateTime

c***********************************************************************
      subroutine CMS_GetSSTData( FileNameIn, Start, CountToRead, Stride,
     1     sstOut)
c***********************************************************************
c     
c     This subroutine will get the input SST data for MSG. 
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
c     ix, jy - do loop parameters
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

c     ConditionedSST - SST after subtracting Offset and truncating.
      integer*2, allocatable :: ConditionedSST(:,:)

c     ncInID - ID for netCDF file
      integer ncInID
c     nxDimID, nyDimID - netCDF ID for the x, y dimension
      integer nxDimID, nyDimID
c     nxN, nyN - x, y dimension size for SST according to the file. Only
c       used to test input from run_config file.
      integer nxN, nyN

c     sstIDin - netCDF ID for this variable.
      integer sstIDin
c     sstIn - SST read in.
      real*4, allocatable :: sstIn(:,:)
c     sstOut - SST written out.
      integer*2  sstOut(LenX, LenY) 
c     sstTemp - Temporary SST value, read in as integer.
      integer*2, allocatable :: sstTemp(:,:) 
c     sstQualIDin - an array of quality values corresponding to SST.
      integer sstIDout
c     sstQualIDIn - the netCDF identifier for the quality variable.
      integer sstQualIDin
c     SSTQualIn - pixel quality.
      byte, allocatable :: sstQualIn(:,:)

     
      integer*2 MinSST, MaxSST, NumberTooSmall, NumberTooLarge

      integer*4 ix, jy

c----------------------------------start here --------------------------
c     

      if(Debug .eq. 1) print *, 'CMS_GetSSTData #000: ',
     2     'Start: ', Start, ' LenX, LenY: ', LenX, LenY 

      if(Debug .eq. 1) print *, 'CMS_GetSSTData #001: ',
     1     'trim(FileNameIn)::', trim(FileNameIn), ':: and ',
     2     'Start, CountToRead: ', Start, CountToRead,
     3     ' LenX, LenY: ', LenX, LenY 

c     Allocate input and output variables.

      allocate( sstIn(LenX,LenY),
     1     sstTemp(LenX,LenY),
     2     sstQualIn(LenX,LenY),
     3     ConditionedSST(LenX,LenY),
     *     stat=ierr)
      
      if (ierr .ne. 0) then
         write(UnitLog,*) 'CMS_GetSSTData #130: Allocation error, ',
     1        'exiting.'
         stop 'CMS_GetSSTData #131: Allocation error, exiting.'
      endif

      if(debug .eq. 1) print *, 'CMS_GetSSTData #002: Start, ',
     1     'CountToRead', Start, CountToRead

c     Open the input file

      status = nf90_open( FileNameIn, nf90_nowrite, ncInID)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Get the input variable IDs

      status = nf90_inq_varid( ncInID, 'sst', sstIDin)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Check that the dimensions of the sst variable agrees with LenX
c      and LenY.

      status = nf90_inq_dimid( ncInID, "nx", nxDimID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_inq_dimid( ncInID, "ny", nyDimID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_inquire_dimension( ncInID, nxDimID, len = nxN)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_inquire_dimension( ncInID, nyDimID, len = nyN)
      if(status .ne. nf90_noerr) call handle_err(status)

      if( (nxN .ne. LenX) .or. (nyN .ne. LenY) ) then
         write(UnitLog,*) 'WriteSSTData #115: Dimensions in netCDF ',
     1        'file do not match LenX and LenY.'
         stop 'WriteSSTData #115: Problem with dimensions in input file'
      endif

      if(debug .eq. 1) print *, 'CMS_GetSSTData #003: Start, ',
     1     'CountToRead', Start, CountToRead

c     Get the input fillvalue, offset and scale factor for the SST field

      status = nf90_get_att( ncInID, sstIDin, '_FillValue',
     1     SSTFillValueIn)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'CMS_GetSSTData #142: SSTFillValueIn: ',
     1     SSTFillValueIn

      status = nf90_get_att( ncInID, sstIDin, 'scale_factor',
     1     SSTScaleFactorIn)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_att( ncInID, sstIDin, 'add_offset',
     1     SSTOffsetIn)
      if(status .ne. nf90_noerr) call handle_err(status)

c      Test offset and scale factor read in to make sure that they are
c      the same as those input in the run control file.

      if(SSTScaleFactorIn .ne. InputScaleFactor) then
         print *, 'CMS_GetSSTData #145: Problem with InputscaleFactor ',
     1        'InputscaleFactor, SSTScaleFactorIn: ', 
     2        InputscaleFactor, SSTScaleFactorIn, ' not equal.'
         stop
      endif
      
      if(SSTOffsetIn .ne. InputOffset) then
         print *, 'CMS_GetSSTData #145: Problem with InputOffset ',
     1        'InputOffset, SSTOffsetIn: ', 
     2        InputOffset, SSTOffsetIn, ' not equal.'
         stop
      endif

c     Read the data

      if(debug .eq. 1) print *, 'CMS_GetSSTData #150: Start, ',
     1     'CountToRead', Start, CountToRead

      status = nf90_get_var( ncInID, sstIDin, sstTemp,
     1     Start, CountToRead, Stride)

      If(status .ne. nf90_noerr) call handle_err(status)

c     Get quality flags.
      
      status = nf90_inq_varid( ncInID, 
     1     'sst_confidence_level', sstQualIDin)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_var( ncInID, sstQualIDin, sstQualIn, 
     1     Start, CountToRead, Stride)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Convert input SST from integer*2 to realand apply the quality flag
      
      do 100 jy=1,LenY
         do 110 ix=1,LenX
            if( (sstQualIn(ix,jy) .ge. 2) .and. 
     1           (sstTemp(ix,jy) .ne. SSTFillValueIn) ) then
               sstIn(ix,jy) = sstTemp(ix,jy)
            else
               sstIn(ix,jy) = FillValueReal
            endif

 110    continue
 100  continue

c     Now condition SST data. ******************************************

      call ConditionInput( sstIn, ConditionedSST)

c     Uncomment the next line and comment out the call to Median below
c     if you want to process without median filtering first.
      
c     call ConditionInput( sstIn, SSTOut)
      
c     Next Median filter the conditioned data.

      call median( ConditionedSST, SSTOut, 3)

      if(Debug .eq. 1) print *, 'CMS_GetSSTData #999: ',
     1     'Median SST generated. Closing input file.'

c     Now clean up.

      status = nf90_close(ncInID)
      if(status .ne. nf90_noerr) call handle_err(status)

      end subroutine CMS_GetSSTData

c**********************************************************************
      subroutine CMS_CreateOutputSSTFile( FileNameOut, ncOutID, 
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

c     MSG OR GOES

      SatName_uc = trim(SatName)
      call upper_case(SatName_uc)
      
      Title = BlankFill(:len(Title))
      Title  = 'Conditioned ' // trim(SatName_uc) // ' '
     1     // trim(ParameterName) // ' fields from the Centre de '
     2     // 'Meteorologie Spatiale (CMS) of MeteoFrance.'

      if(Debug .eq. 1) print *, 'CreateMedian: #100: SatName::', 
     1     trim(SatName), '::, SatName_uc::', trim(SatName_uc),
     2     '::, Title::', trim(Title)

c     Build the summary for this data set in 3 parts. Three parts are
c     needed because of formatting required for the variables in the 
c     middle part. SummaryPart2 and SummaryPart3 are built in the main
c     program (AppQualMed_Main-n.nn.f)

      SummaryPart1 = BlankFill(1:len(SummaryPart1))
      SummaryPart1 = 'The field in this file was generated from the '
     1     // trim(SatName_uc) // ' ' // trim(ParameterName) 
     2     // ' field obtained the Centre de Meteorologie Spatiale '
     3     // '(CMS) of MeteoFrance. ' // trim(ParameterName)
     4     // ' values for which the sst_confidence_level was greater '
     5     // 'than or equal to 2 were scaled as follows: Counts_out '
     6     // '= nint( Counts_in * '

c     % Next do source, creator_name,...
      
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
      acknowledgement = 'The Graduate School of Oceanography '
     1     // 'generated this file with support from the NASA MEaSUREs '
     2     // 'program (Grant #NNX09AD81G), from NSF (OCE-1060397) and '
     3     // 'from MeteoFrance. Please acknowledge the use of these '
     4     // 'data with the following: {The ' // trim(ParameterName)
     5     // ' front and gradient data, based on ' 
     6     // trim(ParameterName) // ' data from the EUMETSAT OSI SAF, '
     7     // 'were produced by the University of Rhode Island '
     8     // 'and Meteo-France with sponsorship from NASA, NSF, the '
     9     // 'State of Rhode Island and Meteo-France.} See the '
     1     // 'license attribute for information regarding use and '
     2     // 're-distribution of these data.'

      license = BlankFill(:len(license))
      license = 'The conditioned ' // trim(ParameterName)  
     1     // ' fields, fields ending in ''_Median.nc'', were not '
     2     // 'publicly available when the file was created in June '
     3     // 'of 2016. All other products derived from '
     4     // 'these fields using the URI edge/gradient workflow are. '
     5     // 'Please acknowledge use of these data with the text '
     6     // 'given in the acknowledgement attribute.'

      contributor_name = BlankFill(:len(contributor_name))
      contributor_name = 'EUMETSAT OSI SAF' //
     1     'http://www.eumetsat.int/Home/Main/Satellites/' //
     2     'GroundNetwork/ApplicationGroundSegment/SAFs/' //
     3     'SAFProjects/SP_2010053117495263?l=en'

      contributor_role = BlankFill(:len(contributor_role))
      contributor_role = 'Generated ' // trim(ParameterName) //
     1     ' fields from the ' // trim(SatName_uc) // ' data stream.'

      publisher_name = BlankFill(:len(publisher_name))
      publisher_name = 'SST Fronts'

      publisher_url = BlankFill(:len(publisher_url))
      publisher_url = 'http://www.sstfronts.org'

      publisher_institution = BlankFill(:len(publisher_institution))
      publisher_institution = 'University of Rhode Island - ' //
     1     'Graduate School of Oceanography'

      cdm_data_type = BlankFill(:len(cdm_data_type))
      cdm_data_type = 'Grid'

      ProgVersionNo = AppQual_CMS_VersionNo

      call CreateOutputSSTFile( FileNameOut, ncOutID, ProgName, 
     1     nxDimIDout, nyDimIDout, sstDimsOut, sstIDout, ClearIDout, 
     2     FrontsIDout, DayOrNightIDout, SecondsSinceID, Title, Source, 
     3     creator_name, creator_url, creator_email,
     4     license, contributor_name, contributor_role, publisher_name, 
     5     publisher_url, publisher_institution, cdm_data_type)

      return

      end subroutine CMS_CreateOutputSSTFile

c***********************************************************************
      subroutine CMS_GetGeoData( Start, CountToRead, Stride,
     1     LatIn, LonIn)
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
c     ix, jy - do loop parameters
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
c     LatLonTemp - temporary variable read in.
      integer*4, allocatable :: LatLonTemp(:,:)
c     LonIDin - netCDF identifier for the longitude variable.
      integer LonIDin
c     LonIn - longitude at each pixel locaion, read in.
      real*4 LonIn(LenX,LenY)

c     ncInID - netCDF identifier for the input file.
      integer ncInID

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

c     Allocate the temporary input variable.

      allocate( LatLonTemp(LenX,LenY), stat=ierr)
      
c     Build the filename with lat,lon information, assuming that it is
c     in BaseDir/SupportingFiles/ and is called meteosat_latlon.nc

      TempFile = trim(BaseDir) // 'SupportingFiles/meteosat_latLon.nc'
      
      if(debug .eq. 1) print *, 'CMS_GetGeoData #101: TempFile ::'
     1     // trim(TempFile) // '::'

c     Open the input file first.

      status = nf90_open( TempFile, nf90_nowrite, ncInID)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Get the input variable IDs

      status = nf90_inq_varid( ncInID, 'lat', LatIDin)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_inq_varid( ncInID, 'lon', LonIDin)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Get the scale factor, offset and missing values for the input

      status = nf90_get_att( ncInID, LatIDin, 'scale_factor',
     1     LatScaleFactor)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_att( ncInID, LonIDin, 'scale_factor',
     1     LonScaleFactor)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_att( ncInID, LatIDin, 'add_offset',
     1     LatOffset)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_att( ncInID, LonIDin, 'add_offset',
     1     LonOffset)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_att( ncInID, LatIDin, '_FillValue',
     1     LatFillValue)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_att( ncInID, LonIDin, '_FillValue',
     1     LonFillValue)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(debug .eq. 1) print *, 'WriteGeoData #120: Scale factors, ',
     1     'offset and missing values: ', LatScaleFactor,
     2     LonScaleFactor, LatOffset, LonOffset, LatFillValue, 
     3     LonFillValue

c     Read the data.

      status = nf90_get_var( ncInID, LatIDin, LatLonTemp,
     1     Start, CountToRead, Stride)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Scale the input lat data

      do ix=1,LenX
         do jy=1,LenY
            if( LatLonTemp(ix,jy) .eq. LatFillValue) then
               LatIn(ix,jy) = FillValueReal
            else
               LatIn(ix,jy) = LatLonTemp(ix,jy) * LatScaleFactor +
     1              LatOffset
             endif
          enddo
      enddo

      if(debug .eq. 1) print *, 'WriteGeoData #120: Done withLatitude.'
      
c     Repeat for longitude
      
      status = nf90_get_var( ncInID, LonIDin, LatLonTemp, 
     1     Start, CountToRead, Stride)
      if(status .ne. nf90_noerr) call handle_err(status)

      do ix=1,LenX
         do jy=1,LenY
            if( LatLonTemp(ix,jy) .eq. LonFillValue) then
               LonIn(ix,jy) = FillValueReal
            else
               LonIn(ix,jy) = LatLonTemp(ix,jy) * LonScaleFactor +
     1              LonOffset
             endif
          enddo
      enddo

      if(debug .eq. 1) print *, 'WriteGeoData #110 ',
     1        ' LatIDin, LonIDin: ', LatIDin, LonIDin

c     Now clean up.

      status = nf90_close(ncInID)
      if(status .ne. nf90_noerr) call handle_err(status)

c     OK, have read all of the necessary metadata, now scale the
c     the latitude and longitude.

               
      if(debug .eq. 1) print *, 'GetGeoData: #999'

      end subroutine CMS_GetGeoData
