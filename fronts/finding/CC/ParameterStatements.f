c***********************************************************************
c     
c     This file contains the explicit statements defining variable used
c     by a number of edge detection programs and subtroutine. The 
c     idea is that one adds a variable here or changes a value in a
c     parameter statment here and it will add or change the value of 
c     the variable in all programs and subroutines simultaneously. It is
c     likely that many of these will not be used in the various
c     programs/subroutines, but the cost of including them is low. This
c     file should be included at the beginning of all programs and
c     subroutines after the 'implicit none' statement and function
c     statements. 
c
c     3/24/11 - PCC - Added ChunkSize, ChunkSzLenX and ChunkSzLenY to
c     . be used to chunk with NetCDF4.
c     4/6/11 - PCC - Added DummyCharString, xType & DeflateLevel for 
c     . nf90 calls.
c
c     5/24/11 - PCC - Added LogicalTemp1 and LogicalTemp2. These 
c      variables are used as arguments in some of the calling programs.
c      The original version of the Fortran compiler that I used did not
c      require that a logical*1 variable be explicitly defined in the 
c      call; i.e., you could simply use .false. This is not true for 
c      the most recent version.
c
c     6/1/11 - PCC - Added Version number variables for AppQuals other
c      than CMS. Also changed the variable name to include archive.
c      Added Msg50 argument for some subroutines that require 50 
c      character arguments. 
c
c     6/2/11 - PCC - Added UnitTempIn.
c 
c     7/18/11 - PCC - Changed SIED from 1.10 to 1.11
c      Added check for number of pixel elements, nep, for variable ppv.
c      Abort if number exceeds MaxEdge.
c      Changed MaxEdge in ParameterStatements.f from 400,000 to 
c       1,000,000.
c
c     12/28/11 - PCC - Changed size of LocFileNamePrefix from 10 to 20 
c      to accommodate 'Pathfinder4km'.
c
c     1/2/12 - PCC - Changed BaseDir length to 319 characters.
c
c     1/11/12 - PCC - Changed the length of Command from 250 to 319; 250
c       was too short for Pathfinder4km tempfile stuff.
c
c     1/12/12 - DI - Added Archive index for hycom_1m_NonAssim field
c
c     2/3/12 - DH - Added debug_max to extend potential debugging
c
c     3/7/12 - DH - Added YearDayC to support yearday naming convention
c                   Added DegreesK to identify input temperature units
c                   as degrees Celsius or degrees Kelvin
c                   Added 'pathfinder1kmL3' to SatIDList(), and
c                   incremented ArchiveIDList(), xLenArchive(), and
c                   YearArchiveList() arrays
c
c     3/29/12 - DH - Added parameter for Modis4kmL3 Program Version
c                   Updated various Archive arrays to add Modis4kmL3.
c
c     4/28/12 - PCC - added a lot of variables for version 2.0.
c
c     6/15/12 - PCC - added AppQualMed_AMSRE_VersionNo
c     
c     9/14/12 - PCC - rearranged non-common variable alphabetically.
c      Moved iEnd and jEnd to the Debug common. Can not move iStrt and
c      jStrt there since there are defined in parameters statements and
c      can not appear in a common statement.
c
c     1/28/12 - PCC - Added statements for version 3.00 of SIED, 
c      variables for the netCDF output.
c
c     1/20/13 - PCC - Added ParameterName to SatelliteDesignation . 
c      common. Used to speficy the name of the parameter that is being
c      processed. It is read in from the runcontrol files. 
c     
c     1/23/13 - PCC - Added iVarName and jVarName to Dimensions common.
c
c     1/27/13 - PCC - Added common for fill values, scale factors and 
c       offsets for gradient fields, lat/lon fields, sst fields and
c       zenithangle fields.
c
c     1/28/13 - PCC - got rid of MedianVersionNo, no longer used.
c
c     1/29/13 - PCC - Moved SatName to ArchiveDefinition common.
c
c     3/15/13 - PCC - Removed reference to ArchiveIDList
c
c     4/2/13 - PCC - Added LSTorGMT to SatelliteDesignation common.
c
c     4/10/13 - PCC - Added SSTorChlorophyll to SatelliteDesignation 
c       common block.
C     5/15/13 - JPS - Added parameter AppQualMed_Ecco2_VersionNo
c     12/23/13 - JPS - Added AppQualMed_AVHR_VersionNo
c_______________________________________________________________________
c
c**   Variables with ** after the c are variables that are used ONLY in
c     sied subroutines. They are parameters that could be changed but
c     are generally not.
c
c--   Variables with -- after the c are the two variables are are 
c     changed for different size input images.
c     
c     All variables are in alphabetical order except Version number
c
c           123456789-123456789-123456789-123456789-123456789-
c     AppQualVersionNo - Version number for AppQual. It will be 
c     . incremented for every change of AppQual and the changes will be
c     . documented in AppQual just before the program starts.

      character*4, parameter :: AppQualMed_VersionNo = 
     1     '3.03'

      character*4, parameter :: AppQual_l2_AMSRE_v7_VersionNo = 
     1     '1.01'

      character*4, parameter :: AppQual_l3_AMSRE_v7_VersionNo = 
     1     '1.00'

      character*4, parameter ::  AppQual_l2p_AVHRR_gac_v1_VersionNo =
     1     '1.01'

      character*4, parameter ::  AppQual_ECCO2_4km_VersionNo =
     1     '1.01'
c$$$      character*4, parameter :: AppQualMed_CMS_VersionNo = 
c$$$     1     '2.13'
c$$$
c$$$      character*4, parameter :: AppQual_HYCOM_VersionNo = 
c$$$     1     '1.00'
c$$$
c$$$      character*4, parameter :: AppQualMed_ATSR_VersionNo = 
c$$$     1     '1.00'
c$$$
c$$$      character*4, parameter :: AppQualMed_AMSRE_VersionNo = 
c$$$     1     '1.61'
c$$$
c$$$      character*4, parameter :: AppQualMed_Pathfinder4km_VersionNo = 
c$$$     1     '2.00'
c$$$
c$$$      character*4, parameter :: AppQual_Pathfinder1km_L2_VersionNo =
c$$$     1     '1.13'
c$$$
c$$$      character*4, parameter :: AppQual_Pathfinder1km_L3_VersionNo =
c$$$     1     '1.04'
c$$$
c$$$      character*4, parameter :: AppQualMed_MODIS4km_VersionNo =
c$$$     1     '1.08'
c$$$
c$$$      character*4, parameter :: AppQualMed_Aquarius_VersionNo = 
c$$$     1     '1.00'
c$$$
c$$$      character*4, parameter :: AppQualMed_Ecco2_VersionNo = 
c$$$     1     '1.01'

c     GenerateZenithAngleImagesVersionNo - Version number for 
c     . GenerateZenithAngleImages. See AppQual for details.
      character*4, parameter :: GenerateZenithAngleImagesVersionNo =
     1     '2.25'

c     GeoSobelVersionNo - Version number for SoGeobel. See AppQual for 
c     . details
      character*4, parameter :: GeoSobelVersionNo = 
     1     '3.20'

c     PmergeVersionNo - Version number for PmergeThin. See AppQual
c     . for details.
      character*4, parameter :: PmergeVersionNo = 
     1     '3.05'

c     ThinVersionNo - Version number for PmergeThin. See AppQual
c     . for details.
      character*4, parameter :: ThinVersionNo = 
     1     '2.09'

c     SIEDVersionNo - Version number for SIED. See AppQual for details.
      character*4, parameter :: SIEDVersionNo = 
     1     '3.24'

c     StatsByDegreeSquare - Version number for Stats. 
      character*4, parameter :: StatsByDegreeSquareVersionNo = 
     1     '1.00'

c*********************************** Commons ****************************

c---------- Fill values read in

c     LatLonFillValueIn - missing value for latitude and longitude
c      fields written by AppQual
      integer*4 LatLonFillValueIn
c     LatLonOffsetIn - offset for latitude and longitude fields written
c      by AppQual
      real*4  LatLonOffsetIn
c     LatLonScaleFactorIn - scale factor for latitude and longitude
c      fields written by AppQual
      real*4  LatLonScaleFactorIn

c     GradFillValueIn - missing value for gradient fields written by 
c      GeoSobel
      real*4 GradFillValueIn
c     GradOffsetIn - offset for latitude and longitude fields written
c      by AppQual
      real*4  GradOffsetIn
c     GradScaleFactorIn - scale factor for gradient fields written by 
c      GeoSobel
      real*4  GradScaleFactorIn

c     sstFillValueIn - missing value for sst fields written by AppQual
      integer*2 SSTFillValueIn
c     sstOffsetIn - offset for sst fields written by AppQual
      real*4 sstOffsetIn
c     sstScaleFactorIn - scale factor for sst fields written by AppQual
      real*4 sstScaleFactorIn

c     ZenithAngleFillValueIn - missing value for zenith angle fields
c       written by GenerateZAImages.
      integer*2 ZenithAngleFillValueIn
c     ZenithAngleOffsetIn - offset for zenith angle fields
c       written by GenerateZAImages.
      real*4  ZenithAngleOffsetIn
c     ZenithAngleScaleFactorIn - scale factor for zenith angle fields
c       written by GenerateZAImages.
      real*4  ZenithAngleScaleFactorIn

      common/FillValues/ 
     1     LatLonFillValueIn, LatLonOffsetIn, LatLonScaleFactorIn, 
     2     GradFillValueIn, GradOffsetIn, GradScaleFactorIn, 
     3     ZenithAngleOffsetIn, ZenithAngleScaleFactorIn, 
     5     sstOffsetIn, SSTScaleFactorIn, SSTFillValueIn, 
     6     ZenithAngleFillValueIn

c---------- Blank array used to clear character strings.

c     BlankFill - a string of blanks to make sure that all generated 
c     . strings end in trailing blanks.
      character*8192 BlankFill
      Common/Blanks/ BlankFill

c---------- Where the data are

c     BaseDir - base directory for input and output directories of 
c     - front and gradient detector programs. 
      character*319 BaseDir

      common /DataLocation/ BaseDir

c     posEnd - position of end-of-string, [posEnd = len_trim(...)]
      integer posEnd

c     posSlash - position of slash character, [posSlash = scan(...)]
      integer posSlash

c---------- Debug

c     Debug - 1 for all debug output, 2 for timing only, 0 for none.
      integer*2 debug
c     DebugMax - for lots of printouts and fileouts in AppQualMed
      integer*2 debug_max

c     iend - the ending location in x of regions to be printed out
c     . in debug mode. 
      integer*2 iend
c     jend - the ending location in y of regions to be printed out
c     . in debug mode. 
      integer*2 jend
      common /Debug/ debug, debug_max, iEnd, jEnd

c---------- Metadata fields

c     CreatorName - name of person who ran the input through the
c     . fronts/gradient workflow
      character*160 CreatorName
c     CreatorURL - URL of creator
      character*160 CreatorURL
c     CreatoreMail - eMail address of creator
      character*160 CreatoreMail
c     Institution - Institution of creator - usually URI.
      character*160 Institution
c     Project - Name of project for which the workflow was run.
      character*160 Project
c     Reference - Reference for original fields input to AppQual
      character*320 Reference
c     Acknowledgement - Funding source for the production of this field.
      character*1024 Acknowledgement
c     ContributorName - Source of original data
      character*160 ContributorName
c     ContributorRole - What the contributor actually did
      character*1024 ContributorRole
c     PublisherName - Using 'SST Front's here. Not sure what it should b
      character*160 PublisherName
c     PublisherURL - Using sstfronts.net, again ot sure about this one.
      character*160 PublisherURL
c     PublisherInstitution - Another one I'm not sure about. Using
c     . URI/GSO.
      character*160 PublisherInstitution
c     cdmDataType - the type of data. Using 'Grid'.
      character*160 cdmDataType 
c     GeospatialLatMin - miniumum latitude of array. Determined from the
c     . data
      real GeospatialLatMin
c     GeospatialLatMax - maximum latitude of array. Determined from the
c     . data
      real GeospatialLatMax
c     GeospatialLatRes - latitude spatial resolution.  Determined from
c     . the data
      real GeospatialLatRes
c     GeospatialLonMin - miniumum longitude of array. Determined from
c     . the data
c     GeospatialLonMin - miniumum longitude of array. Determined from 
c     . the data
      real GeospatialLonMin
c     GeospatialLonMax - maximum longitude of array. Determined from the
c     . data
      real GeospatialLonMax
c     GeospatialLonRes - longitude spatial resolution. Determined
c     . from the data
      real GeospatialLonRes
c     GeospatialVerticalMin - minumum longitude of array. Determined 
c     . from the data
      real GeospatialVerticalMin
c     GeospatialVerticalMax - maximum longitude of array. Determined 
c     . from the data
      real GeospatialVerticalMax
c     GeospatialVerticalUnits - units of the vertical coordinate
      character*20 GeospatialVerticalUnits
c     GeospatialVerticalResolution - maximum longitude of array.  
c     . Determined from the data
      real GeospatialVerticalResolution
c     GeospatialVerticalPositive - direction of the vertical coordinate
      character*4 GeospatialVerticalPositive
c     SingleLocFile - .true. if all input files have the same
c     . geolocation file.
      logical SingleLocFile
c     Summary - the summary description of the contents of a netCDF
c     . file.
      character*4000 Summary
c     SummaryPart1, 2 and 3 - parts of the summary used to build up 
c     . Summary
      character*3000 SummaryPart1, SummaryPart2, SummaryPart3
c     TimeCoverageStart - start time for data used in this field. A 
c     . string yyyymmddhhmmss
      character*16 TimeCoverageStart
c     TimeCoverageDuration - duration of coverage in seconds.
      real TimeCoverageDuration

      common /Metadata/ CreatorName, CreatorURL, CreatoreMail,
     1 Institution, Project, Reference, Acknowledgement,
     2 ContributorName, ContributorRole, 
     3 PublisherName, PublisherURL, PublisherInstitution,
     4 cdmDataType,
     5 GeospatialLatMin, GeospatialLatMax, 
     6 GeospatialLonMin, GeospatialLonMax,
     7 GeospatialVerticalMin, GeospatialVerticalMax, 
     8 GeospatialVerticalUnits, GeospatialVerticalResolution,
     9 GeospatialVerticalPositive, SingleLocFile,
     1 TimeCoverageStart, TimeCoverageDuration,
     2 Summary, SummaryPart1, SummaryPart2, SummaryPart3


c---------- In/out unit numbers

c**   UnitCnt1 - unit for inventory file
      integer*2 UnitCnt
c**   UnitDim1 - unit for inventory file. If  equal to 99, this unit
c     . will not be opened; i.e., only a cnt file will be written.
      integer*2 UnitDim
c**   UnitGeoMetadata - unit for geophysical metadata file. 
      integer*2 UnitGeoMetadata
c     UnitInput - unit for input data, 5 if reading from the terminal,
c     .otherwise use 10
      integer*2 UnitInput
c     UnitInventory - unit for inventory file
      integer*2 UnitInventory
c     UnitLog - unit number to which run history will be written
      integer*2 UnitLog
c     UnitPeaks - the output unit for the peaks and troughs. This
c     . will be reused for all Peaks files in that the files are
c     . closed after writing, so there should be no collisions.
      integer*2 UnitPeaks
c     UnitSumsDay - unit for daily sums of files
      integer*2 UnitSumsDay
c     UnitSumsMonth - unit for monthly sums of files
      integer*2 UnitSumsMonth
c     UnitSumYear - unit for yearly sums of files.
      integer*2 UnitSumsYear
c     UnitTempIn - Temporay unit number used for the list of input files
c      read by AppQual_Pathfinder4km and maybe other programs.
      integer*2 UnitTempIn
c     UnitInputSST - Unit for input files opened by Fortran as opposed
c      to NetCDF
      integer*2 UnitInputSST

      common /Units/ UnitCnt, UnitGeoMetadata, UnitDim, UnitInput, 
     1     UnitInventory, UnitLog, UnitPeaks, UnitSumsDay, 
     2      UnitSumsMonth, UnitSumsYear, UnitTempIn, UnitInputSST

c---------- Time range

c     SecondsSinceEnd - the number of hours since 00:00 of 1 January 
c     . 1900 corresponding to the ending time of this run.
      real*8 SecondsSinceEnd
c     SecondsSinceStart - the number of hours since 00:00 of 1 January 
c     . 1900 corresponding to the starting time of this run.
      real*8 SecondsSinceStart

      common /TimeRange/ SecondsSinceStart, SecondsSinceEnd

c---------- Array sizes - Meteosat are 3712x3712 and goes are 1952x2400

c     AlongScanDimension - 1 if the 'x' dimension is parallel to scan
c      lines (or eastward in a level 3 file) and 2 if the 'y' dimension
c      is parallel to scan lines. The other dimension is referred to as
c      the cross-scan dimension. This variable is used as a flg when
c      assigning attributes in the output netCDF files.
      integer AlongScanDimension

c     iVarName - the name of the 1st dimension, either 'along-scan'
c      or 'cross-scan', although there is nothing that precludes it
c      being something different. These variables are defined in 
c      ReadInputs based on the value of AlongScanDimension.
      character*20 iVarName

c     jVarName - the name of the 2nd dimension. See iVarName
      character*20 jVarName

c     LenX - the x dimension of the input and many output images
      integer LenX
c     LenXA - the augmented x dimension of the output arrays to be
c      a multiple of 32: LenXA = (int(LenX / 32) + 1) * 32. Only
c      used in SIED.
      integer LenXA
c     LenXin - the x dimension of the original NetCDF file.
      integer LenXin
c     LenY - the y dimensions of the input and many output images
      integer LenY
c     LenYA - the augmented y dimension of the output arrays to be
c      a multiple of 32: LenYA = (int(LenY / 32) + 1) * 32. Only
c      used in SIED.
c     LenYin - the y dimension of the original NetCDF file.
      integer LenYin
      integer LenYA

c     PlaneFitSize - the size of the region to use when removing a
c      plane from SIED histogram windows if a plane is to be removed.
      integer PlaneFitSize
c     PlaneFitSizeChar - the character equivalent of PlaneFitSize to 
c      write out to the output netCDF file.
      character*8 PlaneFitSizeChar

c     WinSize - the size of the windows used in SIED
      integer WinSize
c     WinSizeChar - the character equivalent of window size to write out
c     . to the output netCDF file.
      character*8 WinSizeChar

      common /Dimensions/ LenX, LenY, LenXA, LenYA, LenXin, LenYin,
     1     AlongScanDimension, iVarName, jVarName, WinSize, WinSizeChar,
     2     PlaneFitSize, PlaneFitSizeChar

c---------- Cross-front extnet

c     CFPixels - CFPixels to transfer in subroutine argument, the 
c      compiler doesn't seem to like to pass parameters.
      integer CFPixels

      common/CF/ CFPixels

c---------- Maximum array sizes for edges, segments and peaks.

c     MaxEdge - the maximum number of edge pixels in an image.
      integer*4 MaxEdge
c     MaxNumberOfFiles - the maximum number of entries allowed in the
c     . inventory.
      integer MaxNumberOfFiles
c     MaxNumOfSegments - the maximum number of frontal segments in an 
c     . image.
      integer MaxNumOfSegments

      common /MaxNumbers/  MaxEdge, MaxNumberOfFiles, MaxNumOfSegments

c---------- Conditioning the input

c     InputScaleFactor - SST_in = Counts * InputScaleFactor + 
c      InputOffset. Called 'scale_factor' in CF-1.5.defined above. 
      real InputScaleFactor
c     InputOffset - defined above. Called add_offset in CF-1.5.
      real InputOffset
c     DegreesK - 0 if the input data are degrees Celsius, 1 if Kelvin.
      integer DegreesK
c     OutputScaleFactor - SST_out = Counts * OutputScaleFactor +
c      OutputOffset. Called 'scale_factor' in CF-1.5.
      real OutputScaleFactor
c     OutputOffset - defined above. This is also the smallest allowed
c      SST value in Kelvin on output. Input SST values that are smaller 
c      than this are set to _FillValue. This is the required by the
c      histogramming algorithm used by SIED which requires that numbers
c      go from 0 digital counts to an upper limit of Range. Called 
c      add_offset in CF-1.5.
      real OutputOffset
c     MaxSSTOut - maximum allowed SST value on output. This number,
c      together with OutputOffset and OutputScaleFactor is used to
c      calculate Range, used to limit the number of digital counts.
      real MaxSSTOut
c     Range - the range of digital counts used for SST on output. 
c      The range is limited because SIED uses a histogram algorithm that
c      operates on a fixed range of values from 0 to a maximum value:
c      Range. I generally use 400 to cover the range of SST values
c      in Kelvin at 0.1K steps. 
      integer Range
c     ScaleInput - the factor that converts from the input scale factor
c      to the output scale factor. Used in conversion of input counts to
c      output counts. SST_Counts_out = ScaleInput * SST_Counts_in + 
c       InputOffset2OutputOffset.
      real ScaleInput
c     InputOffset2OutputOffset - the offset scale factor used to convert
c      input counts to output counts.  If SST_In is in degrees Celsius,
c      then add 273.15 ot InputOffset.
      real InputOffset2OutputOffset

      common /ConditionParameters/ InputScaleFactor, InputOffset,
     1     DegreesK, OutputScaleFactor, OutputOffset, MaxSSTOut,
     2     Range, ScaleInput, InputOffset2OutputOffset

c---------- Which satellite, variable and what units? ------------------

c     ParameterName - Name of variable that is being analyzed. This is 
c      used to define metadata in the netCDF files.
      character*30 ParameterName
c     ParameterName_lc - lowercase version of ParameterName.
c      used to define metadata in the netCDF files.
      character*30 ParameterName_lc
c     ParameterUnits - Units for the variable that is being analyzed. 
c      This is used to define metadata in the netCDF files.
      character*30 ParameterUnits
c     SSTorChlorophyll - 0 if this is an SST run 1 if Chlorophyll
      integer SSTorChlorophyll
c     WhichSatellite - Satellite type used to make sure that we 
c     . are using the correct parameter statements.
      Character*1  WhichSatellite
c     LSTorGMT - variable indicating whether the input time is local
c      sun time, LSTorGMT= 0, or GMT, LSTorGMT=1; FillValueInt if this
c      attribute is missing in the input file.
      integer LSTorGMT

      common /SatelliteDesignation/ LSTorGMT, SSTorChlorophyll,
     1     WhichSatellite, ParameterName, ParameterName_lc, 
     2     ParameterUnits

c---------- lat, lon filename ------------------------------------------

c     GeoName - the directory above the base directory plus th
      character*319 GeoName

c     LocFileNamePrefix - prefix to use for the lat, lon location file
c     - if there is one file for the entire archive. If the prefix is
c     - empty, AppQual assumes that there is one lat, lon file per input
c     - file and it will put them in subdirectory GeoLoc under a year
c     - month file structure that is indentical to that of Original. It
c     - will use the same basic name structure for the lat, lon files as
c     - for the sst files with _SST replaced with _LOC.

      common /LatLonCommon/ GeoName, LocFileNamePrefix

c---------- Miscellaneous variables to be read in (maybe someday) ------

c     CommonAppQualSubsVersionNo - version number for AppQual special 
c      subroutines.
      character*4 CommonAppQualSubsVersionNo
c     NumClearThresh - threshold for the number of acceptable clear
c     - pixels in an image. Will not process the image if less than
c     - this number.
      integer NumClearThresh
c     NumberOfImages - number of files/images in the input archive; 
c      i.e., on ArchiveInventory.
      integer NumberOfImages

      common /Miscellaneous/ NumClearThresh, NumberOfImages,
     1     CommonAppQualSubsVersionNo

c---------- Windows for Pmerge -----------------------------------------

c     HourWindow - the window in time to use in merging contours. 
c     - Images within this time of an image of interest will be merged 
c     - in Pmerge. If no time, then this number will be set to 0
      integer HourWindow
c     ImageWindow - the number of images to use when merging contours 
c     . pmerge loops over all images on the list that are within +/- 
c     . ImageWindow of the image of interest and then tests the time of 
c     . each image, i.e., +/- SecondsWindow. If there is no time in  
c     . there input files; i.e., year, yearday, hour, minute and  
c     . second is missing the time is returned as 0, so ImageWindow 
c     . will define the number of images to use. If the time is 
c     . present in the input file and ImageWindow is sufficiently 
c     . large, then it will use SecondsWindow for the pmerge step. 
      integer*2 ImageWindow

      common /PmergeWindows/ HourWindow, ImageWindow

c***************** Archive variables  **********************************
c
c     This common statement holds all the information pertinent to the
c     . archives for which the workflow is setup. It is used primarily
c     . in CheckArchive. YOU MUST INITIATLIZE THESE VARIABLES IN
c     . COMMONSUBROUTINES.F/CHECKARCHIVE WHEN ADDING A NEW ARCHIVE!!!

c     NumberOfArchives - the number of archives for which the workflow
c     . is currently setup.
      integer NumberOfArchives
c      parameter NumberOfArchives/5/

c     SatName - either 'goes' or 'meteosat'
      character*20 SatName
c     SatNameList - the names of the archives.
      character*30 SatNameList(100)

      common /ArchiveDefinition/ NumberOfArchives, SatName, SatNameList

c***************** Front/Segment variables  ****************************
c----------------- Front segment control variables

c     MinStep - the minimum temperature step for an edge to be selected 
c     . at the window machine. This threshold can be set to 0 but 
c     . because of the digital nature of the data it should be noted 
c     . that only steps of 3 and above are significant.
      integer*2 minstep
c     MinLength - the minimum length of a frontal segment. Segments 
c     . shorter than this are eliminated.
      integer*2 minlength
c     MinClear - the minimum number of clear pixels in the 32x32 
c     - SIED window
      integer*2 MinClear
c      parameter(MinClear = 100)

c     RemovePlane - Switch to determine if a linear background gradient should be
c     - removed before SIED is run
      logical RemovePlane

      common/FrontSegmentControlVariables/ RemovePlane, MinStep, 
     1     MinLength, MinClear

c----------------- General variables.

c     dim2ID - the netCDF ID for the output 'dim' file.
      integer Dim2ID

      Common/netCDFFronts/ Dim2ID

c----------------- Front pixel variables.

c     CFBGEastGradID - netCDF ID for the background eastward Sobel
c      gradient (background is 5 or more pixels from the front.
      integer CFBGEastGradID
c     CFBGNorthGradID - netCDF ID for the background northward Sobel
c      gradient (background is 5 or more pixels from the front.
      integer CFBGNorthGradID
c     CFEastGradID - netCDF ID for the cross-front eastward Sobel grad
      integer CFEastGradID
c     CFNorthGradID - netCDF ID for the cross-front northward Sobel grad
      integer CFNorthGradID
c     CFEastGradMaxID - the maximum eastward Sobel gradient within 1 
c      pixel of the front (normal to the local front).
      integer CFEastGradMaxID
c     CFNorthGradMaxID - the maximum northward Sobel gradient within 1 
c      pixel of the front (normal to the local front).
      integer CFNorthGradMaxID
c     CFSSTDimID - netCDF ID for the cross-front SST values.
      integer CFSSTDimID
c     CFRecIndex - the cross-front record index. This index should
c      correspond to RecIndex except that the cross-front parameters
c      are determined after a front segment has between written out,
c      so this index needs to be incremented separately. BE CAREFUL.
      integer CFRecIndex
c     CFRecDimID - the cross-front record ID
      integer CFRecDimID

c     I_MaxGradMagID - netCDF ID for the location of the maximum 
c      gradient magnitude along a cross-frontal line.
      integer I_MaxGradMagID
c     iGradID - netCDF ID for the SIED type gradient of the 1st dim.
      integer iGradID

c     jGradID - netCDF ID for the SIED type gradient of the 1st dim.
      integer jGradID

c     LonRecID - the netCDF ID for longitude.
      integer lonRecID
c     LatRecID - the netCDF ID for latitude.
      integer latRecID

c     CFmID - netCDF ID for the across-scan location of the maximum
c      gradient in the cross-front array.
      integer CFmID

c     CFnID - netCDF ID for the across-scan location of the maximum
c      gradient in the cross-front array.
      integer CFnID

c     RecDimID - netCDF ID for the front pixel record number.
      integer RecDimID
c     RecIndex - the record index for the given frontal pixel.
      integer recIndex

c     SSTaID - netCDF ID for the JFs mean SST for the cold population
      integer SSTaID
c     SSTbID - netCDF ID for the JFs mean SST for the warm population
      integer SSTbID
c     SSTcID - netCDF ID for the JFs mean SST value separating the 
c      two populations
      integer SSTcID
c     SSTDiffID - netCDF ID for the temperature difference across the 
c      front. The mean temperature is determined between 1 and 8 pixels
c      from the front approximately normal to it and then differenced.
c      two populations.
      integer SSTDiffID
c     SSTCFID - netCDF ID for the cross-front sst array.
      integer SSTcfID

c     xLocID - the netCDF ID for the i location in the input SST array.
      integer xLocID

c     yLocID - the netCDF ID for the j location in the input SST array.
      integer yLocID

c     ZenithAngRecID - the netCDF ID for the solar zenith angle.
      integer zenithAngRecID

      Common/netCDFFrontPixelIDs/ CFRecDimID, CFSSTDimID, CFRecIndex, 
     1     RecDimID, CFBGEastGradID, CFBGNorthGradID, 
     2     CFEastGradID, CFNorthGradID, 
     3     CFEastGradMaxID, CFNorthGradMaxID, I_MaxGradMagID, iGradID,
     4     jGradID, LatRecID, LonRecID, CFmID, CFnID,
     5     SSTaID, SSTbID, SSTcID, SSTDiffID, SSTCFID, 
     6     xLocID, yLocID, ZenithAngRecID

c----------------- Front segment variables; referred to as segments.

c     LatMinID - netCDF ID for the minimum latitude of this segment.
      integer latMinID
c     LatMaxID - netCDF ID for the maximum latitude of this segment.
      integer latMaxID
c     LonMinID - netCDF ID for the minimum longitude of this segment.
      integer lonMinID
c     LonMxaID - netCDF ID for the maximum longitude of this segment.
      integer lonMaxID

c     SegDimID - netCDF ID for the front segment.
      integer SegDimID
c     SegmentIndex - the index for a front segment record number.
      integer SegmentIndex
c     SegLengthID - netCDF ID for the number of pixels in a 
c      segment.
      integer SegLengthID
c     SegStartID - netCDF ID for the first pixel in this segment.
      integer SegStartID

c     xMinID - netCDF ID for the minimum x location for the segment.
      integer xMinID
c     xMxaID - netCDF ID for the maximum x location for the segment.
      integer xMaxID

c     yMinID - netCDF ID for the minimum y location for the segment.
      integer yMinID
c     yMaxID - netCDF ID for the maximum y location for the segment.
      integer yMaxID

      Common/netCDFFrontSegmentIDs/ SegDimID,
     1     latMinID, latMaxID, lonMinID, lonMaxID, 
     2     SegLengthID, SegStartID 

c-------------- Window variables 

c     WinNumClearID - netCDF ID for the number of clear pixels in each
c     . SIED window.
      integer WinNumClearID
c     WinClearCohesionID - netCDF ID for the cohesion of clear
c     . pixels in each SIED window.
      integer WinClearCohesionID
      
      common/CohesionIDs/ WinNumClearID, WinClearCohesionID

c---------------- Program ID information

c     ProgName - the name of this program 'SobelAndPeaks', used to 
c     - generate the log file name.
      character*50 ProgName
c     ProgVersionNo - the version number for this program.
      character*4 ProgVersionNo

      common/ProgIDInfo/ ProgVersionNo

c***********All the other  variables  **********************************

c     AttNum - attribute number returned from an inquire_attribute call
      integer :: AttNum

c     ChunkSize - the number of elements to use for chunking in x and y.
      integer ChunkSize(2)
c     ChunkSzLenX - the actual number of elements to using in chunking 
c     . in X
      integer, parameter :: ChunkSzLenX = 1024
c     ChunkSzLenY - the actual number of elements to using in chunking 
c     . in Y
      integer, parameter :: ChunkSzLenY = 1024
c     Command - string used to build the compress/decompress commands.
      character*319 Command
c     CountToRead - The number of (rows,columns) to read from the input
c     - file for output. Used only by AppQual.
      integer CountToRead(2)

c     DateTimeEnd - array of values for the end date and time 
c      returned by the GNU subroutine date_and_time.
      integer DateTimeEnd(8)
c     DateTimeStart - array of values for the start date and time 
c      returned by the GNU subroutine date_and_time.
      integer DateTimeStart(8)
c     DayC - character representation of a month day
      character*2 DayC
c     DayEnd - month day of the end of the range to process. 
      integer DayEnd
c     DayStart - month day of the beginning of the range to process. 
      integer DayStart
c     DeflateLevel - degress of compression from 1 to 7. Dan says that
c     . anything after 3 doesn't result in much more compression, but 
c     . does take more time
      integer, parameter :: DeflateLevel=9
c     DummyComment - used when reading input to describe the variable 
c     - to be read in the input file. 
      character*60 DummyComment
c     DummyCharString - used in netCDF.
      character*60 DummyCharString
c     DummyCharacter - used in ReadInputs
      character*20 DummyCharacter
c     DumYr - Dummy character for printout of year.
      character*4 DumYr
c     DumMn - Dummy character for printout of month.
      character*2 DumMn
c     DumDa - Dummy character for printout of day.
      character*2 DumDa
c     DumHo - Dummy character for printout of hour.
      character*2 DumHo
c     DumMi - Dummy character for printout of minute.
      character*2 DumMi
c     DumSe - Dummy character for printout of second.
      character*2 DumSe
c     DurationInMinutes - the number of minutes (and fractions thereof)
c      for this run.
      real*4 DurationInMinutes

c     EndOfString - end of string
      character*1 EndOfString
      parameter(EndOfString=char(000))
c     FileListFlag - a logical .true. if the list of files to be 
c     - processed are to be read from the terminal, 0 if it is to be
c     - determined from ArchiveInventory
      logical FileListFlag
c     FilesPerMinute - the number of files per minute processed in this
c      run.
      real*4 FilesPerMinute
c     FillValueInt1 - the number to use for an integer*1, byte fillvalue
      integer*1 FillValueInt1
      parameter(FillValueInt1=-128)
c     FillValueInt2 - the number to use for an integer*2 fill value.
      integer*2 FillValueInt2
      parameter(FillValueInt2=-32768)
c     FillValueInt4 - the number to use for an integer*4 fill value.
      integer*4 FillValueInt4
      parameter(FillValueInt4=-2147483647)
c     FillValueReal - the number to use for sst a real fill value.
      real FillValueReal
      parameter(FillValueReal=huge(0.))

c     GeoNameIn - the name of the input file with lat, lon locations of
c     - array elements. Only needed if there is one file for an entire
c     - archive. Only used by AppQual
      character*319 GeoNameIn
c     GeoNameIn_lc - lower case version of GeoNameIn.
      character*319 GeoNameIn_lc
c     GeoNameFull - the complete file name for lat, lon file. Not in
c     the Geo common because I want to pass it as an argument so that 
c     it is clear what is being passed in here.
      character*319 GeoNameFull

c     HourC - character representation of a month day
      character*2 HourC
c     HourEnd - Hour in the day of the end of the range to process. 
      integer HourEnd
c     HourStart - Hour in the day of the end of the range to process. 
      integer HourStart

c     iFiles - loop parameter in ReadInventory
      integer iFiles
c     Infinity - a VERY, VERY large number, not quite infinity, but...
      real*4, parameter :: Infinity = 1.0e38
c     InventoryFileName - the ascii name of the file with the list of
c     - files to be processed in this run. In the format of 
c     - ArchiveInventory; could be ArchiveInventory
      character*319 InventoryFileName
c     Error return in allocate statement
      integer ierr
c     iFirst - the index in the file list corresponding to the
c     . beginning of the range to process
      integer iFirst
c     iLast - the index in the file list corresponding to the
c     . end of the range to process
      integer iLast
c     inputputfilename - name of file with input data. If it is equal 
c     . to terminal, the input will be read from the terminal
      character*319 InputFileName

c     istrt - the starting location in x of regions to be printed out
c     . in debug mode. 
      integer*2, parameter :: istrt = 10
c     jstrt - the starting location in y of regions to be printed out
c     . in debug mode. 
      integer*2, parameter :: jstrt = 10

c     Loc1 - used to mark the location of a substring within a string
      integer Loc1
c     Loc2 - used to mark the location of a substring within a string
      integer Loc2
c     Loc3 - used to mark the location of a substring within a string
      integer Loc3
c     Loc4 - used to mark the location of a substring within a string
      integer Loc4
      character*20 LocFileNamePrefix
c     LogicalTemp1 - a variable used to get around the fact that 
c     - compilers handle logical variables differently when passed in
c     - as arguments.
      logical LogicalTemp1
c     LogicalTemp2 - Similar to LogicalTemp1.
      logical LogicalTemp2

c     MinuteC - character representation of the minute of an hour
      character*2 MinuteC
c     MinuteEnd - the minute of the hour at the end of the range to 
c     . process. 
      integer MinuteEnd
c     MinuteStart - the minute of the hour at the start of the range 
c     . to process. 
      integer MinuteStart
c     MonthC - character representation of a month
      character*2 MonthC
c     MonthEnd - the month at the end of the range to process. 
      integer MonthEnd
c     MonthStart - the month at the start of the range to process. 
      integer MonthStart
c     Msg50 - 50 character variable for printout messages passed to 
c     . some subroutines
      character*50 Msg50

c     PercentDone - the fraction of files processed.
      real*4 PercentDone
c     pr_n - previous run number. Is '0' for this run. Used to build 
c     - output filenames.
      character*1 pr_n

c     r_n - run number. Is '1' for this run. Used to build output 
c     - filenames.
      character*1 r_n
c     Resolution - the resolution in km/pixel. This variable is used in 
c      determining the size of the window to use in generating the 
c      merged images. Not sure exactly how this works. It looks like
c      pmerge turns on all pixels within 10/Resolution pixels of a 
c      frontal pixel; i.e., it makes a fat contour to avoid a lot of
c      thin lines with gaps between them. PCC -1/25/13 - not sure that
c      is actually km/pixel. Looks more like it is the number of 
c      pixels
      integer*4 Resolution
      parameter(Resolution=4)

c     SecondEnd - Second in  hour of end of range to process. 
      integer SecondEnd
c     SecondStart - Second in hour of start of  range to process. 
      integer SecondStart
c     SecondsSince1970T - the number of seconds since 00:00 of
c     - 1 January. 1970 corresponding to the time of a given image. 
c     - Read from. the image file.
      real*8 SecondsSince1970T
c     SpaceC - a blank space.
      character*1, parameter :: SpaceC=' '
c     SpaceCread - a variable read in from the inventory - in pricnciple 
c      a space.
      character*1 SpaceCread
c     SSTFillValueChar - not sure what this is for.
      character*20 SSTFillValueChar
c     Start - the (line,column) to start reading the input array from.
c     - Used only in AppQual.
      integer Start(2)
c     StartEndTypeFlag -  1 if start and end time are to be  
c     - entered as yyyymmddhhmmss or 0 if they are to be entered 
c     - as seconds since 1/1/1970 00:00:00
      integer StartEndTypeFlag
c     status - return status from a NetCDF call
      integer status
c     Stride - the stride to take when reading input (rows, columns).
c     - Used only by AppQual.
      integer Stride(2)
c     StringLoc - loaction of first character in string
      integer*2 StringLoc 

c     TempDoubleVar1 - a temporary double variable.
      real*8 TempDoubleVar1
c     TempDoubleVar2 - a temporary double variable.
      real*8 TempDoubleVar2
c     TempIntVar1 - a temporary integer variable.
      integer TempIntVar1
c     TempIntVar2 - a temporary integer variable.
      integer TempIntVar2
c     TempIntVar3 - a temporary integer variable.
      integer TempIntVar3
c     TempString1 - temporary string used to generate file names. This
c     . string is for the directory portion of the name. 
      character*30 TempString1
c     TempString2 - temporary string used to generate file names. This
c     . string is for the file extension  portion of the name. 
      character*30 TempString2
c     TempRealVar1 - a temporary real variable.
      real*4 TempRealVar1
c     TempRealVar2 - a temporary real variable.
      real*4 TempRealVar2

c     WhichComputer - 0 for the CMS (Redhat) machine and 1 for OSX
      integer WhichComputer
      parameter(WhichComputer=1)

c     xType - returned variable type from netCDF call
      integer :: xType

c     YearC - character representation of a Year
      character*4 YearC
c     YearEnd - year of the end of the range to process. 
      integer YearEnd
c     YearStart - year of the start of the range to process. 
      integer YearStart
c     YearStartC - the year for the starting point of this run. Used
c     - to build filenames to make them a bit more unique.
      character*4 YearStartC
c     YearDayC - character representation of a YearDay (Range 001:366)
      character*3 YearDayC

c     UUID - A universally unique identifier used to identify
c     - 'practically unique' files.
      character*36 UUID


c     X_str - temporary string for writing metadata from parameter values
      character*50 minstep_str, minlength_str, minclear_str

c     End of explicit statements for parameters used by all or most
c     - programs

c***********************************************************************

