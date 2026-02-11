c***********************************************************************
c   Modifications 
c
c     3/16/11 PCC - GenerateGlobals - changed CommonSubs... to 1.01
c       Changed  'print *, PreviousSource(1:Loc-1)' to
c         'print *, PreviousSource(1:Loc1)'; 
c       Added 'implicit none';
c       Commented out line beginning with 'iLoc'
c     4/4/11 PCC - All subroutines 
c       Changed all logical*1 to logical 
c       Changed 2000 to '2000' in call FileStatus( in WriteMergedThinned
c       Changed type of n in Median from integer*2 to integer.
c     5/12/11 PCC - All subroutines - changed to version 1.10
c       Changed to netCDF4.
c
c     5/24/11 - PCC - 
c       Added assignement of .false and .true. to explicitly defined
c       logical*1 variables rather than just putting them in the call.
c       Did this for the call to filestatus WriteMergedThinned. 
c
c     6/2/11 - PCC - Moved opening of daily and monthly files from 
c      OpenOutputFiles to AppQual_CMS.
c
c     6/6/11 - PCC - Changed length of SatName from 8 to 20 characters.
c     
c     6/7/11 - PCC - Major changes to simplify and correct handling of
c      strings. Replaced all length() with len(trim(). Removed all 
c      EndOfStrings. Added BlankFill before any definition of a string
c      variable to force the end of the string to be all blanks.
c
c     6/7/11 - PCC - FileStatus - added code to remove .gz if the 
c      filename already has it since FileStatus was written assuming 
c      that the incoming name does not have compression extension on it,
c      even if the file it compressed; it will check for both compressed
c      and uncompressed.
c
c     6/14/11 - PCC - Changed the way CheckArchive works to make it 
c      easier to add another archive. 
c
c     Changes from 1.10 to 1.11
c
c     7/8/11 PCC - Added debug lines in ReadSST.
c
c     7/10/11 PCC - Added debug lines in Median.
c
c     7/14/11 PCC - Formally allocated histo in Median. Not sure how it
c      was working before since it seemed to be allocated in the 
c      definition statment as opposed to the allocate statement, but it
c      was not allocated anywhere else. GULP.
c
c     Changes from 1.11 to 1.20 - major changes in metadata
c
c     8/20/11 PCC - Added StandardName and modified LongName in 
c      DefineSecondsSinceVar
c
c     8/22/11 - PCC - Changed the metadata substantially in 
c      GenerateGlobals. Also added a summary to the argument for this
c      subroutine. Added summary for merged and thinned in 
c      hycom_1m_Assim.
c
c     Changes from 1.11 to 1.21
c
c     12/30/11 - PCC - Changed the way that it handles an error return
c       from a netCDF copy command in GenerateGlobals. In the previous 
c       version it would end the run. Now, if debug is on, it prints out
c       a message otherwise it goes on its merry way hoping that the 
c       attribute was missing in the input file and that there was not 
c       some other really bad problem.
c
c     Changes from 1.21 to 1.22
c
c     1/2/12 - PCC - Changed all occurrences of 150 to 319 when 
c       referring to a string length - to 319 because 150 is too short
c       and because 319 is pretty unique.
c
c     Changes from 1.22 to 1.23
c
c     1/2/12 - PCC - Changed the filename used by uuid_gen to be based 
c       on a current filename to make it unique. The problem was that
c       Median, Sobel and other programs were using the same temporary
c       files resulting in collisions that caused the job to fail. There
c       may also have been files created with the same uuid as a result.
c       The fix was to pass a current filename into uuid_gen. This 
c       however required mucking with GenerateGlobals and a couple of
c       the calling programs. 
c
c     Changes from 1.23 to 1.24
c
c     1/11/12 - DH - Added MergeArchiveInventories() and
c       ReadArchiveInventories() subroutines.  MergeArchiveInv..()
c       merges all BaseDir/ArchiveInventory_UUID files into a single
c       BaseDir/ArchiveMaster_UUID file.
c
c     Changes from 1.24 to 1.25
c
c     1/21/12 - PCC - Changesd some more 150s to 319s for character
c        lengths, mainly in format statements. Not sure how I missed
c        them earlier.
c       Modified CheckArchive to break out of the loop that looks for
c        an archive. Aborts if no archive found.
c
c     Changes 1.25 to 1.26
c
c     1/26/12 - PCC - Removed / from construction of ArchiveInventory 
c       filename in OpenOutputFiles. BaseDir should have a / at the
c       end
c
c     Changes 1.26 to 1.27
c
c     2/12/12 - PCC - Rewrote SecondsSince to use true astronomical 
c        Julian days for calculation of time; added JulianDay.
c       Also removed real scnds and real rand statements - they were
c        just lying around, serving no purpose.
c
c     Changes 1.27 to 1.28
c
c     4/7/12 - PCC - Added functions for converting integer*2, 
c       integer*4 and real*4 numbers to fixed character strings, 
c       floating character strings and general character strings.
c
c     4/17/12 - PCC - Changed printout in OpenLogFile - minor mod. 
c      Changed CheckArchive so that it no longer checks LenX (see
c      CheckArchive for description why) and it no longer provides
c      the starting year for AppQual.
c
c     4/15/12 - PCC - added ReadInputs, ReadArchiveInventory. Modified
c      other routines to make consistent with changes in AppQualMed
c      and SIED. Should have changed to 2.00, but didn't. Will now.
c
c     Changes 1.28 to 2.00 (1.28 should have been 2.00 - sorry)
c
c     4/28/12 - PCC - Changed print statement (#100) in ReadInputs to
c      say ReadInputs instead of ReadInventory. 
c      Added NumberOfImages.
c      Changed destination directory for log files.
c
c     Changes 2.00 to 2.01
c
c     5/18/12 -PCC - Changed printout for CheckArchive.
c      Incremented NumberOfArchives.
c      Removed printout of OldMaster in MergeArchives - doesn't exist.
c      Changed format in Number2StringFloat from F10.5 to F15.5
c
c     Changes 2.01 to 2.02
c
c     5/23/12 - PCC - removed ncID from passed parameter to ReadLatLon.
c       Commented out some debug statements in ReadInventory.
c
c     Changes 2.02 to 2.03
c
c     5/23/12 - DJI - added satellite for hycom_EPR_assim to ReadInputs 
c      subfunction.
c      Made SpaceC a parameter. Added a new variable SpaceCread to be
c      read in inventory reads.
c     
c     Changes 2.03 to 2.04
c
c     5/25/12 - PCC - Added code to read the offset and scale factor 
c       for the median SST data in ReadMedianFile
c
c     Changes 2.04 to 2.05
c
c     6/13/12 - PCC - Added atsr-D3b to the satellite archive list. Also
c      changed atsr to atsr-D2m for sat 11.
c     
c     6/16/12 - PCC - Added atsr-N3b to the satellite archive list.
c
c     Changes 2.05 to 2.06
c     
c     7/25/12 - PCC - Floated Minute in GetSolarZenithAngle
c
c     Changes 2.06 to 2.07
c     
c     8/6/12 - PCC - Added GenerateFileList subroutine.
c
c     Changes 2.07 to 2.08
c     
c     8/9/12 - PCC - Added code to calculate InputOffset2OutputOffset
c      if input is in degrees C.
c
c     Changes 2.08 ==> 2.09
c
c     9/5/12 - PCC - Added calculation of iEnd and jEnd in ReadInputs.
c      This will avoid the problem of the user trying to write out a 
c      different number of i variables than 10. This is only used in 
c      debug.
c 
c     Changes 2.09 ==> 2.10
c
c     10/11/12 - PCC - Increased MaxNumberOfFiles from 100,000 to
c      300,000. Hit 100,000 for Pathfinder 1km.
c
c     Changes 2.10 ==> 2.11
c
c     10/19/12 - PCC - Change ReadLatLon to read integer lat and lon
c      values from the GeoLoc file and to return lat and lon as 
c      real*4 variables to the calling program. This means that the
c      calling programs need not be changed.
c
c     Changes 2.11 ==> 2.12
c
c     10/26/12 - PCC - LatLonFillValue was read in as a real variable in
c      ReadLatLon. It should have been read in as an integer variable 
c      then set equal to a real variable to be returned to the calling
c      program. It has been fixed in this version.
c
c     Changes 2.12 ==> 2.13
c
c     12/29/12 - PCC - Added an nc-close to ReadMergedThinned. No need
c      to leave open. This was left over from when the merged and 
c      thinned files were compressed.
c
c     12/31/12 - PCC - Increased dimensions of ProcessingHistory and
c      PreviousProcessingHistory from 400 to 2000 to accommodate 
c      SIED in GenerateGlobals.
c
c     1/6/13 - PCC - modified ReadSobel to read GeoSobel data and to 
c      get the eastward gradient, the northward gradient and the gradien
c      magnitude if the (1,1) element of the array passed in is 100.
c
c     1/12/13 - PCC - Changed the type of MissingValueInt2 and ...In4 in
c      GetAttribues from real*4 to integer*2 and integer*4 respectively.
c      Have no idea why they were real.
c     Modified printout in GenerateFileName to avoid extra blank line.
c     Modified printout in ReadMedianFile to avoid extra blank line.
c
c     Changes 2.13 ==> 2.14
c
c     1/15/13 - PCC - Added Duration function to calculate elapsed time
c      for this run and the number of files processed per minute.
c
c     Changes 2.14 ==> 2.15
c
c     1/19/13 - PCC - Added code to retrieve SSTFillValue in 
c       ReadMedianFile. Also, updated CommonSubsVersionNo from 2.13
c       to 2.15. Files created using CommonSubroutines 2.14 will show
c       2.13. Bad, bad, bad.
c      Commented out the parameter statement assigning SSTFillValue to
c       0 in parameterstatements.
c
c     Changes 2.15 ==> 2.16
c
c     1/20/13 - PCC - Added code to read ParameterName. This variable
c       is used to determine what metadata to write in the output 
c       netCDF files. Will generally be SST, but later could be
c       'Ocean Color', 'Chlorophyll', 'Salinity',... Added ParameterName
c       ParameterStatements.
c      Added code to read ParameterUnits. This variable give the units
c       of the field to be processed and is used for netCDF metadata.
c      Changed SSTFillValue to SSTFillValueIn so that it is clear that
c       this is the value read in from ReadMedianFile.
c      Added code in the first set of loops over the array in Median to
c       to check for an input value less than or equal to zero or 
c       greater than Range. If found, an error is written and the 
c       program is stopped. Added code at the end of Median to set all
c       values of inpict and pict that are equal to zero to 
c       FillValueInt2
c      Cleaned-up debug statements uuid_gen. Also added trim to command
c       statements in uuid_gen.
c
c     Changes 2.16 ==> 2.17
c
c     1/23/13 - PCC - Added definition of iVarName and jVarName to 
c       ReadInputs. Also, added these variables to ParameterStatements.
c
c     Changes 2.17 ==> 2.18
c
c     1/25/13 - PCC - Changed print statements in PrintArray2 so that
c       FillValueInt2 does not print out as ****.
c      Added code to generate a lowercase version of ParameterName to
c       use in variable names in the output netCDF files. Added
c       ParameterName_lc to ParameterStatements.
c      Added lower_case subroutine.
c
c     Changes 2.18 ==> 2.19 
c
c      Added subroutines to read fill values,
c       scale factors and offsets for sst, lat/lon, gradients and
c       zenith angle. The reason for separate subroutines was to allow
c       for new common in ParameterStatements for these values. This
c       simplifies passing them from one subroutine to the next. I 
c       left the general subroutine GetAtrributes to allow these
c       attributes to be read for a file other than the ones produced
c       by front/gradient workflow. Becareful when using the names of
c       these attributes in a call to the general subroutine though -
c       the common statement used names xxxFillValueIn, xxxOffsetIn and
c       xxxScaleFactorIn. 
c      Also changed the name of the lat/lon fill value returned from
c       ReadLatLon. This routinue is handled differently from reads
c       for files written by this workflow because this routine also
c       applies the offset and scale to the data converting it from an
c       integer to a real value. This required a change in the type
c       of LatLonFillValue so a new fill value was defined in the 
c       subroutine and returned with the call.
c
c     1/28/13 -PCC - SatName moved to ArchiveDefinition common in 
c       ParameterStatements. This required removing SatName from 
c       subroutine calls and definitions for ReadInputs, CheckArchive 
c       and ReplaceBaseDir.
c
c     Changes 2.19 ==> 2.20 
c
c     1/30/13 - PCC - Added ReadPixSpacing subroutine. Used in Pmerge.
c     2/1/13 - PCC - removed code to close the MergedThinned file before
c     .  leaving ReadMergedThinned. Thin required that it be left open.
c
c     Changes 2.20 ==> 2.21
c
c     2/3/13 - PCC - Changed ConvertIntegerToString to suppress leading
c     .  zeros.
c     . Added code to generate the character version of WinSize in 
c     .  ReadInputs.
c
c     Changes 2.21 ==> 2.22
c
c     3/13/13 - JPS - Added Aquarius to SatNameList as number 16
c
c     Version 2.22 ==> 2.23
c
c     3/15/23 - PCC - Added/modified entry for MODIS4km SST. 
c      Removed reference to ArchiveIDList - it is no longer used.
c
c     Version 2.23 ==> 2.24
c
c     4/2/23 - PCC - Added print statements in ReadInputs for scan
c       direction.
c      Added an attribute, LSTorGMT, defining LST or GMT for DateTime  
c       in DefineSecondsSinceVar. 
c      Added GetSecondsSince1970 to read SecondsSince1970 and, and this
c       is the reason that this subroutine was added, LSTorGMT.
c      Added code to GetSecondsSince1970 in all subroutines that read
c       variables output by AppQual and subsequent programs, except
c       for the reading of lat,lon.
c
c     2.24 ==> 2.25
c
c     4/9/13 - PCC - Added modis_chl to the list of archives.
c
c     2.25 ==> 2.26
c
c     4/10/13 - PCC - Modified code to read a variable. Used 
c       ParameteraName variable to construct the name of the parameter
c       to be read. This required minor changes in ReadMedianFile and
c       GetSSTAttributes.
c      Added SecondsSince1970ID to call to GetSecondsSince...
c      Closed the open netCDF zenith file in ReadZenithAngleFile.
c      Moved the place where the geosoble netCDF file is closed to 
c       after the call for attributes.
c
c     2.26 ==> 2.27
c
c     4/12/13 - PCC - Changed ParameterName to ParameterName_lc in 
c       ReadMedian... and in GetSSTAttributes.
c
c     2.27 ==> 2.28
c
c     4/21/13 - PCC - Added Read_int2_variable and Read_real_variable
c       subroutines to read LenX, LenY arrays.
c
c     2.28 ==> 2.29
c
c     4/21/13 - PCC - Changed check on time in ReadInventory from getting
c       the index of the last image prior to or equal to the start time to
c       getting the next image after the last image prior to the start time.
c      Also added code to make sure that the start time is not beyond the 
c       end of the interval with data.
c
c     2.29 ==> 2.30
c
c     6/7/13 - DI - Added hycom archives
c
c     2.30 ==> 2.31
c
c     6/10/13 - JPS - Added subroutines to calculate the linear fit to
c        remove the background linear trend from the 32x32 SIED window
c     Also added a subroutine to calculate the inverse of the least
c        squares fit matrix in the background removal
c
c     2.31 ==> 2.32
c
c     9/2/13 - PCC - For reasons that are beyond me, there were no
c       changes made in going from 2.31 to 2.32 other than the version
c       number. Soooo.... to avoid having to change AppQual, GeoSobel
c       and GenerateZenithAngle, I will not update the version number
c       for the changes that I'm making now.
c     Error in error detection in ReadInputs. This will have NO effect 
c       on operation; it only relates to a check on the value of 
c       MinLength that was entered. 
c
c     2.32 ==> 2.33
c
c     1/13/14 - PCC - Added UnitInputSST for input SST files opened in 
c       Fortran; i.e., not netcdf.
c       
c     2.33 ==> 2.34
c
c     1/26/14 - PCC - Cleaned up debug statements at the beginning of
c       ReadInputs.
c      Replaced amsre and trim in archive list with l2_... and 
c       l3_amsre_v7 to archive list. I am now adding level, version 
c       number of data if appropriate and version number of AppQual.
c      Changed how the SatNameList is updated - only a little bit - 
c       Added instructions for how one adds a new archive name to this
c       list.
c
c     2.34 ==> 2.35
c
c     1/27/14 - PCC - Redefined the chunking parameter in both x and y
c       as the minimum of LexX(Y) and ChunkSzLenX(Y). This only affected
c       WriteMergedThinned.
c
c     2.35 ==> 2.36
c
c     1/29/14 - PCC - Modified Plane_fit such that if fits a plane to
c       a region that is 1.5 time the SIED window size rather than 
c       being hardwired to 48. I had to change a number of lines to
c       accomplish this. 
c      Replaced -32768 with SSTFillValueIn in Plane_fit.
c      Also added debug statements to this Plane_fit.
c      Put Summary, SummaryPart1, 2 and 3 in a common statement in 
c       ParameterStatements
c      Changed ConvertIntegerToString to handle 8 characters instead of
c       5. 
c       
c     12/29/14 - JPS - Changed the read lengths of LenX(Y)in and LenX(Y)
c      to i5 to accomodate the long orbits of AVHRR gac data
c
c     1/13/15 - JPS - Changed AVHRR_gac related stuff to be general and not
c      reference satellites, which are also found in file names. 
c     Added ecco2_2km and avhrr_hrpt to sat name list
c
c     7/22/15 - JPS - changed MaxEdge to 10,000,000 because ECCO2 llc_2160
c     was exceeding previous limit
c     
c     8/25/15 - JPS - Added a flag that lets you suppress output files from
c     Sobel
c
!     2.36 ==> 2.37
!
!     7/29/16 - PCC - Changed all includes for ParameterStatements to
!      ParameterStatements (with no f). Caused problems in Eclipse.
!
c***********************************************************************
      character*4 function CommonSubsVersionNo()
c***********************************************************************
c
c  This function returns the version number for all of the subroutines 
c  in this file. All programs using any of these subroutines should 
c  call this function and write the version number as a global 
c  variable in the output files generated by the program. All changes
c  to any subroutine in this file should be documented here and the
c  version number should be incremented to indicate the change. Version
c  1.0 corresponds to the code in this file on 24 August 2010.
c

      CommonSubsVersionNo = '2.37'
      
      return
      end function CommonSubsVersionNo

c***********************************************************************
      character*20 function Num2StrInt( Number)
c***********************************************************************
c
c     Convert a fixed number to a string.
c
      implicit none

c******Parameter statements

      include 'ParameterStatements'

      integer*4 Number, i, j

c     Convert the number to a string. Set output to blanks first.

      TempString1 = BlankFill(:len(TempString1))
      write( TempString1, '(i10)') Number

c     Remove any leading blanks from the string. Set to blanks first.

      Num2StrInt = BlankFill(:len(Num2StrInt))

      j = 0
      do 100 i=1,30
         if(TempString1(i:i) .ne. ' ') then
            j = j + 1
            Num2StrInt(j:j) = TempString1(i:i)
         endif
 100  continue

      return
      end function Num2StrInt

c***********************************************************************
      character*20 function Num2StrFloat( Number)
c***********************************************************************
c
c     Convert a fixed number to a string.
c
      implicit none

c******Parameter statements

      include 'ParameterStatements'

      integer*4 i, j
      real*4 Number

c     Convert the numer to a string. Set output to blanks first.

      TempString1 = BlankFill(:len(TempString1))
      write( TempString1, '(f15.5)') Number

c     Remove any leading blanks from the string. Set to blanks first.

      Num2StrFloat = BlankFill(:len(Num2StrFloat))

      j = 0
      do 100 i=1,30
         if(TempString1(i:i) .ne. ' ') then
            j = j + 1
            Num2StrFloat(j:j) = TempString1(i:i)
         endif
 100  continue

      return
      end function Num2StrFloat      

c***********************************************************************
      character*20 function Num2StrG( Number)
c***********************************************************************
c
c     Convert a fixed number to a string.
c
      implicit none

c******Parameter statements

      include 'ParameterStatements'

      integer*4 i, j
      real*4 Number

c     Convert the numer to a string. Set output to blanks first.

      TempString1 = BlankFill(:len(TempString1))
      write( TempString1, '(g15.5)') Number

c     Remove any leading blanks from the string. Set to blanks first.

      Num2StrG = BlankFill(:len(Num2StrG))

      j = 0
      do 100 i=1,30
         if(TempString1(i:i) .ne. ' ') then
            j = j + 1
            Num2StrG(j:j) = TempString1(i:i)
         endif
 100  continue

      return
      end function Num2StrG     

c***********************************************************************
      character*36 function UUID_Gen(TempFile)
c***********************************************************************
c     
c     This function will generate a UUID using the unix command uuidgen.
c     It will write this to the file temptemp.data in /tmpdir under in
c     the base directory area and then it will read it in.
c     
      implicit none

c******Parameter statements

      include 'ParameterStatements'

c     TemoFileName - the temporary name of the input file - uncompressed
      character*319 TempFileName
      character*319 TempFile

c---------------------------------start here --------------------------

c     Find /sst/ and _sst in the filename. Make sure the are present.
c     - If not, there is a problem, so abort the run.

      if(debug .eq. 1) then
         print *, 'UUID_Gen #000: BaseDir::', trim(BaseDir), '::'
         print *, 'TempFile::', trim(TempFile), '::'
         write(UnitLog,*) 'UUID_Gen #000: BaseDir::',trim(BaseDir),'::'
      endif

      Loc2 = 1
 100  continue
      Loc1 = index( TempFile(Loc2:len(TempFile)), '/') 
      if(Loc1 .gt. 0) then
         Loc2 = Loc1+Loc2
         if(debug .eq. 1) print *, 'UUID_Gen #102.',
     1        trim(TempFile), Loc1, Loc2
         go to 100
      endif

      TempFileName = BlankFill(:len(TempFileName))
      TempFileName = trim(BaseDir) // 'tmpdir/' //
     1     TempFile(Loc2:len(TempFile))

      Command = BlankFill(:len(Command))
      Command = 'uuidgen > ' // trim(TempFileName)

      call system(Command)

      if(debug .eq. 1) then
         print *, 'UUID_Gen #110. Command::', trim(Command), '::'
         print *, 'TempFileName::', trim(TempFileName), '::'
      endif

      open(unit=100, file=TempFileName)

      read ( 100, fmt='(a36)') UUID_Gen

      close(unit=100)

      Command = BlankFill(:len(Command))
      Command = 'rm ' // trim(TempFileName)
      
      call system(Command)
      
      return
      
      end function UUID_Gen

c***********************************************************************
      character*319 function GenerateFileName( SSTFileName, 
     1     DirComponent, FNComponent)
c***********************************************************************
c
c  This function takes an SST filename in ArchiveInventory for which the
c  BaseDirectory has been replaced and generates the name for the sub-
c  directory specified
c

c******Parameter statements

      include 'ParameterStatements'

c     DirComponent - the component of the directory to replace.
      character*30 DirComponent

c     FNComponent - the component of the filename to replace.
      character*30 FNComponent

c     SSTFileName - the temporary name of the input file - uncompressed
      character*319 SSTFileName

c---------------------------------start here --------------------------

c     Find /sst/ and _sst in the filename. Make sure the are present.
c     - If not, there is a problem, so abort the run.

         if(debug .eq. 1) print *, 'GenerateFileName #100. ',  
     1       'SSTFileName::', trim(SSTFileName), '::'
         if(debug .eq. 1) print *, 'GenerateFileName #101. ',  
     1       'DirComponent::', trim(DirComponent), '::'
         if(debug .eq. 1) print *, 'GenerateFileName #102. ',  
     1       'FNCompnent::', trim(FNComponent), '::'

         Loc1 = index( SSTFileName, '/Median/') 
         if(Loc1 .eq. 0) then
            print *, 'GenerateFileName #110: Problem with filename. ',
     1           'No /median/. ''Tis case sensitive'
            write(UnitLog, *) 'Problem with filename. No /sst/.'
            stop
         endif

         Loc2 = index(SSTFileName, '_Median.')
         if(Loc2 .eq. 0) then
            print *, 'GenerateFileName #120: Problem with filename. ',
     1           'No _SST.  ''Tis case sensitive'
            write(UnitLog, *) 'Problem with filename. No _SST.'
            stop
         endif

         if(debug .eq. 1) print *, 'GenerateFileName #120. Loc1: ',  
     1        Loc1, ' SSTFileName(1:Loc1): ', SSTFileName(:Loc1)
         if(debug .eq. 1) print *, 'GenerateFileName #121. Loc2: ',  
     1        Loc2, ' SSTFileName(Loc1+5:Loc2): ', 
     2        SSTFileName(Loc1+8:Loc2)

c     Replace sst in /sst/ and _sst with Median and Sobel to generate
c     - Median and Sobel filenames. Need to remove end of string markers
c     - on the filename elements.

         GenerateFileName = BlankFill(:len(GenerateFileName))
         GenerateFileName =  SSTFileName(:Loc1) // trim(DirComponent) 
     1     // SSTFileName(Loc1+8:Loc2-1) // trim(FNComponent) 

         if(debug .eq. 1) print *, 'GenerateFileName #999. Filename::', 
     1    trim(GenerateFileName), '::'
      
      return
      end function GenerateFileName

c***********************************************************************
      logical function CheckDimensions(InName)
c***********************************************************************
c     
c    This function will read the dimensions from the netCDF file 
c     passed in and compare them with the LenX and LenY. If they are
c     different, it returns DimensionsEqual as .false. Otherwise, it
c     returns them as .true.
c
      use netcdf

      implicit none

c     Functions

c******Parameter statements

      include 'ParameterStatements'

c     General variables

      character*319 InName

      integer ncid, LenXTID, LenYTID, LenXT, LenYT

c----------------------------------start here --------------------------

c     Write out which satellite this is for and the base directory

      if(Debug .eq. 1) print *, 'CheckDimensions #100: ', 
     1     'InputFileName::', trim(InName), '::'
      write(UnitLog,*) 'InputFileName::', trim(InName), '::'

      status = nf90_open( InName, nf90_nowrite, ncid)
      if (status .ne. nf90_noerr) call handle_err(status)

      status = nf90_inq_dimid( ncid, 'nx', LenXTID)
      if(status .ne. nf90_noerr) call handle_err(status)
      
      status = nf90_inquire_dimension( ncid, LenXTID,  
     1     DummyCharString, LenXT)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_inq_dimid( ncid, 'ny', LenYTID)
      if(status .ne. nf90_noerr) call handle_err(status)
      
      status = nf90_inquire_dimension( ncid, LenYTID, 
     1     DummyCharString, LenYT)
      if(status .ne. nf90_noerr) call handle_err(status)

      if( (LenXT .ne. LenX) .or. (LenYT .ne. LenY) ) then
         CheckDimensions = .false.
      else 
         CheckDimensions = .true.
      endif

      if(Debug .eq. 1) then
         print *, 'CheckDimensions #110: LenX:', LenX, 
     1        '. File value: ', LenXT, '. LenY:', LenY, 
     2        '. File value: ', LenYT
      endif
      status = nf90_close(ncid)
      if (status .ne. nf90_noerr) call handle_err(status)

      end function CheckDimensions

c***********************************************************************
      subroutine ReadInputs( ProgName, Start, CountToRead, GeoNameIn,
     1     YearStart, YearStartC, YearEnd, MonthStart, MonthEnd,
     2     InventoryFileName)
c***********************************************************************
c     
c     This subroutine is used to initialize values of variables that 
c     will be used in this run. First, determine which program this is.
c     If is it AppQual, get the size of the original fields as well as
c     the satellite type.
c     
c     This subroutine will also set values that were defined in  
c     the ParameterStatements file. Some of these need not be
c     changed which is why they are set here. Some, at least one 
c     depends on other values hence can no longer be defined in the
c     ParameterStatements file.
c     

      implicit none

c******Functions
      
      real*8 SecondsSince
      character*20 Num2StrInt, Num2StrFloat, Num2StrG
      integer*2 MinFrontStep, MinFrontLength
      logical PlaneRemover
      integer*2 PRtmp
      integer*2 Sobeltmp
      integer*2 SIEDtmp
c     i - do loop parameter
      integer i

!     Debug_Eclipse - .true. if a debug run on my laptop, otherwise a
!      regular run asking for input.
      logical, parameter :: Debug_Eclipse=.false.

c******General Variables.    

      include 'ParameterStatements'

c$$$  c     TempFileName - filename used for
c$$$  character*319 TempFileName 

c----------------------------------start here --------------------------

c     Write out which satellite this is for and the base directory

      print *, 'ReadInputs #000'

c     This chunk of code is used to initialize values of variables that 
c     will be used in this run. First, determine which program this is.
c     If is it AppQual, get the size of the original fields as well as
c     the satellite type.
c     
c     This program will also provide values that were set in parameter 
c     statements before in the ParameterStatements file. Some of these
c     need not be changed which is why they are set here. Some, at least
c     one depends on other values hence can no longer be defined in the
c     ParameterStatements file.

c     
c     1/2/12 - PCC - Changed a couple of character length statements
c     from 150 to 319.
c     
c     1/12/12 - DI - Added Archive information for NonAssimilated HYCOM
c     
c     2/3/12 - DH - Add check for trailing slash on BaseDir, add if
c     not present.
c     
c     3/16/12 - PCC - Changed xLenArchive(9) to a multiple of 32
c     
c     4/7/12 - PCC - Changed upper limit of do 5432 from 1000 to 8192
c     
c***********************************************************************
c**** Start by initializing variables.
c     
c**** This section will change from archive-to-archive.

c     Need to initialize BlankFill first however since it is used to
c     construct SatNameList.

      do 5432 Loc1=1,8192
         BlankFill(Loc1:Loc1) = ' '
 5432 continue

c     WinSize is the size of the window used to by SIED to find fronts.
c     . It is normally 32 but could be changed here.

      WinSize = 32
      call ConvertIntegerToString( WinSize, WinSizeChar)

c     Get the size of the plane to be fit to the histrogram window and
c      its character equivalent. This is based on the size of the 
c      histogram window.

      PlaneFitSize = WinSize * 3.5
      call ConvertIntegerToString( PlaneFitSize, PlaneFitSizeChar)


c     The next group of variabels is for the definition of the archive
c     to be processed. It is used primarily in checks to make sure 
c     that the input parameters correspond to the proper archive. 
c
c%%%%%%%%%%% Instructions for adding a new archive to the list %%%%%%%%%
c
c     If you are adding a new archive to the suite of archives that the
c     workflow can process, add it here. To do so, do the following:
c      1. Copy the last two lines before the do 4000 loop.
c      2. Insert these lines at the end of the current list.
c      3. Increment the number in these two lines.
c      4. Replace the archive name in 2nd of these two lines with 
c         your archive name. A subset of this name has to be exactly 
c         as appears in the file structure created by MakeDirectory.py;
c         i.e., I have a directory /Volumes/AMSRE/l3_amsre_v7-3.01/. The
c         corresponding name on SatNameList is l3_amsre_v7. The 3.01 is
c         the number of AppQualMed_Main and will likely change as this
c         program is updated. I don't want to have to add an entry for
c         each version number so I leave this off of the SatNameList 
c         name. Also this name must be the same as LocFileNamePrefix
c         which is in YourArchiveName_Run_Configuration_File in
c         .../YourArchiveName/SupportingFiles/.
c         Also, your archive name is restricted to 30 characters. 
c      5. Increment the starting element of the do 4000 loop to avoid
c         overwriting the new name.
c
c     Hopefully, there will not be more than 100 archives named on this
c      list. If so, you will have to increase the following name as 
c      well as the dimension of SatNameList in ParameterStatments.

      NumberOfArchives = 100

c     SatNameList - the names of the archives.
      Loc1 = len(SatName)

      SatNameList(1) = BlankFill(1:Loc1)
      SatNameList(1) = 'msg'
      SatNameList(2) = BlankFill(1:Loc1)
      SatNameList(2) = 'goes'
      SatNameList(3) = BlankFill(1:Loc1)
      SatNameList(3) = 'l2_amsre_v7'
      SatNameList(4) = BlankFill(1:Loc1)
      SatNameList(4) = 'l3_amsre_v7'
      SatNameList(5) = BlankFill(1:Loc1)
      SatNameList(5) = 'pathfinder4km'
      SatNameList(6) = BlankFill(1:Loc1)
      SatNameList(6) = 'hycom_1m_Assim'
      SatNameList(7) = BlankFill(1:Loc1)
      SatNameList(7) = 'pathfinder1km_L2'
      SatNameList(8) = BlankFill(1:Loc1)
      SatNameList(8) = 'hycom_1m_NonAssim'
      SatNameList(9) = BlankFill(1:Loc1)
      SatNameList(9) = 'pathfinder1km_L3'
      SatNameList(10) = BlankFill(1:Loc1)
      SatNameList(10) = 'modis4km_sst'
      SatNameList(11) = BlankFill(1:Loc1)
      SatNameList(11) = 'atsr-D2m'
      SatNameList(12) = BlankFill(1:Loc1)
      SatNameList(12) = 'hycom_EPR_Assim'
      SatNameList(13) = BlankFill(1:Loc1)
      SatNameList(13) = 'atsr-D3b'
      SatNameList(14) = BlankFill(1:Loc1)
      SatNameList(14) = 'atsr-N3b'
      SatNameList(15) = BlankFill(1:Loc1)
      SatNameList(15) = 'ecco2_92'
      SatNameList(16) = BlankFill(1:Loc1)
      SatNameList(16) = 'Aquarius'
      SatNameList(17) = BlankFill(1:Loc1)
      SatNameList(17) = 'modis4km_chl'
      SatNameList(18) = BlankFill(1:Loc1)
      SatNameList(18) = '05.8'
      SatNameList(19) = BlankFill(1:Loc1)
      SatNameList(19) = '60.5'
      SatNameList(20) = BlankFill(1:Loc1)
      SatNameList(20) = '90.2'
      SatNameList(21) = BlankFill(1:Loc1)
      SatNameList(21) = '90.3'
      SatNameList(22) = BlankFill(1:Loc1)
      SatNameList(22) = '90.6'
      SatNameList(23) = BlankFill(1:Loc1)
      SatNameList(23) = '90.8'
      SatNameList(24) = BlankFill(1:Loc1)
      SatNameList(24) = '90.9'
      SatNameList(25) = BlankFill(1:Loc1)
      SatNameList(25) = 'avhrr_gac'
      SatNameList(26) = BlankFill(1:Loc1)
      SatNameList(26) = 'ecco2_4km'
      SatNameList(27) = BlankFill(1:Loc1)
      SatNameList(27) = 'avhrr_hrpt'

      do 4000 i=28,NumberOfArchives
         SatNameList(i) = BlankFill(1:Loc1)
         SatNameList(i) = '%%%%%%%%'
 4000 enddo

c     SatNameList(ii) = BlankFill(1:Loc1)
c     SatNameList(ii) = 'YourArchiveName'

c***********************************************************************
c**** Initialize other variables and read in variables.
c     
c**** This section should not change from archive-to-archive.

c     Set iEnd and jEnd based on iStart and jStart. All of the print
c     statements will print out 10 variable across. The length can
c     be anything, but set it to 10 here.

      iEnd = iStrt + 9
      jEnd = jStrt + 9

c     Now set values that are not likely to be changed between runs.

      MaxEdge = 10000000
      MaxNumberOfFiles = 300000
      MaxNumOfSegments = 1000000

c     MinClear is the minimum number of clear pixels acceptable in a
c     histogram window

      MinClear = 100

c     Set all unit numbers except for the input unit which was set based
c     . on the presence of an input file.

      UnitCnt = 21
c     If equal to 99, this unit will not be opened; i.e., only a cnt 
c     . file will be written.
      UnitGeoMetadata = 28
      UnitDim = 22
      UnitInventory = 23
      UnitLog = 24
      UnitPeaks = 25
      UnitSumsDay = 26
      UnitSumsMonth = 27
      UnitInputSST = 28

c     Initialize some other variables.

      Start(1) = 1
      Start(2) = 1

c********************Now read in values for this run. *****************
c     
c     Start by reading the following from the terminal. These values
c     tell the programs where to find the data, what time period to
c     process and whether or not this is a debug run. Following these,
c     the program will open a file for variables that do not change for
c     this archive. The variables to read from the terminal are:
c     
c     1) The base directory
c     2) Flag to say if start and end times are in yyyymmddhh or seconds
c     since 1970.
c     3) The start time of the period to process
c     4) The end time of the period to process.
c     5) The debug flag.

c     Is this a debug run?

      if(Debug_Eclipse) then
        debug = 1
      else
        print *, 'Is this a debug run? (0 for no, 1 for yes): '
        read ( 5, fmt='(i1)') debug
      endif

      print *, 'ReadInputs #100: debug: ', debug

c     Next the base directory.

      BaseDir = BlankFill(:len(BaseDir))
      if(Debug_Eclipse) then
        BaseDir = '/Volumes/Data/msg'
!      BaseDir = '/Volumes/MSG-GOES-AMSR-MODEL/msg'
      else
        print *, 'Enter base directory name:'
        read ( 5, fmt='(a319)') BaseDir
      endif

c     Make sure there is a '/' at the end of BaseDir; need that later.

      posEnd = len_trim(BaseDir)
      posSlash = scan(BaseDir,'/',.true.)

      if ( posEnd .GT. posSlash) then
         if(Debug .eq. 1) print *, 'ReadInputs #110. Adding a slash ',
     1        'the end of BaseDir'
         BaseDir = trim(BaseDir) // '/'
      endif

      print *, 'ReadInputs #120: BaseDir::', trim(BaseDir), '::'

c************Read in start and end time information *******************

      if(Debug_Eclipse) then
        StartEndTypeFlag = 1

        YearStart = 2010
        MonthStart = 3
        DayStart = 1
        HourStart = 0
        MinuteStart = 0
        SecondStart = 0

        YearEnd = 2010
        MonthEnd = 3
        DayEnd = 3
        HourEnd = 0
        MinuteEnd = 0
        SecondEnd = 0

c     Get the number of seconds since 1970 to the start and end times.

         SecondsSinceStart = SecondsSince( YearStart, MonthStart,
     1        DayStart, HourStart, MinuteStart, SecondStart,
     2        1970, 1, 1, 0, 0, 0)

         SecondsSinceEnd = SecondsSince( YearEnd, MonthEnd,
     1        DayEnd, HourEnd, MinuteEnd, SecondEnd,
     1        1970, 1, 1, 0, 0, 0)

      else
        print *, 'Enter 1 if start and end time are to be ',
     1      'entered as yyyymmddhhmmss or 0 if they are to be ',
     2      'entered as seconds since 1970: '
        read(5,fmt='(I1)') StartEndTypeFlag

        if(StartEndTypeFlag .eq. 1) then
            print *, 'Enter the starting year, month, day, hour, ',
     1          'minute and second of the period to process: '
            read(5,fmt='(I4,5I2)') YearStart, MonthStart,
     1          DayStart, HourStart, MinuteStart, SecondStart

            print *, 'ReadInputs #125: Start Time: ', MonthStart, '/',
     1          DayStart, '/', YearStart, '   ', HourStart, ':',
     2          MinuteStart, ':', SecondStart

            print *, 'Enter the ending values: '
            read(5,fmt='(I4,5I2)') YearEnd, MonthEnd, DayEnd,
     1         HourEnd, MinuteEnd, SecondEnd
            print *, 'ReadInputs #130: End Time: ', MonthEnd, '/',
     1          DayEnd, '/', YearEnd, '   ', HourEnd, ':',
     2          MinuteEnd, ':', SecondEnd

c     Get the number of seconds since 1970 to the start and end times.

         SecondsSinceStart = SecondsSince( YearStart, MonthStart,
     1        DayStart, HourStart, MinuteStart, SecondStart, 
     2        1970, 1, 1, 0, 0, 0)           

         SecondsSinceEnd = SecondsSince( YearEnd, MonthEnd, 
     1        DayEnd, HourEnd, MinuteEnd, SecondEnd,
     1        1970, 1, 1, 0, 0, 0)

        else
            print *, 'Enter the seconds since 1/1/1970: 00:00 ',
     1          ' for the start time of the period to process: '
            read(5,fmt='(a60,f14.2)') DummyComment, SecondsSinceStart

            print *, 'Enter the seconds since for the end time : '
            read(5,fmt='(a60,f14.2)') DummyComment, SecondsSinceEnd
        endif
      endif

      print *, 'ReadInputs #140: Start Time in seconds since 1970: ',
     1     SecondsSinceStart

      print *, 'ReadInputs #141: End Time in seconds since 1970: ',
     1     SecondsSinceEnd

c     Open a file to log the output of this run. First, get the 
c     the character representation for the starting year and month
c     to use in the log file name. 

      DummyCharacter = Num2StrInt( YearStart)
      YearStartC = DummyCharacter(1:4)

      DummyCharacter = Num2StrInt( MonthStart)
      if(MonthStart .lt. 10) then
         MonthC = '0' // DummyCharacter(1:1)
      else
         MonthC = DummyCharacter(1:2)
      endif

      call OpenLogFile( YearStartC, MonthC, ProgName)

c     Write input parameters read thus far to log file.

      write(UnitLog,*) 'BaseDir: ', trim(BaseDir)

      DummyCharacter = Num2strInt(YearStart)
      DumYr = DummyCharacter(1:4)
      DummyCharacter = Num2strInt(MonthStart)
      DumMn = DummyCharacter(1:2)
      DummyCharacter = Num2strInt(DayStart)
      DumDa = DummyCharacter(1:2)
      DummyCharacter = Num2strInt(HourStart)
      DumHo = DummyCharacter(1:2)
      DummyCharacter = Num2strInt(MinuteStart)
      DumMi = DummyCharacter(1:2)
      DummyCharacter = Num2strInt(SecondStart)
      DumSe = DummyCharacter(1:2)
      print *, 'ReadInputs #150: Start Time: ', DumMn, '/', DumDa, 
     1     '/', DumYr, ' ', DumHo, ':', DumMi, ':', DumSe
      write(UnitLog,*) 'Start Time: ', DumMn, '/', DumDa, '/', DumYr, 
     1     ' ', DumHo, ':', DumMi, ':', DumSe

      DummyCharacter = Num2strInt(YearEnd)
      DumYr = DummyCharacter(1:4)
      DummyCharacter = Num2strInt(MonthEnd)
      DumMn = DummyCharacter(1:2)
      DummyCharacter = Num2strInt(DayEnd)
      DumDa = DummyCharacter(1:2)
      DummyCharacter = Num2strInt(HourEnd)
      DumHo = DummyCharacter(1:2)
      DummyCharacter = Num2strInt(MinuteEnd)
      DumMi = DummyCharacter(1:2)
      DummyCharacter = Num2strInt(SecondEnd)
      DumSe = DummyCharacter(1:2)
      print *, 'ReadInputs #160: End Time: ', DumMn, '/', DumDa, 
     1     '/', DumYr, ' ', DumHo, ':', DumMi, ':', DumSe
      write(UnitLog,*) 'End Time: ', DumMn, '/', DumDa, '/', DumYr, 
     1     ' ', DumHo, ':', DumMi, ':', DumSe

      write(UnitLog,*) 'Start Time in seconds since 1970: ',
     1     SecondsSinceStart

      write(UnitLog,*) 'End Time in seconds since 1970: ',
     1     SecondsSinceEnd

      if(Debug .eq. 1) print *, 'ReadInput #170: Finished terminal ',
     1     'input, now will read the satellite_Run_Configuration_file.'


c*******End of section to read in start and end time information ******

c***********************************************************************
c*****Now read control variables from the run configuration file. *****
      
c     Now read the input file. Start by getting the satellite name. 
c     This name is used to define the input filename. Open the input 
c     file.

      call CheckArchive

      InputFileName = BlankFill(:len(InputFileName))
      InputFileName =  trim(BaseDir) // 'SupportingFiles/' //
     1     trim(SatName) // '_Run_Configuration_File'

      print *, 'ReadInput #180: Reading ::', trim(InputFileName), '::'

      if( StringLoc .ne. 1) then
         UnitInput = 10
         open( Unit=UnitInput, file = InputFileName)
      endif

c     Save the satellite name since this is the subdirectory for this 
c     archive.

c     A dummy line used in input file to help user keep variable and 
c     formats straight'

      read ( UnitInput, fmt='(a60)') DummyComment

c**** Name of Variable being processed *********************************
c     
c     This variable is used to correctly specify the metadata to be
c     written out to the netCDF files.

      read (UnitInput, fmt='(a60,a30)') DummyComment, ParameterName
      print *, 'ReadInput #190: ParameterName: ', ParameterName
      write(UnitLog,*) 'ParameterName: ', ParameterName

c     Get the lower case version of the parameter name to use in 
c      the names of variables in the output netCDF file.
      ParameterName_lc = BlankFill(:len(ParameterName_lc))
      ParameterName_lc = trim(ParameterName)
      call lower_case(ParameterName_lc)

c**** This variable is used to correctly specify the metadata to be
c     written out to the netCDF files.

      read (UnitInput, fmt='(a60,a30)') DummyComment, ParameterUnits
      print *, 'ReadInput #200: ParameterUnits: ', ParameterUnits
      write(UnitLog,*) 'ParameterUnits: ', ParameterUnits

c**** Which is the along-scan dimension ********************************

      read (UnitInput, fmt='(a60,i4)') DummyComment, AlongScanDimension
      print *, 'ReadInput #210: AlongScanDimension: ', 
     1     AlongScanDimension
      write(UnitLog,*) 'AlongScanDimension: ', AlongScanDimension

c     Define iVarName and jVarName used in metadata.

      iVarName = BlankFill(1:len(iVarName))
      jVarName = BlankFill(1:len(jVarName))
      if(AlongScanDimension .eq. 1) then
         iVarName = 'along-scan'
         jVarName = 'cross-scan'
         print *,'ReadInput #211: First dimension is along-scan'
      else
         iVarName = 'cross-scan'
         jVarName = 'along-scan'
         print *,'ReadInput #212: First dimension is cross-scan'
      endif


c**** Read in the number of rows and columns of the original fields ****
      
c     First dimension of the input field.

      read (UnitInput, fmt='(a60,i5)') DummyComment, LenXin
      print *, 'ReadInput #220: LenXin: ', LenXin
      write(UnitLog,*) 'LenXin: ', LenXin

c     Second dimension of the input fields.

      read (UnitInput, fmt='(a60,i5)') DummyComment, LenYin
      print *, 'ReadInput #230: LenYin: ', LenYin
      write(UnitLog,*) 'LenYin: ', LenYin

c**** Read in the length of the x and y dimensions of the output file.

c     First dimension of the output field.

      read (UnitInput, fmt='(a60,i5)') DummyComment, LenX
      print *, 'ReadInput #240: LenX: ', LenX
      write(UnitLog,*) 'LenX: ', LenX

c     Second dimension of the output field.

      read (UnitInput, fmt='(a60,i5)') DummyComment, LenY
      print *, 'ReadInput #250: LenY: ', LenY
      write(UnitLog,*) 'LenY: ', LenY

c     Normally, L3 and L4 fields, LenXin and LenYin are the same as 
c     LenX and LenY. Print out a warning if they are not.

      if( (LenXin .ne. LenX) .or. (LenYin .ne. LenY) ) then
         print *, 'ReadInput #260: ******** WARNING LenXin (', 
     1        LenXin, ') .ne. LenX (' , LenX, ') or LenYin (', 
     2        LenYin, ') .ne. LenY (', LenY, '). Continuing, just ',
     3        'wanted to let you know; OK for L2.'
      endif

c**** Minimum number of clear pixels allowed per image. This parameter
c     is used later to skip fields with less than this number.

      read (UnitInput, fmt='(a60,i4)') DummyComment, NumClearThresh
      print *, 'ReadInput #270: NumClearThresh: ', NumClearThresh
      write(UnitLog,*) 'NumClearThresh: ', NumClearThresh
      
c***********************************************************************
c***  Next get information needed to generate the geolocation file ******

      LocFileNamePrefix = BlankFill(:len(LocFileNamePrefix))
      read (UnitInput, fmt='(a60,a20)') DummyComment, LocFileNamePrefix
      print *, 'ReadInput #280: LocFileNamePrefix::', 
     1     trim(LocFileNamePrefix), '::'
      write(UnitLog,*) 'LocFileNamePrefix: ', trim(LocFileNamePrefix)
      
      if(LocFileNamePrefix(20:20) .ne. SpaceC) then
         write(UnitLog,*) 'ReadInputs #184: Problem with ',
     1        'LocFileNamePrefix::', trim(LocFileNamePrefix), ':: ',
     2        'Aborting.'
         print *, 'ReadInputs #290: Problem with ',
     1        'LocFileNamePrefix::', trim(LocFileNamePrefix), ':: ',
     2        'Aborting.'
      endif

      SingleLocFile = .false.
      if(len(trim(LocFileNamePrefix)) .gt. 0) SingleLocFile = .true.

      if(SingleLocFile .eqv. .true.) then
         
c     Make sure that the location filename, if there is one, corresponds
c     to SatName. This will be used downstream in the workflow.

         Loc1 = index( SatName, LocFileNamePrefix)
         if(Loc1 .eq. 0) then
            write( UnitLog, *) 'LocFileNamePrefix, ',
     1           trim(LocFileNamePrefix), ', does not correspond to ',
     2           ' SatName: ', trim(SatName)
            write( 6, *) 'LocFileNamePrefix, ',
     1           trim(LocFileNamePrefix), ', does not correspond to ',
     2           ' SatName: ', trim(SatName)
            stop
         endif

         DummyCharString = BlankFill(:len(DummyCharString))
         print *, 'Enter the filename for the input LatLon file.'
         read (UnitInput, fmt='(a60,A60)') DummyComment, DummyCharString

         if(Debug .eq. 1) print *, 'ReadInputs #300: ', 
     1        'trim(DummyCharString)::',  trim(DummyCharString), '::'

         if(DummyCharString(60:60) .ne. ' ') then
            stop 'GeoNameIn more than 60 characters'
         endif

         GeoNameIn = BlankFill(:len(GeoNameIn))
         GeoNameIn = trim(BaseDir) // 'SupportingFiles/' 
     1        // trim(DummyCharString)

         print *, 'ReadInput #310: GeoNameIn ::', trim(GeoNameIn), '::'
         write(UnitLog,*) 'GeoNameIn: ', trim(GeoNameIn)
      endif

c***********************************************************************
c**** Now get the parametes used to scale the input.
c     
c     NOTE: INPUT AND OUTPUT SCALE FACTORS AND OFFSET APPLY TO SST FROM
C      ORIGINAL FILE TO THE FIRST OUTPUT FILE, THE MEDIAN FIELD. AFTER 
C      THAT USE THE SCALE FACTOR AND OFFSET IN THE FILE THAT IS BEING
C      READ IN NOT THESE VALUES.
c
c     Here's how the scaling works. All temperatures are Kelvin:
c     
c     Given Counts_in, InputScaleFactor, InputOffset and whether the
c     input SSTs are in Kelvin or degrees C.
c     
c     SST_in = Counts_In * InputScaleFactor + InputOffset (+273.15 if
c     the SSTs are in degrees C).
c     
c     Will derive Counts_Out based on OutputScaleFactor, OutputOffset,
c     and SST_In (in Kelvin) such that SST_Out = SST_In.
c     
c     SST_out = Counts_Out * OutputScaleFactor + OutputOffset
c     
c     The following variables will be used:
c     
c     InputScaleFactor - defined above. Called 'scale_factor' in CF-1.5 
c     InputOffset - defined above. Called add_offset in CF-1.5
c     DegreesK - 0 if the input data are degrees Celsius, 1 if Kelvin.
c     OutputScaleFactor - defined above.
c     OutputOffset - defined above. This is also the smallest allowed
c     SST value in Kelvin on output. Input SST values that are smaller 
c     than this are set to _FillValue. This is the required by the
c     histogramming algorithm used by SIED which requires that numbers
c     go from 0 digital counts to an upper limit of Range.
c     MaxSSTOut - maximum allowed SST value on output. This number,
c     together with OutputOffset and OutputScaleFactor is used to
c     calculate Range.
c     Range - the range of digital counts used for SST on output. 
c     The range is limited because SIED uses a histogram algorithm that
c     operates on a fixed range of values from 0 to a maximum value:
c     Range. I generally use 400 to cover the range of SST values
c     in Kelvin at 0.1K steps. 
c     ScaleInput - the factor that converts from the input scale factor
c     to the output scale factor. Used in conversion of input counts to
c     output counts.
c     InputOffset2OutputOffset - the offset scale factor used to convert
c     input counts to output counts
c     
c     CountOut = ScaleInput * CountIn + InputOffset2OutputOffset,

c     First need to get information about the input values.

      read (UnitInput, fmt='(a60,f9.5)') DummyComment, InputScaleFactor
      print*, 'ReadInput #320:  InputScaleFactor: ', InputScaleFactor
      write(UnitLog,*) 'InputScaleFactor: ', InputScaleFactor

      read (UnitInput, fmt='(a60,f9.4)') DummyComment, InputOffset
      print*, 'ReadInput #330:  InputOffset: ', InputOffset
      write(UnitLog,*) 'InputOffset: ', InputOffset

      read (UnitInput, fmt='(a60,i2)') DummyComment, DegreesK
      if(DegreesK .eq. 0) then
         InputOffset = InputOffset + 273.15
         print *, 'ReadInput #340: Input data are in degrees C.; ',
     1        '273.15 added to InputOffset. InputOffset now: ', 
     2        InputOffset
         write(UnitLog,*) 'Input data are in degrees C.; ',
     1        '273.15 added to InputOffset. InputOffset now: ', 
     2        InputOffset
      else
         print *, 'ReadInput #350: Input data are in Kelvin.'
         write(UnitLog,*) 'Input data are in Kelvin.'
      endif

c     Next get the output scaling information 

      read (UnitInput, fmt='(a60,f9.5)') DummyComment, OutputScaleFactor
      print*, 'ReadInput #360:  OutputScaleFactor: ', OutputScaleFactor
      write(UnitLog,*) 'OutputScaleFactor: ', OutputScaleFactor

      read (UnitInput, fmt='(a60,f9.4)') DummyComment, OutputOffset
      print*, 'ReadInput #370: OutputOffset: ', OutputOffset
      write(UnitLog,*) 'OutputOffset: ', OutputOffset

      read (UnitInput, fmt='(a60,f9.5)') DummyComment, MaxSSTOut
      print*, 'ReadInput #380:  MaxSSTOut: ', MaxSSTOut
      write(UnitLog,*) 'MaxSSTOut: ', MaxSSTOut

c     And calculate Range, OutPutScaleFactor and ScaleInput from  
c     these values. 

      Range = (MaxSSTOut - OutputOffset) / OutputScaleFactor
      print*, 'ReadInput #390: Range: ', Range          
      write(UnitLog,*) 'Range: ', Range          

      ScaleInput = InputScaleFactor / OutputScaleFactor
      print*, 'ReadInput #400:  ScaleInput: ', ScaleInput
      write(UnitLog,*) 'ScaleInput: ', ScaleInput

      if(DegreesK .eq. 1) then
         InputOffset2OutputOffset = (InputOffset - OutputOffset) / 
     1        OutputScaleFactor
      else
         InputOffset2OutputOffset = 
     1        (InputOffset - OutputOffset) / 
     2        OutputScaleFactor
      endif

      print*, 'ReadInput #410:  InputOffset2OutputOffset: ', 
     1     InputOffset2OutputOffset
      write(UnitLog,*) 'InputOffset2OutputOffset: ', 
     1     InputOffset2OutputOffset


c***********************************************************************

c     Next, get the windows for the merge step. These parameters will be
c     ignored when not being used; i.e., by programs other than pmerge
c     in the workflow. There are two parameters used here - ImageWindow
c     and HourWindow. ImageWindow designates the number of images to 
c     use on each side of the image of interest and HourWindow 
c     indicates the number of hours within which to search or images
c     from which fronts will be merged for the image of interest. The
c     window resulting in the smaller number of adjacent images will be
c     used in the merging process.

c     ImageWindow - the # of images from each side of the target image.

      read(UnitInput,fmt='(a60,i2)') DummyComment, ImageWindow
      print *, 'ReadInput #420: ImageWindow: ', ImageWindow
      write(UnitLog,*) 'ImageWindow: ', ImageWindow

c     HourWindow - the # of hours on each side of the target window.

      read(UnitInput,fmt='(a60,i2)') DummyComment, HourWindow
      print *, 'ReadInput #430: HourWindow: ', HourWindow
      write(UnitLog,*) 'HourWindow: ', HourWindow

c***********************************************************************

c     Next, get the minimum digital count step allowed for fronts, and 
c     the minimum length of accepted fronts

c     MinStep - the minimum # of digital counts that must seperate the 
c     mean of the two populations for SIED to find a front

      read(UnitInput,fmt='(a60,i2)') DummyComment, MinStep
      print *, 'ReadInput #420: MinStep: ', MinStep      
      write(UnitLog,*) 'MinStep: ', MinStep

c     MinLength - the minimum # of pixels required for a front to be 
c     saved

      read(UnitInput,fmt='(a60,i1)') DummyComment, MinLength
      if(MinLength > 9) then
         print*, '****************************************'
         print*, 'Warning: MinLength was set as ', MinLength, ' Was ',
     1        ' this intentional?'
      endif
      print *, 'ReadInput #420: MinLength: ', MinLength
      write(UnitLog,*) 'MinLength: ', MinLength
         

c     PlaneRemover is a logical that tells SIED to remove a background
c     linear plane before applying the algorithm

      read(UnitInput,fmt='(a60,i1)') DummyComment, PRtmp
      if (PRtmp .eq. 1) then
         RemovePlane = .true.
      else
         RemovePlane = .false.
      endif

      print *, 'ReadInput #420: SIEDRemovePlane: ', RemovePlane
      write(UnitLog,*) 'SIEDRemovePlane: ', RemovePlane

      if(Debug .eq. 1) print *, 'ReadInput #422: PRtmp: ', PRtmp

      read(UnitInput, fmt='(a60,i1)') DummyComment, Sobeltmp
      if (Sobeltmp .eq. 1) then
         SobelFileFlag = .true.
      else
         SobelFileFlag = .false.
      endif
      print*, 'ReadInput #420:  SobelFileFlag: ', SobelFileFlag

      read(UnitInput, fmt='(a60,i1)') DummyComment, SIEDtmp
      if (SIEDtmp .eq. 1) then
         SIED_Full = .true.
      else
         SIED_Full = .false.
      endif
      print*, 'ReadInput #420:  SobelFileFlag: ', SobelFileFlag


c     Print out window size and plane fit size. Didn't do it where they
c      were defined because debug had not be defined yet.

      print  *, 'ReadInput #425: WinSize: ', WinSize,
     1     ' PlaneFitSize: ', PlaneFitSize

      if(Debug .eq. 1) print  *, 'ReadInput #427: WinSizeChar::',  
     1     trim(WinSizeChar), ':: PlaneFitSizeChar::', 
     2     trim(PlaneFitSizeChar), '::'

c***********************************************************************
c**** Now setup some more parameters that depend on parameters read in.

      InventoryFileName = BlankFill(:len(InventoryFileName))
      InventoryFileName = trim(BaseDir) // 'ArchiveInventory'
      print *, 'ReadInput #440: InventoryFileName ::', 
     1     trim(InventoryFileName), '::'

      CountToRead(1) = LenX
      CountToRead(2) = LenY

c$$$  MaxNumOfPeaks = LenX * LenY / 10

c***********************************************************************
c**** Finished entering run control parameters.

      if(Debug .eq. 1) print *, 'ReadInputs #999'

      end subroutine ReadInputs

c***********************************************************************
      subroutine ReadInventory(InventoryFileName, InvSecondsSince1970, 
     1     InvNumClear, InvYear, InvMonth, InvDay, InvHour, 
     2     InvMinute, InvSecond, InvFileName, iFirst, iLast)
c***********************************************************************
c
c  Subroutine to read ArchiveInventory and find the first and last files
c     to process.
c
c****** Functions
c     
c****** General Variables.
c     
      implicit none

      include 'ParameterStatements'

c     InvDay - day in month corresponding to an image in the archive.
      integer InvDay(MaxNumberOfFiles)
c     InvFileName - the name of the input file read from the inventory
      character*319 InvFileName(MaxNumberOfFiles)
c     InvHour - hour in day corresponding to an image in the archive.
      integer InvHour(MaxNumberOfFiles)
c     InvSecond - Second in minute corresponding to an image in the 
c     - archive.
      integer InvSecond(MaxNumberOfFiles)
c     InvSecondsSince1970 - the number of second since 00:00 of 1  
c     . January 1970 corresponding to the time of a given image. Read 
c     . from the inventory
      real*8 InvSecondsSince1970(MaxNumberOfFiles)
c     InvMinute - minute of the hour corresponding to an image in the 
c     . archive.
      integer InvMinute(MaxNumberOfFiles)
c     InvMonth - month of the year corresponding to an image in the 
c     . archive.
      integer InvMonth(MaxNumberOfFiles)
c     InvNumClear - Number of clear pixels in the image read from the 
c     . inventory. 
      integer InvNumClear(MaxNumberOfFiles)
c     InvYear - Year corresponding to an image in the archive.
      integer InvYear(MaxNumberOfFiles)

c     TempFileName - filename used for
      character*319 TempFileName 

c----------------------------------start here --------------------------

      if(Debug .eq. 1) print *, 'ReadInventory #000'

c     Write out which satellite this is for and the base directory

      if(Debug .eq. 1) print *, 'ReadInventory #100: ',
     1     'InventoryFileName::', trim(InventoryFileName), '::'

c     Open the inventory file

      open(unit=UnitInventory, file=trim(InventoryFileName), 
     1     status='old')

      if(Debug .eq. 1) print *, 'ReadInventory #110: Opened input file '

c     Get the image list indices of the 1st and last images to process.

         if(Debug .eq. 1) print *, 'ReadInventory #115: ',
     1        'SecondsSinceStart', SecondsSinceStart,
     2        'SecondsSinceEnd', SecondsSinceEnd

      iFirst = 1
      iLast = 0
      NumberOfImages = 0
      do 1100 iFiles=1,MaxNumberOfFiles
         read(UnitInventory, fmt='(F14.2,I11,6I5,A1,A319)', end=1110)
     1        InvSecondsSince1970(iFiles), InvNumClear(iFiles), 
     2        InvYear(iFiles), InvMonth(iFiles), InvDay(iFiles), 
     3        InvHour(iFiles), InvMinute(iFiles), InvSecond(iFiles), 
     4        SpaceCread, InvFileName(iFiles)

         NumberOfImages = NumberOfImages + 1

c$$$         if(Debug .eq. 1) print *, 'ReadInventory #120: InvFileName(',
c$$$     1        iFiles, ')::', trim(InvFileName(iFiles)), '::'
         
c     Get first and last file in the specified time interval. 

         if(InvSecondsSince1970(iFiles) .lt. SecondsSinceStart) then
             iFirst = iFiles + 1

c$$$             if(Debug .eq. 1) print *, 'ReadInventory #130: iFirst ',
c$$$     1            iFirst
         endif

         if(InvSecondsSince1970(iFiles) .le. SecondsSinceEnd) then
             iLast = iFiles
c$$$             if(Debug .eq. 1) print *, 'ReadInventory #131: iLast ',
c$$$     1            iLast
         endif

c$$$         if(Debug .eq. 1) print *, 'ReadInventory #140: ',
c$$$     1        'InvSecondsSince1970(',iFiles,')=', 
c$$$     2        InvSecondsSince1970(iFiles), 
c$$$     2        '; InvNumClear(',iFiles,')=', InvNumClear(iFiles), 
c$$$     3        ' NumClearThresh=', NumClearThresh
c$$$
c$$$         if(Debug .eq. 1) print *, 'ReadInventory #141: ',
c$$$     1        'trim(InvFileName(iFiles))::',
c$$$     2        trim(InvFileName(iFiles)), '::'

 1100 continue
      stop '# of records in the inventory exceeds MaxNumberOfFiles.'

c     Here when the entire inventory has been read.

 1110 continue

      if(iFirst .gt. NumberOfImages) then
         print *, 'ReadInventory #150. Problem with time interval. ', 
     1        'iFirst (', iFirst, ') > NumberOfImages (', 
     2        NumberOfImages, '). Aborting.'
         stop '*************** ERROR ABORTING **************'
      endif
         
      write(UnitLog,*) 'ReadInventory #150 ', NumberOfImages, 
     1     ' file names from ArchiveInventory.'
      print *, 'ReadInventory #150 ', NumberOfImages, 
     1     ' file names from ArchiveInventory.'

      if(Debug .eq. 1) print *, 'ReadInventory #160. From SecondsSince(' 
     1     , iFirst, ')=', InvSecondsSince1970(iFirst), 
     2     ' to SecondsSince(', iLast, ')=', InvSecondsSince1970(iLast)

      if( (iFirst .eq. 0) .and. (iLast .eq. 0) ) then
         write(UnitLog,*) 'No files found in the time range.'
         stop 'ReadInventory #150: No files found in the time range.'
      endif

      if(Debug .eq. 1) print *, 'ReadInventory #999'

      end subroutine ReadInventory

c***********************************************************************
      subroutine Duration( DateTimeStart, DurationInMinutes, 
     1     NumberOfFilesProcessed, FilesPerMinute)
c***********************************************************************
c
c     Given a starting time in years, months, days, hours,... this 
c      subroutine will get the current date and time can calculate 
c      the duration in minutes since the starting time.
c
      implicit none

c******Functions

c     SecondsSince - returns the number of hours since the reference.
c     - Arguments are year, month, day, hour, minute and the 
c     - values for these.
      real*8 SecondsSince

c******Parameter statements

      include 'ParameterStatements'

c     NumberOfFilesProcessed - the number of fields processed in this 
c     - run.
      integer NumberOfFilesProcessed

c---------------------------  Start here -------------------------------

c     Get the ending date and time so that we can determine the 
c      duration of this run and the number of files/minute processed.

      call date_and_time(VALUES=DateTimeEnd)

c     Convert the starting and ending date/time to secondssince 2010.

      SecondsSinceStart = SecondsSince( DateTimeStart(1), 
     1     DateTimeStart(2), DateTimeStart(3), DateTimeStart(5),
     2     DateTimeStart(6), DateTimeStart(7), 2010, 1, 1, 1, 1, 1)

      SecondsSinceEnd = SecondsSince( DateTimeEnd(1), 
     1     DateTimeEnd(2), DateTimeEnd(3), DateTimeEnd(5),
     2     DateTimeEnd(6), DateTimeEnd(7), 2010, 1, 1, 1, 1, 1)


c     Print out the start time and the end time.

      print *, 'Start time: ', DateTimeStart(1:3), DateTimeStart(5:7) 
      print *, '  End time: ', DateTimeEnd(1:3), DateTimeEnd(5:7) 

      DurationInMinutes = (SecondsSinceEnd - SecondsSinceStart) / 60

      FilesPerMinute = NumberOfFilesProcessed / DurationInMinutes

      return
      end subroutine Duration

c***********************************************************************
      subroutine OpenOutputFiles(InputFileName)
c***********************************************************************
c     
c     Open the following files for this run:
c     
c     1) History file - this file contains processing information in  
c     ...it; i.e., stuff written to the terminal,
c     2) Inventory file - this file will contain an inventory of the 
c     ...entire run. The inventory includes hours since 00:00 on 
c     ...1/1/1900, number of clear pixels in the image, year, month, 
c     ...day, hour, minute of the image and the full filename.
c     3) Daily sums file - this is a list of the number of files in 
c     ...archive for each day.
c     4) Monthly sums file - this is a list of the number of files in 
c     ...archive for each month.
c
c**********Functions
c     
c**********General Variables.
c     
c     UnitLog - unit number to which run history will be written
c     UnitInventory - unit for inventory file
c     UnitSumsDay - unit for daily sums of files
c     UnitSumsMonth - unit for monthly sums of files
c     
c     BaseDir - the directory in which the subdirectories for the input
c     ...images and the subdirectories for the output images exist.
c     
c     HistoryFileName - filename for output of this run of the program
c     InventoryFileName - the ascii name of the file
c     DailySums - name of output file for the daily sums of files.
c     MonthlySums - name of output file for the monthly sums of files.
c     
c     Numbrs - ascii representation of the ten digits
c     
c     xLenArchive - the length for the x-dimension of the archive being
c     ... processed. This is set in ReadInputs.f. It is used to check 
c     ... that we are preocessing the correct archive.
c     
c     exis -  .true. if file exists when inquiring
c     EndOfString - end of string
c     
      implicit none

c     Functions

c******Parameter statements

      include 'ParameterStatements'

c     size of arrays - Meteosat are 3712x3712 and goes are 1980x2431

c      integer LenXin, LenYin
c      parameter(LenXin=3712, LenYin=3712)
c      parameter(LenXin=1980, LenYin=2431)
c      parameter(LenXin=1952, LenYin=2400)

c     General variables

      character*319 HistoryFileName

      character*1 Numbrs(0:9)

      character*36 UUID_Gen

      integer i

      logical exis

c     Data statements

      data Numbrs/ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'/

c----------------------------------start here --------------------------

c     Write out which satellite this is for and the base directory

      print *, 'OpenOutputFiles #100: BaseDir:', BaseDir
      write(UnitLog,*) 'OpenOutputFiles #100: BaseDir: ', BaseDir
c$$$c      if(LenXin .eq. xLenMeteo) then
c$$$      if(LenX .eq. xLenMeteo) then
c$$$         print *, 'This run is for meteosat'
c$$$         write(UnitLog,*) 'This run is for meteosat'
c$$$c      elseif(LenXin .eq. xLenGoes) then
c$$$      elseif(LenX .eq. xLenGoes) then
c$$$         print *, 'This run is for GOES'
c$$$         write(UnitLog,*)  'This run is for GOES'
c$$$      else
c$$$         print *, 'x-dimension is neither 3712 or 1980. Fix this'
c$$$      endif

c     Open inventory file

      UUID = UUID_Gen(InputFileName)

      InventoryFileName = BlankFill(:len(InventoryFileName))
      InventoryFileName = trim(BaseDir) //
     1   'SupportingFiles/ArchiveInventories/ArchiveInventory_' //
     2   UUID

c      InventoryFileName = trim(BaseDir) // 'ArchiveInventory'

      if(debug .eq. 1) print *, 'OpenOutputFiles #110: ', 
     1     'InventoryFileName: ', InventoryFileName

      open(unit=UnitInventory, file=InventoryFileName)

      end subroutine OpenOutputFiles

c***********************************************************************
      subroutine GetSolarZenithAngle( YearDay, Hour, Minute, Lat, Lon,
     1     ZenithAngle)
c***********************************************************************

      implicit none

c******Variables in
c     
c     Hour - hour of day GMT
      integer Hour
c     Lat - latitude of the pixel of interest in degrees
      real*4 Lat
c     Lon - longitude of the pixel of interest in degrees
      real*4 Lon
c     Minute - minute of hour GMT
      integer Minute
c     YearDay - day of year
      integer YearDay

c******Variables out
c     
c     ZenithAngle - Solar zenith angle at pixel. If greater than 90
c     . degrees it is night time.
      integer*2 ZenithAngle

c******General variables
c     
c     AngleTime - radians corresponding to the longitude corrected
c     . time.
      real*4 AngleTime
c     cah - cosine of AngleTime
      real*4 cah
c     cde - cosine of declination
      real*4 cde
c     cla - cosine of latitude
      real*4 cla
c     Declinatione - declination of the sun in radians
      real*4 Declination
c     eqtm - equation of time 
      real*4 eqtm
c     LatRad - latitude in radians
      real*4 LatRad
c     LonTime - correction to time for longitude
      real*4 LonTime
c     pi - 3.14159
      real*4 pi
c     RadiansPerDay
      real*4 RadiansPerDay
c     sla - sine of latitude
      real*4 sla
c     sde - sine of declination
      real*4 sde

c----------------------------------start here --------------------------
      pi = 3.14159

      RadiansPerDay = 2 * pi / 365.25

      Declination = asin( 0.398 * sin( RadiansPerDay * 
     1     ( YearDay - 81.0 + 2.0 * 
     2     sin(RadiansPerDay * (YearDay - 2.0) ) ) ) )

      eqtm = 0.128 * sin(RadiansPerDay * (YearDay - 2.0) ) +
     1     0.164 * sin(2.0 * RadiansPerDay * (YearDay + 10.0) )

      LonTime = Hour + float(Minute) / 60 - eqtm + Lon / 15.0
      AngleTime = pi * (LonTime - 12.0) / 12.0
      LatRad = Lat * pi / 180.0
      sla = sin(LatRad)
      cla = cos(LatRad)
      sde = sin(Declination)
      cde = cos(Declination)
      cah = cos(AngleTime)
      ZenithAngle = ifix(acos( sla * sde + cla * cde * cah) * 180 / pi)

c$$$      if(Debug .eq. 1) print *, 'GetSolarZenithAngle #100: YearDay, ',
c$$$     1     'Hour, Minute, Lat, Lon, Declination, eqtm, LonTim, ',
c$$$     2     'AngleTime, LatRad, sla, cla, sde, cde, cah, ZenithAngle: ',
c$$$     3      YearDay, Hour, Minute, Lat, Lon, Declination, eqtm, 
c$$$     4      LonTim, AngleTime, LatRad, sla, cla, sde, cde, cah, 
c$$$     5      ZenithAngle

      end subroutine GetSolarZenithAngle

c***********************************************************************
      subroutine ReadLatLon( GeoNameFull, Lat, Lon, LatLonFillValueReal)
c***********************************************************************
c     
c     This subroutine will open and read a Sobel file.
c     
c     Written by Peter Cornillon, University of Rhode Island,
c     pcornillon@me.com 28 April 2009
c     
      use netcdf

      implicit none

c     Parameter statements

      include 'ParameterStatements'
      
c****** Output varaiables

      integer*4 ix, jy

c     ncID - NetCDF ID for the input data set
      integer ncID
c     LatID - NetCDF ID for latitude variable
      integer LatID
c     LonID - NetCDF ID for latitude variable
      integer LonID

c     LatLonFillValueReal - the value used for missing data. Be careful
c      here. The fill value in the netCDF file written out for this 
c      variable in integer, but the latitude and longitude are converted
c      to real in the call to ReadLatLon so the fill value is redefined
c      in that subroutine to be real. In order not to confuse it with 
c      the fill value read in, it is renamed as well. 
      real LatLonFillValueReal
c     Lat - latitude array
      real Lat(1:LenX,1:LenY)
c     Lon - longitude array
      real Lon(1:LenX,1:LenY)
c     LonFillValue - temporary value to make sure that the latitude and
c      longitude fill values are the same.
      real LonFillValue
c     LonOffset - temporary value to make sure that the latitude and
c      longitude offsets are the same.
      real LonOffset
c     LonScaleFactor - temporary value to make sure that the latitude and
c      longitude scale factors are the same.
      real LonScaleFactor

      integer, allocatable :: ilon(:,:)
      integer, allocatable :: ilat(:,:)

c      include 'netcdf.inc'

c----------------------------------start here --------------------------

      allocate( ilon(1:LenX,1:LenY),
     1     ilat(1:LenX,1:LenY),
     3     stat=ierr)
      
      if (ierr .ne. 0) then
         print *, 'Allocation error for LenXA, LenYA arrays. Exiting.'
         stop
      endif
      if(Debug .eq. 1) print *, 'ReadLatLon #000: GeoNameFull::', 
     1     trim(GeoNameFull), '::'

      status = nf90_open( GeoNameFull, nf90_nowrite, ncID)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Read the latitude first.

      status = nf90_inq_varid( ncID, 'latitude', LatID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_var( ncID, LatID, iLat)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_att( ncID, LatID, 'scale_factor',
     1     LatLonScaleFactorIn)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_att( ncID, LatID, 'add_offset',
     1     LatLonOffsetIn)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_att( ncID, LatID, '_FillValue',
     1     LatLonFillValueIn)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Next read the longitude.

      status = nf90_inq_varid( ncID, 'longitude', LonID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_var( ncID, LonID, iLon)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_att( ncID, LonID, 'scale_factor',
     1     LonScaleFactor)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_att( ncID, LonID, 'add_offset',
     1     LonOffset)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_att( ncID, LonID, '_FillValue',
     1     LonFillValue)
      if(status .ne. nf90_noerr) call handle_err(status)

      if( (LatLonFillValueIn .ne. LonFillValue) . or. 
     1     (LatLonOffsetIn .ne. LonOffset) .or.
     1     (LatLonScaleFactorIn .ne. LonScaleFactor) ) then
         write(UnitLog,*) 'ReadLatLon #100: Problem with input ',
     1        'attributes. LatFillValue and LonFillValue: ', 
     2        LatLonFillValueIn, LonFillValue, ' LatOffset and ',
     3        'LonOffset: ', LatLonOffsetIn, LonOffset, ' and ',
     4        'LatScaleFactor and LonScaleFactor: ', 
     5        LatLonScaleFactorIn, LonScaleFactor
         print *, 'ReadLatLon #100: Problem with input ',
     1        'attributes. LatFillValue and LonFillValue: ', 
     2        LatLonFillValueIn, LonFillValue, ' LatOffset and ',
     3        'LonOffset: ', LatLonOffsetIn, LonOffset, ' and ',
     4        'LatScaleFactor and LonScaleFactor: ', 
     5        LatLonScaleFactorIn, LonScaleFactor
      endif

c     Close the input file

      status = nf90_close( ncID)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Now convert from integer to real with scale factor

      LatLonFillValueReal = FillValueReal

      do 1021 jy=1,LenY
         do 1022 ix=1,LenX

            if(iLat(ix,jy) .eq. LatLonFillValueIn) then
               Lat(ix,jy) = LatLonFillValueReal
               Lon(ix,jy) = LatLonFillValueReal
            else
               Lat(ix,jy) = iLat(ix,jy) * LatLonScaleFactorIn 
     1              + LatLonOffsetIn
               Lon(ix,jy) = iLon(ix,jy) * LatLonScaleFactorIn 
     1              + LatLonOffsetIn
            endif

 1022    continue
 1021 continue
      
 
      end subroutine ReadLatLon

c***********************************************************************
      subroutine ReadPixelSpacing( xPixelSpacing, yPixelSpacing)
c***********************************************************************
c     
c     This subroutine will open and read a Sobel file.
c     
c     Written by Peter Cornillon, University of Rhode Island,
c     pcornillon@me.com 28 April 2009
c     
      use netcdf

      implicit none

c     Parameter statements

      include 'ParameterStatements'
      
c******LatLon common

c     LatLonFileName - the name of the file containing the latitud and
c     . longitude arrays.
      character*319 LatLonFileName
c     LatLonFileNameSave - the saved LatLonFileName used to see if the
c     . the filename is new for this SST field.
      character*319 LatLonFileNameSave

      common /LatLon/ LatLonFileName, LatLonFileNameSave

c******  varaiables

c     ix, jy - Do loop parameters
      integer*4 ix, jy
c     ixSave - the index in the x pixel separation array at which the
c      maximum number of separations occurs.
      integer ixSave
c     jySave - the index in the y pixel separation array at which the
c      maximum number of separations occurs.
      integer jySave

c     ncID - NetCDF ID for the input data set
      integer ncID
c     nxN - the actual length of the x-dimension separation and count
c      vectors.
      integer*4 nxN
c     nyN - the actual length of the y-dimension separation and count
c      vectors.
      integer*4 nyN

c     resxDimID - NetCDF ID for length of the pixel separation and 
c      count vectors for the x-dimension.
      integer resxDimID
c     resyDimID - NetCDF ID for length of the pixel separation and 
c      count vectors for the y-dimension.
      integer resyDimID

c     xMaxCount - the maximum count in the x pixel count vector.
      integer xMaxCount
c     xPixCountID - netCDF ID for pixel count in x.
      integer xPixCountID
c     xPixCount - pixel the counts histogram in x.
      integer*4, allocatable :: xPixCount(:)
c     xPixSepID - netCDF ID for pixel separation in x.
      integer xPixSepID
c     xPixSep - pixel the separation vector for the histogram in x.
      integer*4, allocatable :: xPixSep(:)
c     xPixelSpacing - The typical x pixel separation in integer km.
      integer*4 xPixelSpacing

c     yMaxCount - the maximum count in the y pixel count vector.
      integer yMaxCount
c     yPixCountID - netCDF ID for pixel count in y.
      integer yPixCountID
c     yPixCount - pixel the counts histogram in y.
      integer*4, allocatable :: yPixCount(:)
c     yPixSepID - netCDF ID for pixel separation in y.
      integer yPixSepID
c     yPixSep - pixel the separation vector for the histogram in y.
      integer*4, allocatable :: yPixSep(:)
c     yPixelSpacing - The typical y pixel separation in integer km.
      integer*4 yPixelSpacing


c      include 'netcdf.inc'

c----------------------------------start here --------------------------

      if(Debug .eq. 1) print *, 'ReadPixelSpacing #000: GeoNameFull::', 
     1     trim(LatLonFileName), '::'

      status = nf90_open( LatLonFileName, nf90_nowrite, ncID)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Get the size of the x and y vectors.

      status = nf90_inq_dimid( ncID, "resx", resxDimID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_inquire_dimension( ncID, resxDimID, len = nxN)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_inq_dimid( ncID, "resy", resyDimID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_inquire_dimension( ncID, resyDimID, len = nyN)
      if(status .ne. nf90_noerr) call handle_err(status)

      if( (nxN .le. 0) .or. (nyN .le. 0) ) then
         write(UnitLog, *) 'ReadPixelSpacing #100: Something ',
     1        'wrong with the length of the pixel spacing ',
     2        'dimensions. nxN, nyN: ', nxN, nyN
        print *, 'ReadPixelSpacing #100: Something ',
     1        'wrong with the length of the pixel spacing ',
     2        'dimensions. nxN, nyN: ', nxN, nyN
        stop
      endif

       allocate( xPixSep(1:nxN),
     1     xPixCount(1:nxN),
     2     yPixSep(1:nyN),
     3     yPixCount(1:nyN),
     4     stat=ierr)
      
      if (ierr .ne. 0) then
         print *, 'Allocation error for LenXA, LenYA arrays. Exiting.'
         stop
      endif

c     Read the x dimension information

      status = nf90_inq_varid( ncID, 'XPixelSeparation', xPixSepID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_var( ncID, xPixSepID, xPixSep)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_inq_varid( ncID, 'XPixelCount', xPixCountID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_var( ncID, xPixCountID, xPixCount)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Read the y dimension information

      status = nf90_inq_varid( ncID, 'YPixelSeparation', yPixSepID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_var( ncID, yPixSepID, yPixSep)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_inq_varid( ncID, 'YPixelCount', yPixCountID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_var( ncID, yPixCountID, yPixCount)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Close the input file

      status = nf90_close( ncID)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Now find the x and y pixel separation for the maximum 
c      counts in the respective histograms.

      xMaxCount = -1

      do 1021 ix=1,nxN
            if(xPixCount(ix) .gt. xMaxCount) then
               xMaxCount = xPixCount(ix)
               ixSave = ix
            endif
 1021 continue
      
      if( (xMaxCount .le. 0) .or. (xPixSep(1) .eq. 0) ) then
         write(UnitLog, *) 'ReadPixelSpacing #200: The maximum count ',
     1        'in the x pixel separation histogram <= 0. nxN: ', nxN
        print *, 'ReadPixelSpacing #200: The maximum count ',
     1        'in the x pixel separation histogram <= 0. xMaxCount, ',
     2        ' xPixSep(1), nxN: ', xMaxCount, xPixSep(1), nxN
        stop
      endif

      xPixelSpacing = xPixSep(ixSave)

c     And for y.

      yMaxCount = -1

      do 1022 jy=1,nyN
            if(yPixCount(jy) .gt. yMaxCount) then
               yMaxCount = yPixCount(jy)
               jySave = jy
            endif
 1022    continue

      if( (yMaxCount .le. 0) .or. (yPixSep(1) .eq. 0) ) then
         write(UnitLog, *) 'ReadPixelSpacing #205: The maximum count ',
     1        'in the x pixel separation histogram <= 0. nyN: ', nyN
        print *, 'ReadPixelSpacing #205: The maximum count ',
     1        'in the x pixel separation histogram <= 0. yMaxCount, ',
     2        'yPixSep(1), nyN: ', yMaxCount, yPixSep(1), nyN
        stop
      endif

      yPixelSpacing = yPixSep(jySave)

      if(Debug .eq. 1) print *, 'ReadPixelSpacing #210: ixSave: ',
     1     ixSave, 'xPixelSpacing: ', xPixelSpacing, ' jySave: ',
     1     jySave, 'yPixelSpacing: ', yPixelSpacing

      if(Debug .eq. 1) print *, 'ReadPixelSpacing #999'

      end subroutine ReadPixelSpacing

c***********************************************************************
      subroutine GetAttributes( FileName, Variable, VariableType, 
     1     Offset, ScaleFactor,
     2     MissingValueReal, MissingValueInt2, MissingValueInt4)
c***********************************************************************
c     
c     This subroutine will open and read the offset, scale factor and
c      missing value attribes from FileName for Variable. 
c
c     INPUT
c
c      FileName - fully qualified name of input file.
c      Variable - string name of variable for which to read attributes.
c      VariableType - 'real' if Variable type is real
c                     'integer*2' if Variable type is integer*2
c                     'integer*4' if Variable type is integer*4
c
c     OUTPUT
c
c      MissingValueReal - Missing value if Variable is real
c      MissingValueInt2 - Missing value if Variable is integer*2
c      MissingValueInt4 - Missing value if Variable is integer*4
c      Offset - the offset to add to the input variable.
c      ScaleFactor - the scale factor by which to multiply the input.

      use netcdf

      implicit none

c     Parameter statements

      include 'ParameterStatements'
      
c****** Output varaiables

c     FileName - the name of the file from which to read attributes.
      character*319 FileName

c     MissingValueInt2 - the missing value passed if the value read in
c     is integer*2.
      integer*2 MissingValueInt2
c     MissingValueInt4 - the missing value passed if the value read in
c     is integer*4.
      integer*4 MissingValueInt4
c     MissingValueReal - the missing value passed if the value read in
c     is real.
      real*4 MissingValueReal

c     ncID - NetCDF ID for the input data set
      integer ncID

c     Offset - the offset to add to the input value.
      real Offset

c     ScaleFactor - the scale factor by which to multiply the input.
      real ScaleFactor

c     Variable - string with the variable name.
      character*50 Variable
c     VariableType - string with the variable name.
      character*12 VariableType

c     VarID - NetCDF ID for variable
      integer VarID

c      include 'netcdf.inc'

c----------------------------------start here --------------------------
      
      if(Debug .eq. 1) print *, 'GetAttributes #000: FileName::',
     1     trim(FileName), '::'

      status = nf90_open( FileName, nf90_nowrite, ncID)
      if (status .ne. nf90_noerr) call handle_err(status)

      status = nf90_inq_varid( ncID, trim(Variable), VarID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_att( ncID, VarID, 'add_offset', OffSet)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_att( ncID, VarID, 'scale_factor', ScaleFactor)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'GetAttributes #140: Offset: ', Offset,
     1     ' Scale factor: ', ScaleFactor

      if(VariableType .eq. 'real') then
         status = nf90_get_att( ncID, VarID, '_FillValue',
     1        MissingValueReal)
      elseif(VariableType .eq. 'integer*2') then
         status = nf90_get_att( ncID, VarID, '_FillValue',
     1        MissingValueInt2)
      elseif(VariableType .eq. 'integer*4') then
         status = nf90_get_att( ncID, VarID, '_FillValue',
     1        MissingValueInt4)
      endif
      if(status .ne. nf90_noerr) call handle_err(status)
      
      status = nf90_close(ncID)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'GetAttributes #999'

      end subroutine GetAttributes

c***********************************************************************
      subroutine GetLatLonAttributes(FileName)
c***********************************************************************
c     
c     This subroutine will open and read the offset, scale factor and
c      missing value attribes from the named lat/lon file for latitude.
c      The assumption is that the attributes for the longitude are the 
c      same.
c
c     INPUT
c
c      FileName - fully qualified name of input file.

      use netcdf

      implicit none

c     Parameter statements

      include 'ParameterStatements'
      
c****** Output varaiables

c     FileName - the name of the file from which to read attributes.
      character*319 FileName

c     ncID - NetCDF ID for the input data set
      integer ncID

c     Variable - string with the variable name.
      character*50 Variable

c     VarID - NetCDF ID for variable
      integer VarID

c      include 'netcdf.inc'

c----------------------------------start here --------------------------
      
      if(Debug .eq. 1) print *, 'GetLatLonAttributes #000: FileName::',
     1     trim(FileName), '::'

      status = nf90_open( FileName, nf90_nowrite, ncID)
      if (status .ne. nf90_noerr) call handle_err(status)

      status = nf90_inq_varid( ncID, 'latitude', VarID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_att( ncID, VarID, 'add_offset', LatLonOffsetIn)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_att( ncID, VarID, 'scale_factor', 
     1     LatLonScaleFactorIn)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_att( ncID, VarID, '_FillValue',
     1        LatLonFillValueIn)
      if(status .ne. nf90_noerr) call handle_err(status)
      
      status = nf90_close(ncID)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'GetLatLonAttributes #999: Offset: ', 
     1     LatLonOffsetIn, ' Scale factor: ', LatLonScaleFactorIn, 
     2     ' fill value: ', LatLonFillValueIn

      end subroutine GetLatLonAttributes

c***********************************************************************
      subroutine GetGradAttributes(FileName)
c***********************************************************************
c     
c     This subroutine will open and read the offset, scale factor and
c      missing value attribes from the named geoSobel file for
c      eastward_gradient. The assumption is that the attributes for the
c      wesward_gradient and gradient magnitude are the same.
c
c     INPUT
c
c      FileName - fully qualified name of input file.

      use netcdf

      implicit none

c     Parameter statements

      include 'ParameterStatements'
      
c****** Output varaiables

c     FileName - the name of the file from which to read attributes.
      character*319 FileName

c     ncID - NetCDF ID for the input data set
      integer ncID

c     Variable - string with the variable name.
      character*50 Variable

c     VarID - NetCDF ID for variable
      integer VarID

c      include 'netcdf.inc'

c----------------------------------start here --------------------------
      
      if(Debug .eq. 1) print *, 'GetGradAttributes #000: FileName::',
     1     trim(FileName), '::'

      status = nf90_open( FileName, nf90_nowrite, ncID)
      if (status .ne. nf90_noerr) call handle_err(status)

      status = nf90_inq_varid( ncID, 'eastward_gradient', VarID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_att( ncID, VarID, 'add_offset', GradOffsetIn)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_att( ncID, VarID, 'scale_factor', 
     1     GradScaleFactorIn)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_att( ncID, VarID, '_FillValue',
     1        GradFillValueIn)
      if(status .ne. nf90_noerr) call handle_err(status)
      
      status = nf90_close(ncID)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'GetGradAttributes #999: Offset: ', 
     1     GradOffsetIn, ' Scale factor: ', GradScaleFactorIn, 
     2     ' fill value: ', GradFillValueIn

      end subroutine GetGradAttributes

c***********************************************************************
      subroutine GetSSTAttributes(FileName)
c***********************************************************************
c     
c     This subroutine will open and read the offset, scale factor and
c      missing value attribes from the named SST file for sst.
c
c     INPUT
c
c      FileName - fully qualified name of input file.

      use netcdf

      implicit none

c     Parameter statements

      include 'ParameterStatements'
      
c****** Output varaiables

c     FileName - the name of the file from which to read attributes.
      character*319 FileName

c     ncID - NetCDF ID for the input data set
      integer ncID

c     Variable - string with the variable name.
      character*50 Variable
c     VarID - NetCDF ID for variable
      integer VarID

c      include 'netcdf.inc'

c----------------------------------start here --------------------------
      
      if(Debug .eq. 1) print *, 'GetSSTAttributes #000: FileName::',
     1     trim(FileName), '::'

      status = nf90_open( FileName, nf90_nowrite, ncID)
      if (status .ne. nf90_noerr) call handle_err(status)

      Variable = Blankfill(:len(Variable))
      Variable = 'median_' // trim(ParameterName_lc)
      status = nf90_inq_varid( ncID, Variable, VarID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_att( ncID, VarID, 'add_offset', sstOffSetIn)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_att( ncID, VarID, 'scale_factor', 
     1     sstScaleFactorIn)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_att( ncID, VarID, '_FillValue',
     1        sstFillValueIn)
      if(status .ne. nf90_noerr) call handle_err(status)
      
      status = nf90_close(ncID)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'GetSSTAttributes #999: Offset: ', 
     1     sstOffsetIn, ' Scale factor: ', sstScaleFactorIn, 
     2     ' fill value: ', sstFillValueIn

      end subroutine GetSSTAttributes

c***********************************************************************
      subroutine GetZenithAngleAttributes(FileName)
c***********************************************************************
c     
c     This subroutine will open and read the offset, scale factor and
c      missing value attribes from the named zenith angle file for 
c      the zenith angle.
c
c     INPUT
c
c      FileName - fully qualified name of input file.

      use netcdf

      implicit none

c     Parameter statements

      include 'ParameterStatements'
      
c****** Output varaiables

c     FileName - the name of the file from which to read attributes.
      character*319 FileName

c     ncID - NetCDF ID for the input data set
      integer ncID

c     Variable - string with the variable name.
      character*50 Variable

c     VarID - NetCDF ID for variable
      integer VarID

c      include 'netcdf.inc'

c----------------------------------start here --------------------------
      
      if(Debug .eq. 1) print *, 'GetZenithAngleAttributes #000: ',
     1     'FileName::', trim(FileName), '::'

      status = nf90_open( FileName, nf90_nowrite, ncID)
      if (status .ne. nf90_noerr) call handle_err(status)

      status = nf90_inq_varid( ncID, 'zenith_angle', VarID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_att( ncID, VarID, 'add_offset', 
     1     ZenithAngleOffsetIn)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_att( ncID, VarID, 'scale_factor', 
     1     ZenithAngleScaleFactorIn)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_att( ncID, VarID, '_FillValue',
     1        ZenithAngleFillValueIn)
      if(status .ne. nf90_noerr) call handle_err(status)
      
      status = nf90_close(ncID)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'GetZenithAngleAttributes #999: ',
     1     'Offset: ', ZenithAngleOffsetIn, 
     2     ' Scale factor: ', ZenithAngleScaleFactorIn, 
     3     ' fill value: ', ZenithAngleFillValueIn

      end subroutine GetZenithAngleAttributes

c***********************************************************************
      subroutine handle_err(status)
c***********************************************************************

      use netcdf

      implicit none

      integer*4 status

c      include 'netcdf.inc'

c----------------------------------start here --------------------------

      if(status .ne. nf90_noerr)then
         print *, nf90_strerror(status)
         print *, 'netCDF returned Error code ', status 
         stop 'Stopped'
      endif
      end

c***********************************************************************
      subroutine calend( year, month, days)
c***********************************************************************
c
c  This function returns the number of days in the month and year
c  provided.
c
      integer year, month, days


c----------------------------------start here --------------------------

      go to (310, 280, 310, 300, 310, 300, 
     1     310, 310, 300, 310, 300, 310) month
c            Jan  Feb  Mar  Apr  May  Jun  
c            Jul  Aug  Sep  Oct  Nov  Dec

c February: has 29 days in leap years, 28 otherwise

 280  if( (mod(year,400) .eq. 0) .or. ( (mod(year,100) .ne. 0)
     1  .and. (mod(year,4) .eq. 0) ) ) then

         days = 29

      else

         days = 28

      endif
      
      go to 1000

c Short months

 300  days = 30

      go to 1000

c Long months

 310  days = 31

c Here when all done

 1000 continue

      end

c***********************************************************************
      subroutine OpenLogFile( YearC, MonthC, ProgName)
c***********************************************************************
c     
c     Open the Log file - this file contains processing information
c     in it; i.e., stuff written to the terminal,
c     
c     Written by Peter Cornillon, University of Rhode Island,
c     pcornillon@me.com 18 April 2009
c
c     4/17/12 - PCC - Changed printout for filename.
c     4/28/12 - PCC - Moved log files from BaseDir to 
c      SupportingFiles/LogFiles/ under BaseDir
c     
c.......................................................................

      implicit none

c**********Parameter statements

      include 'ParameterStatements'
      
c**********Functions
c     

c**********General Variables.

c     exis -  .true. if file exists when inquiring
      logical exis
c     i - a general loop parameter
      integer i
c     LastChar1 - the location of the last character of a string. 
      integer LastChar1
c     LastChar2 - the last character of another string. 
      integer LastChar2
c     LogFileName - filename for output of this run of the program
      character*319 LogFileName
c     Numbrs - ascii representation of the ten digits
      character*1 Numbrs(0:9)

c**********Data statements

      data Numbrs/ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'/

c----------------------------------start here --------------------------

      i = 0
 554  continue

      LogFileName = BlankFill(:len(LogFileName))
      LogFileName = trim(BaseDir) // 'SupportingFiles/LogFiles/Log_' // 
     1     trim(ProgName) // '_' // YearC // MonthC // '_' // Numbrs(i)

      inquire(file=LogFileName, Exist=exis)
      if(exis .eqv. .true.) then
         i = i + 1
         if(i .gt. 9) then
            stop 'Too many old history files, only 10 allowed'
         endif
         go to 554
      endif

      if(debug .eq. 1) print *, 'OpenLogFile #100::', 
     1     'LogFileName: ', trim(LogFileName), '::'

      open(unit=UnitLog, file=LogFileName, status='new')

      end subroutine OpenLogFile

c***********************************************************************
      subroutine ReplaceBaseDir(FileName)
c***********************************************************************
c     
c     The input disk may be mounted on a different file system than
c     it was when the inventory was written OR it may have been 
c     copied to another disk. Since the inventory used the FULL
c     filename, we need to replace the base portion of the filename
c     read in with that of the disk that we hope the data are on.
c
c     Written by Peter Cornillon, University of Rhode Island,
c     pcornillon@me.com 18 April 2009
c     
c********** Functions
c     
c********** Subroutine arugments
c     
c     BaseDir - the directory in which the subdirectories for the input
c     . images and the subdirectories for the output images exist.
c     UnitLog - unit number to which run history will be written
c
     
c********** General Variables.
c
c     FileName - the filename read from the inventory. The modified
c     . filename will be here also.
c     ModifiedFileName - the name with replaced basedir. Copied to
c     . filename for return
c     
c     Numbrs - ascii representation of the ten digits
c     
c     LocBaseDir - the location of the satellite name in BaseDir
c     LocFileName  - the location of the satellite name in FileName
c
      implicit none

c     Functions

c     Parameter statements

      include 'ParameterStatements'
      
c     General variables

      character*319 FileName
      character*319 ModifiedFileName
c     LastChar1 - the location of the last character of a string. 
      integer LastChar1

      integer LocBaseDir, LocTemp, LocFileName, i

      logical exis

c----------------------------------start here --------------------------

      if(Debug .eq. 1) then
         print *, 'ReplaceBaseDir #100. SatName::', trim(SatName), '::'
         print *, 'FileName::', trim(FileName), '::'
      endif

      LocBaseDir = index( BaseDir, trim(SatName) ) - 1

      LocTemp = index( BaseDir(LocBaseDir+2:), trim(SatName) )

      if(LocTemp .gt. 0) then
         write(6,*) 'Be careful the satellite name occurs more ',
     1        'than once in BaseDir. Continuing using 1st occurrence.'
         write(UnitLog,*) 'Be careful the satellite name occurs more ',
     1        'than once in BaseDir. Continuing using 1st occurrence.'
       endif

c     Now get location of satellite name in FileName

      LocFileName = index( FileName, trim(SatName) )

c     Modify the file name.

      ModifiedFileName = BlankFill(:len(ModifiedFileName))
      ModifiedFileName = trim(BaseDir(:LocBaseDir)) // 
     1     trim(FileName(LocFileName:) )

      if(Debug .eq. 1) then
         print *, 'ReplaceBaseDir #100: trim(SatName)::', 
     1         trim(SatName), '::'
         print *, 'ReplaceBaseDir #101: LocBaseDir:', LocBaseDir,
     1        ' trim(BaseDir)::', trim(BaseDir), '::'
         print *, 'ReplaceBaseDir #102: LocFileName: ', LocFileName,
     1        ' trim(FileName)::', trim(FileName), '::'
         print *, 'ReplaceBaseDir #103: LocFileName: ', LocFileName,
     1        ' trim(ModifiedFileName)::', trim(ModifiedFileName), '::'
      endif

      FileName = ModifiedFileName

      end

c***********************************************************************
      subroutine GetDateCharStrings( year, month, day, hour, minute,
     1     YearC, MonthC, DayC, HourC, MinuteC)
c***********************************************************************
c     
c     This subroutine returns strings for the year, month, day, hour
c     and minute to use in building filenames. 
c     
c     Written by Peter Cornillon, University of Rhode Island,
c     pcornillon@me.com 18 April 2009
c     

c**********General Variables.
c     
c     Numbrs - ascii representation of the ten digits
c     
c     Year, Month, Day, Hour - year, month, day and hour working on
c     Days - day in month for the month and year working on
c     
c     YearC, MonthC, DayC, HourC, MinuteC - more dummy variables
c     Yr1, Yr2, Yr3, Yr4. Mn1, Mn2, Dy1, Dy2, Hr1 and Hr2
c     ...dummy variables used to build file name.
c     
      implicit none

c     Functions

c     Parameter statements

      include 'ParameterStatements'
      
c     General variables

      character*1 Numbrs(0:9)

      integer Year, Month, Days, Day, Hour, Minute

      integer Yr1, Yr2, Yr3, Yr4, Mn1, Mn2, Dy1, Dy2, Hr1, Hr2

c     Data statements

      data Numbrs/ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'/


c----------------------------------start here --------------------------

c     First generate a character string for the year

      Yr1 = Year / 1000
      Yr2 = (Year - Yr1 * 1000) / 100
      Yr3 = (Year - Yr1 * 1000 - Yr2 * 100) / 10
      Yr4 = Year - Yr1 * 1000 - Yr2 * 100 - Yr3 * 10
      YearC = Numbrs(Yr1) // Numbrs(Yr2) // 
     1     Numbrs(Yr3) // Numbrs(Yr4)

c     Next a character string for the month

      Mn1 = Month / 10
      Mn2 = Month - Mn1 * 10

      MonthC = Numbrs(Mn1) // Numbrs(Mn2)

c     And the day

      Dy1 = Day / 10
      Dy2 = Day - Dy1 * 10

      DayC = Numbrs(Dy1) // Numbrs(Dy2)

c     And now for the hour


      Hr1 = Hour / 10
      Hr2 = Hour - Hr1 * 10

      HourC = Numbrs(Hr1) // Numbrs(Hr2)

c     And minutes

      MinuteC = '00'

      end

c***********************************************************************
      subroutine GetDateCharStringsYrDay( year, yearday, month, day,
     1     hour, minute, YearC, MonthC, DayC, HourC, MinuteC, YearDayC)
c***********************************************************************
c
c     This subroutine returns strings for the year, month, day, hour
c     and minute to use in building filenames.
c

c**********General Variables.
c
c     Numbrs - ascii representation of the ten digits
c
c     Year, Month, Day, Hour - year, month, day and hour working on
c     Days - day in month for the month and year working on
c
c     YearC, MonthC, DayC, HourC, MinuteC - more dummy variables
c     Yr1, Yr2, Yr3, Yr4. Mn1, Mn2, Dy1, Dy2, Hr1 and Hr2
c     ...dummy variables used to build file name.
c
      implicit none

c     Functions

c     Parameter statements

      include 'ParameterStatements'

c     General variables

      character*1 Numbrs(0:9)

      integer Year, YearDay, Month, Days, Day, Hour, Minute

      integer Yr1, Yr2, Yr3, Yr4, Mn1, Mn2, Dy1, Dy2, Hr1, Hr2

      integer Ydy1, Ydy2, Ydy3

      integer remainder

      logical leapYear

c     Data statements

      data Numbrs/ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'/


c----------------------------------start here --------------------------

c     First generate a character string for the year

      Yr1 = Year / 1000
      Yr2 = (Year - Yr1 * 1000) / 100
      Yr3 = (Year - Yr1 * 1000 - Yr2 * 100) / 10
      Yr4 = Year - Yr1 * 1000 - Yr2 * 100 - Yr3 * 10
      YearC = Numbrs(Yr1) // Numbrs(Yr2) //
     1     Numbrs(Yr3) // Numbrs(Yr4)

      remainder = mod(Year,4)

      if (remainder .gt. 0) then
          leapYear = .false.
      else
          leapYear = .true.
          remainder = mod(Year,100)
          if (remainder .eq. 0) then
              leapYear = .false.
          endif
          remainder = mod(Year,400)
          if (remainder .eq. 0) then
              leapYear = .true.
          endif
      endif

      if (yearday .lt. 32) then
        Month = 1
        Day = yearday
      endif

      if (leapYear) then
        if (yearday .ge. 32 .and. yearday .le. 60) then
            Month = 2
            Day = yearday - 31
        endif
        if (yearday .ge. 61 .and. yearday .le. 91) then
            Month = 3
            Day = yearday - 60
        endif
        if (yearday .ge. 92 .and. yearday .le. 121) then
            Month = 4
            Day = yearday - 91
        endif
        if (yearday .ge. 122 .and. yearday .le. 152) then
            Month = 5
            Day = yearday - 121
        endif
        if (yearday .ge. 153 .and. yearday .le. 182) then
            Month = 6
            Day = yearday - 152
        endif
        if (yearday .ge. 183 .and. yearday .le. 213) then
            Month = 7
            Day = yearday - 182
        endif
        if (yearday .ge. 214 .and. yearday .le. 244) then
            Month = 8
            Day = yearday - 213
        endif
        if (yearday .ge. 245 .and. yearday .le. 274) then
            Month = 9
            Day = yearday - 244
        endif
        if (yearday .ge. 275 .and. yearday .le. 305) then
            Month = 10
            Day = yearday - 274
        endif
        if (yearday .ge. 306 .and. yearday .le. 335) then
            Month = 11
            Day = yearday - 305
        endif
        if (yearday .ge. 336 .and. yearday .le. 366) then
            Month = 12
            Day = yearday - 335
        endif
      else
        if (yearday .ge. 32 .and. yearday .le. 59) then
            Month = 2
            Day = yearday - 31
        endif
        if (yearday .ge. 60 .and. yearday .le. 90) then
            Month = 3
            Day = yearday - 59
        endif
        if (yearday .ge. 91 .and. yearday .le. 120) then
            Month = 4
            Day = yearday - 90
        endif
        if (yearday .ge. 121 .and. yearday .le. 151) then
            Month = 5
            Day = yearday - 120
        endif
        if (yearday .ge. 152 .and. yearday .le. 181) then
            Month = 6
            Day = yearday - 151
        endif
        if (yearday .ge. 182 .and. yearday .le. 212) then
            Month = 7
            Day = yearday - 181
        endif
        if (yearday .ge. 213 .and. yearday .le. 243) then
            Month = 8
            Day = yearday - 212
        endif
        if (yearday .ge. 244 .and. yearday .le. 273) then
            Month = 9
            Day = yearday - 243
        endif
        if (yearday .ge. 274 .and. yearday .le. 304) then
            Month = 10
            Day = yearday - 273
        endif
        if (yearday .ge. 305 .and. yearday .le. 334) then
            Month = 11
            Day = yearday - 304
        endif
        if (yearday .ge. 335 .and. yearday .le. 365) then
            Month = 12
            Day = yearday - 334
        endif
      endif

c     Next a character string for the month

      Mn1 = Month / 10
      Mn2 = Month - Mn1 * 10

      MonthC = Numbrs(Mn1) // Numbrs(Mn2)

c     And the day

      Dy1 = Day / 10
      Dy2 = Day - Dy1 * 10

      DayC = Numbrs(Dy1) // Numbrs(Dy2)

c     And now for the hour

      Hr1 = Hour / 10
      Hr2 = Hour - Hr1 * 10

      HourC = Numbrs(Hr1) // Numbrs(Hr2)

c     And minutes

      MinuteC = '00'

      Ydy1 = Yearday / 100
      Ydy2 = (Yearday - Ydy1 * 100) / 10
      Ydy3 = (Yearday - Ydy1 * 100) - Ydy2 * 10
      YeardayC = Numbrs(Ydy1) // Numbrs(Ydy2) // Numbrs(Ydy3)

      end


c***********************************************************************
      subroutine ConvertIntegerToString( Variable, VariableC)
C***********************************************************************
c     
c     This subroutine returns strings for the variable passed in. It
c     . removes leading 0s. It will only work for integers up to 8
c     . digits.
c     
c     Written by Peter Cornillon, University of Rhode Island,
c     pcornillon@me.com 26 June2009
c     

c**********General Variables.
c     
c     Numbrs - ascii representation of the ten digits
c     
      implicit none

c     Functions

c     Parameter statements

      include 'ParameterStatements'
      
c     Variables
c
c     Numbrs - character representation of the numbers from 0 to 9
      character*1 Numbrs(0:9)

c     Va1-8 - single digits of Variable
      integer Va1, Va2, Va3, Va4, Va5, Va6, Va7, Va8
c     Variable - input variable
      integer Variable
c     VariableC - the string representation of the variable.
      character*8 VariableC

c     Data statements

      data Numbrs/ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'/


c----------------------------------start here --------------------------

c     

      Va1 = Variable / 10000000
      Va2 = (Variable - Va1 * 10000000) / 1000000
      Va3 = (Variable - Va1 * 10000000 - Va2 * 1000000) / 100000
      Va4 = (Variable - Va1 * 10000000 - Va2 * 1000000 - 
     1     Va3 * 100000) / 10000
      Va5 = (Variable - Va1 * 10000000 - Va2 * 1000000 - 
     1     Va3 * 100000 - Va4 * 10000) / 1000 
      Va6 = (Variable - Va1 * 10000000 - Va2 * 1000000 - 
     1     Va3 * 100000 - Va4 * 10000 - Va5 * 1000) / 100 
      Va7 = (Variable - Va1 * 10000000 - Va2 * 1000000 - 
     1     Va3 * 100000 - Va4 * 10000 - Va5 * 1000 -
     2     Va6 * 100) / 10 
      Va8 = Variable - Va1 * 10000000 - Va2 * 1000000 - 
     1     Va3 * 100000 - Va4 * 10000 - Va5 * 1000 -
     2     Va6 * 100 - Va7 * 10

      if(Va1 .gt. 0) then
         VariableC = Numbrs(Va1) // Numbrs(Va2) // Numbrs(Va3) // 
     1        Numbrs(Va4) // Numbrs(Va5) // Numbrs(Va6) // Numbrs(Va7)
     2         // Numbrs(Va8)
      elseif(Va2 .gt. 0) then
         VariableC = Numbrs(Va2) // Numbrs(Va3) // Numbrs(Va4) // 
     1        Numbrs(Va5) // Numbrs(Va6) // Numbrs(Va7) // Numbrs(Va8) 
      elseif(Va3 .gt. 0) then
         VariableC = Numbrs(Va3) // Numbrs(Va4) // Numbrs(Va5) //
     1        Numbrs(Va6) // Numbrs(Va7) // Numbrs(Va8) 
      elseif(Va4 .gt. 0) then
         VariableC = Numbrs(Va4) // Numbrs(Va5) // Numbrs(Va6) // 
     1        Numbrs(Va7) // Numbrs(Va8) 
      elseif(Va5 .gt. 0) then
         VariableC = Numbrs(Va5) // Numbrs(Va6) // Numbrs(Va7) // 
     1        Numbrs(Va8) 
      elseif(Va6 .gt. 0) then
         VariableC = Numbrs(Va6) // Numbrs(Va7) // Numbrs(Va8) 
      elseif(Va7 .gt. 0) then
         VariableC = Numbrs(Va7) // Numbrs(Va8) 
      else  
         VariableC = Numbrs(Va8)
      endif

      end

c***********************************************************************
      real*8 function SecondsSince( Year, Month, Day, Hour, Minute,
     1     Second, YearRef, MonthRef, DayRef, HourRef, MinuteRef,
     2     SecondRef)
c***********************************************************************
c     
c     This function returns the seconds since the reference time given.
c     
c     Year, Month, Day, Hour, Minute, Second - year, month, day, hour 
c     minute and second working on
c     YearRef, MonthRef, DayRef, HourRef, MinuteRef, SecondReg - 
c     reference to use in calculating SecondsSince.
c
c Tested with /Users/petercornillon/Desktop/Dropbox/ComputerPrograms/
c Matlab_Utilities/Fronts/TestSecondsSince.f
c     
c (datenum([2011 03 12 05 10 30]) - datenum([1970 1 1 0 0 0]))* 86400
c
c ans =
c
c 1299906630.00
c
      implicit none

c     Functions

      real*4 JulianDay 

c     General variables

      real*8 SS, SSRef

      integer Year, Month, Day, Hour, Minute, Second
      integer YearRef, MonthRef, DayRef, HourRef, MinuteRef, SecondRef

c----------------------------------start here --------------------------
c
c     First get the Julian day corresponding to the reference day.

      SSRef = JulianDay( YearRef, MonthRef, DayRef) 
      SSRef = SSRef * 86400 + HourRef * 3600.d0 + MinuteRef * 60.d0 + 
     1     SecondRef

c     Now the Julian Day corresponding to the day of interest

      SS = JulianDay( Year, Month, Day) 
      SS = SS * 86400 + Hour * 3600.d0 + Minute * 60.d0 + Second

c     Finally the difference between the two plus the 

      SecondsSince = SS - SSRef

      end

c***********************************************************************
      real*4 function JulianDay( Yr, Mn, Da)
c***********************************************************************
c     
c     This function calculates the Julian Day. See: 
c     http://en.wikipedia.org/wiki/Julian_day
c     
c Tested with /Users/petercornillon/Desktop/Dropbox/ComputerPrograms/
c Matlab_Utilities/Fronts/TestSecondsSince.f
c
      implicit none

      integer Yr, Mn, Da
      real*4 Year, Month, Day
      real*4 a, y, m

      Year = Yr
      Month = Mn
      Day = Da

      a = ifix((14.0 - Month) / 12.0)

      y = Year + 4800 - a

      m = Month + 12.0 * a - 3.0

      JulianDay = Day + ifix((153.0 * m + 2.0) / 5.0) + 365.0 * y +
     1     ifix(y / 4.0) - ifix(y / 100.0) + ifix(y / 400.0) - 32045.5

      end function JulianDay

c***********************************************************************
      subroutine FileStatus( FileName, FileExists, Compressed,
     1     YearC, TempFileName, Decompress, DecompressInPlace)
c***********************************************************************
c     
c     This subroutine builds two names based on the one passed in,
c     one ending in .nc and the other in .nc.gz. (We only allow for
c     gz compression.) If neither exists, FileExists is set to
c     .false., otherwise it is .true. If an uncompressed version 
c     exists, Compressed is set to .false., otherwise .true. Compressed
c     has no meaning if the file does not exist. If it exists, 
c     and Decompress is .true., it will be decompressed to 
c     TempFileName if DecompressInPlace is .false. and to the same 
c     location (sans .gz) if DecompressInPlace is .true. If 
c     decompressed to a temporary filename, the filename is built 
c     with random numbers to avoid collisions with other possible 
c     temporary files in the same directory - other programs may be 
c     running at the same time from the same directory creating their 
c     own temporary files. FileName will be returned as the 
c     uncompressed version. If the file is compressed, but Decompress 
c     is .false. or DecompressInPlace is .true., TempFileName is 
c     meaningless.
c     
c     The reason for decompressing some files in place is that you may
c     be adding to them, for example in the case of the merge/thin 
c     file. In other cases we are only decompressing temporarily so
c     no need to decompress and then compress when we are done. 
c     
c     Written by Peter Cornillon, University of Rhode Island,
c     pcornillon@me.com 18 April 2009
c     
c     PCC 11-May-2009 - Call to system should be after 2nd to last 
c     endif, not before. I moved it. There was also a problem with
c     the construction of the compressed filename. I fixed this as
c     well. (It was truncating the gile name at .gz, so it looked like
c     the original filename, not the compressed filename.)
c     
      implicit none

c******Functions

c******Parameter statements

      include 'ParameterStatements'
      
c******Input variables

c     Decompress - .true. if a compressed file is found and it is to
c     . be decompressed to TempFileName, otherwise .false.
      logical Decompress
c     DecompressInPlace - .true. if a compressed file is found and it 
c     . is to be decompressed to with the same name (no .gz extension
c     . of course. .false. it will build and decompress to a temporary
c     . file. It will not decompress at all if Decompress is .false.
      logical DecompressInPlace
c     FileName - the filename passed in.
      character*319 FileName

c******Returned variables
c     
c     FileExists - .true. if either the compressed or the 
c     . decompressed version of this file exists, .false. otherwise.
      logical FileExists
c     Compressed - .true. if the file we are seeking is compressed.
      logical Compressed

c******General variables

c     exis - logical to test for the existence of the input file.
      logical exis
c     exis_gz - logical to test for the existence of the input file 
c     . with .gz appended to it.
      logical exis_gz
c     FileName_gz - the gz compressed input filename
      character*319 FileName_gz
c     i - a loop index
      integer i
c     LastChar1 - the location of the last character of a string. 
      integer LastChar1
c     LastChar2 - the last character of another string.
      integer LastChar2
c     Now - current time, (1) hour, (2) minute, (3) second
      integer*4 now(3)
c     Numbrs - ascii representation of the ten digits
      character*1 Numbrs(0:9)
c     r - a random integer between 0 and 9
      integer r
c     RanNum - the 0 to 1 random number generated
      real RanNum
c     RandNumber - a string of 5 random digits.
      character*5 RandNumber
c     Seed - seed value for random number generator
      integer Seed
c     ShortFileName - the part of the filename from the last '/' to 
c     . the next '.'
      character*319 ShortFileName 
c     TempFileName - the name of the temporary file generated
      character*319 TempFileName

c*******Data statements
      data Numbrs/ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'/

c----------------------------------start here --------------------------

      FileExists = .true.

c     Get base of filename.

      if(debug .eq. 1) print *,'Filestatus # 100: FileName::', 
     1     trim(FileName), '::'

c     Check to see if there is a .gz extension to the input file.

      Loc1 = index( FileName, '.gz')

      inquire( File=trim(FileName), Exist=exis)

c     Check to see if the file exists. If it does AND if there is not a
c     . .gz extension, set compressed flag to false and return, 
c     . otherwise continue.

      if( (exis .eqv. .true.) .and. (Loc1 .eq. 0) ) then
         if(debug .eq. 1)  print *, 'Filestatus # 105. exis: ',
     1        exis
         Compressed = .false.
         return
      endif

c     Didn't find a file yet or the file that was found has a .gz 
c     . extension. Check to see if a compressed version of the file
c     . exists.

      if(Loc1 .gt. 0) then
         FileName_gz = FileName
      else
         FileName_gz = BlankFill(:len(FileName_gz))
         FileName_gz = trim(FileName) // '.gz'
      endif

      inquire( File=trim(FileName_gz), Exist=exis_gz)

      if(debug .eq. 1)  print *, 'Filestatus #110 ',
     1     'FileName_gz::', trim(FileName_gz), '::'

      if(exis_gz .eqv. .true.) then

         if(debug .eq. 1)  print *, 'Filestatus #112. exis_gz: ',
     1        exis_gz

         Compressed = .true.

         if(Decompress .eqv. .false.) return

         if(debug .eq. 1)  print *, 'Filestatus #113. Decompressing.'

c     Found a compressed version, decompress it.

         if(DecompressInPlace .eqv. .true.) then

            if(debug .eq. 1)  print *, 'Filestatus #114. ',
     1           'Decompressing in place.'

c     Here if the file is to be decompressed in place. An uncompressed
c     version of this file should not exist because we returned above
c     if it did - I hope.

            Command = BlankFill(:len(Command))
            Command = 'gunzip ' // trim(FileName_gz)

c            if(debug .eq. 1) print *, 'Command: ', Command
         else

c     Here if a temporary file is to be created. The program will
c     generate a 5 digit random number to append to the tmp file name
c     to avoid collisions with other programs creating a temp file in
c     the same directory.

            Loc2 = 1
 1000       continue
            Loc1 = index(FileName(Loc2:), '/')
            Loc2 = Loc1 + Loc2
            if(Loc1 .gt. 0) go to 1000
            Loc1 = Loc2 - 1

            Loc2 = index(FileName(Loc1:), '.') - 3

            ShortFileName = BlankFill(:len(ShortFileName))
            ShortFileName = trim(BaseDir) // 'TmpDir/' //
     1           FileName(Loc1+1:Loc1+Loc2+1)

c     Generate a random number for the end of the temporary filename.
c     - Use the current seconds for the seed of the random number 
c     - generator.

            call itime(now)     ! now(1)=hour, (2)=minute, (3)=second
            Seed = now(3)
            RanNum = ran(Seed)
            do 200 i=1,5
               RanNum = ran()
               r = floor(RanNum * 10)
               RandNumber(i:i) = Numbrs(r)
 200        continue
 
            TempFileName = BlankFill(:len(TempFileName))
            TempFileName = trim(ShortFileName) // 
     1           '_' // RandNumber // '.nc'

            if(debug .eq. 1)  print *, 'Filestatus #120. ',
     1           'Decompressing to::', trim(TempFileName), '::'

            Command = BlankFill(:len(Command))
            Command = 'gzip -dc ' // trim(FileName_gz) // ' > '
     1           // trim(TempFileName)

         endif

         if(debug .eq. 1)  print *, 'Filestatus #130. ',
     1        'Decompressing command: ', trim(Command)

         call system(Command)

         return

      endif

      FileExists = .false.

      end

c***********************************************************************
      subroutine CleanUp( ncID, FileName, Compressed, TempFileName,
     1     Compress)
c***********************************************************************
c     
c     This subroutine cleans up a file that has been opend for read.
c     It closes it and, if it was a temporary file, it deletes it,
c     otherwise it compresses it. If ncID < 0, then this file is not
c     a netCDf file, so do not try to close it.
c     
c     Written by Peter Cornillon, University of Rhode Island,
c     pcornillon@me.com 18 April 2009
c     
c     PCC 11 May 2009 - added check for netCDF; i.e., ncID >= 0. There
c     was also a problem in the logic that could lead to a compressed
c     version of a file being deleted when an uncompressed version 
c     did not exist - I'm not talking about the temporary file here, so
c     I put in a check to be sure that we don't accidentally delete a
c     file that we want to keep.
c     

      use netcdf

      implicit none

c******Functions
c     

c******Parameter statements
c     
      include 'ParameterStatements'

c******Input variables
c     
c     ncID - NetCDF ID for the file to cleanup
      integer ncID
c     FileName - the original name of thefile that is to be dealt with.
c     . If Compressed is .true. this filename is ignored. If false,
c     . this file will be gz comressed.
      character*319 FileName
c     Compressed - .true. if the file was compressed.
      logical Compressed
c     TempFileName - the name of the temporary file. This file will be
c     . deleted if the file was compressed.
      character*319 TempFileName
c     Compress - .true. if the file is to be compressed, .false. 
c     - otherwise.
      logical Compress
      
c******General variables

c     AnotherTempFileName - dummy temporary file to use in the call
c     . to FileStatus.
      character*319 AnotherTempFileName
c     Decompress - dummy flag in this case
      logical Decompress
c     DecompressInPlace - dummy flag in this case
      logical DecompressInPlace
c     FileExists - .true. if either the compressed or the 
c     . decompressed version of this file exists, .false. otherwise.
      logical FileExists
c     LastChar1 - the location of the last character of a string. 
      integer LastChar1

c      include 'netcdf.inc'

c----------------------------------start here --------------------------

      if(debug .eq. 1) then
         print *, 'Cleanup #100 ncID: ', ncID
         print *, 'Cleanup #101 FileName: ', FileName
         print *, 'Cleanup #102 Compressed: ', Compressed
         print *, 'Cleanup #103 TempFileName: ', TempFileName
         print *, 'Cleanup #104 Compress: ', Compress
      endif

      if( ncID .gt. -1) then
         status = nf90_close(ncID)
         if(status .ne. nf90_noerr) call handle_err(status)
      endif

      if(Compressed .eqv. .true.) then
         Command = BlankFill(:len(Command))
         Command = 'rm ' // trim(TempFileName) 
         call system(Command)
      else

c     Check to see if a compressed version of this file also exists.
c     If it does remove the compressed version first. DANGEROUS.

         TempFileName = BlankFill(:len(TempFileName))
         TempFileName = trim(FileName) // '.gz' 

c     Does a compressed file for the input file exist?

            inquire(File=trim(TempFileName), Exist=FileExists)

            if(debug .eq. 1) print *, 'Cleanup #110: ', FileName

            if(FileExists .eqv. .true.) then

c     A compressed version of this file already exists. Check to see
c     if a decompressed version exists.

               inquire(File=trim(FileName), Exist=FileExists)

               if(FileExists .eqv. .true.) then

c     OK, a decompressed version exists delete the compressed version 
c     and compress the decompressed version.

                  Command = BlankFill(:len(Command))
                  Command = 'rm ' // trim(TempFileName)
                  call system(Command)
                  if(Compress) then
                     Command = BlankFill(:len(Command))
                     Command = 'gzip ' // trim(FileName)
                     call system(Command)
                  endif
               endif

               if(debug .eq. 1) print *, 'Cleanup #120: ', FileName

            else

c     Here if a compressed version does not exist - I'm assuming that
c     there is a decompressed version - if not crash and burn. I guess
c     that I could check for it, but I'm too lazy.

               if(Compress) then
                  Command = BlankFill(:len(Command))
                  Command = 'gzip ' // trim(FileName)
                  call system(Command)
               endif

               if(debug .eq. 1) print *, 'Cleanup #130: ', 
     1              FileName

            endif
      endif

      end

c***********************************************************************
      subroutine ElapsedTime( LastTime, NewTime, Message)
C***********************************************************************
c     
c     This subroutine calculates the elapsed processor time since
c     the last invocation of this routinue. If LastTime is -999, the
c     routinue assumes that this is the first call and writes nothing
c     out.
c     
c     Written by Peter Cornillon, University of Rhode Island,
c     pcornillon@me.com 18 April 2009
c     

c**********Functions
c     
c     etime - Fortran intrinsic for elapsed time.

c************System timing variables
c     
c     CPUTime - CPU time used by this program to date
c     LastCPUTime - CPUTime the last time ElapsedTime was called.
c     TimeArgs - after call to etime, the first TimeArgs is the user
c     . portion of the elapsed time and the second one is the system
c     . portion.
c     LastTimeArgs

c***********Other variables
c     
c     Message - the message to print out.
c     
      implicit none

c     Functions

      real etime

c****** Parameter statements

      include 'ParameterStatements'
      
c     Timing Variables

      real LastTime, NewTime
      real TimeArgs(2), LastTimeArgs(2)

c     Other variables

c     LocEOS - location of the string EOS in the Message string passed c     . in.
      integer LocEOS

      character*50 Message

c----------------------------------start here --------------------------

      if(LastTime .eq. -99.0) then
         LastTime = etime(LastTimeArgs)
      else
         NewTime = etime(TimeArgs)
         LocEOS = index( Message, 'EOS')
         print *, Message(:LocEOS), 
     1        NewTime - LastTime, ' User portion: ',
     2        TimeArgs(1) - LastTimeArgs(1)
         LastTime = NewTime
         LastTimeArgs(2) = TimeArgs(2)
         LastTimeArgs(2) = TimeArgs(2)
      endif

      end

c***********************************************************************
      subroutine PrintArray2( ArrayToPrint, Message)
C***********************************************************************
c     
c     This subroutine is used as part of the sied debug suite. It
c     will print the array passed in. This is assumed to be a subset
c     of an sied array. The 4 in the name means that the array being
c     in is integer*2.
c     
c     Written by Peter Cornillon, University of Rhode Island,
c     pcornillon@me.com 27 April 2009
c     
c**********Functions
c     
c     etime - Fortran intrinsic for elapsed time.

c***********Other variables
c     
c     Message - the message to print out.
c     
      implicit none

c     Functions

      real etime

c******Parameter statements

      include 'ParameterStatements'
      
c****** Input variables
c
c     ArrayToPrint - the portion of the array to print.
      integer*2 ArrayToPrint(10,10)
c     Message - message to print out 
      character*50 Message

c****** General variables
c
c     i - do loop index
      integer i
c     ind - value of j in the original matrix
      integer ind
c     j - do loop index
      integer j
c     LocEOS - position of the letters EOS in the input string.
      integer LocEOS
c     xToPrint - the location in the original array for this element.
      integer xToPrint(10)

c----------------------------------start here --------------------------

c     First make sure that the range specified for the debug printout
c     in ParameterStatements (istrt-iend,jstrt-jend) does not exceed
c     the size of the array to print.

      if( (iend-istrt+1 .gt. 10) .or. (jend-jstrt+1 .gt. 10) ) stop
     1     'Range of arrays to print in debug exceeds 10x10.'

c     Everything OK, print it out.

      LocEOS = index( Message, 'EOS')
      write(6,fmt='(/a50/)'), Message(:LocEOS-1)

      do 100 i=1,iend-istrt+1
         xToPrint(i) = istrt + i - 1
 100  continue

      print 1000, xToPrint
 1000 format(10x,10i9)

      ind = jstrt-1
      do 200 j=1,jend-jstrt+1
         ind = ind + 1
         print 1010, ind, ArrayToPrint(1:iend-istrt+1,j)
 1010    format(i5,5x,10i9)
 200  continue

      end

c***********************************************************************
      subroutine PrintArray4( ArrayToPrint, Message)
C***********************************************************************
c     
c     This subroutine is used as part of the sied debug suite. It
c     will print the array passed in. This is assumed to be a subset
c     of an sied array. The 4 in the name means that the array being
c     passed in is integer*4
c     
c     Written by Peter Cornillon, University of Rhode Island,
c     pcornillon@me.com 27 April 2009
c     
c**********Functions
c     
c     etime - Fortran intrinsic for elapsed time.

c***********Other variables
c     
c     Message - the message to print out.
c     
      implicit none

c     Functions

      real etime

c******Parameter statements

      include 'ParameterStatements'
      
c****** Input variables
c
c     ArrayToPrint - the portion of the array to print.
      integer*4 ArrayToPrint(10,10)
c     Message - message to print out 
      character*50 Message

c****** General variables
c
c     i - do loop index
      integer i
c     ind - value of j in the original matrix
      integer ind
c     j - do loop index
      integer j
c     LocEOS - position of the letters EOS in the input string.
      integer LocEOS
c     xToPrint - the location in the original array for this element.
      integer xToPrint(10)

c----------------------------------start here --------------------------

c     First make sure that the range specified for the debug printout
c     in ParameterStatements (istrt-iend,jstrt-jend) does not exceed
c     the size of the array to print.

      if( (iend-istrt+1 .gt. 10) .or. (jend-jstrt+1 .gt. 10) ) stop
     1     'Range of arrays to print in debug exceeds 10x10.'

c     Everything OK, print it out.

      LocEOS = index( Message, 'EOS')
      write(6,fmt='(/a50/)'), Message(:LocEOS-1)

      do 100 i=1,iend-istrt+1
         xToPrint(i) = istrt + i - 1
 100  continue

      print 1000, xToPrint
 1000 format(10x,10i5)

      ind = jstrt-1
      do 200 j=1,jend-jstrt+1
         ind = ind + 1
         print 1010, ind, ArrayToPrint(1:iend-istrt+1,j)
 1010    format(i5,5x,10i5)
 200  continue

      end

c***********************************************************************
      subroutine PrintArrayReal( ArrayToPrint, Message)
C***********************************************************************
c     
c     This subroutine is used as part of the sied debug suite. It
c     will print the array passed in. This is assumed to be a subset
c     of an sied array. The real in the name means that the array being
c     in is real*
c     
c     Written by Peter Cornillon, University of Rhode Island,
c     pcornillon@me.com 27 April 2009
c     
c**********Functions
c     
c     etime - Fortran intrinsic for elapsed time.

c***********Other variables
c     
c     Message - the message to print out.
c     
      implicit none

c     Functions

      real etime

c******Parameter statements

      include 'ParameterStatements'
      
c****** Input variables
c
c     ArrayToPrint - the portion of the array to print.
      real*4 ArrayToPrint(10,10)
c     Message - message to print out 
      character*50 Message

c****** General variables
c
c     i - do loop index
      integer i
c     ind - value of j in the original matrix
      integer ind
c     j - do loop index
      integer j
c     LocEOS - position of the letters EOS in the input string.
      integer LocEOS
c     xToPrint - the location in the original array for this element.
      integer xToPrint(10)

c----------------------------------start here --------------------------

c     First make sure that the range specified for the debug printout
c     in ParameterStatements (istrt-iend,jstrt-jend) does not exceed
c     the size of the array to print.

      if( (iend-istrt+1 .gt. 10) .or. (jend-jstrt+1 .gt. 10) ) stop
     1     'Range of arrays to print in debug exceeds 10x10.'

c     Everything OK, print it out.

      LocEOS = index( Message, 'EOS')
      write(6,fmt='(/a50/)'), Message(:LocEOS-1)

      do 100 i=1,iend-istrt+1
         xToPrint(i) = istrt + i - 1
 100  continue

      print 1000, xToPrint
 1000 format(10x,10i10)

      ind = jstrt-1
      do 200 j=1,jend-jstrt+1
         ind = ind + 1
         print 1010, ind, ArrayToPrint(1:iend-istrt+1,j)
 1010    format(i5,5x,10F10.4)
 200  continue

      end

c***********************************************************************
c$$$      subroutine CheckArchive( ArchiveID, SatName, Year)
      subroutine CheckArchive
c***********************************************************************
c     
c     This subroutine will determine the name of the satellite 
c     corresponding to the selected archive.
c     

c**********General Variables.
c     
c     BaseDir - the directory in which the subdirectories for the input
c     ...images and the subdirectories for the output images exist.
c     
c     FracLen - a temporary variable used in array length calculation.
c     
c     Loc - location of one string in another
c     
c     
c$$$c     ArchiveID - is 1 for meteosat and 2 for goes
c     
c     LenXin, LenYin - the x, y dimensions of the input image
c     
      implicit none

c     Parameter statements

      include 'ParameterStatements'
      
c     General variables

      integer iArchive

c$$$      integer ArchiveID

c----------------------------------start here --------------------------

c     Initialize list of archives for which this program works. SatName
c     . must match the archive name in the base directory name.

      do 100 iArchive=1,NumberOfArchives
         Loc1 = index(BaseDir, trim(SatNameList(iArchive)))

         if(debug .eq. 1) then
            print *, 'CheckArchive #100: trim(BaseDir)::',
     1           trim(BaseDir), '::'
            print *, 'CheckArchive #110: trim(SatNameList(iArchive))::',
     1           trim(SatNameList(iArchive)), '::'
         endif

         if(Loc1 .gt. 0) then
            SatName = SatNameList(iArchive)
            write( UnitLog, *) 'CheckArchive #120: Processing ', 
     1           trim(SatNameList(iArchive))
            print *, 'CheckArchive #120: Processing ', 
     1           trim(SatNameList(iArchive))
            return
         endif
 100  continue

      write( UnitLog, *) 'CheckArchive #130: Not a recognized ',
     1     'archive.'
      write( UnitLog, *) 'Either the archive name in BaseDir: ',
     1     trim(BaseDir), ' does not correspond to one of the ',
     2     'SatNames.'
      write( UnitLog, *) 'Or the workflow has not been setup ',
     1     'for this archive. '
      write( UnitLog, *) 'To setup for an archive, go to subroutine ',
     1     'ReadInputs in CommonSubroutines.f and follow the ',
     2     'instructions which preceed the NumberOfArchives = 100 ',
     3     'statement. '

      write( 6, *)  'CheckArchive #130: Not a recognized ',
     1     'archive'
      write( 6, *) 'Either the archive name in BaseDir: ',
     1     trim(BaseDir), ' does not correspond to one of the ',
     2     'SatNames.'
      write( 6, *) 'Or the workflow has not been setup ',
     1     'for this archive. '
      write( 6, *) 'To setup for an archive, go to subroutine ',
     1     'ReadInputs in CommonSubroutines.f and follow the ',
     2     'instructions which preceed the NumberOfArchives = 100 ',
     3     'statement. '

      stop

      End subroutine CheckArchive

c***********************************************************************
      Subroutine ReadSSTFile( FileName, ncID, 
     1     sstID, SecondsSince1970ID, 
     2     sst,   SecondsSince1970, ReadSST)
c***********************************************************************
c     
c     This subroutine will open and read an SST file.
c     
c     Written by Peter Cornillon, University of Rhode Island,
c     pcornillon@me.com 18 April 2009
c     

c********General Variables.
c     
c     FileName - the name of the input file
c     
c     SecondsSince1970 - the number of seconds since 00:00 of 1 January 
c     .1970 corresponding to the time of a given image. Read from
c     .the inventory
c     
c     ReadSST - a flag .true. to read the SST variable file, .false. 
c     . to simply open, get NetCDF IDs and read SecondsSince.
c     . NetCDF IDs may be needed to copy attributes when creating
c     . other files.

c*********NetCDF Variables
c     
c     status - return status after a netCDF call
c     
c     ncID - IDs for the input and output files
c     sstID, sstQualIDin, sstIDout - variable IDs
c     geosID, geosIDout - IDs for geos
c     SecondsSince1970ID - ID for hours since 
c     
c     sst - the sst input and output fields.
c     SecondsSince1970 - the number of hours since 1970 
c     geos - the metadata defining the projection.
c     
      use netcdf

      implicit none

c     Parameter statements

      include 'ParameterStatements'
      
c     General variables

      character*319 FileName
      logical ReadSST

c     netCDF variables

      integer ncID

      integer sstID, SecondsSince1970ID

      integer*2 sst(1:LenX,1:LenY)

      real*8 SecondsSince1970

c      include 'netcdf.inc'

c----------------------------------start here --------------------------

      if(Debug .eq. 1) print *, FileName

      status = nf90_open(FileName, nf90_nowrite, ncID)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Get the SST data, if requested, otherwise just the ID

      status = nf90_inq_varid( ncID, 'sst', sstID)
      if(status .ne. nf90_noerr) then
         status = nf90_inq_varid( ncID, 'swt', sstID)
         if(status .ne. nf90_noerr) call handle_err(status)
      endif

      if(Debug .eq. 1) print *, 'ReadSST #100'

      if(ReadSST .eqv. .true.) then
         status = nf90_get_var( ncID, sstID, sst)
         if(status .ne. nf90_noerr) call handle_err(status)
      endif

c     Get the time data.

      call GetSecondsSince1970( ncID, SecondsSince1970ID,
     1     SecondsSince1970)

      if(Debug .eq. 1) print *, 'ReadSST #120'

      end

c***********************************************************************
      Subroutine GetSecondsSince1970( ncID, SecondsSince1970ID, 
     1     SecondsSince1970)
c***********************************************************************
c     
c     This subroutine will read in the DateTime variable and the 
c      LSTorGMT attribute which says whether this time is local sun
c      time or GMT.
c     
c     Written by Peter Cornillon, University of Rhode Island,
c     pcornillon@me.com 2 April 2013
c     

c********General Variables.
c     
c     FileName - the name of the input file
c     
c     SecondsSince1970 - the number of seconds since 00:00 of 1 January 
c     .1970 corresponding to the time of a given image. Read from
c     .the inventory
c     
c*********NetCDF Variables
c     
c     status - return status after a netCDF call
c     
c     ncID - IDs for the input and output files
c     sstID, sstQualIDin, sstIDout - variable IDs
c     geosID, geosIDout - IDs for geos
c     SecondsSince1970ID - ID for hours since 
c     
c     sst - the sst input and output fields.
c     SecondsSince1970 - the number of hours since 1970 
c     geos - the metadata defining the projection.
c     
      use netcdf

      implicit none

c     Parameter statements

      include 'ParameterStatements'
      
c     netCDF variables

      character*20 LSTorGMTAtt

      integer ncID

      integer SecondsSince1970ID

      real*8 SecondsSince1970

c      include 'netcdf.inc'

c----------------------------------start here --------------------------

      if(Debug .eq. 1) print *, 'GetSecondsSince1970 #000: ncID', ncID

      status = nf90_inq_varid( ncID, 'DateTime', SecondsSince1970ID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_var( ncID, SecondsSince1970ID, SecondsSince1970)
      if(status .ne. nf90_noerr) call handle_err(status)
      
c     Read in LSTorGMT if present. In pre-Common 2.24 it was not written
c      out by AppQual hence subsequent programs so could be absent.

      status = nf90_get_att( ncID, SecondsSince1970ID, 'LSTorGMT',
     1     LSTorGMTAtt)
      if(status .ne. nf90_noerr) then
         LSTorGMT = FillValueInt4
      else

c     OK, there is a value, check to see if it LST or GMT.

         Loc1 = index( LSTorGMTAtt, 'LST')
         Loc2 = index( LSTorGMTAtt, 'local sun')

         if( ( Loc1 .gt. 0) .or. (Loc2 .gt. 0) ) then
            LSTorGMT = 0
         else
            Loc1 = index( LSTorGMTAtt, 'GMT')
            if(Loc1 .gt. 0) then
               LSTorGMT = 1
            else

c     If neither, that is very bad, write out a message and stop the run

               write(UnitLog,*) 'GetSecondsSince1970 #100: ',
     1              'LSTorGMTAtt::', trim(LSTorGMTAtt), ':: but ',
     2              'this is not an acceptable value. Aborting.'
               print *, 'GetSecondsSince1970 #100: ',
     1              'LSTorGMTAtt::', trim(LSTorGMTAtt), ':: but ',
     2              'this is not an acceptable value. Aborting.'
               stop
            endif
         endif
      endif

      if(Debug .eq. 1) print *, 'GetSecondsSince1970 #999'

      end

c***********************************************************************
      subroutine ReadMedianFile( FileName, ncID, MedID, 
     1     SecondsSince1970ID, MedSST, SecondsSince1970)
c***********************************************************************
c     
c     This subroutine will open and read a Median file.
c     
c     Written by Peter Cornillon, University of Rhode Island,
c     pcornillon@me.com 20 April 2009
c     

c********General Variables.
c     
c     FileName - the name of the input file
c     

c*********NetCDF Variables
c     
c     status - return status after a netCDF call
c     
c     ncID - IDs for the input and output files
c     MedsstID - variable ID
c     
c     MedSST - the median sst field
c     
      use netcdf

      implicit none

c     Parameter statements

      include 'ParameterStatements'
      
c     General variables
      character*319 FileName

c     netCDF variables
      integer ncID, MedID

      integer SecondsSince1970ID

      integer*2 MedSST(1:LenX,1:LenY)

      real*8 SecondsSince1970

c     Compressed - .true. if the file being tested for in FileStatus
c     . is compressed.
      logical Compressed

c     Decompress - .true. if the fileis to be decompressed
      logical Decompress
c     DecompressInPlace - .true. if the file is to be decompressed in
c     . place; i.e., to the same name.
      logical DecompressInPlace

c     FileExists - .true. if the file being tested for in FileStatus
c     . exists.
      logical FileExists

c     TempFileName - the name of the temporary file into which the
c     . input file is to be decomressed.
      character*319 TempFileName

c     Variable - string with the variable name.
      character*50 Variable

c----------------------------------start here --------------------------

c.............This section to determine status of input file ...........
c     

      if(debug .eq. 1) print *, 'ReadMedianFile #100. FileName::', 
     1     trim(FileName), '::'

      status = nf90_open( FileName, nf90_nowrite, ncid)
      if (status .ne. nf90_noerr) call handle_err(status)

c     Read median data.

      if(debug .eq. 1) print *, 'ReadMedianFile #110.' 

      Variable = Blankfill(:len(Variable))
      Variable = 'median_' // trim(ParameterName_lc)

      if(debug .eq. 1) print *, 'ReadMedianFile #120. Variable to ',
     1     'read::', trim(Variable), '::'

      status = nf90_inq_varid( ncID, Variable, MedID)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(debug .eq. 1) print *, 'ReadMedianFile #130.' 

      status = nf90_get_var( ncID, MedID, MedSST)
      if(status .ne. nf90_noerr) call handle_err(status)

      status =  nf90_get_att( ncID, MedID, 'scale_factor', 
     1     sstScaleFactorIn)
      if(status .ne.  nf90_noerr) call handle_err(status)

      if(debug .eq. 1) print *, 'ReadMedianFile #140.' 

      status =  nf90_get_att( ncID, MedID, 'add_offset', 
     1     sstOffsetIn)
      if(status .ne.  nf90_noerr) call handle_err(status)

      status =  nf90_get_att( ncID, MedID, '_FillValue', 
     1     sstFillValueIn)
      if(status .ne.  nf90_noerr) call handle_err(status)

c     Get the time data.

      if(debug .eq. 1) print *, 'ReadMedianFile #150.' 

      call GetSecondsSince1970( ncID, SecondsSince1970ID,
     1     SecondsSince1970)

      if(debug .eq. 1) print *, 'ReadMedianFile #999.' 

      end
c***********************************************************************
      subroutine Read_int2_variable( FileName, VariableName, Variable)
c***********************************************************************
c     
c     This subroutine will open FileName and read the variable with
c      VariableName. The variable is assumed to be LenX by LenY. It also
c      assumes that all variables are integer*2
c
c     Written by Peter Cornillon, University of Rhode Island,
c     pcornillon@me.com 21 April 2013

      use netcdf

      implicit none

      include 'ParameterStatements'
      
c     FileName - the name of the input file
      character*319 FileName

c     ncID - netCDF file identifer
      integer ncID, MedID

c     ncVarID - netCDF variable identifer
      integer ncVarID

c     Variable - array of 0/1, 1 if clear.
      integer*2 Variable(1:LenX,1:LenY)

c     VariableName - string with the variable name.
      character*50 VariableName

c----------------------------------start here --------------------------

      if(debug .eq. 1) print *, 'ReadInt2 #100. FileName::', 
     1     trim(FileName), '::'

      status = nf90_open( FileName, nf90_nowrite, ncID)
      if (status .ne. nf90_noerr) call handle_err(status)

c     Read clear or cloudy field first.

      if(debug .eq. 1) print *, 'ReadInt2 #110. Variable to ',
     1     'read::', trim(VariableName), '::'

      status = nf90_inq_varid( ncID, VariableName, ncVarID)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(debug .eq. 1) print *, 'ReadInt2 #120.' 

      status = nf90_get_var( ncID, ncVarID, Variable)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_close(ncID)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(debug .eq. 1) print *, 'ReadInt2 #999.' 

      end subroutine Read_int2_variable

c***********************************************************************
      subroutine Read_real_variable( FileName, VariableName, Variable)
c***********************************************************************
c     
c     This subroutine will open FileName and read the variable with
c      VariableName. The variable is assumed to be LenX by LenY. It also
c      assumes that all variables are real
c
c     Written by Peter Cornillon, University of Rhode Island,
c     pcornillon@me.com 21 April 2013

      use netcdf

      implicit none

      include 'ParameterStatements'
      
c     FileName - the name of the input file
      character*319 FileName

c     ncID - netCDF file identifer
      integer ncID, MedID

c     ncVarID - netCDF variable identifer
      integer ncVarID

c     Variable - array of 0/1, 1 if clear.
      real Variable(1:LenX,1:LenY)

c     VariableName - string with the variable name.
      character*50 VariableName

c----------------------------------start here --------------------------

      if(debug .eq. 1) print *, 'ReadInt2 #100. FileName::', 
     1     trim(FileName), '::'

      status = nf90_open( FileName, nf90_nowrite, ncID)
      if (status .ne. nf90_noerr) call handle_err(status)

c     Read clear or cloudy field first.

      if(debug .eq. 1) print *, 'ReadInt2 #110. Variable to ',
     1     'read::', trim(VariableName), '::'

      status = nf90_inq_varid( ncID, VariableName, ncVarID)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(debug .eq. 1) print *, 'ReadInt2 #120.' 

      status = nf90_get_var( ncID, ncVarID, Variable)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_close(ncID)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(debug .eq. 1) print *, 'ReadInt2 #999.' 

      end subroutine Read_real_variable

c***********************************************************************
      subroutine ReadGeoSobelFile( FileName, VariableName, Gradient)
c***********************************************************************
c     
c     This subroutine will open and read one variable out of the
c     Sobel or GeoSobel file.
c     
c     Written by Peter Cornillon, University of Rhode Island,
c     pcornillon@me.com 20 April 2009
c
c     1/6/13 - PCC - modified to read GeoSobel data and to get the
c      eastward gradient, the northward gradient and the gradient 
c      magnitude if the (1,1) element of the array passed in is 100.

c********General Variables.
c     
c     FileName - the name of the input file
c     VariableName - this is the name of the variable in the netCDF
c      file that is desired.
c 
c     The reason for passing in either 1 or LenX/LenY for these 
c     variables is to save space if 
c*********NetCDF Variables
c     
c     status - return status after a netCDF call
c     
c     ncID - IDs for the input and output files
c     GradID - ID for the variable

      use netcdf

      implicit none

c     Parameter statements

      include 'ParameterStatements'
      
c     General variables
      character*319 FileName
      character*50 VariableName

      integer SecondsSince1970ID

c     netCDF variables
      integer ncID, GradID

      real Gradient(1:LenX,1:LenY)

      real*8 SecondsSince1970

c      include 'netcdf.inc'

c----------------------------------start here --------------------------

      if(Debug .eq. 1) then
         print *, 'ReadGeoSobelFile #000: VariableName::', 
     1        trim(VariableName), '::'
         print *, '#001: FileName::', trim(FileName), '::'
      endif

      status = nf90_open(FileName, nf90_nowrite, ncID)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'ReadGeoSobelFile #100: ncID: ', ncID

      status = nf90_inq_varid( ncID, trim(VariableName), GradID)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'ReadGeoSobelFile #110'

      status = nf90_get_var( ncID, GradID, Gradient)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Get DateTime, really want to set LSTorGMT.

      if(Debug .eq. 1) print *, 'ReadGeoSobelFile #120: ncID: ', ncID

      call GetSecondsSince1970( ncID, SecondsSince1970ID,
     1     SecondsSince1970)

      status = nf90_close(ncID)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'ReadGeoSobelFile #999'

      end subroutine ReadGeoSobelFile

c***********************************************************************
      subroutine ReadZenithAngleFile( FileName, ncID, ZenithAngle)
c***********************************************************************
c     
c     This subroutine will open and read a Sobel file.
c     
c     Written by Peter Cornillon, University of Rhode Island,
c     pcornillon@me.com 29 April 2009
c     
      use netcdf

      implicit none

c****** Parameter statments
c
      include 'ParameterStatements'
 
c******Functions
c

c****** Input variables
c
c     FileName - name of input file to read
      character*319 FileName

c****** Output variables
c
c     ncID - NetCDF ID for the input file
      integer ncID
c     ZenithAngle - local solar zenith angle at the pixel location - 
c     . greater than 90 degrees is night
      integer*2 ZenithAngle(1:LenX,1:LenY)

c****** General variables
c
c     ZAID - NetCDF ID for the zenith angle variable
      integer ZAID

      integer SecondsSince1970ID

      real*8 SecondsSince1970

c      include 'netcdf.inc'

c----------------------------------start here --------------------------

      status = nf90_open(trim(FileName), nf90_nowrite, 
     1     ncID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_inq_varid( ncID, 'zenith_angle', ZAID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_var( ncID, ZAID, ZenithAngle)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Get DateTime, really want to set LSTorGMT.

      call GetSecondsSince1970( ncID, SecondsSince1970ID,
     1     SecondsSince1970)

      status = nf90_close(ncID)
      if(status .ne. nf90_noerr) call handle_err(status)

      end

c***********************************************************************
      subroutine median( inpict, pict, n)
c***********************************************************************
c     
c     Fast Median algorithm applies an nxn median filter to inpict.
c     The result is in pict. Values in inpict are assumed to range
c     from 1 to Range. Values equal to the fill value, FillValueInt2,
c     for the present configuration are set to 0 on input. If values
c     of inpict are less than or equal to 0 or greater than Range on
c     input an error is flagged and the program is stopped. Prior to
c     returning values of pict, the median filtered version of inpict,
c     equal to 0 are set to the fill value. This means that if the
c     majority of values in an nxn region are fill values, they will
c     be returned as fill values.
c     
c     Skip if n is 1 or less; i.e., no median filtering.
c     
c     
c     HISTO(0:255)         -  histogram accumulator
c     N                    -  window size (side)
c     N2D2                 -  ((N*N)-1)/2
c     MID1                 -  (N-1)/2 window center
c     COUNT                -  Number of pixels < median
c     OLD_VALUE            -  pixel on line removed
c     NEW_VALUE            -  pixel on line added
c     I,J                  -  center pixel coordinates
c     IL,JL                -  neighbors coordinates
c     JM                   -  line removed (J coord)
c     JN                   -  line added (J coord)
c     K                    -  index/previous median
c     L                    -  index
c     
c     PCC added Offset and Range. The numbers were 
c     hard coded in as 255 in the .rat version.
c     
      implicit none

c     Parameter statements

      include 'ParameterStatements'
      
c     inpict - input array. Values equal to the fill value will be
c      set to 0 on input and back to the fill value on output.
c      BE CAREFUL, THE FILL VALUE IS SET TO FILLVALUEINT2 IN THIS
c      THIS SUBROUTINE. IF YOU CHANGE THE INTEGER TYPE OF INPICT AND
c      PICT, YOU SHOULD CHANGE THE FILL VALUE TO THE CORRESPONDING
c      MISSING VALUE. FILLVALUEIN2 IS -32768.
      integer*2 inpict(1:LenX,1:LenY)

c     pict - the median filtered version of inpict.
      integer*2 pict(1:LenX,1:LenY)

      integer*2, allocatable :: histo(:)
      integer n
      integer*2 n2d2
      integer*2 mid1
      integer*2 PixelCount
      integer*2 old_value
      integer*2 new_value
      integer*2 i,j
      integer*2 il,jl
      integer*2 jm
      integer*2 jn
      integer*2 k
      integer*2 l

c----------------------------------start here --------------------------

c     Allocate space for the histogram.

      allocate( histo(0:Range), 
     2     stat=ierr)

      if (ierr .ne. 0) then
         print *, 'Allocation error for histo. Exiting.'
         stop
      endif

c     set output array to input array unless the input array equals
c     FillValue in which case set the output value to 0. Otherwise
c     the histogram will screw up.

      do 23144 j=1,LenY 
         do 23142 i=1,LenX 
            if(inpict(i,j) .eq. FillValueInt2) then
               pict(i,j) = 0
               inpict(i,j) = 0
            elseif( (inpict(i,j) .le. 0) .or. (inpict(i,j) .gt. Range) )
     1              then
               print *, 'Median #100: A value in the field input to ',
     1              'the median filter is less than or equal to 0 or ',
     2              'greater than ', Range, '. This is not permitted. ',
     3              'Exiting. Sorry.'
               stop
            else
               pict(i,j) = inpict(i,j)
            endif
23142    continue
23144 continue

      if(Debug .eq. 1) print *, 'Median subroutine #100:'

      if (n .gt. 1) then

         mid1 = (n-1)/2
         n2d2 = ((n*n)-1)/2

         if(Debug .eq. 1) print *, 'Median subroutine #110:'

         do 23146 i=1+mid1,LenX-mid1 

            do 23148 l=0,Range 
               histo(l) = 0
23148       continue

c     Median for the 1st column of each line (no previous information)

            do 23152 jl=1,n 
               do 23150 il=i-mid1,i+mid1 
                  histo(inpict(il,jl)) = histo(inpict(il,jl)) + 1
23150          continue
23152       continue

            j = 1
            k = -1
            PixelCount = 0

23154       if (PixelCount .le. n2d2) then
               k = k+1
               PixelCount = PixelCount+histo(k)
               goto 23154
            endif

            pict(i,j+mid1) = k

c     Median using the median computed in previous column.

            do 23156 j = 2,LenY-(n-1) 
               jm = j-1
               jn = jm+n

               do 23158 il = i-mid1,i+mid1 
                  old_value = inpict(il,jm)
                  new_value = inpict(il,jn)
                  histo(old_value) = histo(old_value)-1

                  if(old_value .le. k) then
                     PixelCount = PixelCount-1
                  endif

                  histo(new_value) = histo(new_value)+1

                  if(new_value .le. k) then
                     PixelCount = PixelCount+1
                  endif

23158          continue

               if(PixelCount .le. n2d2) then

23166             if(PixelCount .le. n2d2) then
                     k = k+1
                     PixelCount = PixelCount+histo(k)
                     goto 23166
                  endif

               else

23168             if(PixelCount .gt. n2d2) then
                     PixelCount = PixelCount-histo(k)
                     k = k-1
                     goto 23168
                  endif

                  k = k+1
                  PixelCount = PixelCount+histo(k)
               endif

               pict(i,j+mid1) = k

23156       continue
23146    continue

      endif

c     Now insert FillValueIn2 in places where the input field had
c      missing values. Do this for both the input field, which has
c      been changed, and the output field.

      do 1000 j=1,LenY
         do 1010 i=1,LenX
            if(pict(i,j) .eq. 0) pict(i,j) = FillValueInt2
            if(inpict(i,j) .eq. 0) inpict(i,j) = FillValueInt2
 1010    continue
 1000 continue

      if(Debug .eq. 1) print *, 'Median subroutine #120:'

      return
      end
c***********************************************************************
      subroutine ReadMergedThinned( in_file, ncID, frID,
     1     SecondsSince1970ID, FrontArray, SecondsSince1970, 
     1     MergedOrThinned)
c***********************************************************************
c     
c     This subroutine is designed to read merged or thinned arrays 
c     written by WriteMergedThinned.
c
c     It will also read the date in hours since 1900 and  vectors of 
c     lat/lon values (all zeros if irrelevant). 
c     
c
c     The program will look for a compressed version of the input file 
c     and decompress if necessary. It assumes gz compression. If it 
c     decompressed, it will remove the temporary file that it created.
c     
c.............................Variables ................................
c
      use netcdf

      implicit none

c****** Parameter statements

      include 'ParameterStatements'

c****** Functions
c

c****** Input
c
c     in_file   - the name of the merged/thinned file 
      character*319 in_file
c     MergedOrThinned - 1 to read MergedFronts and 2 to read 
c     . ThinnedFronts
      integer MergedOrThinned

c****** Output
c
c     SecondsSince1970 - The time in hours since 1/1/1900
      real*8 SecondsSince1970
c     FrontArray(1:SisOfImgx,1:SisOfImgy) - image of merged fronts
      integer*2 FrontArray(1:LenX,1:LenY)

c****** General variables

c     ArrayID - NetCDF ID for the array variable.
      integer ArrayID
c     Compressed - comression flag, true if input file is compressed.
      logical Compressed
c     DateTimeID - NetCDF ID for variable DateTime
      integer*4 DateTimeID
c     Decompress - flag telling FileStatus to decompress the file
c     . passed to it.
      logical Decompress
c     DecompressInPlace - flag telling FileStatus to decompress the
c     . the file passed to it in place, i.e., with the same name sans
c     . .gz. Ignored if Decompress is .false.
      logical DecompressInPlace
c     FileExists - flag returned from FileStatus .true. if the file
c     . passed into FileStatus exists.
      logical FileExists
c     frID      - NetCDF ID for merged/thinned fronts variable.
      integer frID
c     i - do loop index
      integer i
c     j - do loop index
      integer j
c     ncArrayID - NetCDF ID for the merged or thinned data file.
      integer ncArrayID
c     ncid   - NetCDF ID for input data set.
      integer ncid
c     TempFileName - temporary filename built by FileStatus for 
c     . unzipped output.
      character*319 TempFileName
c     titlength - the length of the output file name.
      integer*4 titlength
c     SecondsSince1970ID - NetCDF ID for seconds since variable
      integer SecondsSince1970ID

c      include 'netcdf.inc'

c----------------------------------start here --------------------------

      if( debug .eq. 1) print *, 'ReadMergedThinned #100: ',
     1     'in_file: ', in_file

c     Read the merged frontal field

      if((MergedOrThinned .ne. 1) .and. (MergedOrThinned .ne. 2)) then 
        stop 'ReadMergedThinned #110: Neither thinned nor merged fronts 
     1requested. Bad, bad'
      endif

c     Open the file.

      status = nf90_open( in_file, nf90_nowrite, ncID)
      if (status .ne. nf90_noerr) call handle_err(status)

      if( debug .eq. 1) print *, 'ReadMergedThinned #120: '

c     Get the NetCDF ID for the merged array

      if(MergedOrThinned .eq. 1) then
         status = nf90_inq_varid( ncID, 'MergedFronts', frID)
         if (status .ne. nf90_noerr) call handle_err(status)
      endif

c     Get the NetCDF ID for the thinned array

      if(MergedOrThinned .eq. 2) then
         status = nf90_inq_varid( ncID, 'ThinnedFronts', frID)
         if (status .ne. nf90_noerr) call handle_err(status)
      endif

      if( debug .eq. 1) print *, 'ReadMergedThinned #130: ncID, frID: ',
     1     ncID, frID

c     Read the array, whichever one you were told to read.

      status = nf90_get_var( ncID, frID, FrontArray)
      if (status .ne. nf90_noerr) call handle_err(status)

      if( debug .eq. 1) print *, 'ReadMergedThinned #140: '

c     Get the time data.

      status = nf90_inq_varid( ncID, 'DateTime', SecondsSince1970ID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_get_var( ncID, SecondsSince1970ID,
     1     SecondsSince1970)
      if(status .ne. nf90_noerr) call handle_err(status)

c$$$      status = nf90_close(ncID)
c$$$      if(status .ne. nf90_noerr) call handle_err(status)

      if( debug .eq. 1) print *, 'ReadMergedThinned #150: '

      end subroutine ReadMergedThinned

c***********************************************************************
      subroutine WriteMergedThinned( out_file, hdate, lats, lons, 
     1     FrontArray, MergedOrThinned, ncInID, sstID, 
     2     SecondsSince1970InID)
c***********************************************************************
c     
c     This subroutine is designed to write merged arrays in netcdf.
c     It will also write the date in hours since 1900 and  vectors of 
c     lat/lon values (all zeros if irrelevant). The output file will 
c     be compressed using gz when done.
c     
c...........................Variables ..................................
c     
c     SecondsSince1970OutID - the ID of the number of seconds since  
c     . 0 1 Jan 1970 corresponding to the time of the output arrays.
c     out_file   - the output file name
c     out_file_gz - the output file name with the gz extension. This
c     is the argument passed in.
c     MergeString - used for attribute description of merge function
c     FrontArray(1:SisOfImgx,1:SisOfImgy) - image of merged fronts
c     ImageWindow - the maximum number of images used in the merge step
c     HourWindow - if time is available the number of hours used in 
c     merge step.
c     
c     status    - netCDF status
c     ncid   - dataset ID   
c     LatDim    - ID for lat dimension vector in netCDF file.
c     LonDim    - ID for lon dimension vector in netCDF file.
c     MergedID  - ID for merged fronts variable in netCDF file.
c     ThinnedID  - ID for thinned fronts variable in netCDF file.
c     frDims(2) - Defines the order of the dimensions for 
c     
c     Dummy - dummy variable required by gfort on Sofa
c     sizex, sizey - could not use SizImgX and Y in nf90_def_var, 
c     because they were defined as integer*2, need to be integer.
c     
c     exis, exis_gz  - a logical that is used when testing for the 
c     existence of a file, tests both compressed and uncompressed names
c     Lats  - y (2nd) dimension of array to write
c     Lons  - x (1st) dimension of array to write
c     
c     YearID, YearDayID, HourID, MinID, SecID and DateTimeID - are all
c     variable IDs.
c     Year, YearDay, Hour, Min, Sec and DateTime - are the correspoding
c     values
c     hDate - The time in hours since 1/1/1900
c     
c     titlength - the length of the output file name.
c     i, j - do loop indices

      use netcdf

      implicit none

c******Functions

c******Parameter statements

      include 'ParameterStatements'

c     MergedOrThinned - 1 to read MergedFronts and 2 to read 
c     . ThinnedFronts
      integer*2 MergedOrThinned

c******General variables
      
      character*319 out_file, out_file_gz
c     TempFileName - dummy filename for test of file existence
      character*319 TempFileName
      character*54 MergeString

      character*100 ProcessingProgram
      character*100 NewTitle

c$$$      character*4000 Summary

      integer*2 FrontArray(1:LenX,1:LenY)

      integer ncID, ArrayID
      integer MergedID, ThinnedID
      integer sizex, sizey

      integer SecondsSince1970InID, SecondsSince1970OutID

      integer nxDimIDout, nyDimIDout
      integer DimsOut(2)

      integer Dummy

      integer*4 DateTimeID
      real*8 hDate

      integer*4 titlength
      integer i, j

      real*4 lats(1:LenY), lons(1:LenX)

      integer MaxVal, MinVal

      integer LenXinID, LenYinID

      integer ncInID, sstID

      real*8 SecondsSince1970

c     FileExists - .true. if file being tested for exists
      logical FileExists
c     Compressed - dummy logical for test of file existence.
      logical Compressed

c      include 'netcdf.inc'

c----------------------------------start here --------------------------

c     First make sure that this is a legitimate call.
      
      if((MergedOrThinned .ne. 1) .and. (MergedOrThinned .ne. 2)) then 
         stop 'WriteMergedThinned #90: Neither thinned nor merged fronts
     1 requested. Bad, bad'
      endif

c     Get the range of values in the field. Really not necessary, just
c     to check and make sure that there is at least one merged or
c     thinned pixel in the field.

      if( debug .eq. 1) then
         print *, 'WriteMergedThinned #100: outfile: ', out_file

         MaxVal = -10000
         MinVal = 10000
         do 9875 j=1,LenY
            do 9876 i=1,LenX
               if( FrontArray(i,j) .lt. MinVal) then
                  MinVal = FrontArray(i,j)
               endif
               if( FrontArray(i,j) .gt. MaxVal) then
                  MaxVal = FrontArray(i,j)
               endif
 9876       continue
 9875    continue

         if(MaxVal .eq. 0) print *, 'WriteMergedThinned #110: Max ',
     1        'and Min of merged or thinned array: ', MinVal, MaxVal

      endif

c............This section to determine status of output file ...........
c     
c     Check that the dimensions of the SST array agree with those
c     - of the Sobel array.

      if(Debug .eq. 1) print *, 'WriteMergedThinned #111: '

      status = nf90_inq_dimid( ncInID, 'nx', LenXinID)
      if(status .ne. nf90_noerr) call handle_err(status)
      
      status = nf90_inquire_dimension( ncInID, LenXinID,  
     1     DummyCharString, LenXin)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'WriteMergedThinned #112: '

      status = nf90_inq_dimid( ncInID, 'ny', LenYinID)
      if(status .ne. nf90_noerr) call handle_err(status)
      
      if(Debug .eq. 1) print *, 'WriteMergedThinned #113: '

      status = nf90_inquire_dimension( ncInID, LenYinID, 
     1     DummyCharString, LenYin)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'WriteMergedThinned #115: LenXin, ',
     1     'LenYin: ', LenXin, LenYin

      if( (LenXin .ne. LenX) .or. (LenYin .ne. LenY) ) then
         write(UnitLog,*) 'WriteMergedThinned #115: Merged or ',
     1        'thinned array dimensions do not match Median ',
     2        'dimensions. STOP'
         stop 'WriteMergedThinned #115: Merged or thinned array dimensio
     2ns do not match Median dimensions. STOP'
      endif

c     Test for the existence of the output file. If it exists stop, it
c     shouldn't. If it does not exist, create it. 

      call FileStatus( out_file, FileExists, Compressed, '2000', 
     1     TempFileName, .false., .false.)

      if(FileExists .eqv. .false.) then
         status = nf90_create( out_file, 
     1        OR(nf90_netcdf4, nf90_noclobber), ncID)
         if (status .ne. nf90_noerr) call handle_err(status)
      else
         write(UnitLog,fmt='(a72/a319)') 
     1        'WriteMergedThinned #120: File already exists, should not 
     2be rewriting: ', out_file
         print *, 'WriteMergedThinned #120: File already exists, ',
     1        'should not be rewriting: ', out_file
         stop
      endif

      if(debug .eq. 1) print *, 'WriteMergedThinned #130: Created:',
     1     out_file

c     Create the dimensions for the output fields.

      status = nf90_def_dim( ncID, 'nx', LenX, nxDimIDout)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_def_dim( ncID, 'ny', LenY, nyDimIDout)
      if(status .ne. nf90_noerr) call handle_err(status)

      DimsOut(1) = nxDimIDout
      DimsOut(2) = nyDimIDout

      chunksize(1) = min( LenX, chunkszLenX)
      chunksize(2) = min( LenY, chunkszLenY)

      if(Debug .eq. 1) print *, 'WriteMergedThinned #140: '

c     Create the output variable.

      if(MergedOrThinned .eq. 1) then

c     Define the merged variable

         status = nf90_def_var( ncid, 'MergedFronts', nf90_short, 
     1        DimsOut, ArrayID)
         if (status .ne. nf90_noerr) call handle_err(status)

         status = nf90_def_var_chunking( ncID, ArrayID, 0, chunksize)
         if(status .ne. nf90_noerr) call handle_err(status)
         
         status = nf90_def_var_deflate( ncID, ArrayID, 0, 1, 4)
         if(status .ne. nf90_noerr) call handle_err(status)

         status = nf90_put_att( ncID, ArrayID, 'long_name', 
     1        'Merged fronts from Cayula-Cornillon algorithm')
         if(status .ne. nf90_noerr) call handle_err(status)

         status = nf90_put_att( ncID, ArrayID, 'units', 'none')
         if(status .ne. nf90_noerr) call handle_err(status)

         status = nf90_put_att( ncID, ArrayID,
     1        'standard_name', 'merged_fronts')
         if(status .ne. nf90_noerr) call handle_err(status)
      else

c     Define the thinned variable

         status = nf90_def_var( ncid, 'ThinnedFronts', nf90_short, 
     1        DimsOut, ArrayID)
         if (status .ne. nf90_noerr) call handle_err(status)

         status = nf90_def_var_chunking( ncID, ArrayID, 0, chunksize)
         if(status .ne. nf90_noerr) call handle_err(status)
         
         status = nf90_def_var_deflate( ncID, ArrayID, 0, 1, 4)
         if(status .ne. nf90_noerr) call handle_err(status)

         status = nf90_put_att( ncID, ArrayID, "long_name",
     1        "Thinned fronts from Cayula-Cornillon algorithm")
         if(status .ne. nf90_noerr) call handle_err(status)

         status = nf90_put_att( ncID, ArrayID, 'units', 'none')
         if(status .ne. nf90_noerr) call handle_err(status)

         status = nf90_put_att( ncID, ArrayID,
     1        'standard_name', 'thinned_fronts')
         if(status .ne. nf90_noerr) call handle_err(status)
      endif

      if(Debug .eq. 1) print *, 'WriteMergedThinned #150: '

      status = nf90_put_att( ncID, ArrayID, 'add_offset', 0.0)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncID, ArrayID, 'scale_factor', 1.0)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncID, ArrayID, '_FillValue', FillValueInt2)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'WriteMergedThinned #160: '

c     create the time - seconds since 1970 and define its attributes

      call DefineSecondsSinceVar( ncID, SecondsSince1970OutID)

c     Develop string for number of hours over which the data were merged

c     Now generate all of the global attributes

      if(Debug .eq. 1) print *, 'WriteMergedThinned #170: '

      Summary = BlankFill(1:len(Summary))

      if(MergedOrThinned .eq. 1) then
         write( NewTitle, '(a19, i4, a42 )') 
     1        'Fronts merged over ', HourWindow * 2, 
     2        ' hours centered on this image for SIED of '
         ProcessingProgram = 'Pmerge_Main'

         write(Summary, '(a40,a47,a14,a3,a11,a18,a4,a43,a37)') 
     1        'The field in this file was generated by ',
     1        'merging all fronts found by SIED in all images ',
     2        'either within ', ImageWindow,
     3        ' images or within ', HourWindow,
     4        ' hours of the image of interest, whichever ',
     5        ' results in the larger number images.'

         call GenerateGlobals( ncInID, ncID, NewTitle,
     1        ProcessingProgram, PmergeVersionNo, out_file)
      else
         write( NewTitle, '(a19)') 
     1        'Thinned fronts for '
         ProcessingProgram = 'Thin_Main'

         Summary = 'The field in this file was generated by ' //
     1        'thinning the field of merged SIED fronts.'

         call GenerateGlobals( ncInID, ncID, NewTitle,
     1        ProcessingProgram, ThinVersionNo, out_file)
      endif

      status = nf90_enddef(ncid)
      if (status .ne. nf90_noerr) call handle_err(status)

c......................All done defining variables......................
c     
c     Now write out the data. 

c     Get the time from the SST file and write it to the Median file. 

      if(Debug .eq. 1) print *, 'WriteMergedThinned #175: '

      status = nf90_get_var( ncInID, SecondsSince1970InID, 
     1     SecondsSince1970)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'WriteMergedThinned #180: '

      status = nf90_put_var( ncID, SecondsSince1970OutID, 
     1     SecondsSince1970)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'WriteMergedThinned #190: '

c     Write the merged or thinned variable

      status = nf90_put_var( ncid, ArrayID, FrontArray)
      if (status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'WriteMergedThinned #200: '

c     All done. close file and exit
      
c$$$      call cleanup( ncID, out_file, .false., TempFileName, .true.)

      status = nf90_close(ncID)
      if(status .ne. nf90_noerr) call handle_err(status)

      end subroutine WriteMergedThinned


c***********************************************************************
      subroutine DefineSecondsSinceVar( ncOutID, SecondsSinceID)
c***********************************************************************
c     
c     This subroutine creates the SecondsSince variable for an .nc 
c     . file.
c
c     Written by Peter Cornillon, University of Rhode Island,
c     pcornillon@me.com 22 August 2010
c
c     8/20/11 PCC
c      Added StandardName
c      Modified LongName to include since...
c
c********** Subroutine arguments
c     
c     status - return status after a netCDF call
c     ncOutID - ID for the output files.
c
c********** Other variables
c     
c     SecondsSince1970 - the number of seconds since 0 of 1 Jan 1970
c     . corresponding to the time of the sst array. This value will
c     . be read from the input file and writte to the output file.
c
      use netcdf

      implicit none

c     Parameter statements

      include 'ParameterStatements'

c     General variables

      character*3 AttributeDescription

      integer Dummy,  ncOutID,  SecondsSinceID

      integer MedID

c      include 'netcdf.inc'

c----------------------------------start here --------------------------

      if(debug .eq. 1) print *, 'DefineSecondsSinceVar #000'

      status = nf90_def_var( ncOutID, 'DateTime', nf90_double, 
     1     SecondsSinceID)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, SecondsSinceID, 
     1     'standard_name', 'time')
      if(status .ne. nf90_noerr) call handle_err(status)

      if(debug .eq. 1) print *, 'DefineSecondsSinceVar #150'

      status = nf90_put_att( ncOutID, SecondsSinceID, 
     1     'long_name', 'time since 1970-01-01 00:00:00.0')
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_put_att( ncOutID, SecondsSinceID, 
     1     'units', 'seconds since 1970-01-01 00:00:00.0')
      if(status .ne. nf90_noerr) call handle_err(status)

c     Attribute for LST or GMT. Some files, such as global Pathfinder,
c      the data are all at the same local sun time and for others such 
c      as MSG, space-view, the times are GMT. This attribute says which.

      if(LSTorGMT .eq. 0) then
         AttributeDescription = 'LST'
      elseif(LSTorGMT .eq. 1) then
         AttributeDescription = 'GMT'
      else
         print *, 'DefineSecondsSinceVar #100: LSTorGMT=',
     1        LSTorGMT, ' not an acceptable value. Aborting.'
      endif

      status = nf90_put_att( ncOutID, SecondsSinceID, 
     1     'LSTorGMT', AttributeDescription)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(debug .eq. 1) print *, 'DefineSecondsSinceVar #999'

      end subroutine DefineSecondsSinceVar

c***********************************************************************
      subroutine GenerateGlobals( ncInID, ncOutID, NewTitle, 
     1     ProcessingProgram, VersionNumber, TempFile)
c***********************************************************************
c     
c     This subroutine will copy global attributes common to all of the
c     - fronts processing path (Conventions and Institution) and it will
c     - generate the history of previous processing steps from global
c     - attributes in the main input file.
c
c     Written by Peter Cornillon, University of Rhode Island,
c     pcornillon@me.com 16 October 2010
c
c     3/16/11 PCC - Changed  'print *, PreviousSource(1:Loc-1)' to
c         'print *, PreviousSource(1:Loc1)'; 
c       Added 'implicit none';
c       Commented out line beginning with 'iLoc'
c     5/15/11 PCC - Changed variable DateTime to ProcessingDateTime so
c       that it would not get confused with the variable in the netCDF
c       outfile called DateTime
c     8/22/11 - PCC - Changed the metadata substantially.
c     12/30/11 - PCC - Changed the way that it handles an error return
c       from a netCDF copy command. In the previous version it would
c       end the run. Now, if debug is on, it prints out a message 
c       otherwise it goes on its merry way hoping that the attribute was
c       missing in the input file and that there was not some other 
c       really bad problem.
c     12/31/12 - PCC - Increased dimensions of ProcessingHistory and
c      PreviousProcessingHistory from 400 to 2000 to accommodate 
c      SIED.
c
c.......................................................................
c
      use netcdf

      implicit none
c
c     Parameter statements

      include 'ParameterStatements'

c     Functions

      character*4 CommonSubsVersionNo, VersionNumber
      character*36 UUID_Gen
 
c     General variables

      character*100 NewTitle
      character*400 Title
      character*100 ProcessingProgram
      character*100 Source
      character*200 PreviousTitle, PreviousSource
      character*200 PreviousProcessingDate
      character*2000 ProcessingHistory, PreviousProcessingHistory
c$$$      character*4000 Summary

      integer ncInID, ncOutID

      integer nDims, nVars, ngAtts, unLimited

c     Variables used to determine the current date and time

      integer Values(8)
      character*5 Zone
      character*10 Time
      character*8 Date
      character*19 ProcessingDateTime

      character*319 TempFile

c      include 'netcdf.inc'

c---------------------------------start here --------------------------

      if(Debug .eq. 1) print *, 'GenerateGlobals #100: ', ncInID

c     Generate the title from the input file.

      status = nf90_get_att( ncInID, nf90_global,
     1     'title', PreviousTitle)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_inquire_attribute( ncInID, nf90_global, 'title', 
     1     xType, Loc1, AttNum)
      if(status .ne. nf90_noerr) call handle_err(status)

      Title = BlankFill(1:len(Title))
      Title  = trim(NewTitle) // ' ' // PreviousTitle(:Loc1)
      status = nf90_put_att( ncOutID, nf90_global, 'title', Title)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'GenerateGlobals #105: ', ncInID

c     Write out the summary.

      status = nf90_put_att( ncOutID, nf90_global, 'summary', Summary)
      if(status .ne. nf90_noerr) call handle_err(status)

c     What conventions were used for this file?
c     Note that the c in the Conventions attribut is upper case, all
c     other attributes appear to start with lower case letters.

c$$$      status = nf90_put_att( ncOutID, nf90_global, 'Conventions',
c$$$     1     'CF-1.5')
c$$$      if(status .ne. nf90_noerr) call handle_err(status)
c$$$
c$$$      status = nf90_put_att( ncOutID, nf90_global, 
c$$$     1 'standard_name_vocabulary', 'CF-1.5')
c$$$      if(status .ne. nf90_noerr) call handle_err(status)
c$$$
c$$$      status = nf90_put_att( ncOutID, nf90_global, 
c$$$     1 'Metadata_Conventions', 'Unidata Dataset Discovery v1.0')
c$$$      if(status .ne. nf90_noerr) call handle_err(status)
c$$$

      status = nf90_copy_att( ncInID, nf90_global, 
     1     'Conventions', ncOutID, nf90_global)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_copy_att( ncInID, nf90_global, 
     1     'standard_name_vocabulary', ncOutID, nf90_global)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_copy_att( ncInID, nf90_global, 
     1     'Metadata_Conventions', ncOutID, nf90_global)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Next generate a UUID for this file based on a random number

      UUID = UUID_Gen(TempFile) 
      status = nf90_put_att( ncOutID, nf90_global, 'uuid', UUID)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Next processing date and time

      Zone = '+0000'
      call date_and_time( Date, Time, Zone, Values)

      ProcessingDateTime = BlankFill(1:len(ProcessingDateTime))
      ProcessingDateTime = trim(Date) // ' ' // trim(Time)
      status = nf90_put_att( ncOutID, nf90_global,
     1     'date_created', ProcessingDateTime)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'GenerateGlobals #110: '
      
c************* Generate processing history for this file **************

c     Which programs used to produce this file?

      Source = BlankFill(1:len(Source))
      Source = trim(ProcessingProgram) // ' version ' // 
     1     trim(VersionNumber) // '; CommonSubroutines version ' // 
     2     trim(CommonSubsVersionNo())

c     Get the history to date.

      status = nf90_get_att( ncInID, nf90_global, 'history', 
     1     PreviousProcessingHistory)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Put it all together

      ProcessingHistory = BlankFill(1:len(ProcessingHistory))
      ProcessingHistory = '{' // trim(ProcessingDateTime) // ' : ' //
     1     trim(Source) // ' : ' // trim(UUID) // '} ' //
     2     trim(PreviousProcessingHistory)
 
      if(Debug .eq. 1) then
         print *, 'GenerateGlobals #125: ',
     1        'trim(PreviousProcessingHistory)::',
     2        trim(PreviousProcessingHistory), '::'
         print *, 'GenerateGlobals #126: trim(Source)::',
     1        trim(Source), '::'
         print *, 'GenerateGlobals #127: trim(ProcessingHistory):: ',
     1        trim(ProcessingHistory), '::'
      endif

c     And write it to the file.

      status = nf90_put_att( ncOutID, nf90_global, 'history', 
     1     ProcessingHistory)
      if(status .ne. nf90_noerr) call handle_err(status)

c*********************** End processing history ************************

c     Copy creator information

      status = nf90_copy_att( ncInID, nf90_global, 'creator_name', 
     1     ncOutID, nf90_global)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_copy_att( ncInID, nf90_global, 'creator_url', 
     1     ncOutID, nf90_global)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_copy_att( ncInID, nf90_global, 'creator_email', 
     1     ncOutID, nf90_global)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Copy Institution from input file

      status = nf90_copy_att( ncInID, nf90_global, 'institution', 
     1     ncOutID, nf90_global)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'GenerateGlobals #130: '

c     Copy project from input file

      status = nf90_copy_att( ncInID, nf90_global, 'project', 
     1     ncOutID, nf90_global)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Copy Reference from input file

      status = nf90_copy_att( ncInID, nf90_global, 'reference', 
     1     ncOutID, nf90_global)
c      if(status .ne. nf90_noerr) call handle_err(status)         
      if((status .ne. nf90_noerr) .and. (Debug .eq. 1))
     1     print *, 'No reference global GenerateGlobals #132: '

c     Copy Acknowledgement from input file

      status = nf90_copy_att( ncInID, nf90_global, 'acknowledgement', 
     1     ncOutID, nf90_global)
c      if(status .ne. nf90_noerr) call handle_err(status)         
      if((status .ne. nf90_noerr) .and. (Debug .eq. 1))
     1     print *, 'No acknowledgement global GenerateGlobals #134: '

c     To account for stupit spelling error.

      status = nf90_copy_att( ncInID, nf90_global, 'acknowledegement', 
     1     ncOutID, nf90_global)
      if((status .ne. nf90_noerr) .and. (Debug .eq. 1))
     1     print *, 'No acknowledgement global GenerateGlobals #136: '

c     Copy contributor information from input file

      status = nf90_copy_att( ncInID, nf90_global, 'contributor_name', 
     1     ncOutID, nf90_global)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_copy_att( ncInID, nf90_global, 'contributor_role', 
     1     ncOutID, nf90_global)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'GenerateGlobals #138: '

c     Copy publisher information from input file

      status = nf90_copy_att( ncInID, nf90_global, 'publisher_name', 
     1     ncOutID, nf90_global)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_copy_att( ncInID, nf90_global, 'publisher_url', 
     1     ncOutID, nf90_global)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_copy_att( ncInID, nf90_global, 
     1     'publisher_institution', ncOutID, nf90_global)
      if(status .ne. nf90_noerr) call handle_err(status)

c     Copy data type from input file

      status = nf90_copy_att( ncInID, nf90_global, 'cdm_data_type', 
     1     ncOutID, nf90_global)
      if(status .ne. nf90_noerr) call handle_err(status)

      if(Debug .eq. 1) print *, 'GenerateGlobals #140: '

c     Copy the name of the file containing the lat and lon arrays
c     - Make sure there is a LatLon file first.

      status = nf90_copy_att( ncInID, nf90_global, 'LatLonFileName', 
     1     ncOutID, nf90_global)
      if(status .ne. nf90_noerr) then
         write(UnitLog,*) 'Did not find a LatLon file. ', 
     1        'Assume we are generating zenith angle files and ',
     2        'generate the LatLon filename.'
         write(6,*)  'Did not find a LatLon file. ', 
     1        'Assume we are generating zenith angle files and ',
     2        'generate the LatLon filename.'

         status = nf90_put_att( ncOutID, nf90_global,
     1        'LatLonFileName', trim(GeoName))
         if(status .ne. nf90_noerr) call handle_err(status)

      endif

      if(Debug .eq. 1) print *, 'GenerateGlobals #150: '

c     Copy geospatial range data

      status = nf90_copy_att( ncInID, nf90_global, 
     1     'geospatial_lat_min', ncOutID, nf90_global)
      if(status .ne. nf90_noerr) call handle_err(status)
      status = nf90_copy_att( ncInID, nf90_global, 
     1     'geospatial_lat_max', ncOutID, nf90_global)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_copy_att( ncInID, nf90_global, 
     1     'geospatial_lon_min', ncOutID, nf90_global)
      if(status .ne. nf90_noerr) call handle_err(status)
      status = nf90_copy_att( ncInID, nf90_global, 
     1     'geospatial_lon_max', ncOutID, nf90_global)
      if(status .ne. nf90_noerr) call handle_err(status)

      status = nf90_copy_att( ncInID, nf90_global, 
     1     'geospatial_vertical_min', ncOutID, nf90_global)
      if(status .ne. nf90_noerr) call handle_err(status)
      status = nf90_copy_att( ncInID, nf90_global, 
     1     'geospatial_vertical_max', ncOutID, nf90_global)
      if(status .ne. nf90_noerr) call handle_err(status)
      status = nf90_copy_att( ncInID, nf90_global, 
     1     'geospatial_vertical_units', ncOutID, nf90_global)
      if(status .ne. nf90_noerr) call handle_err(status)
      status = nf90_copy_att( ncInID, nf90_global, 
     1     'geospatial_vertical_resolution', ncOutID, nf90_global)
c      if(status .ne. nf90_noerr) call handle_err(status)         
      if((status .ne. nf90_noerr) .and. (Debug .eq. 1))
     1     print *, 'No vertical resolution global GenerateGlobals #152
     2'
      status = nf90_copy_att( ncInID, nf90_global, 
     1     'geospatial_vertical_positive', ncOutID, nf90_global)
c      if(status .ne. nf90_noerr) call handle_err(status)         
      if((status .ne. nf90_noerr) .and. (Debug .eq. 1))
     1     print *, 'No vertical positive global GenerateGlobals #154'

      if(Debug .eq. 1) print *, 'GenerateGlobals #160: '

c     Copy time coverage of these data

      status = nf90_copy_att( ncInID, nf90_global, 
     1     'time_coverage_start', ncOutID, nf90_global)
c      if(status .ne. nf90_noerr) call handle_err(status)         
      if((status .ne. nf90_noerr) .and. (Debug .eq. 1))
     1     print *, 'No time staart global GenerateGlobals #162'
      status = nf90_copy_att( ncInID, nf90_global, 
     1     'time_coverage_duration', ncOutID, nf90_global)
c      if(status .ne. nf90_noerr) call handle_err(status)         
      if((status .ne. nf90_noerr) .and. (Debug .eq. 1))
     1     print *, 'No time end global GenerateGlobals #164'

      if(Debug .eq. 1) print *, 'GenerateGlobals #170: '

      end subroutine GenerateGlobals


c***********************************************************************
      subroutine MergeArchiveInventories
c***********************************************************************
c
c     This subroutine will merge all Archive Inventory files into a
c     single Archive Master file.
c     - Input: All ArchiveInventory_[UUID] files located at BaseDir
c     - Output: Single ArchiveMaster_[UUID] file into BaseDir
c     - Does not delete input ArchiveInventory files.

c******Parameter statements

      include 'ParameterStatements'

      integer*2 UnitArchiveIn, UnitInputList
      integer*2 UnitMasterIn, UnitMasterOut
      character*319 DirectoryIn, ListOfArchives, ArchiveName
      character*319 NewMaster, OldMaster

      real*8 SS1970In(100000)
      integer NumClearIn(100000)
      integer YearIn(100000), MonthIn(100000), DayIn(100000)
      integer HourIn(100000), MinuteIn(100000), SecondIn(100000)
      character*319 FileNameOutIn(100000)

      real*8 SS1970Ms(100000)
      integer NumClearMs(100000)
      integer YearMs(100000), MonthMs(100000), DayMS(100000)
      integer HourMs(100000), MinuteMs(100000), SecondMs(100000)
      character*319 FileNameOutMs(100000)

      integer readErr
      integer i,j
      integer NumberOfArchivesFiles, NumberOfFiles
      integer NumberOfMasterRecs, NumberOfArchiveRecs

      character*36 UUID_Gen

      UnitArchiveIn = 28
      UnitMasterIn = 27
      UnitMasterOut = 29
      UnitInputList = 30
      NumberOfArchiveFiles = 0

c     Get list of archive files for this processor

      DirectoryIn = BlankFill(:len(DirectoryIn))
      DirectoryIn = trim(BaseDir) //
     1     'SupportingFiles/ArchiveInventories/ArchiveInventory_*'

      ListOfArchives = BlankFill(:len(ListOfArchives))
      ListOfArchives =  trim(BaseDir) //
     1     'TmpDir/ListOfArchiveFileNames.txt'

      Command = BlankFill(:len(Command))
      Command = 'ls -1 ' // trim(DirectoryIn) // ' > ' //
     1     trim(ListOfArchives)

      if(debug .eq. 1) print *, 'MergeArchives #100: Command::',
     1     trim(Command), '::'

      call system(Command)

c     Read the temporary file just written

      open (unit=UnitInputList,file=ListOfArchives,status='old',
     1     access='SEQUENTIAL')

c     Loop over file names in the temporary list of filenames.

      do 1120 iFiles=1,MaxNumberOfFiles

c     Read Input Archive from Listing File

         read(UnitInputList,'(A319)',end=1121) ArchiveName

         if(debug .eq. 1) print *, 'MergeArchives #102: In::',
     1        trim(ArchiveName), '::'

         open (unit=UnitArchiveIn, file=ArchiveName,
     1        status='old',access='SEQUENTIAL')

         NumberOfArchiveRecs = 0

         do 1130 j=1,MaxNumberOfFiles

            read(UnitArchiveIn, fmt='(F14.2,I11,6I5,A1,A319)',
     1           end=1131)
     2           SS1970In(j), NumClearIn(j), YearIn(j),
     3           MonthIn(j), DayIn(j), HourIn(j), MinuteIn(j),
     4           SecondIn(j), SpaceCread, FileNameOutIn(j)

c     print *, 'ss1970:',SS1970In(j)

            NumberOfArchiveRecs = NumberOfArchiveRecs + 1

 1130    continue

 1131    continue

         close (unit=UnitArchiveIn)

         NumberOfArchiveFiles = NumberOfArchiveFiles + 1

c     Open new inventory master file

         UUID = UUID_Gen(ArchiveName)

         NewMaster = BlankFill(:len(NewMaster))
         NewMaster = trim(BaseDir) //
     1        'SupportingFiles/ArchiveInventories/ArchiveMaster_' //
     2        UUID

         if(debug .eq. 1) then
            print *, 'MergeArchives #101: New::', trim(NewMaster), '::'
         endif

         open(unit=UnitMasterOut, file=NewMaster, status='new',
     1        access='SEQUENTIAL')


c     Write Archive to new Master Archive

         if (iFiles .eq. 1) then

            if(debug .eq. 1) print *, 'MergeArchives #201: Out::',
     1           trim(NewMaster), '::'

            do 2130 j=1,NumberOfArchiveRecs

               write(UnitMasterOut, fmt='(F14.2,I11,6I5,A1,A319)')
     2              SS1970In(j),NumClearIn(j),YearIn(j),
     3              MonthIn(j),DayIn(j),HourIn(j),MinuteIn(j),
     4              SecondIn(j),SpaceC,FileNameOutIn(j)

c     print *, 'ss1970In:',SS1970In(j)

 2130       continue

c     print *, 'NumberOfArchiveRecs:', NumberOfArchiveRecs

            close(unit=UnitMasterOut)

            OldMaster = NewMaster

         else

c     Merge Archive with Old Master Archive

            if(debug .eq. 1) then
               print *, 'MergeArchives #301: Out::', 
     1              trim(NewMaster), '::'
               print *, 'MergeArchives #301: In::', 
     2              trim(OldMaster), '::'
            endif

            open(unit=UnitMasterIn, file=OldMaster, status='old')

            NumberOfMasterRecs = 0

            do 3130 j=1,MaxNumberOfFiles

               read(UnitMasterIn, fmt='(F14.2,I11,6I5,A1,A319)',
     1              end=3131)
     2              SS1970Ms(j),NumClearMs(j),YearMs(j),
     3              MonthMs(j), DayMs(j), HourMs(j), MinuteMs(j),
     4              SecondMs(j), SpaceCread, FileNameOutMs(j)

c     print *, 'ss1970Ms:',SS1970Ms(j)

               NumberOfMasterRecs = NumberOfMasterRecs + 1

 3130       continue

 3131       continue

c     print *, 'NumberOfMasterRecs:', NumberOfMasterRecs
            close (unit=UnitMasterIn)

c     Merge Master and Archive Records into new Archive Master

            j = 1
            i = 1

            if(debug .eq. 1) print *, 'MergeArchives #304: ',
     1           NumberOfMasterRecs, ':', NumberOfArchiveRecs

            do while (i .le. NumberOfMasterRecs)
               if (j .le. NumberOfArchiveRecs) then
                  if (ss1970In(j) .lt. ss1970Ms(i)) then
                     write(UnitMasterOut,
     1                    fmt='(F14.2,I11,6I5,A1,A319)')
     2                    SS1970In(j),NumClearIn(j),YearIn(j),
     3                    MonthIn(j),DayIn(j),HourIn(j),
     4                    MinuteIn(j),
     5                    SecondIn(j),SpaceC,FileNameOutIn(j)
c     print *, 'ss1970Out:',SS1970In(j)
                     j = j + 1
                  else
                     if ( ss1970In(j) .eq. ss1970Ms(i)) then
                        j = j + 1
                     endif
                     write(UnitMasterOut,
     1                    fmt='(F14.2,I11,6I5,A1,A319)')
     2                    SS1970Ms(i),NumClearMs(i),YearMs(i),
     3                    MonthMs(i),DayMs(i),HourMs(i),
     4                    MinuteMs(i),
     5                    SecondMs(i),SpaceC,FileNameOutMs(i)
c     print *, 'ss1970Out:',SS1970Ms(i)
                     i = i + 1
                  endif

               else

c     Write Out remaining Master Records
                  write(UnitMasterOut,
     1                 fmt='(F14.2,I11,6I5,A1,A319)')
     2                 SS1970Ms(i),NumClearMs(i),YearMs(i),
     3                 MonthMs(i),DayMs(i),HourMs(i),
     4                 MinuteMs(i),
     5                 SecondMs(i),SpaceC,FileNameOutMs(i)
c     print *, 'ss1970Out:',SS1970Ms(i)
                  i = i + 1
               endif
            enddo

c     Check for any remaining input records
            if(debug .eq. 1) print *, 'MergeArchives #306: ',
     1           j, ':', NumberArchiveRecs

            do while (j .le. NumberOfArchiveRecs)

               write(UnitMasterOut,
     1              fmt='(F14.2,I11,6I5,A1,A319)')
     2              SS1970In(j),NumClearIn(j),YearIn(j),
     3              MonthIn(j),DayIn(j),HourIn(j),
     4              MinuteIn(j),
     5              SecondIn(j),SpaceC,FileNameOutIn(j)
c     print *, 'ss1970:',SS1970In(j)

               j = j + 1
            enddo

            close(unit=UnitMasterOut)

            Command = BlankFill(:len(Command))
            Command = 'rm ' // trim(OldMaster)

            if(debug .eq. 1) print *, 'MergeArchives #303: Cmnd::',
     1           trim(Command), '::'

            call system(Command)

            OldMaster = NewMaster

            if(debug .eq. 1) then
               print *, 'MergeArchives #304: New::', trim(NewMaster), 
     1              '::'
               print *, 'MergeArchives #305: Old::', trim(OldMaster), 
     1              '::'
            endif

            close (unit=UnitMasterOut)

         endif

 1120 continue

 1121 continue

      close(UnitInputList)

      DirectoryIn = BlankFill(:len(DirectoryIn))
      DirectoryIn = trim(BaseDir) //
     1     'ArchiveInventory'

      Command = BlankFill(:len(Command))
      Command = 'cp ' // trim(NewMaster) // ' ' //
     1     trim(DirectoryIn)

      if(debug .eq. 1) print *, 'MergeArchives #400: Cmnd::',
     1     trim(Command), '::'

      call system(Command)

      end subroutine MergeArchiveInventories


c***********************************************************************
      subroutine ReadArchiveInventories
c***********************************************************************
c     
c     This subroutine will copy global attributes common to all of the
c     - fronts processing path (Conventions and Institution) and it will
c     - generate the history of previous processing steps from global
c     - attributes in the main input file.
c     

c******Parameter statements

      include 'ParameterStatements'

      integer*2 UnitArchiveIn, UnitInputList
      integer*2 UnitMasterOut

      character*319 DirectoryIn, ListOfArchives, ArchiveName

      real*8 SS1970
      integer NumClear, Year, Month
      integer Day, Hour, Minute, Second
      character*319 FileNameOut

      integer i,j
      integer NumberOfArchiveFiles, NumberOfFiles
      integer NumberOfMasterRecs, NumberOfArchiveRecs

      UnitArchiveIn = 28
      UnitInputList = 30

      debug = 1

      NumberOfArchiveFiles = 0

      if(debug .eq. 1) print *, 'ReadArchives #100: MaxFiles::',
     1        MaxNumberOfFiles

c     Get list of archive files for this processor

      DirectoryIn = BlankFill(:len(DirectoryIn))
      DirectoryIn = trim(BaseDir) //
     1   'SupportingFiles/ArchiveInventories/ArchiveInventory_*'

      ListOfArchives = BlankFill(:len(ListOfArchives))
      ListOfArchives =  trim(BaseDir) //
     1     'TmpDir/ListOfArchiveFileNames.txt'

      Command = BlankFill(:len(Command))
      Command = 'ls -1 ' // trim(DirectoryIn) // ' > ' //
     1     trim(ListOfArchives)

      if(debug .eq. 1) print *, 'ReadArchives #100: Command::',
     1     trim(Command), '::'

      call system(Command)

c     Read the temporary file just written

      open (unit=UnitInputList,file=ListOfArchives,status='old',
     1      access='SEQUENTIAL')

c     Loop over file names in the temporary list of filenames.

      do 1120 iFiles=1,MaxNumberOfFiles

c     Read Input Archive from Listing File

        read(UnitInputList,'(A319)',end=1121) ArchiveName

        if(debug .eq. 1) print *, 'ReadArchives #102: In::',
     1       trim(ArchiveName)

        open (unit=UnitArchiveIn, file=ArchiveName,
     1       status='old',access='SEQUENTIAL')

        NumberOfArchiveRecs = 0

        do 1130 j=1,MaxNumberOfFiles

            read(UnitArchiveIn, fmt='(F14.2,I11,6I5,A1,A319)',
     1          end=1131)
     2          SS1970, NumClear, Year,
     3          Month, Day, Hour, Minute,
     4          Second, SpaceCread, FileNameOut

             print *, 'ss1970:',SS1970
             NumberOfArchiveRecs = NumberOfArchiveRecs + 1

 1130   continue

 1131   continue

        if(debug .eq. 1) print *, 'NumberOfArchiveRecs:',
     1          NumberOfArchiveRecs

        close (unit=UnitArchiveIn)

        NumberOfArchiveFiles = NumberOfArchiveFiles + 1

 1120 continue

 1121 continue

      if(debug .eq. 1) print *, 'NumberOfArchiveFiles:',
     1    NumberOfArchiveFiles

      close(UnitInputList)

      end subroutine ReadArchiveInventories

c**********************************************************************
      subroutine GenerateFileList( DirectoryIn, ListOfDailyFiles)
c**********************************************************************
c     
c     This subroutine will generate the list of input SST files to the
c     workflow in a DirectoryIn. The calling program generally loops
c     over year and month directories, creating and processing a list
c     for each.
c     
      use netcdf

      implicit none

c****** Parameter statements

      include 'ParameterStatements'

c****** Function statements

      character*36 UUID_Gen

c     DirectoryIn - the directory with the input files.
      character*319 DirectoryIn

c     ListOfDailyFiles - the name of the file with the list of filenames
c     for one of the directories to traverse.
      character*319 ListOfDailyFiles
c     ListOfFiles - 
      character*319 ListOfFiles

c     TmpListOfFiles - 
      character*319 TmpListOfFiles

c---------------------------------start here --------------------------
      if(debug .eq. 1) print *, 'GenerateFileList #100: DirectoryIn::',
     1     trim(DirectoryIn), '::'

c     Names generated below are for files that hold the list of input
c     files. UUIDs have been added to them to make unique so that we 
c     do not have collisions when running several jobs at once.

      InputFileName = BlankFill(:len(InputFileName))
      InputFileName = 'Temp1_pcc.Data'
      UUID = UUID_Gen(InputFileName)

      ListOfFiles = BlankFill(:len(ListOfFiles))
      ListOfFiles =  trim(BaseDir) // 
     1     'TmpDir/ListOfInputFileNames_' // trim(UUID) // '.txt'

      InputFileName = BlankFill(:len(InputFileName))
      InputFileName = 'Temp2_pcc.Data'
      UUID = UUID_Gen(InputFileName)

      TmpListOfFiles = BlankFill(:len(TmpListOfFiles))
      TmpListOfFiles =  trim(BaseDir) //
     1     'TmpDir/TmpListOfInputFileNames_' // trim(UUID) // '.txt'

      InputFileName = BlankFill(:len(InputFileName))
      InputFileName = 'Temp3_pcc.Data'
      UUID = UUID_Gen(InputFileName)

      ListOfDailyFiles = BlankFill(:len(ListOfDailyFiles))
      ListOfDailyFiles =  trim(BaseDir) //
     1     'TmpDir/ListOfInputDailyFiles_' // trim(UUID) // '.txt'

      Command = BlankFill(:len(Command))
      Command = 'ls -1 ' // trim(DirectoryIn) // ' > ' // 
     1     trim(ListOfDailyFiles)

      if(debug .eq. 1) print *, 'GenerateFileList #110: Command::', 
     1     trim(Command), '::'

      call system(Command)

      Command = BlankFill(:len(Command))
      Command = 'cat ' // trim(ListOfFiles) // ' ' //
     1     trim(ListOfDailyFiles) // ' > ' //
     2     trim(TmpListOfFiles)

      if(debug .eq. 1) print *, 'GenerateFileList #120: Command::',
     1     trim(Command), '::'

      call system(Command)

      Command = BlankFill(:len(Command))
      Command = 'mv ' // trim(TmpListOfFiles) // ' ' //
     1     trim(ListOfFiles)

      if(debug .eq. 1) print *, 'GenerateFileList #130: Command::',
     1     trim(Command), '::'

      call system(Command)

      end subroutine GenerateFileList

c***********************************************************************
      subroutine lower_case(word)
c***********************************************************************
c     convert a word to lower case

      character (len=*) , intent(in out) :: word
      integer                            :: i,ic,nlen

      nlen = len(word)
      do i=1,nlen
         ic = ichar(word(i:i))
         if (ic >= 65 .and. ic < 90) word(i:i) = char(ic + 32)
      end do 
      end subroutine lower_case

c***********************************************************************
      subroutine upper_case(word)
c***********************************************************************
c     convert a word to lower case

      character (len=*) , intent(in out) :: word
      integer                            :: i,ic,nlen

      nlen = len(word)
      do i=1,nlen
         ic = ichar(word(i:i))
         if(ic >= 97 .and. ic <= 122)  word(i:i) = char(ic - 32)
      end do 
      end subroutine upper_case

c***********************************************************************
      subroutine plane_fit(sst_in,fit_out,start_i_in,start_j_in)
c***********************************************************************
      implicit none
      include 'ParameterStatements'
c     Computing Inverse matrix
c     Method: Based on the Doolittle LU method

c     x1 contains [xdummy,ydummy,sst] values determined by the first
c     dimension of x1: x1(1,:) = xvalues, x1(2,:) = yvalues, x1(3,:) =
c     sst values etc
      integer*2 :: x1(3,PlaneFitSize*PlaneFitSize)

      integer, parameter :: n=3

c     a and c exist only as temp arrays to pass to the inversion subroutine
      real,dimension(n,n) :: a, c
      integer*2 i,j
      integer counter_j
      real, dimension(n) :: RHS, eval

      integer*2, dimension(LenXA,LenYA), intent(in) :: sst_in
      integer*2, dimension(WinSize,WinSize), intent(out) :: fit_out
      integer*2, intent(in) :: start_i_in, start_j_in
      integer :: start_i, start_j

c     WinBorder - the width of the border around the histogram window 
c      to include in the region for which a plane is fit. 
      integer WinBorder

c***********************************************************************
c     Start Here

      if(Debug .eq. 1) print *, 'Plane_fit #000'

c     Get the width of the border around the histogram window to include
c      in the region for which a plane is fit. 

      WinBorder = (PlaneFitSize - WinSize) / 2

      start_i = start_i_in - WinBorder;
      start_j = start_j_in - WinBorder;
c     Values passed in for start_i, start_j are the locations where the
c     WinSize x WinSize window start in the original image. Changing 
c     them now for clarity to where the PlaneFitSize x PlaneFitSize
c     matrix I use to fit would start(note, this means they might be 
c     negative if the edges go out of bounds!)


c     first fill matrix of values to be fitted, where out of bounds
c     values are set to NaN (-999)

      counter_j = 0
      do i = 1,PlaneFitSize
         do j = 1,PlaneFitSize
            if((i-1)+start_i>0.and.(j-1)+start_j>0.and.(j-1)+start_j
     1           .le.LenY.and.(i-1)+start_i.le.LenX) then
c     windows that have no out of bounds locations are assigned as normal
               x1(3,i+(j-1)*PlaneFitSize) = 
     1              sst_in((i-1)+start_i,(j-1)+start_j) 
            else
c    and are set to be ignored otherwise
               x1(3,i+(j-1)*PlaneFitSize) = -999
               counter_j = counter_j + 1
            endif
         enddo
      enddo
c     count the number of NaN's passed in with pict and ignore them; 
c     This might have been messing things up very badly before this fix

      do i = 1,PlaneFitSize
         do j = 1,PlaneFitSize
            if(x1(3,i+PlaneFitSize*(j-1)).eq. SSTFillValueIn) then
               x1(3,i+PlaneFitSize*(j-1)) = -999
               counter_j = counter_j + 1
            endif
         enddo
      enddo



      do i = 1,PlaneFitSize
         do j = 1,PlaneFitSize
            x1(1,i+PlaneFitSize*(j-1))=0 + (i-1)*1
            x1(2,i+PlaneFitSize*(j-1))=0 + (j-1)*1            
         enddo
      enddo

      a = 0.0
      RHS = 0.0
      if(counter_j.lt.PlaneFitSize*PlaneFitSize) then
         do i = 1,PlaneFitSize*PlaneFitSize
            if(x1(3,i).gt.-999)then
               a(1,1) = a(1,1) + real(x1(1,i))*real(x1(1,i))
               a(1,2) = a(1,2) + real(x1(2,i))*real(x1(1,i))
               a(1,3) = a(1,3) + real(x1(1,i))
               
               a(2,1) = a(2,1) + real(x1(2,i))*real(x1(1,i))
               a(2,2) = a(2,2) + real(x1(2,i))*real(x1(2,i))
               a(2,3) = a(2,3) + real(x1(2,i))
               
               a(3,1) = a(3,1) + real(x1(1,i))
               a(3,2) = a(3,2) + real(x1(2,i))
               a(3,3) = a(3,3) + 1;
               
               RHS(1) = RHS(1) + real(x1(3,i))*real(x1(1,i))
               RHS(2) = RHS(2) + real(x1(3,i))*real(x1(2,i))
               RHS(3) = RHS(3) + real(x1(3,i))
            endif
         enddo         
         
         call inverse(a,c,n)
         
         eval(1) = c(1,1)*RHS(1) + c(1,2)*RHS(2) + c(1,3)*RHS(3)
         eval(2) = c(2,1)*RHS(1) + c(2,2)*RHS(2) + c(2,3)*RHS(3)
         eval(3) = c(3,1)*RHS(1) + c(3,2)*RHS(2) + c(3,3)*RHS(3)
         
         
c     now finally, recreate the 32x32 matrix with the linear background
c     to remove
         
         do i = 1,WinSize
            do j = 1,WinSize
               fit_out(i,j) = int(eval(1),2) * 
     1              x1(1,(i+WinBorder)+(j+WinBorder-1)*PlaneFitSize) + 
     2              int(eval(2),2) * 
     3              x1(2,(i+WinBorder)+(j+WinBorder-1)*PlaneFitSize) +
     4              int(eval(3),2)
            enddo
         enddo
      else
         do i = 1,WinSize
            do j = 1,WinSize
               fit_out(i,j) = 0
            enddo
         enddo
      endif

      if(Debug .eq. 1) print *, 'Plane_fit #999'

      end subroutine plane_fit
      
      subroutine inverse(a,c,n)
c============================================================
c     Inverse matrix
c     Method: Based on Doolittle LU factorization for Ax=b
c     Alex G. December 2009
c-----------------------------------------------------------
c     input ...
c     a(n,n) - array of coefficients for matrix A
c     n      - dimension
c     output ...
c     c(n,n) - inverse matrix of A
c     comments ...
c     the original matrix a(n,n) will be destroyed 
c     during the calculation
c===========================================================
      implicit none 
      integer n
      real a(n,n), c(n,n)
      real L(n,n), U(n,n), b(n), d(n), x(n)
      real coeff
      integer i, j, k

c     step 0: initialization for matrices L and U and b
c     Fortran 90/95 aloows such operations on matrices
      L=0.0
      U=0.0
      b=0.0

c     step 1: forward elimination
      do k=1, n-1
         do i=k+1,n
            coeff=a(i,k)/a(k,k)
            L(i,k) = coeff
            do j=k+1,n
               a(i,j) = a(i,j)-coeff*a(k,j)
            end do
         end do
      end do

c     Step 2: prepare L and U matrices 
c     L matrix is a matrix of the elimination coefficient
c     + the diagonal elements are 1.0
      do i=1,n
         L(i,i) = 1.0
      end do
c     U matrix is the upper triangular part of A
      do j=1,n
         do i=1,j
            U(i,j) = a(i,j)
         end do
      end do

c     Step 3: compute columns of the inverse matrix C
      do k=1,n
         b(k)=1.0
         d(1) = b(1)
c     Step 3a: Solve Ld=b using the forward substitution
         do i=2,n
            d(i)=b(i)
            do j=1,i-1
               d(i) = d(i) - L(i,j)*d(j)
            end do
         end do
c     Step 3b: Solve Ux=d using the back substitution
         x(n)=d(n)/U(n,n)
         do i = n-1,1,-1
            x(i) = d(i)
            do j=n,i+1,-1
               x(i)=x(i)-U(i,j)*x(j)
            end do
            x(i) = x(i)/u(i,i)
         end do
c     Step 3c: fill the solutions x(n) into column k of C
         do i=1,n
            c(i,k) = x(i)
         end do
         b(k)=0.0
      end do
      end subroutine inverse

