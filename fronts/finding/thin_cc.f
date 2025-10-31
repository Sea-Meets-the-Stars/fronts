c**********************************************************************
      subroutine thin( MTFileName, hdate, MedSST, MergedFronts,
     1                 ncMergedID, MergedID, SecondsSince1970ID)
c**********************************************************************
c
c     MTFILENAME - (input/output file name)
c     MEDSST(1:SizOfImg,1:SizOfImg) - med-fltrd image (0,255)
c     MergedFronts(1:SizOfImg,1:SizOfImg) - original edge img
c     Thinnedfronts(1:SizOfImg,1:SizOfImg) - thinned edge image
c
c     I,J - generic indices
c     K - current image
c
c     DIFTEMP - difference of temperature
c     MAXDIF - maximum DIFTEMP
c     IMAX,JMAX - MAXDIF coordinates
c     COUNT - distance from last edge
c     win - median filter window
c     dumhdate - dummy value for date passed from Read_Image
c     dumlats(1:SizOfImg),dumlons(SizOfImg) - dummy values for
c               lats/lons from Read_Image
c
c     DataRange - maximum minus minimum value read in.
c     NAboveRange - number of input data values above Range.
c     UnitOut - the unit number to which ProcInfo will be written
c
c     21-May-2009 - PCC - The upper limits of the 23032 and 23034 loops
c                   were backward; LenY for the first dimension and LenX
c                   for the second dimension. I reversed them.
c
      implicit none

c     ******Functions
c     HoursSince - returns the number of hours since the reference.
c                  Arguments are year, month, day, hour, minute and the
c                  values for these.
      real*8 HoursSince

c     ******Parameter statements
      include 'ParameterStatements.f'

c     ******General variables
      character*319 MTFileName

c     Variables used to test for thinned fronts in input file.
      integer ncMTID, status2
      integer frID, frDims(2)
      integer*2 MedSST(1:LenX, 1:LenY)
      integer*2 MergedFronts(1:LenX, 1:LenY)
      integer*2, allocatable :: ThinnedFronts(:,:)
      integer*2 i, j
      integer*2 k
      integer*2 diftemp
      integer*2 maxdif
      integer*2 imax, jmax
      integer*2 count
      integer*2 win
      character filtyp
      real*8 hdate
      real*4, allocatable :: dumlats(:), dumlons(:)
      integer*4 DataRange, NAboveRange

c     MergedOrThinned - 1 to read MergedFronts and 2 to read
c                       ThinnedFronts
      integer*2 MergedOrThinned
      integer ncMergedID, MergedID, SecondsSince1970ID

      include 'netcdf.inc'

c     Allocate arrays
      allocate( ThinnedFronts(1:LenX,1:LenY),
     1          dumlats(1:LenY),
     2          dumlons(1:LenX),
     3          stat=ierr)
      if (ierr .ne. 0) then
          stop 'thin #100: Allocation error for LenX, LenY arrays.'
      endif

      win = 5

      if (debug .eq. 1) write(6,*) MTFileName
      if (debug .eq. 1) write(UnitLog,*) MTFileName

      if (debug .eq. 1) then
          Msg50 = '--- thin #330: SST after median in thin --- EOS'
          call PrintArray2( MedSST(istrt:iend,jstrt:jend), Msg50)
      endif

c     Set output thinned front data to 0.
      do 23014 i = 1, LenX
          do 23016 j = 1, LenY
              ThinnedFronts(i,j) = 0
23016     continue
23014 continue

c     ====================================================================
c     FIRST PASS: Thin fronts in the J-direction (vertical)
c     ====================================================================
      diftemp = 32767
      do 23018 i = 1, LenX
          maxdif = 2
          do 23020 j = 2, LenY-1
              if (MergedFronts(i,j) .eq. 4) then
                  count = 0
                  if (min(MedSST(i,j-1),MedSST(i,j+1)) .gt. 8) then
                      diftemp = abs(MedSST(i,j+1) - MedSST(i,j-1))
                  endif
                  if (maxdif .lt. diftemp) then
                      maxdif = diftemp
                      imax = i
                      jmax = j
                  endif
              else
                  if (maxdif .ne. 2) then
                      if (count .ge. 2) then
                          ThinnedFronts(imax,jmax) = 4
                          maxdif = 2
                      endif
                      count = count + 1
                  endif
              endif
23020     continue
23018 continue

c     ====================================================================
c     SECOND PASS: Thin fronts in the I-direction (horizontal)
c     ====================================================================
      do 23032 j = 1, LenY
          maxdif = 2
          do 23034 i = 2, LenX-1
              if (MergedFronts(i,j) .eq. 4) then
                  count = 0
                  if (min(MedSST(i-1,j),MedSST(i+1,j)) .gt. 8) then
                      diftemp = abs(MedSST(i+1,j) - MedSST(i-1,j))
                  endif
                  if (maxdif .lt. diftemp) then
                      maxdif = diftemp
                      imax = i
                      jmax = j
                  endif
              else
                  if (maxdif .ne. 2) then
                      if (count .ge. 2) then
                          ThinnedFronts(imax,jmax) = 4
                          maxdif = 2
                      endif
                      count = count + 1
                  endif
              endif
23034     continue
23032 continue

c     Write the thinned fronts to the output file
      MergedOrThinned = 2
      call WriteMergedThinned( MTFileName, hdate, dumlats,
     1                         dumlons, ThinnedFronts, MergedOrThinned,
     2                         ncMergedID, MergedID, SecondsSince1970ID)

      return
      end
