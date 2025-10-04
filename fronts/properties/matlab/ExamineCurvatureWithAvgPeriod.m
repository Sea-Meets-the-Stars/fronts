% Examine curvature with averaging period.
%
% This is matlab code to examine the amount of curvature in
% geostrophic velocities estimated from ocean currents.
%
% There are two options: 1) geostrophic velocity
%                        2) total velocity (with mean dynamic height + winds accounted for)
%
%
% This particular code is set up to examine U and V from OSCAR currents
% obtained from PO.DAAC. These are located in subfolders by year.
%
% Christian Buckingham
% Suraj Singh
% Amit Tandon
%
% December 2023

% Add paths.
addpath('~/OSCAR/code/UsefulFunctions')
addpath('~/OSCAR/code/UsefulFunctions/geostrophic')
addpath('~/OSCAR/code/UsefulFunctions/seawater_ver3_3')
addpath('~/OSCAR/code/UsefulFunctions/m_map_1.4h')
addpath('~/OSCAR/code/UsefulFunctions/time.epoch')

% User-defined parameters.
sw_geo = 1; % 0 off, 1 on; switch to read geostrophic velocities
sw_graphics = 1; % 0 off, 1 on; switch to create plot
sw_saveplot = 1; % 0 off, 1 on; switch to save plots to file
sw_savedata = 1; % 0 off, 1 on; switch to save data to file

% Graphics options.
FontSize = 18;
FontName = 'Times';
figsize = [10 5]*1.25;
ProjName = 'Miller Cylindrical'; %'Azimuthal Equal-area'; %'Hammer-Aitoff';
latlim = [-1 1]*70;
lonlim = [0 360];

% Define main input directory.
basedirin = '~/OSCAR';

%=====================================================
% SEASONAL AVERAGE.
%=====================================================
% Define input parameters.
yrs_of_interest = [2023];
mos_of_interest = (1:12); % jan,feb,mar
ndays_per_month = [31,28,31,30,31,30,31,31,30,31,30,31]; % we handle leap years below

% The following code only gets the relevant filenames.
files = {}; % set to zero initially
ifile = 1; % set to one initially

for iyr = 1:length(yrs_of_interest) % loop over years
for imo = 1:length(mos_of_interest) % loop over months
    yr = yrs_of_interest(iyr);
    mo = mos_of_interest(imo);
    
    if mo == 2 && isleap(yr) % handle leap years
        ndays_per_month(mo) = 29;
    end
    nd = ndays_per_month(mo); % total number of days
    dys_of_interest = (1:nd); % number of days of interest
    for ida = 1:nd % loop over days of this month
        
        % Define the date.
        da = dys_of_interest(ida);
        yyyy = num2str(yr,'%04d');
        mm = num2str(mo,'%02d');
        dd = num2str(da,'%02d');
        yyyymmdd = [yyyy,mm,dd];
        prefx = 'oscar_currents_interim';
        subdir = yyyy;
        pname = fullfile(basedirin,subdir);
        fname = [prefx,'_',yyyymmdd,'.nc'];
        infile = fullfile(pname,fname);
        
        % Store this filename if it exists.
        if exist(infile,'file')
            disp(['Identified file of interest ... ',infile])
            files{ifile} = infile;
            ifile = ifile + 1;
        end
        
    end % loop over days of this month
     
end % loop over months
end % loop over years
%=====================================================

nfile = length(files);
etime = nan(nfile,1); % allocate memory
for ifile = 1:nfile % loop over files of interest

    infile = files{ifile}; % present file
    [pname,fname,ext] = fileparts(infile);
    dummy = strtok_apl(fname,'_');
    yyyymmdd = dummy{4}; %fname(24:end-3);
    yyyy = yyyymmdd(1:4);
    mm = yyyymmdd(5:6);
    dd = yyyymmdd(7:8);
    yr = str2double(yyyy);
    mo = str2double(mm);
    da = str2double(dd);
    gtimei = [yr mo da 12 0 0];
    etimei = greg2epoch(gtimei);
    
    disp(['Reading file ... ',infile])
    %if ifile == 1
    %    ncdisp(infile) % display contents of first file
    %    keyboard
    %end
    
    lat = ncread(infile,'lat');
    lon = ncread(infile,'lon');
    if sw_geo % turn on reading of geostrophic velocities
        U = ncread(infile,'ug'); % there are geostrophic velocities in this file too
        V = ncread(infile,'vg'); % there are geostrophic velocities in this file too
    else
        U = ncread(infile,'u'); % total velocities
        V = ncread(infile,'v'); % total velocities
    end
    
    %=====================================================
    % Allocate memory for data.
    %=====================================================
    
    %% Allocate memory for data.
    if ifile == 1 % allocate memory
        
        sdata = size(U);
        U_sum = zeros(sdata); % allocate
        U_count = zeros(sdata); % allocate
        V_sum = zeros(sdata); % allocate
        V_count = zeros(sdata); % allocate
        
    end % allocate memory

    U_count = U_count + 1;
    V_count = V_count + 1;

    iselect = ~isnan(U);
    U_sum(iselect) = U_sum(iselect) + U(iselect);
    iselect = ~isnan(V);
    V_sum(iselect) = V_sum(iselect) + V(iselect);
    
    %=====================================================
    % Compute means.
    iselect = U_count > 2;
    U(iselect) = U_sum(iselect)./U_count(iselect);
    U(~iselect) = nan;

    iselect = V_count > 2;
    V(iselect) = V_sum(iselect)./V_count(iselect);
    V(~iselect) = nan;

    %=====================================================
    % Compute curvature quantities from these time-averaged fields.
    %=====================================================
    % Define new variables.
    [LON,LAT] = meshgrid(lon,lat);
    UMAG2 = U.^2 + V.^2;
    UMAG = sqrt(UMAG2);

    % Estimate velocity gradients.
    [LON,LAT] = meshgrid(lon,lat);
    sw_sobel = 1;
    [DUDX,DUDY] = EstimateGradXY(LAT,LON,U,sw_sobel);
    [DVDX,DVDY] = EstimateGradXY(LAT,LON,V,sw_sobel);
    DUDX = DUDX./1000;
    DUDY = DUDY./1000;
    DVDX = DVDX./1000;
    DVDY = DVDY./1000;

    % Estimate vertical component of relative vorticity.
    VORT = DVDX - DUDY;
    ff = sw_f(LAT);
    ROSSBY = VORT./ff;

    % Now compute radius of curvature.
    Rc = RadOfCurvFromVel(LAT,LON,U,V);
    Rc_km = Rc./1000; % meters to kilometers
    kappa = 1./Rc_km;

    % Approximate curvature number from magnitude of velocity.
    CURVATURE = 2.*UMAG./ff./(Rc + 1);

    % Compute horizontal kinetic energy.
    KE = 0.5.*UMAG2; % m^2/s^2

    % Define Suraj's V/Rc^2 term (curvature beta).
    BETA_C = UMAG./Rc./Rc; % 1/(m*s) % same units as planetary beta

    % Get rid of data at equatorial latitudes.
    latlimnan = [-1 1]*6; % not sure what is best here, but seems like a good idea
    inan = LAT >= latlimnan(1) & LAT <= latlimnan(2);
    VORT(inan) = nan;
    ROSSBY(inan) = nan;
    CURVATURE(inan) = nan;

    % Determine planetary beta.
    ff = sw_f(LAT);
    [~,DFDY_perkm] = EstimateGradXY(LAT,LON,ff);
    % DLAT = 0.5.*(LAT(1:end-1,:) - LAT(2:end));
    BETA_P = DFDY_perkm./1000;
    
    %=====================================================
    % SAVE DATA TO FILE.
    %=====================================================
    if sw_savedata % save data to file
    
        if ifile == 1 % if first file
            
            gtime_start = gtimei;
            etime_start = etimei;
            yyyymmdd_start = yyyymmdd;
            
            gtime_stop = gtimei;
            etime_stop = etimei;
            yyyymmdd_stop = yyyymmdd;
            
            averaging_period_dy = 1;
            
        else
        
            gtime_stop = gtimei;
            etime_stop = etimei;
            yyyymmdd_stop = yyyymmdd;
            
            averaging_period_dy = (etime_stop - etime_start + 1)/24/3600; % number of days
            
        end % ifile == 1
        
        subdirout1 = 'data-output';
        subdirout2 = 'RunningAverage';
        subdirout3 = yyyy;
        outdir = fullfile(basedirin,subdirout1,subdirout2,subdirout3);
        if ~exist(outdir,'dir')
            mkdir(outdir)
        end
        pnameout = outdir;
        fnameout = ['RunningAverage_U_V_BetaC_BetaP','.','',yyyymmdd_start,'.','',yyyymmdd_stop,'.mat'];
        outfile = fullfile(pnameout,fnameout);
        disp(['Writing file ... ',outfile])
        save(outfile, 'yyyymmdd_start', 'yyyymmdd_stop', 'gtime_start', 'gtime_stop',...
                      'etime_start','etime_stop', 'averaging_period_dy',...
                      'lat','lon','U','V','U_sum','V_sum','U_count','V_count',...
                      'BETA_C*','BETA_P*');
    
    end % sw_savedata
    
    %=====================================================
    % GRAPHICS. (ratio of beta_curvature to beta_planetary)
    %=====================================================
    nd = ndays_per_month(mo);
    if sw_graphics && rem(da,nd) == 0 % turn on graphics if first of the month

        figure(7)
        clf

        RATIO = double(BETA_C)./double(BETA_P);
        thold = 1;
        thold = 0.25;
        iselect = RATIO < thold;
        RATIO(iselect) = nan;

        RATIO_alt = double(BETA_C)./double(BETA_P);
        iselect = RATIO_alt > 100;
        RATIO_alt(iselect) = nan;
        ratio_mean = mean(RATIO_alt,2,'omitnan');
        ratio_median = median(RATIO_alt,2,'omitnan');

        % Transform latitude to mapping coordinates.
        latrad = lat.*pi/180;
        latlimrad = latlim.*pi/180;
        ynew = (5/4).*log(tan(pi/4 + 2*latrad/5)); %ynew = (5/4).*asinh(tan(4/5.*lat));
        ynewlim = (5/4).*log(tan(pi/4 + 2*latlimrad/5));

        set(gcf,'PaperSize',figsize)
        set(gcf,'PaperPosition',[0 0 figsize]);
        previewmap

        m_proj(ProjName,'lon',lonlim,'lat',latlim);
        m_pcolor(LON,LAT,RATIO);
        shading('flat')
        hold on

        m_coast('patch',[1 1 1]*0.75);
        m_grid('xaxis','bottom','FontName',FontName,'FontSize',FontSize);
        title(gca,'Curvature Beta $\beta_{c}$ to Planetary Beta $\beta_{p}$','Interpreter','Latex',...
            'FontWeight','Normal','FontName',FontName,'FontSize',FontSize);

        hbar = colorbar('v');
        caxis([0 10])
        set(hbar,'FontSize',FontSize)
        title(hbar,'$\beta_{c}/\beta_{p}$','Interpreter','Latex')
        set(hbar,'FontName',FontName)
        cmocean('matter')
        % colormap parula
        % set(gca,'ColorScale','log')

        cmap = cmocean('matter');
        CLIM = caxis;
        tmp = linspace(CLIM(1),CLIM(2),length(cmap));

        inan = tmp < thold;
        indx = find(inan,1,'last');
        cmapnew = cmap;
        npts = indx;
        cmapnew(1:indx,:) = ones(npts,3);
        colormap(cmapnew);

        hpos = get (gca,'Position');
        set(gca,'Position',[hpos(1)-0.07, hpos(2), hpos(3), hpos(4)]);
        hax_main = gca;
        YTick = get(hax_main,'YTick');
        YTickLabel = get(hax_main,'YTickLabel');

        hax_sub = axes(gcf,'Position',[hpos(1)+hpos(3)-0.012, hpos(2)+0.06, 0.1, hpos(4)-0.12]);
        h1 = plot(ratio_mean,ynew,'-','linewidth',2);
        % hold on
        % h2 = plot(ratio_mean,ynew,'--','linewidth',2);
        hold on
        h2 = plot(ratio_median,ynew,'--','linewidth',2);
        grid on
        set(gca,'FontName',FontName,'FontSize',FontSize)
        set(gca,'xdir','normal')
        xlim(CLIM)
        YLIM = ynewlim;
        ylim(YLIM)
        set(gca,'YAxisLocation','right')

        YTickRad = [-60 -30 0 30 60]*pi/180;
        YTickNew = (5/4).*log(tan(pi/4 + 2*YTickRad/5));
        set(hax_sub,'YTick',YTickNew);
        %set(hax_sub,'YTickLabel',{})
        set(hax_sub,'YTickLabel',{'60{\circ}S','30{\circ}S','0{\circ}','30{\circ}N','60{\circ}N'})%,'Interpreter','Tex'); %,'FontName',FontName)
        title(hax_sub,'Bulk Ratio','FontSize',FontSize,'FontWeight','normal','Interpreter','Latex');
        % xlabel(hax_sub,'$\left < \beta_{c}/\beta_{p} \right >$','Interpreter','Latex','FontSize',FontSize)
        xlabel(hax_sub,'$Mn\left < \beta_{c}/\beta_{p}\right >,Med\left < \beta_{c}/\beta_{p} \right >$','Interpreter','Latex','FontSize',FontSize-4)
        %xlabel(hax_sub,'Mn(-),~Med(-~-)','Interpreter','Latex','FontSize',FontSize)
        %xlabel(hax_sub,'Mean,~Median','Interpreter','Latex','FontSize',FontSize-4)

        hold on
        h3 = plot([2 2],ylim,'-','Color',[1 1 1]*0.5,'linewidth',0.25);

        hleg = legend([h1,h2],{'mean','med.'},'Location','west');
        %=====================================================
        % save to file
        if sw_saveplot
        
            subdirout1 = 'graphics';
            subdirout2 = 'RunningAverage';
            subdirout3 = yyyy;
            outdir = fullfile(basedirin,subdirout1,subdirout2,subdirout3);
            if ~exist(outdir,'dir')
                mkdir(outdir)
            end
            pnameout = outdir;
            fnameout = ['Beta_Ratio.RunningAverage','.','',yyyymmdd_start,'.','',yyyymmdd_stop,'.png'];
            outfile = fullfile(pnameout,fnameout);
            disp(['Writing file ... ',outfile])
            
            suffx = datestr(datenum([yr mo da]),'yyyymmm');
            exportgraphics(gcf, outfile, 'Resolution', 300);
        end
        %=====================================================
        
    end % sw_graphics

end % loop over files of interest