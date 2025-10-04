function varargout = RadOfCurvFromVel(varargin)
% Compute the radius of curvature from the velocity field.
%
% radius = RadOfCurvFromVel(lat,lon,u,v);
% radius = RadOfCurvFromVel(lat,lon,u,v,u_grad_tensor);
%
% INPUT(S):
%
%   lat -- [M x N] latitude in degrees
%   lon -- [M x N] longitude in degrees
%   u -- [M x N] eastward horizontal velocity in meters per second
%   v -- [M x N] northward horizontal velocity in meters per second
%   u_grad_tensor -- (optional) a structure with the following fields:
%
%       .dudx = [M x N matrix] shear du/dx
%       .dudy = [M x N matrix] shear du/dy
%       .dvdx = [M x N matrix] shear dv/dx
%       .dvdy = [M x N matrix] shear dv/dx
%
%   if precomputed, it increases efficiency.     
%
% OUTPUT(S):
%
%   radius -- radius of curvature in meters
%   u_grad_tensor -- (optional) horizontal velocity gradient tensor
%
% Christian E. Buckingham
% UMASSD

%% Parse inputs.
if nargin == 4

    lat = varargin{1};
    lon = varargin{2};
    u = varargin{3};
    v = varargin{4};

elseif nargin == 5

    lat = varargin{1};
    lon = varargin{2};
    u = varargin{3};
    v = varargin{4};
    u_grad_tensor = varargin{5};

else

    error('Incorrect number of input args.')
    
end

sdata = size(u);
ndims = length(sdata);

%% Horizontal velocity tensor.
if exist('u_grad_tensor','var')

    % do nothing

else

    % Estimate velocity gradients.
    sw_sobel = 1; % reduces noise
    [DUDX,DUDY] = EstimateGradXY(lat,lon,u,sw_sobel);
    DUDX = DUDX./1000; % divide by distance, convert km to m
    DUDY = DUDY./1000; % divide by distance, convert km to m
    [DVDX,DVDY] = EstimateGradXY(lat,lon,v,sw_sobel);
    DVDX = DVDX./1000; % divide by distance, convert km to m
    DVDY = DVDY./1000; % divide by distance, convert km to m

    u_grad_tensor.dudx = DUDX;
    u_grad_tensor.dudy = DUDY;
    u_grad_tensor.dvdx = DVDX;
    u_grad_tensor.dvdy = DVDY;

end

%% Radius of curvature.
tmp1 = u.^2;
tmp2 = v.^2;
ARG1 = (tmp1 + tmp2).^(1.5);
arg2 =  tmp1.*u_grad_tensor.dvdx;
arg3 = -tmp2.*u_grad_tensor.dudy;
%arg3 = -tmp2.*u_grad_tensor.dvdy;
arg4 = u.*v.*(u_grad_tensor.dvdy - u_grad_tensor.dudx);
ARG2 = arg2 + arg3 + arg4;
radius_of_curv = ARG1./ARG2;
%radius_of_curv = radius_of_curv./1000; % convert to kilometers

%% Handle outputs.
if nargout == 1

    varargout{1} = radius_of_curv;

elseif nargout == 2

    varargout{1} = radius_of_curv;
    varargout{2} = u_grad_tensor;

end

return % end of function
