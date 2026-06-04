function phantom = make_cyst_phantom(cfg, cyst_center, cyst_radius, cyst_contrast, seed)
%MAKE_CYST_PHANTOM Create a speckle phantom with a circular cyst.
%   cyst_center is [x0, z0] in meters. cyst_contrast scales amplitudes
%   inside the cyst: 0 anechoic, 0.1 strongly hypoechoic, 1 same as
%   background.

if nargin < 5 || isempty(seed)
    seed = 1;
end

if nargin < 4 || isempty(cyst_contrast)
    cyst_contrast = 0;
end

rng(seed, 'twister');

num_scatterers = cfg.num_scatterers_cyst;

x = cfg.x_limits(1) + diff(cfg.x_limits) * rand(num_scatterers, 1);
y = zeros(num_scatterers, 1);
z = cfg.z_limits(1) + diff(cfg.z_limits) * rand(num_scatterers, 1);
amplitudes = randn(num_scatterers, 1);

dx = x - cyst_center(1);
dz = z - cyst_center(2);
inside_cyst = (dx.^2 + dz.^2) <= cyst_radius^2;
amplitudes(inside_cyst) = cyst_contrast * amplitudes(inside_cyst);

phantom.positions = [x, y, z];
phantom.amplitudes = amplitudes;
phantom.type = 'cyst';

phantom.metadata = struct();
phantom.metadata.seed = seed;
phantom.metadata.num_scatterers = num_scatterers;
phantom.metadata.x_limits = cfg.x_limits;
phantom.metadata.y_limits = cfg.y_limits;
phantom.metadata.z_limits = cfg.z_limits;
phantom.metadata.coordinate_order = 'x_y_z';
phantom.metadata.positions_dimensions = '[N x 3]';
phantom.metadata.amplitudes_dimensions = '[N x 1]';
phantom.metadata.units = 'SI: meters, seconds, Hz';
phantom.metadata.cyst_center = cyst_center;
phantom.metadata.cyst_radius = cyst_radius;
phantom.metadata.cyst_contrast = cyst_contrast;
phantom.metadata.num_scatterers_inside_cyst = nnz(inside_cyst);
end
