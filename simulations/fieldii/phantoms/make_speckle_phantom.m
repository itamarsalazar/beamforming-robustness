function phantom = make_speckle_phantom(cfg, seed)
%MAKE_SPECKLE_PHANTOM Create a homogeneous random speckle phantom.
%   Coordinates follow [x, y, z] = [lateral, elevation, depth].

if nargin < 2 || isempty(seed)
    seed = 1;
end

rng(seed, 'twister');

num_scatterers = cfg.num_scatterers_speckle;

x = cfg.x_limits(1) + diff(cfg.x_limits) * rand(num_scatterers, 1);
y = zeros(num_scatterers, 1);
z = cfg.z_limits(1) + diff(cfg.z_limits) * rand(num_scatterers, 1);

phantom.positions = [x, y, z];
phantom.amplitudes = randn(num_scatterers, 1);
phantom.type = 'speckle';

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
end
