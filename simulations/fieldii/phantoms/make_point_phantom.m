function phantom = make_point_phantom(cfg)
%MAKE_POINT_PHANTOM Create a small set of bright point scatterers.
%   Points are placed at known lateral/depth locations for later sanity
%   checks of beamforming and image geometry.

x_mid = mean(cfg.x_limits);
z_min = cfg.z_limits(1);

positions_xz = [
    x_mid,      z_min + 0.20 * diff(cfg.z_limits);  % superficial center
    x_mid,      z_min + 0.50 * diff(cfg.z_limits);  % mid center
    x_mid,      z_min + 0.80 * diff(cfg.z_limits);  % deep center
    -6e-3,      z_min + 0.50 * diff(cfg.z_limits);  % lateral left
     6e-3,      z_min + 0.50 * diff(cfg.z_limits)   % lateral right
];

num_points = size(positions_xz, 1);
y = zeros(num_points, 1);

phantom.positions = [positions_xz(:, 1), y, positions_xz(:, 2)];
phantom.amplitudes = [10; 10; 10; 8; 8];
phantom.type = 'points';

phantom.metadata = struct();
phantom.metadata.seed = 'deterministic';
phantom.metadata.num_scatterers = num_points;
phantom.metadata.x_limits = cfg.x_limits;
phantom.metadata.y_limits = cfg.y_limits;
phantom.metadata.z_limits = cfg.z_limits;
phantom.metadata.coordinate_order = 'x_y_z';
phantom.metadata.positions_dimensions = '[N x 3]';
phantom.metadata.amplitudes_dimensions = '[N x 1]';
phantom.metadata.units = 'SI: meters, seconds, Hz';
phantom.metadata.description = 'Bright point scatterers at known positions';
end
