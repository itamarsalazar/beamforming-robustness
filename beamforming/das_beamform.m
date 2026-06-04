function das = das_beamform(channel_data, cfg, grid)
%DAS_BEAMFORM Minimal delay-and-sum beamformer for plane-wave RF/IQ data.
%   Uses SI units internally: meters, seconds, and Hz.
%   channel_data.rf is interpreted as [num_time_samples x num_elements].
%   grid.x_axis and grid.z_axis are in meters. Output images are [Nz x Nx].
%
%   Aperture modes:
%     cfg.das.aperture_mode = 'full'    uses all receive elements.
%     cfg.das.aperture_mode = 'dynamic' uses elements satisfying
%       abs(x_element - x_pixel) <= aperture_size/2,
%       where aperture_size = z_pixel / cfg.das.f_number.
%
%   IQ target:
%     If channel_data.channel_iq_raw exists, it is used as [T x C] complex.
%     Otherwise it is derived from RF with rf_to_iq(rf, fs, fc, t0).
%     The IQ DAS uses the same delays, grid, aperture and apodization as RF.

channel_data = validate_channel_data(channel_data);
validate_cfg(cfg);
validate_grid(grid);

rf = channel_data.rf;
fs = channel_data.fs;
fc = get_center_frequency(channel_data, cfg);
c = channel_data.c;
t0 = channel_data.t0;
theta = get_tx_angle(channel_data, cfg);

num_time_samples = size(rf, 1);
num_elements = size(rf, 2);
t_axis = t0 + (0:num_time_samples - 1)' / fs;
channel_iq_raw = get_channel_iq_raw(channel_data, rf, fs, fc, t0, num_time_samples, num_elements);

x_axis = grid.x_axis(:).';
z_axis = grid.z_axis(:);
Nx = numel(x_axis);
Nz = numel(z_axis);

element_x = channel_data.element_positions(:, 1).';
das_cfg = get_das_config(cfg);

rf_image = zeros(Nz, Nx);
das_iq_raw = complex(zeros(Nz, Nx));

for iz = 1:Nz
    z = z_axis(iz);
    line_sum = zeros(1, Nx);
    iq_line_sum = complex(zeros(1, Nx));

    % Delay model for a plane wave and point receive element:
    %   tx_time = (x*sin(theta) + z*cos(theta)) / c
    %   rx_time = sqrt((x - x_element)^2 + z^2) / c
    %   total_time = tx_time + rx_time
    tx_time = (x_axis * sin(theta) + z * cos(theta)) / c;

    % Weights are [num_elements x Nx]. In dynamic mode they depend on x,z.
    aperture_weights = get_aperture_weights(element_x, x_axis, z, das_cfg);

    for elem = 1:num_elements
        rx_time = sqrt((x_axis - element_x(elem)).^2 + z.^2) / c;
        total_time = tx_time + rx_time;

        values = interp1(t_axis, rf(:, elem), total_time, 'linear', 0);
        line_sum = line_sum + aperture_weights(elem, :) .* values;

        iq_values = interp1(t_axis, channel_iq_raw(:, elem), total_time, 'linear', 0);
        iq_values = iq_values .* exp(1j * 2 * pi * fc * total_time);
        iq_line_sum = iq_line_sum + aperture_weights(elem, :) .* iq_values;
    end

    rf_image(iz, :) = line_sum;
    das_iq_raw(iz, :) = iq_line_sum;
end

% Envelope is computed along depth for each lateral line.
envelope = abs(hilbert(rf_image));
bmode_db = log_compress(envelope, das_cfg.dynamic_range_db);

envelope_iq = abs(das_iq_raw);
bmode_iq_db = log_compress(envelope_iq, das_cfg.dynamic_range_db);

das = struct();
das.rf_image = rf_image;          % [Nz x Nx]
das.envelope = envelope;          % [Nz x Nx]
das.bmode_db = bmode_db;          % [Nz x Nx], clipped dB
das.das_iq_raw = das_iq_raw;      % [Nz x Nx] complex
das.envelope_iq = envelope_iq;    % [Nz x Nx]
das.bmode_iq_db = bmode_iq_db;    % [Nz x Nx], clipped dB
das.x_axis = x_axis;              % [1 x Nx], lateral [m]
das.z_axis = z_axis;              % [Nz x 1], depth [m]
das.grid = grid;
das.metadata = struct();
das.metadata.method = 'DAS';
das.metadata.transmit_type = 'plane_wave';
das.metadata.tx_type = 'plane_wave';
das.metadata.tx_angle_deg = get_tx_angle_deg(channel_data, theta);
das.metadata.aperture_mode = das_cfg.aperture_mode;
das.metadata.f_number = das_cfg.f_number;
das.metadata.aperture_formula = 'aperture_size = z / f_number';
das.metadata.rf_input_dimensions = '[num_time_samples x num_elements]';
das.metadata.image_dimensions = '[Nz x Nx]';
das.metadata.iq_dimensions = '[Nz x Nx] complex';
das.metadata.channel_iq_dimensions = '[num_time_samples x num_elements] complex';
das.metadata.apodization = das_cfg.apodization;
das.metadata.dynamic_range_db = das_cfg.dynamic_range_db;
das.metadata.units = 'SI: meters, seconds, Hz';
das.metadata.contains_das = true;
das.metadata.contains_iq = true;
das.metadata.contains_deep_learning = false;
das.metadata.created_by = 'beamforming-robustness';
end

function validate_cfg(cfg)
required_positive = {'fs', 'fc', 'c'};
for idx = 1:numel(required_positive)
    field_name = required_positive{idx};
    if ~isfield(cfg, field_name) || ~isnumeric(cfg.(field_name)) || ...
            ~isscalar(cfg.(field_name)) || cfg.(field_name) <= 0 || isnan(cfg.(field_name))
        error('das_beamform:InvalidConfig', ...
            'cfg.%s must be a positive scalar.', field_name);
    end
end
end

function validate_grid(grid)
if ~isfield(grid, 'x_axis') || ~isfield(grid, 'z_axis')
    error('das_beamform:InvalidGrid', ...
        'grid must contain x_axis and z_axis.');
end

if any(isnan(grid.x_axis(:))) || any(isnan(grid.z_axis(:)))
    error('das_beamform:InvalidGrid', ...
        'grid axes must not contain NaNs.');
end

if numel(grid.x_axis) < 2 || numel(grid.z_axis) < 2
    error('das_beamform:InvalidGrid', ...
        'grid.x_axis and grid.z_axis must contain at least two points.');
end
end

function fc = get_center_frequency(channel_data, cfg)
if isfield(channel_data, 'fc') && ~isempty(channel_data.fc)
    fc = channel_data.fc;
else
    fc = cfg.fc;
end

if ~isnumeric(fc) || ~isscalar(fc) || fc <= 0 || isnan(fc)
    error('das_beamform:InvalidCenterFrequency', ...
        'Center frequency must be a positive scalar [Hz].');
end
end

function channel_iq_raw = get_channel_iq_raw(channel_data, rf, fs, fc, t0, num_time_samples, num_elements)
if isfield(channel_data, 'channel_iq_raw') && ~isempty(channel_data.channel_iq_raw)
    channel_iq_raw = channel_data.channel_iq_raw;
else
    channel_iq_raw = rf_to_iq(rf, fs, fc, t0);
end

if ~isnumeric(channel_iq_raw) || ...
        ~isequal(size(channel_iq_raw), [num_time_samples, num_elements])
    error('das_beamform:InvalidIQ', ...
        'channel_data.channel_iq_raw must be complex/numeric [num_time_samples x num_elements].');
end

if isreal(channel_iq_raw)
    error('das_beamform:InvalidIQ', ...
        'channel_data.channel_iq_raw must be complex [num_time_samples x num_elements].');
end

if any(isnan(channel_iq_raw(:)))
    error('das_beamform:InvalidIQ', ...
        'channel_data.channel_iq_raw must not contain NaNs.');
end
end

function theta = get_tx_angle(channel_data, cfg)
if isfield(channel_data, 'tx_angle_rad')
    theta = channel_data.tx_angle_rad;
elseif isfield(cfg, 'tx_angle_rad')
    theta = cfg.tx_angle_rad;
elseif isfield(channel_data, 'tx_angle_deg')
    theta = channel_data.tx_angle_deg * pi / 180;
else
    theta = 0;
end

if abs(theta) > 1e-12
    error('das_beamform:UnsupportedAngle', ...
        'This minimal DAS currently supports only tx_angle_deg = 0.');
end
end

function tx_angle_deg = get_tx_angle_deg(channel_data, theta)
if isfield(channel_data, 'tx_angle_deg')
    tx_angle_deg = channel_data.tx_angle_deg;
else
    tx_angle_deg = theta * 180 / pi;
end
end

function das_cfg = get_das_config(cfg)
das_cfg = struct();
das_cfg.aperture_mode = 'full';
das_cfg.f_number = 1.5;
das_cfg.apodization = 'rectangular';
das_cfg.dynamic_range_db = 60;

if isfield(cfg, 'das')
    user_fields = fieldnames(cfg.das);
    for idx = 1:numel(user_fields)
        das_cfg.(user_fields{idx}) = cfg.das.(user_fields{idx});
    end
end

das_cfg.aperture_mode = lower(char(das_cfg.aperture_mode));
das_cfg.apodization = lower(char(das_cfg.apodization));

if ~ismember(das_cfg.aperture_mode, {'full', 'dynamic'})
    error('das_beamform:InvalidApertureMode', ...
        'Unsupported cfg.das.aperture_mode: %s.', das_cfg.aperture_mode);
end

if ~ismember(das_cfg.apodization, {'rect', 'rectangular', 'none', 'hann', 'hanning'})
    error('das_beamform:InvalidApodization', ...
        'Unsupported cfg.das.apodization: %s.', das_cfg.apodization);
end

if ~isnumeric(das_cfg.f_number) || ~isscalar(das_cfg.f_number) || ...
        das_cfg.f_number <= 0 || isnan(das_cfg.f_number)
    error('das_beamform:InvalidFNumber', ...
        'cfg.das.f_number must be a positive scalar.');
end
end

function weights = get_aperture_weights(element_x, x_axis, z, das_cfg)
num_elements = numel(element_x);
Nx = numel(x_axis);
weights = zeros(num_elements, Nx);

for ix = 1:Nx
    switch das_cfg.aperture_mode
        case 'full'
            active = true(1, num_elements);
        case 'dynamic'
            aperture_size = z / das_cfg.f_number;
            active = abs(element_x - x_axis(ix)) <= aperture_size / 2;
    end

    active_idx = find(active);
    if isempty(active_idx)
        [~, nearest_idx] = min(abs(element_x - x_axis(ix)));
        active_idx = nearest_idx;
    end

    weights(active_idx, ix) = make_apodization_weights(numel(active_idx), das_cfg.apodization);
end
end

function weights = make_apodization_weights(num_active, apodization)
switch apodization
    case {'rect', 'rectangular', 'none'}
        weights = ones(num_active, 1);
    case {'hann', 'hanning'}
        if num_active == 1
            weights = 1;
        else
            weights = hanning(num_active);
        end
end

weight_sum = sum(weights);
if weight_sum > 0
    weights = weights ./ weight_sum * num_active;
end
end
