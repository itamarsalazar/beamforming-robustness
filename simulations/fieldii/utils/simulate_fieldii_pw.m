function channel_data = simulate_fieldii_pw(phantom, cfg)
%SIMULATE_FIELDII_PW Simulate one normal-incidence plane-wave RF acquisition.
%   Inputs use SI units. phantom.positions is [N x 3] with columns [x, y, z]
%   in meters, and phantom.amplitudes is [N x 1]. channel_data.rf is
%   [num_samples x num_elements], one RF trace per receive element.

validate_cfg(cfg);
validate_phantom(phantom);
ensure_fieldii_available(cfg);

positions = phantom.positions;
amplitudes = phantom.amplitudes(:);

if cfg.tx_angle_deg ~= 0
    error('simulate_fieldii_pw:UnsupportedAngle', ...
        'This toy simulator currently supports only cfg.tx_angle_deg = 0.');
end

field_init(0);
set_sampling(cfg.fs);
set_field('c', cfg.c);

num_sub_x = get_cfg_value(cfg, 'fieldii_num_sub_x', 1);
num_sub_y = get_cfg_value(cfg, 'fieldii_num_sub_y', 1);
far_focus_z = get_cfg_value(cfg, 'fieldii_far_focus_z', 1.0);

tx = [];
rx = [];

% A far transmit focus approximates a normal plane wave for this toy setup.
tx_focus = [0, 0, far_focus_z];
rx_focus = [0, 0, far_focus_z];

tx = xdc_linear_array(cfg.num_elements, cfg.element_width, ...
    cfg.element_height, cfg.kerf, num_sub_x, num_sub_y, tx_focus);
rx = xdc_linear_array(cfg.num_elements, cfg.element_width, ...
    cfg.element_height, cfg.kerf, num_sub_x, num_sub_y, rx_focus);
aperture_cleanup = onCleanup(@() free_apertures(tx, rx));
field_cleanup = onCleanup(@() field_end());

xdc_center_focus(tx, [0, 0, 0]);
xdc_center_focus(rx, [0, 0, 0]);
xdc_focus(tx, 0, tx_focus);
xdc_focus(rx, 0, rx_focus);
xdc_apodization(tx, 0, ones(1, cfg.num_elements));
xdc_apodization(rx, 0, ones(1, cfg.num_elements));

excitation_cycles = get_cfg_value(cfg, 'excitation_cycles', 2);
impulse_cycles = get_cfg_value(cfg, 'impulse_cycles', 2);

t_exc = 0:1/cfg.fs:excitation_cycles/cfg.fc;
excitation = sin(2 * pi * cfg.fc * t_exc);

t_imp = 0:1/cfg.fs:impulse_cycles/cfg.fc;
impulse = sin(2 * pi * cfg.fc * t_imp) .* hanning(numel(t_imp))';

xdc_excitation(tx, excitation);
xdc_impulse(tx, impulse);
xdc_impulse(rx, impulse);

[rf, t0] = calc_scat_multi(tx, rx, positions, amplitudes);

if any(isnan(rf(:))) || isnan(t0)
    error('simulate_fieldii_pw:InvalidRF', 'Field II returned NaNs in rf or t0.');
end

element_positions = compute_element_positions(cfg);

channel_data = struct();
channel_data.rf = rf;  % [num_samples x num_elements]
channel_data.t0 = t0;  % start time of first RF sample [s]
channel_data.fs = cfg.fs;  % sampling frequency [Hz]
channel_data.fc = cfg.fc;  % center frequency [Hz]
channel_data.c = cfg.c;    % speed of sound [m/s]
channel_data.tx_angle_deg = cfg.tx_angle_deg;
channel_data.tx_angle_rad = cfg.tx_angle_rad;
channel_data.element_positions = element_positions;  % [num_elements x 3], [x,y,z] [m]
channel_data.metadata = struct();
channel_data.metadata.simulator = 'Field II';
channel_data.metadata.transmit_type = 'plane_wave';
channel_data.metadata.plane_wave_approximation = 'normal incidence approximated by far transmit focus';
channel_data.metadata.rf_dimensions = '[num_samples x num_elements]';
channel_data.metadata.element_position_dimensions = '[num_elements x 3]';
channel_data.metadata.num_scatterers = size(positions, 1);
channel_data.metadata.num_elements = cfg.num_elements;
channel_data.metadata.units = 'SI: meters, seconds, Hz';
channel_data.metadata.fieldii_path = get_cfg_value(cfg, 'fieldii_path', '');
channel_data.metadata.fieldii_num_sub_x = num_sub_x;
channel_data.metadata.fieldii_num_sub_y = num_sub_y;
channel_data.metadata.fieldii_far_focus_z = far_focus_z;
channel_data.metadata.excitation_cycles = excitation_cycles;
channel_data.metadata.impulse_cycles = impulse_cycles;
end

function validate_cfg(cfg)
required_positive = {'fs', 'fc', 'c', 'num_elements', 'element_width', ...
    'element_height', 'pitch'};

for idx = 1:numel(required_positive)
    field_name = required_positive{idx};
    if ~isfield(cfg, field_name) || ~isnumeric(cfg.(field_name)) || ...
            ~isscalar(cfg.(field_name)) || cfg.(field_name) <= 0 || isnan(cfg.(field_name))
        error('simulate_fieldii_pw:InvalidConfig', ...
            'cfg.%s must be a positive scalar.', field_name);
    end
end

if ~isfield(cfg, 'kerf') || ~isnumeric(cfg.kerf) || ~isscalar(cfg.kerf) || cfg.kerf < 0 || isnan(cfg.kerf)
    error('simulate_fieldii_pw:InvalidConfig', 'cfg.kerf must be a non-negative scalar.');
end
end

function validate_phantom(phantom)
if ~isfield(phantom, 'positions') || ~isfield(phantom, 'amplitudes')
    error('simulate_fieldii_pw:InvalidPhantom', ...
        'phantom must contain positions and amplitudes fields.');
end

positions = phantom.positions;
amplitudes = phantom.amplitudes;

if ~isnumeric(positions) || size(positions, 2) ~= 3
    error('simulate_fieldii_pw:InvalidPhantom', ...
        'phantom.positions must be numeric [N x 3] with columns [x, y, z].');
end

if ~isnumeric(amplitudes) || size(amplitudes, 2) ~= 1
    error('simulate_fieldii_pw:InvalidPhantom', ...
        'phantom.amplitudes must be numeric [N x 1].');
end

if size(positions, 1) ~= size(amplitudes, 1)
    error('simulate_fieldii_pw:InvalidPhantom', ...
        'phantom.positions and phantom.amplitudes must have the same number of rows.');
end

if any(isnan(positions(:))) || any(isnan(amplitudes(:)))
    error('simulate_fieldii_pw:InvalidPhantom', ...
        'phantom.positions and phantom.amplitudes must not contain NaNs.');
end
end

function ensure_fieldii_available(cfg)
if isfield(cfg, 'fieldii_path') && ~isempty(cfg.fieldii_path)
    addpath(cfg.fieldii_path);
end

if exist('field_init', 'file') ~= 2 || exist('calc_scat_multi', 'file') ~= 2
    error('simulate_fieldii_pw:FieldIINotFound', ...
        ['Field II is not available on the MATLAB path. ', ...
         'Add Field II to the path before running this simulation, ', ...
         'or set cfg.fieldii_path / FIELDII_PATH to the Field II directory.']);
end
end

function value = get_cfg_value(cfg, field_name, default_value)
if isfield(cfg, field_name) && ~isempty(cfg.(field_name))
    value = cfg.(field_name);
else
    value = default_value;
end
end

function element_positions = compute_element_positions(cfg)
element_index = (0:cfg.num_elements - 1)' - (cfg.num_elements - 1) / 2;
x = element_index * cfg.pitch;
element_positions = [x, zeros(cfg.num_elements, 1), zeros(cfg.num_elements, 1)];
end

function free_apertures(tx, rx)
if ~isempty(tx)
    xdc_free(tx);
end

if ~isempty(rx)
    xdc_free(rx);
end
end
