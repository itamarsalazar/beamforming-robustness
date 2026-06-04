%SIMULATE_TOY_DATASET Generate the toy Field II ultrasound dataset.
%   Run from simulations/fieldii. Units are SI: meters, seconds, and Hz.
%   This script generates raw RF channel data only; it does not run DAS.

clear;
clc;

script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, 'configs'));
addpath(fullfile(script_dir, 'phantoms'));
addpath(fullfile(script_dir, 'utils'));

cfg = config_toy();
repo_root = normalize_path(fullfile(script_dir, '..', '..'));
output_dir = fullfile(repo_root, cfg.output_dir);

fieldii_env_path = getenv('FIELDII_PATH');
if ~isempty(fieldii_env_path)
    cfg.fieldii_path = fieldii_env_path;
else
    cfg.fieldii_path = fullfile(repo_root, '..', '..', 'Field_II_ver_3_30_linux');
end
if exist(cfg.fieldii_path, 'dir')
    cfg.fieldii_path = normalize_path(cfg.fieldii_path);
end

validate_cfg(cfg);

samples = {};
samples{end + 1} = make_sample(1, 'speckle', make_speckle_phantom(cfg, 101), cfg);
samples{end + 1} = make_sample(2, 'points', make_point_phantom(cfg), cfg);
samples{end + 1} = make_sample(3, 'cyst_center', make_cyst_phantom(cfg, [0, 25e-3], 4e-3, 0, 201), cfg);
samples{end + 1} = make_sample(4, 'cyst_lateral', make_cyst_phantom(cfg, [5e-3, 28e-3], 4e-3, 0.1, 202), cfg);
samples{end + 1} = make_sample(5, 'cyst_small', make_cyst_phantom(cfg, [-4e-3, 20e-3], 2e-3, 0, 203), cfg);

fprintf('Generating toy dataset: %s\n', cfg.dataset_name);
fprintf('Output directory: %s\n', output_dir);
fprintf('Field II path: %s\n\n', cfg.fieldii_path);

for idx = 1:numel(samples)
    sample = samples{idx};
    validate_phantom(sample.phantom);

    fprintf('[%d/%d] Simulating %s (%d scatterers)...\n', ...
        idx, numel(samples), sample.name, size(sample.phantom.positions, 1));

    channel_data = simulate_fieldii_pw(sample.phantom, cfg);
    sample.channel_data = channel_data;
    sample.metadata.simulator = 'Field II';
    sample.metadata.transmit_type = 'plane_wave';
    sample.metadata.tx_angle_deg = cfg.tx_angle_deg;
    sample.metadata.contains_channel_data = true;
    sample.metadata.contains_das = false;
    sample.metadata.rf_dimensions = '[num_samples x num_elements]';

    sample_path = save_sample(sample, output_dir, sample.id);
    rf_size = size(channel_data.rf);
    fprintf('Saved %s | RF [%d x %d] | t0 %.6g s\n\n', ...
        sample_path, rf_size(1), rf_size(2), channel_data.t0);
end

fprintf('Generated %d Field II RF samples.\n', numel(samples));

function sample = make_sample(id, name, phantom, cfg)
%MAKE_SAMPLE Build the standard sample structure used by this toy dataset.
sample = struct();
sample.id = id;
sample.name = name;
sample.phantom = phantom;
sample.cfg = cfg;
sample.metadata = struct();
sample.metadata.created_by = 'simulate_toy_dataset';
sample.metadata.dataset_name = cfg.dataset_name;
sample.metadata.sample_name = name;
sample.metadata.units = 'SI: meters, seconds, Hz';
sample.metadata.coordinate_order = 'x_y_z';
sample.metadata.phantom_positions_dimensions = '[N x 3]';
sample.metadata.phantom_amplitudes_dimensions = '[N x 1]';
sample.metadata.contains_channel_data = false;
sample.metadata.contains_das = false;
sample.metadata.notes = 'Raw RF channel data generated with Field II; DAS is not included yet.';
end

function validate_cfg(cfg)
required_positive = {'fs', 'fc', 'c'};
for idx = 1:numel(required_positive)
    field_name = required_positive{idx};
    if ~isfield(cfg, field_name) || ~isscalar(cfg.(field_name)) || ...
            cfg.(field_name) <= 0 || isnan(cfg.(field_name))
        error('simulate_toy_dataset:InvalidConfig', ...
            'cfg.%s must be a positive scalar.', field_name);
    end
end
end

function validate_phantom(phantom)
if size(phantom.positions, 2) ~= 3
    error('simulate_toy_dataset:InvalidPhantom', ...
        'phantom.positions must be [N x 3] with columns [x, y, z].');
end

if size(phantom.positions, 1) ~= size(phantom.amplitudes, 1)
    error('simulate_toy_dataset:InvalidPhantom', ...
        'phantom.positions and phantom.amplitudes must have the same number of rows.');
end

if any(isnan(phantom.positions(:))) || any(isnan(phantom.amplitudes(:)))
    error('simulate_toy_dataset:InvalidPhantom', ...
        'phantom.positions and phantom.amplitudes must not contain NaNs.');
end
end

function normalized_path = normalize_path(path_value)
current_dir = pwd;
cd(path_value);
normalized_path = pwd;
cd(current_dir);
end
