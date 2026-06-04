%BEAMFORM_TOY_DATASET Run minimal DAS on the toy Field II dataset.
%   Input:  data/simulated/toy_fieldii/sample_*.mat
%   Output: data/beamformed/toy_fieldii/sample_*_das.mat

clear;
clc;

script_dir = fileparts(mfilename('fullpath'));
repo_root = normalize_path(fullfile(script_dir, '..'));

addpath(script_dir);
addpath(fullfile(script_dir, 'utils'));

input_dir = fullfile(repo_root, 'data', 'simulated', 'toy_fieldii');
output_dir = fullfile(repo_root, 'data', 'beamformed', 'toy_fieldii');

if ~exist(input_dir, 'dir')
    error('beamform_toy_dataset:MissingInput', ...
        'Input directory not found: %s. Run the Field II toy simulation first.', input_dir);
end

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

files = dir(fullfile(input_dir, 'sample_*.mat'));
if isempty(files)
    error('beamform_toy_dataset:MissingInput', ...
        'No sample_*.mat files found in %s.', input_dir);
end

fprintf('Beamforming %d toy samples from %s\n', numel(files), input_dir);
fprintf('Output directory: %s\n\n', output_dir);

for idx = 1:numel(files)
    input_path = fullfile(files(idx).folder, files(idx).name);
    loaded = load(input_path);

    if ~isfield(loaded, 'sample')
        error('beamform_toy_dataset:InvalidSample', ...
            'File does not contain variable sample: %s', input_path);
    end

    sample = loaded.sample;
    cfg = sample.cfg;
    cfg.das = get_das_config(cfg);
    grid = make_default_grid(cfg);

    sample.channel_data.channel_iq_raw = rf_to_iq(sample.channel_data.rf, ...
        sample.channel_data.fs, sample.channel_data.fc, sample.channel_data.t0);
     [sample.channel_data.channel_iq_norm, sx, sx_raw, sx_used_fallback] = normalize_complex_rms( ...
        sample.channel_data.channel_iq_raw);
    sample.channel_data.sx = sx;
    sample.channel_data.sx_raw = sx_raw;
    sample.channel_data.sx_used_fallback = sx_used_fallback;
    sample.channel_data.normalization_type = 'per_sample_complex_rms';
    sample.channel_data.channel_iq_raw_dimensions = '[num_time_samples x num_elements] complex';
    sample.channel_data.channel_iq_norm_dimensions = '[num_time_samples x num_elements] complex';

    fprintf('[%d/%d] DAS %s on grid [%d z x %d x], aperture=%s, apod=%s...\n', ...
        idx, numel(files), sample.name, numel(grid.z_axis), numel(grid.x_axis), ...
        cfg.das.aperture_mode, cfg.das.apodization);

    das = das_beamform(sample.channel_data, cfg, grid);
    [das.das_iq_norm, sy, sy_raw, sy_used_fallback] = normalize_complex_rms(das.das_iq_raw);
    das.sy = sy;
    das.sy_raw = sy_raw;
    das.sy_used_fallback = sy_used_fallback;
    das.normalization_type = 'per_sample_complex_rms';
    das.metadata.normalization_type = 'per_sample_complex_rms';
    das.metadata.sx = sx;
    das.metadata.sx_raw = sx_raw;
    das.metadata.sx_used_fallback = sx_used_fallback;
    das.metadata.used_fallback_x = sx_used_fallback;
    das.metadata.sy = sy;
    das.metadata.sy_raw = sy_raw;
    das.metadata.sy_used_fallback = sy_used_fallback;
    das.metadata.used_fallback_y = sy_used_fallback;
    das.metadata.channel_iq_norm_dimensions = '[num_time_samples x num_elements] complex';
    das.metadata.das_iq_norm_dimensions = '[Nz x Nx] complex';

    sample.das = das;
    sample.metadata.contains_das = true;
    sample.metadata.das_created = true;
    sample.metadata.das_method = 'DAS';
    sample.metadata.das_output_dimensions = '[Nz x Nx]';
    sample.metadata.das_aperture_mode = das.metadata.aperture_mode;
    sample.metadata.das_f_number = das.metadata.f_number;
    sample.metadata.das_apodization = das.metadata.apodization;
    sample.metadata.contains_iq = das.metadata.contains_iq;
    sample.metadata.das_iq_dimensions = das.metadata.iq_dimensions;
    sample.metadata.normalization_type = 'per_sample_complex_rms';
    sample.metadata.sx = sx;
    sample.metadata.sx_raw = sx_raw;
    sample.metadata.sx_used_fallback = sx_used_fallback;
    sample.metadata.used_fallback_x = sx_used_fallback;
    sample.metadata.sy = sy;
    sample.metadata.sy_raw = sy_raw;
    sample.metadata.sy_used_fallback = sy_used_fallback;
    sample.metadata.used_fallback_y = sy_used_fallback;
    sample.metadata.channel_iq_raw_dimensions = '[num_time_samples x num_elements] complex';
    sample.metadata.channel_iq_norm_dimensions = '[num_time_samples x num_elements] complex';
    sample.metadata.das_iq_raw_dimensions = '[Nz x Nx] complex';
    sample.metadata.das_iq_norm_dimensions = '[Nz x Nx] complex';

    output_name = regexprep(files(idx).name, '\.mat$', '_das.mat');
    output_path = fullfile(output_dir, output_name);
    save(output_path, 'sample');

    fprintf('Saved %s | B-mode [%d x %d] | sx %.3e | sy %.3e\n\n', ...
        output_path, size(das.bmode_db, 1), size(das.bmode_db, 2), sx, sy);
end

fprintf('Generated %d DAS samples.\n', numel(files));

function grid = make_default_grid(cfg)
%MAKE_DEFAULT_GRID Initial image grid. Reduce these counts if needed.
grid = struct();
grid.x_axis = linspace(cfg.x_limits(1), cfg.x_limits(2), 256);
grid.z_axis = linspace(cfg.z_limits(1), cfg.z_limits(2), 512);
grid.metadata = struct();
grid.metadata.x_axis_units = 'm';
grid.metadata.z_axis_units = 'm';
grid.metadata.dimensions = '[Nz x Nx]';
end

function das_cfg = get_das_config(cfg)
das_cfg = struct();
das_cfg.aperture_mode = 'dynamic';
das_cfg.f_number = 1.5;
das_cfg.apodization = 'hanning';
das_cfg.dynamic_range_db = 60;

if isfield(cfg, 'das')
    user_fields = fieldnames(cfg.das);
    for idx = 1:numel(user_fields)
        das_cfg.(user_fields{idx}) = cfg.das.(user_fields{idx});
    end
end
end

function normalized_path = normalize_path(path_value)
current_dir = pwd;
cd(path_value);
normalized_path = pwd;
cd(current_dir);
end
