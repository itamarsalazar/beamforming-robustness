%INSPECT_TOY_SAMPLE Quick inspection for one toy Field II sample.
%   Loads one generated .mat file, prints basic metadata, and saves simple
%   RF and phantom-position figures. This is not DAS.

clear;
clc;

script_dir = fileparts(mfilename('fullpath'));
repo_root = normalize_path(fullfile(script_dir, '..', '..'));

sample_file = fullfile(repo_root, 'data', 'simulated', 'toy_fieldii', ...
    'sample_005_cyst_small.mat');
output_dir = fullfile(repo_root, 'data', 'simulated', 'toy_fieldii', ...
    'inspection');

if ~exist(sample_file, 'file')
    error('inspect_toy_sample:MissingSample', ...
        'Sample file not found: %s. Run simulate_toy_dataset first.', sample_file);
end

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

loaded = load(sample_file);
sample = loaded.sample;

validate_sample(sample);

rf_size = size(sample.channel_data.rf);
fprintf('Sample: %s\n', sample.name);
fprintf('Phantom type: %s\n', sample.phantom.type);
fprintf('Scatterers: %d\n', size(sample.phantom.positions, 1));
fprintf('RF size: [%d x %d] = [num_samples x num_elements]\n', ...
    rf_size(1), rf_size(2));
fprintf('t0: %.6g s\n', sample.channel_data.t0);
fprintf('Contains DAS: %d\n', sample.metadata.contains_das);

rf_png = fullfile(output_dir, sprintf('%s_rf.png', sample.name));
phantom_png = fullfile(output_dir, sprintf('%s_phantom_xz.png', sample.name));

fig = figure('Visible', 'off');
imagesc(sample.channel_data.rf);
colormap(gray);
colorbar;
xlabel('Element index');
ylabel('Sample index');
title(sprintf('RF channel data: %s', sample.name), 'Interpreter', 'none');
saveas(fig, rf_png);
close(fig);

positions = sample.phantom.positions;
fig = figure('Visible', 'off');
scatter(positions(:, 1) * 1e3, positions(:, 3) * 1e3, ...
    8, sample.phantom.amplitudes, 'filled');
set(gca, 'YDir', 'reverse');
axis equal;
grid on;
colorbar;
xlabel('x lateral [mm]');
ylabel('z depth [mm]');
title(sprintf('Phantom scatterers: %s', sample.name), 'Interpreter', 'none');
saveas(fig, phantom_png);
close(fig);

fprintf('Saved RF image: %s\n', rf_png);
fprintf('Saved phantom image: %s\n', phantom_png);

function validate_sample(sample)
if ~isfield(sample, 'phantom') || ~isfield(sample, 'channel_data') || ...
        ~isfield(sample, 'cfg') || ~isfield(sample, 'metadata')
    error('inspect_toy_sample:InvalidSample', ...
        'sample must contain phantom, channel_data, cfg, and metadata.');
end

if ~isfield(sample.channel_data, 'rf') || ~isfield(sample.channel_data, 't0')
    error('inspect_toy_sample:InvalidSample', ...
        'sample.channel_data must contain rf and t0.');
end

if size(sample.phantom.positions, 2) ~= 3
    error('inspect_toy_sample:InvalidSample', ...
        'sample.phantom.positions must be [N x 3].');
end

if size(sample.phantom.positions, 1) ~= size(sample.phantom.amplitudes, 1)
    error('inspect_toy_sample:InvalidSample', ...
        'sample.phantom.positions and amplitudes row counts differ.');
end

if any(isnan(sample.phantom.positions(:))) || ...
        any(isnan(sample.phantom.amplitudes(:))) || ...
        any(isnan(sample.channel_data.rf(:)))
    error('inspect_toy_sample:InvalidSample', ...
        'sample contains NaNs in phantom or RF data.');
end
end

function normalized_path = normalize_path(path_value)
current_dir = pwd;
cd(path_value);
normalized_path = pwd;
cd(current_dir);
end
