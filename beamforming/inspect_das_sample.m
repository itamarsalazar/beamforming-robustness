%INSPECT_DAS_SAMPLE Inspect one beamformed toy Field II DAS sample.
%   If sample_file is not defined before running, the first *_das.mat file
%   in data/beamformed/toy_fieldii is loaded. Figures are saved as PNG.
%   Ejemplo: sample_file = 'Q:\isalazar\codes\beamforming-robustness\data\beamformed\toy_fieldii\sample_004_cyst_lateral_das.mat'


clearvars -except sample_file;
clc;

script_dir = fileparts(mfilename('fullpath'));
repo_root = normalize_path(fullfile(script_dir, '..'));

input_dir = fullfile(repo_root, 'data', 'beamformed', 'toy_fieldii');
figure_dir = fullfile(repo_root, 'outputs', 'figures', 'toy_fieldii_das');

if ~exist(input_dir, 'dir')
    error('inspect_das_sample:MissingInput', ...
        'Input directory not found: %s. Run beamform_toy_dataset first.', input_dir);
end

if ~exist('sample_file', 'var') || isempty(sample_file)
    files = dir(fullfile(input_dir, '*_das.mat'));
    if isempty(files)
        error('inspect_das_sample:MissingInput', ...
            'No *_das.mat files found in %s.', input_dir);
    end
    sample_file = fullfile(files(1).folder, files(1).name);
elseif isfolder(sample_file)
    files = dir(fullfile(sample_file, '*_das.mat'));
    if isempty(files)
        error('inspect_das_sample:MissingInput', ...
            'No *_das.mat files found in %s.', sample_file);
    end
    sample_file = fullfile(files(1).folder, files(1).name);
end

if ~exist(sample_file, 'file')
    candidate = fullfile(input_dir, sample_file);
    if exist(candidate, 'file')
        sample_file = candidate;
    else
        error('inspect_das_sample:MissingInput', ...
            'Sample file not found: %s.', sample_file);
    end
end

if ~exist(figure_dir, 'dir')
    mkdir(figure_dir);
end

loaded = load(sample_file);
if ~isfield(loaded, 'sample')
    error('inspect_das_sample:InvalidSample', ...
        'File does not contain variable sample: %s', sample_file);
end

sample = loaded.sample;
validate_sample(sample);

rf_size = size(sample.channel_data.rf);
bmode_size = size(sample.das.bmode_db);
x_mm = sample.das.x_axis * 1e3;
z_mm = sample.das.z_axis * 1e3;

fprintf('Sample: %s\n', sample.name);
fprintf('Phantom type: %s\n', sample.phantom.type);
fprintf('Channel RF size: [%d x %d]\n', rf_size(1), rf_size(2));
fprintf('DAS B-mode size: [%d x %d]\n', bmode_size(1), bmode_size(2));
fprintf('Lateral range: %.2f to %.2f mm\n', min(x_mm), max(x_mm));
fprintf('Depth range: %.2f to %.2f mm\n', min(z_mm), max(z_mm));

[~, base_name] = fileparts(sample_file);
base_name = regexprep(base_name, '_das$', '');

bmode_png = fullfile(figure_dir, sprintf('%s_bmode.png', base_name));
scatterers_png = fullfile(figure_dir, sprintf('%s_scatterers.png', base_name));
rf_png = fullfile(figure_dir, sprintf('%s_rf.png', base_name));

dynamic_range_db = get_dynamic_range(sample.das);

fig = figure('Visible', 'off');
imagesc(x_mm, z_mm, sample.das.bmode_db);
axis image;
set(gca, 'YDir', 'reverse');
colormap(gray);
colorbar;
caxis([-dynamic_range_db, 0]);
xlabel('x lateral [mm]');
ylabel('z depth [mm]');
title(sprintf('DAS B-mode: %s', sample.name), 'Interpreter', 'none');
saveas(fig, bmode_png);
close(fig);

positions = sample.phantom.positions;
fig = figure('Visible', 'off');
scatter(positions(:, 1) * 1e3, positions(:, 3) * 1e3, ...
    8, sample.phantom.amplitudes, 'filled');
axis image;
set(gca, 'YDir', 'reverse');
grid on;
colorbar;
xlabel('x lateral [mm]');
ylabel('z depth [mm]');
title(sprintf('Scatterers: %s', sample.name), 'Interpreter', 'none');
saveas(fig, scatterers_png);
close(fig);

fig = figure('Visible', 'off');
imagesc(sample.channel_data.rf);
colormap(gray);
colorbar;
xlabel('Element index');
ylabel('Sample index');
title(sprintf('Raw RF: %s', sample.name), 'Interpreter', 'none');
saveas(fig, rf_png);
close(fig);

fprintf('Saved B-mode figure: %s\n', bmode_png);
fprintf('Saved scatterer figure: %s\n', scatterers_png);
fprintf('Saved RF figure: %s\n', rf_png);

function validate_sample(sample)
required_top = {'phantom', 'channel_data', 'das', 'cfg', 'metadata'};
for idx = 1:numel(required_top)
    if ~isfield(sample, required_top{idx})
        error('inspect_das_sample:InvalidSample', ...
            'sample.%s is required.', required_top{idx});
    end
end

required_das = {'rf_image', 'envelope', 'bmode_db', 'x_axis', 'z_axis'};
for idx = 1:numel(required_das)
    if ~isfield(sample.das, required_das{idx})
        error('inspect_das_sample:InvalidSample', ...
            'sample.das.%s is required.', required_das{idx});
    end
end

if size(sample.phantom.positions, 2) ~= 3
    error('inspect_das_sample:InvalidSample', ...
        'sample.phantom.positions must be [N x 3].');
end

if any(isnan(sample.channel_data.rf(:))) || any(isnan(sample.das.bmode_db(:))) || ...
        any(isnan(sample.phantom.positions(:)))
    error('inspect_das_sample:InvalidSample', ...
        'sample contains NaNs in RF, B-mode, or phantom positions.');
end
end

function dynamic_range_db = get_dynamic_range(das)
dynamic_range_db = 60;
if isfield(das, 'metadata') && isfield(das.metadata, 'dynamic_range_db') && ...
        ~isempty(das.metadata.dynamic_range_db)
    dynamic_range_db = das.metadata.dynamic_range_db;
end
end

function normalized_path = normalize_path(path_value)
current_dir = pwd;
cd(path_value);
normalized_path = pwd;
cd(current_dir);
end
