function cfg = config_toy()
%CONFIG_TOY Configuration for the initial toy Field II dataset.
%   Units are SI: meters, seconds, and Hz.

cfg.project_name = 'beamforming-robustness';
cfg.dataset_name = 'toy_fieldii';

cfg.c = 1540;              % speed of sound [m/s]
cfg.fc = 5e6;              % center frequency [Hz]
cfg.fs = 40e6;             % sampling frequency [Hz]
cfg.lambda = cfg.c / cfg.fc;

cfg.num_elements = 128;
cfg.pitch = 0.3e-3;
cfg.element_width = 0.27e-3;
cfg.kerf = cfg.pitch - cfg.element_width;
cfg.element_height = 5e-3;

cfg.tx_angle_deg = 0;
cfg.tx_angle_rad = 0;

cfg.x_limits = [-12e-3, 12e-3];  % lateral [m]
cfg.y_limits = [0, 0];           % elevation [m]
cfg.z_limits = [5e-3, 45e-3];    % depth [m]

cfg.num_scatterers_speckle = 8000;
cfg.num_scatterers_cyst = 12000;

cfg.fieldii_path = '';      % set by simulate_toy_dataset from FIELDII_PATH or repo-relative path
cfg.fieldii_num_sub_x = 1;
cfg.fieldii_num_sub_y = 1;
cfg.fieldii_far_focus_z = 1.0;
cfg.excitation_cycles = 2;
cfg.impulse_cycles = 2;

cfg.output_dir = fullfile('data', 'simulated', 'toy_fieldii');
end
