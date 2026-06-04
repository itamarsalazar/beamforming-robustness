function channel_iq_raw = rf_to_iq(rf, fs, fc, t0)
%RF_TO_IQ Convert real RF channel data to raw complex baseband IQ.
%   rf is [T x C], where T is time samples and C is receive channels.
%   channel_iq_raw is complex [T x C] with the same dimensions.
%
%   The conversion uses the analytic signal along time:
%       rf_analytic = hilbert(rf);
%   and demodulates to baseband:
%       t_axis = t0 + (0:T-1)'/fs;
%       channel_iq_raw = rf_analytic .* exp(-1j*2*pi*fc*t_axis);

validate_inputs(rf, fs, fc, t0);

num_time_samples = size(rf, 1);
t_axis = t0 + (0:num_time_samples - 1)' / fs;

rf_analytic = hilbert(rf);
channel_iq_raw = rf_analytic .* exp(-1j * 2 * pi * fc * t_axis);
end

function validate_inputs(rf, fs, fc, t0)
if ~isnumeric(rf) || ndims(rf) ~= 2
    error('rf_to_iq:InvalidRF', 'rf must be a numeric 2-D matrix [T x C].');
end

if isempty(rf) || size(rf, 1) < 2 || size(rf, 2) < 1
    error('rf_to_iq:InvalidRF', 'rf must have size [T x C] with T >= 2 and C >= 1.');
end

if any(isnan(rf(:)))
    error('rf_to_iq:InvalidRF', 'rf must not contain NaNs.');
end

if ~is_positive_scalar(fs)
    error('rf_to_iq:InvalidSamplingFrequency', 'fs must be a positive scalar [Hz].');
end

if ~is_positive_scalar(fc)
    error('rf_to_iq:InvalidCenterFrequency', 'fc must be a positive scalar [Hz].');
end

if ~isnumeric(t0) || ~isscalar(t0) || isnan(t0)
    error('rf_to_iq:InvalidStartTime', 't0 must be a numeric scalar [s].');
end
end

function tf = is_positive_scalar(value)
tf = isnumeric(value) && isscalar(value) && value > 0 && ~isnan(value);
end
