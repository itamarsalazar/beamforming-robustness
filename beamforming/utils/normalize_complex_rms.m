function [z_norm, scale, raw_scale, used_fallback] = normalize_complex_rms(z)
%NORMALIZE_COMPLEX_RMS Per-sample RMS normalization for complex arrays.
%   Works for channel IQ [T x C] and beamformed IQ [H x W].
%   Uses the actual RMS even when it is below eps. Only falls back for
%   degenerate scales: zero, NaN, or Inf.

if ~isnumeric(z)
    error('normalize_complex_rms:InvalidInput', ...
        'Input z must be a numeric array.');
end

if any(isnan(z(:)))
    error('normalize_complex_rms:InvalidInput', ...
        'Input z must not contain NaNs.');
end

raw_scale = sqrt(mean(abs(z(:)).^2));
used_fallback = false;

if isfinite(raw_scale) && raw_scale > 0
    scale = raw_scale;
    z_norm = z / scale;
else
    warning('normalize_complex_rms:DegenerateScale', ...
        'Degenerate RMS scale detected. Returning z unchanged with scale = 1.');
    scale = 1;
    z_norm = z;
    used_fallback = true;
end
end
