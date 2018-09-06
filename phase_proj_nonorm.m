function xp = phase_proj_nonorm(x, mags)

xh = fft2(x);

inds_big = find(abs(xh) > sqrt(eps) & mags > sqrt(eps));
xh(inds_big) = xh(inds_big).*mags(inds_big)./abs(xh(inds_big));

inds_small = setdiff(1:length(xh(:)), [inds_big; 1]);
xh(inds_small) = mags(inds_small);

xp = ifft2(xh);
end