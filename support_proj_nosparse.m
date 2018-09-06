function xs = support_proj_nosparse(x, m, n)
xs = zeros(m);
xs(1:n, 1:n) = x(1:n, 1:n);
end