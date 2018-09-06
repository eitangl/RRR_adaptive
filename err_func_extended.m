function err = err_func_extended(x, x_t, n)
x1 = zeros(size(x)); x2 = zeros(size(x));
x1(1:n, 1:n) = x(1:n, 1:n);
x2(1:n, 1:n) = rot90(x(1:n, 1:n), 2);
err_func_norots = @(x) norm(x_t - exp(1i*angle(dot(x(:),x_t(:))))*x,'fro')/norm(x_t,'fro');
err = min(err_func_norots(x1), err_func_norots(x2));
end