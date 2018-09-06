clear all

n = 100;
m = 150;

A = randn(m, n);
x_t = randn(n, 1);
y_t = A*x_t;
y0 = abs(y_t);

CSP = A*pinv(A);
P1 = @(y) CSP*y;
P2 = @(y) y0.*y./abs(y);
P1_ker = @(y) y - P1(y);
P2_ker = @(y) y - P2(y);

f = @(y) norm(y - P1(P2(y)))^2 - (norm(P2_ker(y))^2 + norm(P1_ker(y))^2)/2;
df = @(y) y - P1(P2(y)) - P1_ker(P2_ker(y));

% addpath(genpath('../manopt')) 
% problem.M = euclideanfactory(m^2, 1);
% problem.cost  = @(y) f(y);
% problem.egrad = @(y) df(y);
% checkgradient(problem);
% [y_manopt, manopt_cost] = trustregions(problem);

err_func = @(x) min(norm(x_t-x), norm(x_t+x))/norm(x_t);
% err_func = @(x) norm(x_t - exp(1i*angle(dot(x,x_t)))*x)/norm(x_t);

y_init = randn(m, 1);
y = y_init;
errs = err_func(A\y);
cost_vals = f(y);

err_trunc = 1e-3
err_trunc_GS = 1e-10
eta_arr = []; 

rel_change = Inf; ii = 0;
alpha = 0.05, beta = 0.95, eta_0 = 1, eta_trunc = 0.5
while errs(end) >= err_trunc
    y_new = @(eta) y - eta*df(y);
    eta = eta_0;
    ndf = norm(df(y))^2; func_vals = f(y);
    while abs(f(y_new(eta))) > abs(func_vals) - alpha*eta*ndf && eta > eta_trunc
        eta = beta*eta;
    end
    eta_arr(end+1) = eta;
    if eta <= eta_trunc, eta = eta_trunc; end
    y_new = y_new(eta);
    rel_change = norm(y_new - y)/norm(y);
    y = y_new;
    errs(end+1) = err_func(A\y);
    cost_vals(end+1) = f(y);
    ii = ii + 1;
end

jj=0; rel_change = inf;
while rel_change >= err_trunc_GS
    y_new = P1(P2(y));
    rel_change = norm(y_new - y)/norm(y);
    y = y_new;
    errs(end+1) = err_func(A\y);
    jj = jj + 1;
end

y_optim = y;
iters_optim = ii
iters_GS_optim = jj
figure, semilogy(errs)

y = y_init;
errs = err_func(A\y);
cost_vals = f(y);
rel_change = Inf; eta = 0.5
while errs(end) >= err_trunc
    y_new = y + eta*(2*P1(P2(y)) - P1(y) - P2(y));
    rel_change = norm(y - y_new)/norm(y);
    y = y_new;
    errs(end+1) = err_func(A\y);
    cost_vals(end+1) = f(y);
end
iters_RRR = length(errs) - 1
y_RRR = y;

jj=0; rel_change = inf;
while rel_change >= err_trunc_GS
    y_new = P1(P2(y));
    rel_change = norm(y_new - y)/norm(y);
    y = y_new;
    errs(end+1) = err_func(A\y);
    jj = jj + 1;
end

iters_GS_RRR = length(errs) - iters_RRR

hold on, semilogy(errs)
