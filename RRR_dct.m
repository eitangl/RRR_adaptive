clear all

n = 64
m = 127

% I = imread('circuit.tif');
I = imread('cameraman.tif');
% I = imread('coins.png');
% I = imread('kids.tif');
I = I(77:77+63,105:105+63);
% I = double(imresize(I, [n, n]));
% I = double(I(35:35+63,95:95+63));
% I = double(imread('football.jpg'));
% I = double(imread('tape.png'));
% I = imresize(I(:,:,1), [64,64]);
x_t = zeros(m);
x_t(1:n, 1:n) = I;
y_t = dct2(x_t);
y0 = abs(y_t);

P1 = @(x) support_proj_nosparse(x, m, n);
P2 = @(x) idct2( y0.*sign( dct2(x) ) );
% P2 = @(x) phase_proj_nonorm(x, y0);
P1_ker = @(x) x - P1(x);
P2_ker = @(x) x - P2(x);

f = @(y) norm(y - P1(P2(y)), 'fro')^2 - norm(P2_ker(y), 'fro')^2/2 - norm(P1_ker(y), 'fro')^2/2;
df = @(y) y - P1(P2(y)) - P1_ker(P2_ker(y));

x_init = randn(m)
err_func = @(x) err_func_extended(x, x_t, n);
err_trunc = 1e-3
err_trunc_GS = 1e-10

x = x_init;
errs = err_func(x);
cost_vals = f(x);
eta_arr = []; 

rel_change = Inf; ii = 0;
alpha = 0.25, beta = 0.75, eta_0 = 1, eta_trunc = 0.5
while errs(end) >= err_trunc
    x_new = @(eta) x - eta*df(x);
    eta = eta_0;
    ndf = norm(df(x))^2; func_vals = f(x);
    while f(x_new(eta)) > func_vals - alpha*eta*ndf && eta > eta_trunc
        eta = beta*eta;
    end
    eta_arr(end+1) = eta;
    if eta <= eta_trunc, eta = eta_trunc; end
    x_new = x_new(eta);
    rel_change = norm(x_new - x)/norm(x);
    x = x_new;
    errs(end+1) = err_func(x);
    cost_vals(end+1) = f(x);
    ii = ii + 1;
end

jj=0; rel_change = inf;
while rel_change >= err_trunc_GS
    x_new = P1(P2(x));
    rel_change = norm(x_new - x)/norm(x);
    x = x_new;
    errs(end+1) = err_func(x);
    jj = jj + 1;
end

x_optim = x;
iters_optim = ii
iters_GS_optim = jj
figure, semilogy(errs)

x = x_init;
errs = err_func(x);
cost_vals = f(x);
rel_change = Inf; eta = 0.5
while errs(end) >= err_trunc
    x_new = x + eta*(2*P1(P2(x)) - P1(x) - P2(x));
    rel_change = norm(x - x_new)/norm(x);
    x = x_new;
    errs(end+1) = err_func(x);
    cost_vals(end+1) = f(x);
end
iters_RRR = length(errs) - 1
x_RRR = x;

jj=0; rel_change = inf;
while rel_change >= err_trunc_GS
    x_new = P1(P2(x));
    rel_change = norm(x_new - x)/norm(x);
    x = x_new;
    errs(end+1) = err_func(x);
    jj = jj + 1;
end

iters_GS_RRR = length(errs) - iters_RRR

hold on, semilogy(errs)

% save('res_oversampled_fourier.mat')
