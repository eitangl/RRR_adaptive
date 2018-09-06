clear all

n = 64;
m = 127;

% I = imread('circuit.tif');
% I = imread('cameraman.tif');
% I = imread('coins.png');
% I = imread('kids.tif');
% I = I(77:77+63,105:105+63);
% I = double(imresize(I, [n, n]));
% I = double(I(35:35+63,95:95+63));
I = double(imread('football.jpg'));
% I = double(imread('tape.png'));
I = imresize(I(:,:,1), [64,64]);
x_t = zeros(m);
x_t(1:n, 1:n) = I;
y_t = fft2(x_t);
y0 = abs(y_t);

P1 = @(x) support_proj_nosparse(x, m, n);
P2 = @(x) real(ifft2( y0.*exp(1i*angle( fft2(x) )) ));
% P2 = @(x) phase_proj_nonorm(x, y0);
P1_ker = @(x) x - P1(x);
P2_ker = @(x) x - P2(x);

f = @(y) norm(y - P1(P2(y)), 'fro')^2 - norm(P2_ker(y), 'fro')^2/2 - norm(P1_ker(y), 'fro')^2/2;
df = @(y) y - P1(P2(y)) - P1_ker(P2_ker(y));

trials = 1e3; err_trunc = 1e-5;
iters_optim = zeros(trials, 1); iters_RRR = zeros(trials, 1);
if isempty(gcp('nocreate')), parpool('local', maxNumCompThreads); end
parfor t = 1:trials
    x_init = randn(m);
    x = x_init;
    
    rel_change = Inf; ii = 0;
    alpha = 0.05; beta = 0.85; eta_0 = 1; eta_trunc = 0.25;
    while rel_change >= err_trunc
        x_new = @(eta) x - eta*df(x);
        eta = eta_0;
        ndf = norm(df(x))^2; func_vals = f(x);
        while abs(f(x_new(eta))) > abs(func_vals) - alpha*eta*ndf && eta > eta_trunc
            eta = beta*eta;
        end
        if eta <= eta_trunc, eta = eta_trunc; end
        x_new = x_new(eta);
        rel_change = norm(x_new - x)/norm(x);
        x = x_new;
        ii = ii + 1;
    end

    iters_optim(t) = ii;
        
    x = x_init;
    
    rel_change = Inf; ii = 0; eta = 0.5;
    while rel_change >= err_trunc
        x_new = x + eta*(2*P1(P2(x)) - P1(x) - P2(x));
        rel_change = norm(x - x_new)/norm(x);
        x = x_new;
        ii = ii + 1;
    end

    iters_RRR(t) = ii;
end
save(['res_oversampled_fourier_football_trials_' num2str(trials) '.mat'])
