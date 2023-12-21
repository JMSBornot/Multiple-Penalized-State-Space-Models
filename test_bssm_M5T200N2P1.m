% test_bssm_M5T200N2P1
% This script solves the small-scale simulation with N=2 hidden processes and P=1 autoregressive
% matrix order using the Matlab Econometric Toolbox bssm.estimate (Bayesian parameter estimation of
% state-space models). Change the observation noise level in lines 13-15 to test different
% signal-to-noise ratio.

rng('default');

%% Simulation
Nepoch = 1;
ss = 1;

% so = 0.1;
% so = 0.5;
so = 1;

M = 5;
T = 200;
[x, y, A, B] = mvar1sim1(M, T, Nepoch, ss, so);
N = size(B,2);
p = size(A,3);

figure;
subplot 211;
plot(x', 'LineWidth', 2); axis tight; set(gca, 'FontSize', 24);
legend('1','2');
ylabel('\boldmath$\mathrm{X}$', 'Interpreter', 'latex');

%% Estimate with Matlab Econometric Toolbox
At = [NaN NaN; NaN NaN];
Bt = [NaN 0; 0 NaN];
Ct = B;
Dt = [NaN 0 0 0 0; 0 NaN 0 0 0; 0 0 NaN 0 0; 0 0 0 NaN 0; 0 0 0 0 NaN];
Mdl = ssm(At, Bt, Ct, Dt, 'Mean0', 0, 'Cov0', 10*eye(N), 'StateType', zeros(N,1));

bayesMdl = ssm2bssm(Mdl);

params0 = 0.1*ones(11,1);
options = optimoptions(@fminunc, 'OptimalityTolerance', 1e-6, 'MaxFunctionEvaluations', 1e4);
tic
EstPostMdl = estimate(bayesMdl, y', params0, 'Options', options);
toc

coeff_mean = mean(EstPostMdl.ParamDistribution,2);
coeff_std = std(EstPostMdl.ParamDistribution,0,2);
Ae = reshape(coeff_mean(1:N*N*p), [N N*p]) %#ok<*NOPTS>
Ae_std = reshape(coeff_std(1:N*N*p), [N N*p])

disp('Estimated autoregressive matrix:'); Ae

disp('Ground truth:'); A