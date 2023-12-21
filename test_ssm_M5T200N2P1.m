% test_ssm_M5T200N2P1
% This function solves the small-scale simulation with N=2 hidden processes and P=1 autoregressive
% matrix order using the Matlab Econometric Toolbox ssm.estimate (Change the observation noise level
% in lines 12-14 to test different signal-to-noise ratio.).

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
Mdl = ssm(At, Bt, Ct, Dt, 'Mean0', 0, 'Cov0', 1*eye(N), 'StateType', zeros(N,1));
params0 = 0.1*ones(11,1);
lowBound = [-10*ones(4,1); zeros(N+M,1)];
uppBound = 10*ones(N*N*p +N+M,1);
% options = optimoptions(@fmincon, 'ConstraintTolerance', 1e-6, 'Algorithm', 'sqp', ...
%     'MaxFunctionEvaluations', 10000);
options = optimoptions(@fmincon, 'ConstraintTolerance', 1e-6, 'Algorithm', 'interior-point', ...
    'MaxFunctionEvaluations', 10000);
tic
EstMdl = estimate(Mdl, y', params0, 'lb', lowBound, 'ub', uppBound, 'Options', options) %#ok<*NOPTS> 
toc

disp('Estimated autoregressive matrix:');
Ae = EstMdl.A

disp('Ground truth:'); A