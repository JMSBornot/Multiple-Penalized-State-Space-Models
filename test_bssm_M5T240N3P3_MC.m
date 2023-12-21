% test_bssm_M5T240N3P3_MC
% This script replicates the simulation in the script "test_bssm_M5T240N3P3" using Monte Carlo
% simulations, as represented by the number of simulated epochs (Nepoch). The simulated data is read
% in lines 8-10 from a mat file to guarantee that the same data is used by each method in the
% comparison analysis.

%% Simulation
load('ssals_M5T240N3P3_MCso0.1.mat')
% load('ssals_M5T240N3P3_MCso0.5.mat')
% load('ssals_M5T240N3P3_MCso1.0.mat')

%% Estimation with Matlab Econometric Toolbox
At = [NaN(N) NaN(N) NaN(N); eye(N) zeros(N) zeros(N); zeros(N) eye(N) zeros(N)];
Bt = [diag(NaN(N,1)); zeros(N); zeros(N)];
Ct = [B zeros(M,N) zeros(M,N)];
Dt = diag(NaN(M,1));

Mdl = ssm(At, Bt, Ct, Dt, 'Mean0', 0, 'Cov0', 1*eye(3*N), 'StateType', zeros(3*N,1));
bayesMdl = ssm2bssm(Mdl);
options = optimoptions(@fminunc, 'OptimalityTolerance', 1e-6, 'MaxFunctionEvaluations', 1e4);

Ae = zeros(N,N,P,Nepoch);

tstart = tic;
for i = 1:Nepoch
    fprintf('\n\n============ iter %d of %d ============\n\n', i, Nepoch);

    frepeat = true;
    cont = 0;
    while (frepeat && cont<5)
        cont = cont + 1;
        params0 = [0.1*randn(N*N*P,1); ones(N+M,1)];
        lastwarn('','');

        EstPostMdl = estimate(bayesMdl, y(:,:,i)', params0, 'Options', options);

        [warnMsg, warnId] = lastwarn();
        frepeat = ~isempty(warnMsg);
    end
    coeff_mean = mean(EstPostMdl.ParamDistribution,2);
    Ae(:,:,:,i) = reshape(coeff_mean(1:N*N*P), [N N P]); %#ok<*NOPTS>
end
time_calc = toc(tstart);

% save bssm_M5T240N3P3_MCso0.1.mat
% save bssm_M5T240N3P3_MCso0.5.mat
% save bssm_M5T240N3P3_MCso1.0.mat