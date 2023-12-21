% test_bssm_M5T200N2P1_MC
% This script replicates the simulation in the script "test_bssm_M5T200N2P1" using Monte Carlo
% simulations, as represented by the number of simulated epochs (Nepoch). The simulated data is read
% in lines 8-10 from a mat file to guarantee that the same data is used by each method in the
% comparison analysis.

%% Simulation
% load('backprop_M5T200N2P1_MCso0.1.mat')
% load('backprop_M5T200N2P1_MCso0.5.mat')
load('backprop_M5T200N2P1_MCso1.0.mat')

%% Estimation with Matlab Econometric Toolbox
At = [NaN NaN; NaN NaN];
Bt = [NaN 0; 0 NaN];
Ct = B;
Dt = [NaN 0 0 0 0; 0 NaN 0 0 0; 0 0 NaN 0 0; 0 0 0 NaN 0; 0 0 0 0 NaN];

Mdl = ssm(At, Bt, Ct, Dt, 'Mean0', 0, 'Cov0', 1*eye(N), 'StateType', zeros(N,1));
bayesMdl = ssm2bssm(Mdl);
options = optimoptions(@fminunc, 'OptimalityTolerance', 1e-6, 'MaxFunctionEvaluations', 1e4);

Ae = zeros(N,N,p,Nepoch);

tstart = tic;
for i = 1:Nepoch
    fprintf('\n\n============ iter %d of %d ============\n\n', i, Nepoch);

    frepeat = true;
    cont = 0;
    while (frepeat && cont<5)
        cont = cont + 1;
        params0 = [0.1*randn(N*N*p,1); ones(N+M,1)];
        lastwarn('','');

        EstPostMdl = estimate(bayesMdl, y(:,:,i)', params0, 'Options', options);

        [warnMsg, warnId] = lastwarn();
        frepeat = ~isempty(warnMsg);
    end
    coeff_mean = mean(EstPostMdl.ParamDistribution,2);
    Ae(:,:,:,i) = reshape(coeff_mean(1:N*N*p), [N N p]); %#ok<*NOPTS>
end
time_calc = toc(tstart);

% save bssm_M5T200N2P1_MCso0.1.mat
% save bssm_M5T200N2P1_MCso0.5.mat
% save bssm_M5T200N2P1_MCso1.0.mat