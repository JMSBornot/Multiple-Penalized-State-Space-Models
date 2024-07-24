function [xe, Ae, ce, iter] = SSALS_MCOLS(y, B, Ae, ce, lmbd, l2a, Niter, loss, tol, verbose, intercept)
% State space alternating least squares (SSALS) algorithm:
% [xe, Ae, iter] = SSALS(y, B, Ae, lmbd, l2a, Niter, loss, reltol, verbose);
%
% INPUT:
% y -           Observed time series data. Dimension: MxT, for M sensors and T time instants
% B -           Mixing matrix, for forecasting or if B is unknown, use B = eye(M). Dimension: MxN,
%               where N is the number of latent variables.
% Ae -          Initial value for the estimation of autoregressive coefficients. Dimension: NxNxP,
%               where P is the order of the autoregressive model. If do not have any preferred
%               initialization, then set Ae to zeros(N,N,P).
% ce -          Initial value for the estimation of the intercept. Dimension: Mx1. If do not have
%               any preferred initialization, set it to empty ([]).
% lmbd -        hyperparameter of the latent cost term (scalar, >=0)
% lmbd2 -       hyperparameter implementing the Gaussian a priori of the autoregressive cofficients
%               (scalar, >=0)
% Niter -       Maximum number of iterations (scalar, >0). Default value is 1000
% loss -        Loss function used to implement cross-validation approach. Dimension: 1xT. Default
%               value is ones(1,T). For k-fold cross-validation, initialize it to ones(1,T), then
%               set to zero the entries for the samples in the hold-out fold
% tol -         Tolerance error for minimum successive objetictive function evaluations. Once
%               reached the algorithm will stop earlier (convergence criteria)
% verbose -     Set to print messages to monitor algorithm behaviour
% intercept -   Set to false if do not wish to calculate the intercept. By default intercept = true
%
% OUTPUT:
% xe -      Estimated latent variable. Dimension is NxT
% Ae -      Estimated autoregressive coefficients. Dimension is NxNxP
% ce -      Estimated intercept from the observation equation
% iter -    Number of iterations needed to compute solution. If convergence not reached, then iter
%           equals to Niter

if ~exist('intercept','var') || isempty(intercept)
    intercept = true;
end
if ~exist('verbose','var') || isempty(verbose)
    verbose = false;
end
if ~exist('reltol','var') || isempty(tol)
    tol = 1e-6;
end
if ~exist('Niter','var') || isempty(Niter)
    Niter = 1000;
end

[M,T,Nepoch] = size(y);
N = size(Ae,1);
P = size(Ae,3);

% time reversal
y = fliplr(y);
if ~exist('loss','var') || isempty(loss)
    loss = ones(1,T);
else
    loss = fliplr(loss);
end
t = 1:T-P; % last T-p time points in reverse order

% initialization
if intercept
    if isempty(ce) || all(ce(:)==0)
        ce = sum(y.*repmat(loss, [M 1 Nepoch]),2)/sum(loss);
    end
else
    ce = 0;
end

% XtX and XtY terms which never change
XtX = kron(sparse(diag(loss)), B'*B);

if ~intercept
    XtY = B'*reshape(y.*repmat(loss, [M 1 Nepoch]),M,[]);
end

% auxiliar variables
Z = zeros(P*N,T-P);

tic
for iter = 1:Niter
    % Initialize with Identity
    W = speye(T*N);
    % First, add the linear terms
    for k = 1:P
        vec = zeros(T-k,1);
        vec(1:T-P) = 1;
        W = W - kron(sparse(diag(vec,k)), Ae(:,:,k)); % add to superior triangular part
        W = W - kron(sparse(diag(vec,-k)), Ae(:,:,k)'); % add to lower triangular part
    end
    % Then, add the quadratic terms
    for k = 1:P
        for l = 1:k
            vec = zeros(T-(k-l),1);
            vec(l+(1:T-P)) = 1;
            tmp = Ae(:,:,l)'*Ae(:,:,k);
            W = W + kron(sparse(diag(vec,k-l)), tmp); % added to the main or to one of the super diagonals
            if (k > l)
                W = W + kron(sparse(diag(vec,l-k)), tmp'); % then, add it too the (mirror) lower diagonal
            end
        end
    end    
    % Last, compute in time-reversal order
%     LtL = XtX + lmbd*W + 1e-8*speye(T*N);
    LtL = XtX + lmbd*W;
    if intercept
        yce = (y-ce).*repmat(loss, [M 1 Nepoch]);
        XtY = B'*reshape(yce,M,[]);
    end
    xe = LtL\reshape(XtY,[],Nepoch);
    xe = reshape(xe, [N T Nepoch]);
    
    % -> Update Ae
    xZt = 0;
    ZZt = 0;
    for e = 1:Nepoch
        for k = 1:P
            Z((k-1)*N+(1:N),:) = xe(:,t+k,e); % t+k, instead of t-k, due to time-reversal
        end
        xZt = xZt + xe(:,t,e)*Z';
        ZZt = ZZt + Z*Z';
    end
    Ae(:) = xZt/(ZZt + (T*Nepoch*l2a/lmbd)*speye(P*N));
%     xZt = xZt/(T*Nepoch);
%     ZZt = ZZt/(T*Nepoch);
%     Ae(:) = xZt/(ZZt + (l2a/lmbd)*speye(P*N));
    
    % -> Update intercept
    if intercept
        yhat = reshape(B*reshape(xe,N,[]), [M T Nepoch]);
        ce = sum((y-yhat).*repmat(loss, [M 1 Nepoch]),2)/sum(loss);
    end
    
    % -> Evaluate cost function after all updates at each iteration
    F1 = 0;
    F2 = 0;
    for e = 1:Nepoch
        xerr = xe(:,:,e);
        for k = 1:P
            xerr(:,t) = xerr(:,t) - Ae(:,:,k)*xe(:,t+k,e); % t+k, instead of t-k, due to time-reversal
        end
        if intercept
            F1 = F1 + sum(sum(((y(:,:,e)-B*xe(:,:,e)-ce(:,:,e)).^2).*loss));
        else
            F1 = F1 + sum(sum(((y(:,:,e)-B*xe(:,:,e)).^2).*loss));
        end
        F2 = F2 + sum(sum(xerr.^2));
    end
    F = 0.5*((F1 + lmbd*F2)/(T*Nepoch) + l2a*sum(Ae(:).^2));
    if verbose
        fprintf('iteration #%d (ALS): F = %.8f\n', iter, F);
    end
    
    % -> Convergence checkup and auxiliar operations
    if (iter > 1)
        % relative error
        % relchange = abs(Fprev - F)/Fprev;
        obfchange = abs(Fprev - F);
        if (Fprev < F)
            warning('Objective function has increased.');
        end
        % elapsed time
        if verbose
            % fprintf('100*(F(k)-F(k+1))/F(k) = %.4f%%.\n', 100*relchange);
            fprintf('|F(k)-F(k+1)| = %.8f\n', obfchange);
            tnow = toc;
            fprintf('Elapsed time is %.3f seconds.\n', tnow);
            drawnow;
        end
        % stop criterium
        if (obfchange < tol)
            if verbose
                fprintf('SSALS converged after %d iterations.\n', iter);
            end
            break;
        end
    end
    Fprev = F;
end
xe = fliplr(xe);