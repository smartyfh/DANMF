function [U, V, dnorm, dnormarray] = DANMF(A, layers, option)


%%%%%%%%%%%%%%%%%%%%
% A: n x n
% layers: layer size array
% option: maxiter, tolfun, maxiter_pre, verbose, UpdateVi
%%%%%%%%%%%%%%%%%%%%
maxiter = option.maxiter;
tolfun = option.tolfun;
maxiter_pre = option.maxiter_pre;
verbose = option.verbose;
UpdateVi = option.UpdateVi;
lambda = option.lambda;


p = numel(layers);

U = cell(1, p);
V = cell(1, p);

dnormarray = zeros(maxiter);


%%%%%%%%%%%%%%%%%%%%
% Pre-training
%%%%%%%%%%%%%%%%%%%%
for i_layer = 1:p
    if i_layer == 1
        Z = A;
    else
        Z = V{i_layer - 1};
    end
    
    if verbose
        display(sprintf('Initialising Layer #%d ...', i_layer));
    end
    
    [U{i_layer}, V{i_layer}, ~] = ShallowNMF(Z, layers(i_layer), maxiter_pre, tolfun);
    
    if verbose
        display('Finishing initialization ...');
    end
end

%%%%%%%%%%%%%%%%%%%%
% Fine-tuning
%%%%%%%%%%%%%%%%%%%%

if verbose
    display('Fine-tuning ...');
end


% \Psi -> P; \Phi -> Q; \Pi -> R
D = diag(sum(A));
L = D - A;
Q = cell(1, p + 1);
AAT = A * A';
for iter = 1:maxiter
    Q{p + 1} = eye(layers(p));
    for i_layer = p:-1:2
        Q{i_layer} = U{i_layer} * Q{i_layer + 1};
    end
    
    VpVpT = V{p} * V{p}';
    
    for i = 1:p
        % Update Ui
        if i == 1
            R = U{1} * (Q{2} * VpVpT *  Q{2}') + AAT * (U{1} * (Q{2} * Q{2}'));
            Ru = 2 * A * (V{p}' * Q{2}');
            U{1} = U{1}.* Ru ./ max(R, 1e-10);
        else
            R = P' * P * U{i} * Q{i + 1} * VpVpT * Q{i + 1}' + P' * AAT * P * U{i} * Q{i + 1} * Q{i + 1}';
            Ru = 2 * P' * A * V{p}' * Q{i + 1}';
            U{i} = U{i}.* Ru ./ max(R, 1e-10);
        end
        
        % Update Vi
        if i == 1
            P = U{i};
        else
            P = P * U{i};
        end
        if (i < p) && UpdateVi
            Vu = 2 * P' * A;
            Vd = P' * P * V{i} + V{i};
            V{i} = V{i} .* Vu ./ max(Vd, 1e-10);
        else
            Vu = 2 * P' * A + lambda * V{i} * A;
            Vd = P' * P * V{i} + V{i} + lambda * V{i} * D;
            V{i} = V{i} .* Vu ./ max(Vd, 1e-10);
        end
    end
    
    dnorm = cost_function(A, P, V{p}, L, lambda);
    dnormarray(iter) = dnorm;
    if iter > 1 && abs(dnorm0 - dnorm) <= tolfun
        display(sprintf('Converged at iteration #%d ...', iter));
        break; % converge
    end
    dnorm0 = dnorm;
    
end

% VV = V{p};
% UU = P;
% 
% de = norm(VV - UU' * A, 'fro');
% dd = norm(A - UU * VV, 'fro');



end

function error = cost_function(A, Up, Vp, L, lambda)
    error = norm(A - Up * Vp, 'fro')^2 + norm(Vp - Up' * A, 'fro')^2 + lambda * trace(Vp * L * Vp');
    error = sqrt(error);
end