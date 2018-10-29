function [U, V, dnorm] = ShallowNMF(X, r, maxiter, tolfun)

[m, n] = size(X);

U = rand(m, r);
V = rand(r, n);

dnorm0 = norm(X - U * V, 'fro') + norm(V - U' * X, 'fro');

for i = 1:maxiter
    % update U
    
    U = U .* (2 * X * V') ./ max(U * (V * V') + X * (X' * U), 1e-10);
    
    % update V
    
    V = V .* (2 * U' * X) ./ max(U' * U * V + V, 1e-10);
    
    dnorm = norm(X - U * V, 'fro') + norm(V - U' * X, 'fro');
    
    if abs(dnorm0 - dnorm) <= tolfun
        break; % converge
    end
    
    dnorm0 = dnorm;
    
end
%dnorm = norm(X - U * V, 'fro') + norm(V - U' * X, 'fro');
