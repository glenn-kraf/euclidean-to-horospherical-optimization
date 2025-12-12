% demo_algorithm2.m
import manopt.*;

% Boundary targets on unit circle:
Y  = [0  1  sqrt(0.5);
      1  0  sqrt(0.5)];          % 2x3

alpha = [5; 2; 7];
x0    = [0.5; 0.5];
r0    = 6.0;      % Phase-A radius
eps   = 1e-8;

[x_star, s_star, hist] = algorithm2_h_ellipsoid_ipm(x0, r0, eps, alpha, Y, struct());

% Quick certificate check:
fx = -inf;
for i=1:numel(alpha)
    nx  = dot(x_star,x_star);
    num = 1 - nx;
    den = sum((x_star - Y(:,i)).^2);
    fx  = max(fx, alpha(i) * (-log(num/den)));
end
fprintf('\nResult:\n');
fprintf('x*   = [%.10f, %.10f]\n', x_star(1), x_star(2));
fprintf('f(x*)= %.12f\n', fx);
fprintf('s*   = %.12f\n', s_star);
fprintf('gap  = %.3e\n', s_star - fx);

% --- Plotting ---
iters = 1:numel(hist.fvals);

% Compute best function value reached — approximate f*
f_best = min(hist.fvals);

figure;
semilogy(iters, hist.fvals - f_best, '-o','LineWidth',1.5);
xlabel('Iteration');
ylabel('f(x_k)-min_t f(x_t) (log scale)');
title('Objective values vs iteration');
grid on;

figure;
semilogy(iters(2:end), hist.steps(2:end), '-o','LineWidth',1.5);
xlabel('Iteration');
ylabel('||x_{k+1} - x_k|| (log scale)');
title('Step norm vs iteration');
grid on;