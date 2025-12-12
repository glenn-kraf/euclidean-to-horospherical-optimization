function [x_star, s_star, hist] = algorithm2_h_ellipsoid_ipm(x0, r0, eps, alpha, Y, opts)
    % ALGORITHM2_H_ELLIPSOID_IPM  (MATLAB / Manopt-style)
    %  Hybrid h-ellipsoid–IPM on the Poincaré disk (curvature -1, n=2).
    %
    %  Problem:  minimize f(x) = max_i alpha(i) * b_yi(x),  i=1..m
    %            with Busemann function b_y(x) on the Poincaré disk (||x||<1).
    %
    % Inputs:
    %   x0   : 2x1 start (inside unit disk)
    %   r0   : initial localization radius (Euclidean scalar, as in your Phase A)
    %   eps  : stopping tolerance for certificate s - f(x) <= eps
    %   alpha: mx1 weights (e.g. [5; 2; 7])
    %   Y    : 2xm boundary points on unit circle (each column norm==1)
    %   opts : struct with fields:
    %          .verbose (true), .phaseA_min_r (4.0), .newton_max_it (50)
    %          .newton_tol_dec (1e-10), .t0 (1.0), .mu (2.0),
    %          .ac_iters (15)x
    %
    % Outputs:
    %   x_star, s_star : final primal point and certifying slack
    %   hist           : struct of traces (optional)
    
    if nargin < 6, opts = struct(); end
    opts = set_default(opts, 'verbose', true);
    opts = set_default(opts, 'phaseA_min_r', 4.0);
    opts = set_default(opts, 'newton_max_it', 50);
    opts = set_default(opts, 'newton_tol_dec', 1e-10);
    opts = set_default(opts, 't0', 1.0);
    opts = set_default(opts, 'mu', 2.0);
    opts = set_default(opts, 'ac_iters', 15);

    assert(size(x0,1)==2 && size(x0,2)==1, 'x0 must be 2x1.');
    assert(all(abs(vecnorm(Y)-1) < 1e-12), 'Columns of Y must be unit-norm (on the boundary).');
    assert(norm(x0) < 1, 'x0 must lie strictly inside the unit disk.');
    
    import manopt.*;
    M = poincareballfactory(2, 1);   % a single 2D Poincaré ball
        
    % -- Phase A: localization on H^2 (your exact schedule) -------------------
    x_c = phaseA_localize_poincare(M, x0(:), r0, opts.phaseA_min_r, @f, @grad_sub_f, Y, alpha, opts.verbose);
    
    % -- Build initial feasible (x,s) and analytic center for Phi --------------
    s = f(x_c, Y, alpha) + 1e-2;
    [x_c, s] = analytic_center(M, x_c, s, Y, alpha, opts.ac_iters);
    
    if opts.verbose
        fprintf('Phase A done. Starting IPM path-following...\n\n');
    end
    
    % -- Path-following: damped Newton on F_t = t*s + Phi(x,s) -----------------
    t = opts.t0;
    mu = opts.mu;
    hist = struct('gap',[],'Phi',[],'t',[],'x',[],'s',[],'fvals',[],'steps',[]);
    for k = 1:40
        % store previous iterate to measure step length
        x_old = x_c;
    
        % do one outer iteration (path-following Newton)
        [x_c, s] = newton_on_Ft(M, x_c, s, t, Y, alpha, ...
                                opts.newton_max_it, opts.newton_tol_dec);
    
        % compute scalars for logging & history
        fx_val    = f(x_c, Y, alpha);
        gap       = s - fx_val;                 % duality gap / certificate
        [Phi_val,~,~,~,~,~,~] = phi_and_derivs(M, x_c, s, Y, alpha);
        step_norm = norm(x_c - x_old);          % Euclidean step length
    
        % pretty console log (only if verbose)
        if opts.verbose
            fprintf(['[t=%8.2e]  Phi=% .6e   f(x)=%.6f   s-f(x)=% .3e   ' ...
                     'step=%.3e   x=[% .6f,% .6f]\n'], ...
                    t, Phi_val, fx_val, gap, step_norm, x_c(1), x_c(2));
        end
    
        % append to history traces
        hist.gap(end+1)     = gap;
        hist.Phi(end+1)     = Phi_val;
        hist.t(end+1)       = t;
        hist.x(:,end+1)     = x_c;
        hist.s(end+1)       = s;
        hist.fvals(end+1)   = fx_val;
        hist.steps(end+1)   = step_norm;
    
        % stopping & schedule
        if gap <= eps
            break;
        end
        t = t * mu;
    end
    
    x_star = x_c;
    s_star = s;
    
    % ===================== NESTED / LOCAL FUNCTIONS ===========================
    function val = f(x, Y, a)
        % objective: max_i alpha_i * busemann_yi(x)
        m = size(Y,2);
        vals = zeros(m,1);
        for i=1:m
            vals(i) = a(i)*busemann(x, Y(:,i));
        end
        val = max(vals);
    end
    
    function g = grad_sub_f(x, Y, a)
        % Euclidean subgradient in the disk coordinates (avg of active grads)
        m = size(Y,2);
        vals = zeros(m,1);
        for i=1:m
            vals(i) = a(i)*busemann(x, Y(:,i));
        end
        mval = max(vals);
        active = find(mval - vals <= 1e-12);
        if numel(active) == 1
            i = active;
            g = a(i)*grad_busemann(x, Y(:,i));
        else
            g = zeros(2,1);
            for idx = active.'
                g = g + a(idx)*grad_busemann(x, Y(:,idx));
            end
            g = g/numel(active);
        end
    end

end % ===== end main =====

% -------------------------------------------------------------------------
% Phase A localization (unit-Riemannian subgradient steps on H^2)
function x = phaseA_localize_poincare(M, x0, r0, min_r, f_handle, g_handle, Y, alpha, verbose)
    if r0 <= min_r
        x = x0;
        return;
    end
    N = ceil(4*log(r0/min_r));
    x = x0;
    for k = 0:N-1
        g_e = g_handle(x, Y, alpha);                        % Euclidean subgrad
        ngR = M.norm(x, g_e);                               % Riemannian norm via Manopt
        if ngR < 1e-15, break; end
        g_unit = g_e / ngR;                                 % unit in Riemannian metric
        s = (r0*exp(-k/4))/2;
        x = M.retr(x, -s*g_unit);                           % or M.retr(x, -s*g_unit)
        if verbose && (k==0 || mod(k+1,5)==0 || k+1==N)
            fprintf('[Phase A %02d/%02d] step=% .3e  f(x)=% .6f\n', ...
                    k+1, N, s, f_handle(x,Y,alpha));
        end
    end
end


% -------------------------------------------------------------------------
% Analytic center (Newton on Phi = -sum log h_i)
function [x,s] = analytic_center(M, x, s, Y, alpha, iters)
    if nargin < 6, iters = 30; end
    for it=1:iters
        [Phi, gx, gs, Hxx, Hxs, Hss, h] = phi_and_derivs(M, x, s, Y, alpha);
        if ~isfinite(Phi) || any(h<=0)
            s = max(s, max(alpha.*busemann_vec(x,Y)) + 1e-3);
            continue;
        end
        grad = [gx; gs];
        H = [Hxx, Hxs; Hxs.', Hss];
        % small ridge if needed
        [delta, ok] = safe_solve(H, -grad);
        if ~ok
            H(1:2,1:2) = H(1:2,1:2) + 1e-8*eye(2);
            H(3,3)     = H(3,3)     + 1e-8;
            delta = H \ (-grad);
        end
        step = 1.0; Phi0 = Phi;
        while true
            xn = M.retr(x, step*delta(1:2));
            sn = s + step*delta(3);
            [Phi_try,~,~,~,~,~,htry] = phi_and_derivs(M, xn, sn, Y, alpha);
            if all(isfinite(htry)) && all(htry>0) && Phi_try <= Phi0 - 1e-4*step*(grad.'*(-delta))
                x = xn; s = sn; break;
            end
            step = 0.5*step;
            if step < 1e-8, break; end
        end
    end
end

% -------------------------------------------------------------------------

% Damped Newton on F_t = t*s + Phi(x,s)
function [x,s] = newton_on_Ft(M, x, s, t, Y, alpha, max_it, tol_dec)
    if nargin < 7, max_it = 30; end
    if nargin < 8, tol_dec = 1e-8; end
    for it=1:max_it
        [Phi, gx, gs, Hxx, Hxs, Hss, h] = phi_and_derivs(M, x, s, Y, alpha);
        if ~isfinite(Phi) || any(h<=0)
            s = max(s, max(alpha.*busemann_vec(x,Y)) + 1e-6);
            continue;
        end
        grad = [gx; t + gs];
        H = [Hxx, Hxs; Hxs.', Hss];

        [delta, ok] = safe_solve(H, -grad);
        if ~ok
            H(1:2,1:2) = H(1:2,1:2) + 1e-8*eye(2);
            H(3,3)     = H(3,3)     + 1e-8;
            delta = H \ (-grad);
        end
        lam2 = -(grad.'*delta);
        if lam2/2 <= tol_dec, break; end

        step = 1.0;
        F0 = t*s + Phi;
        while true
            xn = M.retr(x, step*delta(1:2));
            sn = s + step*delta(3);
            [Phi_try,~,~,~,~,~,htry] = phi_and_derivs(M, xn, sn, Y, alpha);
            if all(isfinite(htry)) && all(htry>0) ...
               && (t*sn + Phi_try) <= F0 - 1e-4*step*lam2
                x = xn; s = sn; break;
            end
            step = 0.5*step;
            if step < 1e-8
                % ensure xn, sn exist even if we failed every check
                xn = project_inside_ball(M.retr(x, step*delta(1:2)));
                sn = s + step*delta(3);
        
                x = xn;
                sn_safe = max(sn, max(alpha.*busemann_vec(xn,Y)) + 1e-12);
                s = sn_safe;
                break;
            end
        end
    end
end

% -------------------------------------------------------------------------

% Barrier Phi and derivatives
function [Phi, gx, gs, Hxx, Hxs, Hss, h] = phi_and_derivs(M, x, s, Y, alpha)
    m = size(Y,2);

    b  = zeros(m,1);
    g  = zeros(2,m);
    Hs = zeros(2,2,m);
    for i=1:m
        b(i)     = busemann(x, Y(:,i));
        g(:,i)   = grad_busemann(x, Y(:,i));
        Hs(:,:,i)= hess_busemann(x, Y(:,i));
    end

    h = s - alpha(:).*b;   % slacks
    if any(h <= 0)
        Phi = inf; gx=[]; gs=[]; Hxx=[]; Hxs=[]; Hss=[]; return;
    end
    invh = 1./h;

    Phi = -sum(log(h));
    gs  = -sum(invh);
    gx  = sum((alpha(:).*invh).'.*g, 2);

    Hss = sum(invh.^2);
    Hxs = -sum((alpha(:).*invh.^2).'.*g, 2);

    Hxx = zeros(2,2);
    for i=1:m
        Hxx = Hxx + (alpha(i)*invh(i))*Hs(:,:,i) + ((alpha(i)^2)*(invh(i)^2))*(g(:,i)*g(:,i).');
    end
end

% =================== Hyperbolic geometry helpers ==========================

function b = busemann(x, yb)
    % yb on unit circle
    nx  = dot(x,x);
    num = 1.0 - nx;
    den = sum((x - yb).^2);
    b   = -log(num/den);
end

function bv = busemann_vec(x, Y)
    m = size(Y,2);
    bv = zeros(m,1);
    for i=1:m, bv(i)=busemann(x, Y(:,i)); end
end

function g = grad_busemann(x, yb)
    nx   = dot(x,x);
    num  = 1.0 - nx;
    d    = x - yb;
    den  = dot(d,d);
    grad_num = 2.0*x;
    grad_den = 2.0*d;
    g = (grad_num*den + num*grad_den)/(num*den);
end

function H = hess_busemann(x, yb)
    nx  = dot(x,x);
    num = 1.0 - nx;
    d   = x - yb;
    d2  = dot(d,d);
    I   = eye(2);
    H1  = (2.0/num)*I + (4.0/(num*num))*(x*x.');
    H2  = (2.0/d2 )*I - (4.0/(d2*d2))*(d*d.');
    H   = H1 + H2;
end

% ============================ small utils =================================

function S = set_default(S, key, val)
    if ~isfield(S, key) || isempty(S.(key)), S.(key) = val; end
end

function [x, ok] = safe_solve(A, b)
    ok = true;
    % try Cholesky for SPD; fall back to backslash
    try
        R = chol((A+A.')/2);
        x = R\(R'\b);
    catch
        ok = false;
        x = A\b;
    end
end