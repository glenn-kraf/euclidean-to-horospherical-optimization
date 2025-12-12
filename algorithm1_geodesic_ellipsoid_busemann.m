function [z_star, f_star, trace] = algorithm1_geodesic_ellipsoid_busemann(x0, r0, Y, alpha, delta, opts)
    % Geodesic Ellipsoid (Algorithm 1) for h-convex minimization on
    % Poincare Disc model
    % Objective:  f(x) = max_i alpha(i) * b_{y_i}(x),   y_i on unit circle.
    %
    % INPUTS
    %   x0    : 2x1 start,  ||x0|| < 1
    %   r0    : localization radius (assume argmin ⊆ B(x0, r0))
    %   Y     : 2xm boundary points (columns) with norm == 1
    %   alpha : mx1 positive weights
    %   delta : certificate target δ > 0   (stop when J(r_k)*r_max ≤ 2δ)
    %   opts  : struct (all optional)
    %           .verbose       (true)
    %           .phaseA_min_r  (1.0)
    %           .phaseA_print  (true)
    %           .maxit_B       (200)
    %           .flipprobe_eps (1e-3)
    %           .active_tol    (1e-12)  % tie tolerance for active set
    %
    % OUTPUTS
    %   z_star : final center on the manifold (2x1)
    %   f_star : f(z_star)
    %   trace  : struct with per-iteration history (Phase B):
    %            .fvals, .step_geo, .rmax, .Jr, .z (2xK), .c (2xK)

    if nargin < 6, opts = struct(); end
    opts = set_default(opts, 'verbose', true);
    opts = set_default(opts, 'phaseA_min_r', 1.0);
    opts = set_default(opts, 'phaseA_print', true);
    opts = set_default(opts, 'maxit_B', 200);
    opts = set_default(opts, 'flipprobe_eps', 1e-3);
    opts = set_default(opts, 'active_tol', 1e-12);

    % --- sanity
    assert(size(x0,1)==2 && size(x0,2)==1, 'x0 must be 2x1.');
    assert(all(abs(vecnorm(Y)-1) < 1e-12), 'Columns of Y must be unit-norm (boundary).');
    assert(numel(alpha)==size(Y,2), 'alpha length must match number of columns of Y.');
    assert(all(alpha>0), 'alpha must be positive.');
    assert(norm(x0) < 1, 'x0 must lie strictly inside the unit disk.');

    % --- Manopt Poincaré model ---
    M = poincareballfactory(2,1);

    % --- problem-specific f and (unit) Riemannian subgradient ---
    f_handle = @(x) f_buse(x, Y, alpha);

    function g = subgrad_unit(x)
        % 1) Euclidean subgradient in the disk chart (avg of active grads)
        m = size(Y,2);
        vals = zeros(m,1);
        Ge   = zeros(2,m);
        for i=1:m
            vals(i) = alpha(i)*busemann(x, Y(:,i));
            Ge(:,i) = alpha(i)*grad_busemann(x, Y(:,i));
        end
        mval   = max(vals);
        active = find(mval - vals <= opts.active_tol);
        if numel(active)==1 %#ok<ISCL>
            ge = Ge(:,active);
        else
            ge = mean(Ge(:,active), 2); % average
        end
        % 2) Convert to a Riemannian tangent vector & normalize
        %    (Poincaré metric: g_x = λ(x)^2 I ⇒ grad^R = (1/λ^2) * grad^E)
        lam = 2/(1-norm(x)^2);
        gz  = ge / (lam*lam);
        ng  = M.norm(x, gz);
        if ng < 1e-15, g = zeros(2,1); else, g = gz/ng; end
    end

    % ===================== Phase A: localization ==========================
    if opts.verbose, fprintf('--- Phase A (localization) ---\n'); end
    x_c = phaseA_localize_poincare(M, x0, r0, opts.phaseA_min_r, f_handle, @subgrad_unit, opts.phaseA_print);
    if opts.verbose
        g_c = subgrad_unit(x_c);
        fprintf('x_c = [%.6f, %.6f]   gradf(x_c) = [%.6f, %.6f]\n', ...
            x_c(1), x_c(2), g_c(1), g_c(2));
    end
    if opts.verbose, fprintf('Phase A done. Starting Phase B...\n\n'); end

    % ===================== Phase B: charted ellipsoid =====================
    % Chart: T_{x_c}M via Exp_{x_c}; ellipsoid E_k = {x: (x-c_k)' P_k^{-1} (x-c_k) ≤ 1}
    c = zeros(2,1);               % chart center
    P = (4.0^2) * eye(2);         % cover Exp_{x_c}(B(0,4))
    trace = struct('fvals',[],'step_geo',[],'rmax',[],'Jr',[],'z',[],'c',[]);


    z_prev = M.exp(x_c, c);

    for it = 1:opts.maxit_B
        % center on the manifold
        z = M.exp(x_c, c);

        % subgradient at z (unit Riemannian)
        gz = subgrad_unit(z);

        % pull back by parallel transport to x_c (Manopt)
        a = M.transp(z, x_c, gz);        % back to T_{x_c}M

        % ---------- central cut ellipsoid update in the chart ----------

        Pa    = P*a;
        denom = max(a.'*Pa, 0);
        
        %{
        if denom <= 1e-15
            P = 1.05*P;    % weak cut: inflate a bit and retry
            if opts.verbose && (it==1 || mod(it,20)==0)
                fprintf('[B %04d] f(z)=%.6f  Weak cut because a=[%.6f,%.6f] and P=[%.4e %.4e; %.4e %.4e]\n', ...
                    it, f_handle(z),a,P(1,1), P(1,2), P(2,1), P(2,2));
            end
            continue;
        end
        %}

        invsqrt = 1/sqrt(max(a.'*(P*a), 1e-15));
        Pa_hat  = Pa  * invsqrt;

        n     = 2;
        beta  = 2/(n+1);
        gamma = (n^2)/(n^2 - 1);

        c_new = c - (1/(n+1))*Pa_hat;
        P_new = gamma * (P - beta*(Pa_hat*Pa_hat.'));

        % SPD repair
        eta = 1e-12;
        P_new = 0.5*(P_new+P_new.');

        %[V,D] = eig(P_new); d = max(diag(D), 1e-15); P_new = V*diag(d)*V.';
        %ridge repair:
        mineig = min(eig(P_new));
        
        if mineig <= 0
            % Inflate just enough to make it strictly PD
            P_new = P_new + (abs(mineig) + eta) * eye(2);
        else
            % Still add a small regularization for numerical stability
            P_new = P_new + eta * eye(2);
        end

        % stopping test: J(r_k) * r_max ≤ 2 δ
        eigsP = eig(P_new);
        rmax  = sqrt(max(eigsP)); % How big the ellipsoid is in Tx_cM
        r_geo = M.dist(x_c, z);   % geodesic distance

        if r_geo < 1e-14
            Jk = 1.0;
        else
            Jk = sinh(r_geo) / r_geo;
        end

        % Jacobian distortian factor, stretching
        Jr    = Jk * rmax;        % Actual stretching of ellipsoid when mapping back

        % trace & console
        step_geo = M.dist(z_prev, z);
        trace.fvals(end+1)    = f_handle(z);
        trace.step_geo(end+1) = step_geo;
        trace.rmax(end+1)     = rmax;
        trace.Jr(end+1)       = Jr;
        trace.z(:,end+1)      = z;
        trace.c(:,end+1)      = c;

        if opts.verbose %&& (it==1 || mod(it,20)==0)
            fprintf('[B %04d] f(z)=%.6f  a=[%.6f,%.6f]  r_max=%.3e  J=%.3e  J*r=%.3e  step=%.3e\n', ...
                    it, trace.fvals(end), a, rmax, Jk, Jr, step_geo);
        end
        %{
        if Jr <= 7*delta % 7 is max of alpha, Lipschitzness of objective
            if opts.verbose
                fprintf('Phase B stop, criterion satisfied: J*r_max ≤ 2δ   (%.3e ≤ %.3e)\n', Jr, 2*delta);
            end
            break;
        end
        %}
        if rmax <= delta / 7 % 7 is max of alpha, Lipschitzness of objective
            if opts.verbose
                fprintf('Phase B stop, criterion satisfied: r_max ≤ δ/7   (%.3e ≤ %.3e)\n', rmax, delta/7);
            end
            break;
        end

        % advance
        z_prev = z;
        c = c_new; P = P_new;
    end

    z_star = M.exp(x_c, c);
    f_star = f_handle(z_star);
end

% --------------------- Phase A (same style as Alg. 2) ----------------------
function x = phaseA_localize_poincare(M, x0, r0, min_r, f_handle, grad_unit, printflag)
    if r0 <= min_r, x = x0; return; end
    N = ceil(4*log(r0/min_r));
    x = x0;
    for k = 0:N-1
        g = grad_unit(x);                         % unit Riemannian direction
        s = (r0*exp(-k/4))/2;
        x = M.retr(x, -s*g);
        if printflag && (k==0 || mod(k+1,5)==0 || k+1==N)
            fprintf('[Phase A %02d/%02d] step=% .3e  f(x)=%.6f\n', k+1, N, s, f_handle(x));
        end
    end
end

% ------------------------- Busemann helpers --------------------------------
function b = busemann(x, yb)
    nx  = dot(x,x);
    num = 1.0 - nx;
    den = sum((x - yb).^2);
    b   = -log(num/den);
end

function g = grad_busemann(x, yb)
    nx   = dot(x,x);
    num  = 1.0 - nx;
    d    = x - yb; den = dot(d,d);
    grad_num = 2.0*x; grad_den = 2.0*d;
    g = (grad_num*den + num*grad_den)/(num*den);
end

function val = f_buse(x, Y, alpha)
    m = size(Y,2);
    vals = zeros(m,1);
    for i=1:m
        vals(i) = alpha(i)*busemann(x, Y(:,i));
    end
    val = max(vals);
end

% ------------------------------ utils --------------------------------------
function S = set_default(S, key, val)
    if ~isfield(S, key) || isempty(S.(key)), S.(key) = val; end
end