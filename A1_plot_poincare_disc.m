% ------------------------------------------------------------
% Visualize f(x) = max_i alpha_i * b_{y_i}(x) on the Poincaré disk
% ------------------------------------------------------------

% Define grid over unit disk
res = 400;                     % resolution
x1 = linspace(-0.99, 0.99, res);
x2 = linspace(-0.99, 0.99, res);
[X1, X2] = meshgrid(x1, x2);

% Preallocate value array
F = nan(size(X1));

% Define function params
Y  = [0, 1, sqrt(0.5);
      1, 0, sqrt(0.5)];
alpha = [5; 2; 7];

function b = busemann(x, yb)
    nx  = dot(x,x);
    num = 1.0 - nx;
    den = sum((x - yb).^2);
    b   = -log(num/den);
end

function val = f_buse(x, Y, alpha)
    m = size(Y,2);
    vals = zeros(m,1);
    for i=1:m
        vals(i) = alpha(i)*busemann(x, Y(:,i));
    end
    val = max(vals);
end

% Evaluate objective inside the disk
for i = 1:numel(X1)
    x = [X1(i); X2(i)];
    if norm(x) < 1
        F(i) = f_buse(x, Y, alpha);  % your local function
    end
end

% Mask values outside the disk
F((X1.^2 + X2.^2) >= 1) = NaN;

% -------------------- Plot --------------------
figure; hold on;
h = imagesc(x1, x2, F);                 % color map
set(h, 'AlphaData', ~isnan(F));   % <-- make NaNs (outside disk) transparent
set(gca,'YDir','normal');
axis equal tight
colormap(turbo);                    % nicer gradient than 'parula'
colorbar;
% --- Make background outside the disk white ---
set(gca,'Color','w');               % axes background = white
set(gcf,'Color','w');               % figure background = white (optional
set(gcf,'InvertHardcopy','off');  % keep white on export (PDF/PNG)

title('Objective landscape  f(x) = max_i \alpha_i b_{y_i}(x)');
xlabel('x_1'); ylabel('x_2');

% Draw unit circle boundary
theta = linspace(0, 2*pi, 400);
plot(cos(theta), sin(theta), 'k', 'LineWidth', 1);

% Overlay boundary points Y
plot(Y(1,:), Y(2,:), 'ko', 'MarkerFaceColor','y', 'MarkerSize',6);
for i = 1:size(Y,2)
    text(Y(1,i)*1.05, Y(2,i)*1.05, sprintf('y_%d',i), 'Color','k','FontSize',8);
end

% Overlay phase-A center x_c if available
if exist('x_c','var')
    plot(x_c(1), x_c(2), 'bs', 'MarkerFaceColor','c','MarkerSize',6);
    text(x_c(1)+0.03, x_c(2), 'x_c','Color','c','FontSize',9);
end

% -------------------- IPM Minimum --------------------
ipm_min = [0.3498; 0.2655];
plot(ipm_min(1), ipm_min(2), 'mp', 'MarkerFaceColor','m', 'MarkerSize',10, 'LineWidth',1.5);
text(ipm_min(1)+0.03, ipm_min(2), 'IPM minimum', 'Color','m', 'FontSize',9, 'FontWeight','bold');

% Adjust visual limits
xlim([-1 1]); ylim([-1 1]);
hold off;