function H = densityContour(Y1, Y2, x1, x2, gL, gs)

% H = densityContour(Y1, Y2, x1, x2, gL, gs)
%
% Create contour data for bivariate density
% Y1, Y2        observations
% x1, x2        histogram grid
% gL, gs        size and sd of Gaussian filter

G = fspecial('gaussian',gL,gs);
H = hist3([Y1(:) Y2(:)], {x1, x2});
H = imfilter(H', G, 'conv');
