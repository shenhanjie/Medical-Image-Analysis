cd('/Users/Shen/Desktop/WORK/Image Project (Liver Cancer)/HCC')
addpath('./NIFTI_20130306')
clear
close all

%% Data Section

% Load image from Case 1
filename = '/Users/Shen/Desktop/WORK/Image Project (Liver Cancer)/Anonymized LI-RADS Cases/Case #1/03 PRE CON/03 PRE CON.nii';
hcc(1) = load_untouch_nii(filename);
filename = '/Users/Shen/Desktop/WORK/Image Project (Liver Cancer)/Anonymized LI-RADS Cases/Case #1/03 PRE CON/03 PRE CON-label.nii';
hcc_label(1) = load_untouch_nii(filename);
filename = '/Users/Shen/Desktop/WORK/Image Project (Liver Cancer)/Anonymized LI-RADS Cases/Case #1/05 ARTERIAL/05 ARTERIAL.nii';
hcc(2) = load_untouch_nii(filename);
filename = '/Users/Shen/Desktop/WORK/Image Project (Liver Cancer)/Anonymized LI-RADS Cases/Case #1/05 ARTERIAL/05 ARTERIAL-label.nii';
hcc_label(2) = load_untouch_nii(filename);
filename = '/Users/Shen/Desktop/WORK/Image Project (Liver Cancer)/Anonymized LI-RADS Cases/Case #1/06 PORTAL VENOUS/06 PORTAL VENOUS.nii';
hcc(3) = load_untouch_nii(filename);
filename = '/Users/Shen/Desktop/WORK/Image Project (Liver Cancer)/Anonymized LI-RADS Cases/Case #1/06 PORTAL VENOUS/06 PORTAL VENOUS-label.nii';
hcc_label(3) = load_untouch_nii(filename);
filename = '/Users/Shen/Desktop/WORK/Image Project (Liver Cancer)/Anonymized LI-RADS Cases/Case #1/08 DELAYED/08 DELAYED.nii';
hcc(4) = load_untouch_nii(filename);
filename = '/Users/Shen/Desktop/WORK/Image Project (Liver Cancer)/Anonymized LI-RADS Cases/Case #1/08 DELAYED/08 DELAYED-label.nii';
hcc_label(4) = load_untouch_nii(filename);

hcc(3).img = hcc(3).img(:,:,49:127);
hcc_label(3).img = hcc_label(3).img(:,:,49:127);

Data = hcc;
Label = hcc_label;
tumorind = [110 111 113 115];
unigmmclasses = [3 3 3 3];
multigmmclasses = 3;
phases = {'nc', 'ha', 'pv', 'de'};
cropp = 0.5;


np = length(phases);
ind = cell(1,np);
for i = 1:np,
    ind{i} = find(Label(i).img == tumorind(i));
end

x = cell(1,np);
y = cell(1,np);
z = cell(1,np);

for i = 1:np,
    [xlabel ylabel zlabel] = ind2sub(size(Label(i).img), ind{i});
    x{i} = xlabel;
    y{i} = ylabel;
    z{i} = zlabel;
end

slice = zeros([1 np]);
for i = 1:np,
    slice(i) = round(mean(z{i}));
end


for i = 1:np,
    xmin(i) = round(min(x{i})-(max(x{i})-min(x{i}))*cropp);
    ymin(i) = round(min(y{i})-(max(y{i})-min(y{i}))*cropp);
    zmin(i) = round(min(z{i})-(max(z{i})-min(z{i}))*cropp);
    xmax(i) = round(max(x{i})+(max(x{i})-min(x{i}))*cropp);
    ymax(i) = round(max(y{i})+(max(y{i})-min(y{i}))*cropp);
    zmax(i) = round(max(z{i})+(max(z{i})-min(z{i}))*cropp);
end

% need to check !!!
xm1 = round(mean(xmin));
xm2 = round(mean(xmax));
ym1 = round(mean(ymin));
ym2 = round(mean(ymax));
zm1 = round(mean(zmin));
zm2 = round(mean(zmax));

for i = 1:np,
     Data(i).lesion = Data(i).img(xm1:xm2,ym1:ym2,zm1:zm2);
end


for i = 1:np,
    subplot(2,2,i);
    s = size(Data(i).lesion);
    s = round(s(3)/2);
    imagesc(Data(i).lesion(:,:,s), [0,200]), colormap(gray), axis equal tight,colorbar
    title(phases{i})
end



%% Simulation Section 

% Generate true image
sz = [54 61 10];
[x y z] = meshgrid(1:sz(1),1:sz(2),1:sz(3));
% b2 = 0.015;
% b4 = 0.015;
% b1 = -56*b2;
% b3 = -50*b4;
% b0 = 784*b2 + 625*b4 - 3*56*b2 - 3*50*b4;
% b = [b0 b1 b2 b3 b4];
% mu0 = b0 + b1*x + b2*(x.^2) + b3*y + b4*(y.^2);
% %mu0 = b0 - 60*b1*x + b1*(x.^2) - 50*b2*y + b2*(y.^2);
% p0 = exp(mu0)./(1+exp(mu0));
% figure(1)
% imagesc(p0), colorbar

% Create response
img0 = Data(4).lesion(:,:,:);
%img = [img0 > 60];
%figure(2)
%imagesc(img), colorbar
figure(2)
imagesc(img0(:,:,5)), colorbar



% Fit logistic regression
bhat = glmfit([x(:) x(:).^2 y(:) y(:).^2 x(:).*y(:)],img(:),'binomial','link','logit');
%X = [x(:) x(:).^2 y(:) y(:).^2 z(:) z(:).^2 x(:).*y(:) x(:).*z(:) y(:).*z(:)];
X = [x(:) x(:).^2 y(:) y(:).^2 z(:) z(:).^2];
y = double(img0(:));
mod = fitlm(X,y,'linear','RobustOpts','on');
bhat = mod.Coefficients.Estimate;
yhat = predict(mod,X);
imghat = reshape(yhat, [61 54 10]);

figure(3)
imagesc(imghat(:,:,5)),colorbar

% Prediction
muhat = bhat(1) + bhat(2)*x + bhat(3)*(x.^2) + bhat(4)*y + bhat(5)*(y.^2) + bhat(6)*(x.*y);
phat = exp(muhat)./(1+exp(muhat));
figure(3)
imagesc(phat),colorbar
figure(4)
imagesc(phat > 0.5), colorbar





