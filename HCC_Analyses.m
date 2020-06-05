function HCC_Analyses(Data, Slice, SliceRange, X, Y, D, Phases, Case, unigmmclasses, multigmmclasses)

np = length(Phases);

%% Apply Gaussian smoothing technique
bw = 1.3;

for i = 1:np,   
    Datasm(i).img = smooth3(Data{Case,i}.img,'gaussian',ceil(5*bw),bw); 
end

for i = 1:np,
    lesion(i).sm = Datasm(i).img((X(i)-D):(X(i)+D),(Y(i)-D):(Y(i)+D),Slice(i));
end

figure(1),clf
for i = 1:np,
    subplot(2,2,i);
    img = lesion(i).sm;
    imagesc(img'),axis equal tight,colorbar
    title(Phases{i})
end

figure(2), clf
for i = 1:np,
    subplot(2,2,i)
    %hist(double(lesion(i).sm(:)),40)
    ksdensity(double(lesion(i).sm(:)))
    title(Phases{i})
end

%% Univariate Statistical Analysis
% Fit univariate Gaussian Mixture Model after smoothing
img0 = lesion(1).sm;
szvol = size(img0(:));
sz = size(img0');

% K-means as initial value
kmfit = cell(np,1);
rng(2016)
for i = 1:np,
    kmfit{i} = kmeans(lesion(i).sm(:),unigmmclasses(i),'MaxIter',1000,'Distance','sqeuclidean');
end

% GMM
uniGMM = cell(np,1);
gamma = cell(np,1);
uniind = cell(np,1);
for i = 1:np,
    %gamma{i} = cell(unigmmclasses(i),1);
    gamma{i} = zeros([szvol(1) unigmmclasses(i)]);
end


for i = 1:np,
    uniGMM{i} = fitgmdist(lesion(i).sm(:), unigmmclasses(i), 'Start', kmfit{i});
    for n = 1:unigmmclasses(i),
        mu(n) = uniGMM{i}.mu(n);
        sigma(n) = sqrt(uniGMM{i}.Sigma(1,1,n));
        weight(n) = uniGMM{i}.PComponents(n);
    end
    denom = 0;
    for j = 1:unigmmclasses(i),
        denom = denom + weight(j)*normpdf(lesion(i).sm(:),mu(j),sigma(j));
    end
    for k = 1:unigmmclasses(i),
        gamma{i}(:,k) = weight(k)*normpdf(lesion(i).sm(:),mu(k),sigma(k))./denom;
    end
    [gammax ind0] = max(gamma{i}, [], 2);
    uniind{i} = ind0;
end


figure(3), clf
for i =1:np,
    subplot(2,2,i)
    img = reshape(uniind{i},sz);
    imagesc(img(:,:)', [1 unigmmclasses(i)]), axis equal tight,colorbar  
    title(Phases{i})
end

% Feature
uniGMM = uniGMM;
uniind = uniind;

phase = 2;
HCC_Feature(phase,uniGMM,uniind)
phase = 3;
HCC_Feature(phase,uniGMM,uniind)
phase = 4;
HCC_Feature(phase,uniGMM,uniind)



% Volume and Diameter - PCA
phase = 2;
[Meantumor, indtumor] = max(uniGMM{phase}.mu);
tumor = reshape(uniind{phase} == indtumor, sz);
[x y] = ind2sub(sz,find(tumor == 1));
scale = Data{Case,phase}.hdr.dime.pixdim(2:3);
xs = x*scale(1);
ys = y*scale(2);
volume = sum(tumor(:))*prod(scale)/100;
[coef score] = pca([xs ys]);
diam = range(score);
volume
diam

%% Multivariate Statistical Analysis

X = zeros([szvol(1) np]);
for i = 1:np,
    X(:,i) = lesion(i).sm(:);
end

% K-means for initial values
rng(2016)
multikmfit = kmeans(X,multigmmclasses,'MaxIter',1000,'Distance','sqeuclidean');
initial.value = multikmfit;

% GMM
multiGMM = fitgmdist(X,multigmmclasses,'Start',initial.value);

for n = 1:multigmmclasses,
    weight(n) = multiGMM.PComponents(n);
end

multigamma = zeros([szvol(1) multigmmclasses]);
multidenom = 0;
for j = 1:multigmmclasses,
    multidenom = multidenom + weight(j)*mvnpdf(X,multiGMM.mu(j,:),multiGMM.Sigma(:,:,j));
end
for k = 1:multigmmclasses,
    multigamma(:,k) = weight(k)*mvnpdf(X,multiGMM.mu(k,:),multiGMM.Sigma(:,:,k))./multidenom;
end
[gammamax ind] = max(multigamma, [], 2);


figure(4), clf
img = reshape(ind, sz);
imagesc(img(:,:)', [1 multigmmclasses]), axis image, colorbar
title('hard classification'), axis image

figure(5), clf
for n = 1:multigmmclasses,
    V(n,:) = diag(multiGMM.Sigma(:,:,n));
end
plot(multiGMM.mu')
hold on
plot(multiGMM.mu' + sqrt(V'),'--')
plot(multiGMM.mu' - sqrt(V'),'--')
hold off


end

