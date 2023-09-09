function [metric,fea,An] =CalMetric(Z,truthF,nv,numClust)
Sum_Z = 0;

for iv = 1:nv
    Sum_Z = Sum_Z+Z{iv};
end
Sum_Z = (1/nv)*Sum_Z;
Sum_Z = (Sum_Z+Sum_Z')*0.5;
N = size(Sum_Z,1);
Dd = diag(sqrt(1./(sum(Sum_Z,1)+eps)));
An = Dd*Sum_Z*Dd;
An(isnan(An)) = 0;
An(isinf(An)) = 0;
An = (An + An')*0.5;
try
    [Fng, a] = eigs(An,numClust);
catch ME
    if (strcmpi(ME.identifier,'MATLAB:eigs:ARPACKroutineErrorMinus14'))
        opts.tol = 1e-3;
        [Fng, ~] = eigs(An,numClust,opts.tol);
    else
        rethrow(ME);
    end
end
Fng(isnan(Fng))=0;
Fng = Fng./repmat(sqrt(sum(Fng.^2,2))+eps,1,numClust);  %optional
fea = real(Fng);
pre_labels = kmeans(fea,numClust,'maxiter',1000,'replicates',50,'EmptyAction','singleton');
% kmeans(Fng,numClust,'maxiter',1000,'replicates',50,'EmptyAction','singleton');
metric = getFourMetrics(pre_labels,truthF);
end
