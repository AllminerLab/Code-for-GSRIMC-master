% clear memory
% clear all
% clc
addpath('./data');
addpath('./twist');
addpath(genpath('./tensor_toolbox-v3.2.1'))

%% Select dataset
Dataname = ["100leaves","COIL20","Caltech101-7","Scene-15","NUSwide"];

lambda = [
    0.01,0.1,1;
    0.01,0.1,1;
    0.01,1,0.001;
    
    1e-3,1,100;
    1e-4,10,10;
    1e-3,1e-2,100;
    
    1e-2,0.1,100;
    1e-2,0.1,10;
    0.01,1000,0.001;
    
    1e-3,1e-3,10;
    1e-3,1e-3,1;
    1e-3,1e-2,10;
    
    0.0001,0.01,1;
    0.0001,0.01,0.01;
    0.0001,0.01,1;
    ];

% Fsize represents the number of cases for each missing percent. You could
% select it from 1 to 30. The result reported in paper is mean value of 30
% cases.
Fsize = 30;
% repeat number of kmeans
rep = 5;

%%
percentDel = [0.1,0.3,0.5];
%%
STD = cell(5,1);
for idata = 1:5
    load(char(Dataname(idata)));
    X = data;
    Y = labels;
    clear data labels;
    STD{idata} = zeros(3,4);
    for idel = 1:3
        Datafold = [char(Dataname(idata)),'_percentDel_',num2str(percentDel(idel)),'.mat'];
        Result{idata,idel} = zeros(1,4);
        load(Datafold);
        it = 1;
        Z1 = zeros(30,4);
        Z2 = zeros(30,4);
        S1 = zeros(30,4);
        S2 = zeros(30,4);
        for f = 1:Fsize
            ind_folds = folds{f};
            truthF = Y;
            numClust = length(unique(truthF));
            P = cell(length(X),1);
            missIdx = cell(length(X),1);
            existIdx = cell(length(X),1);
            nv = length(X);
            N = size(X{1},1);
            for iv = 1:nv
                X1 = X{iv}';
                X1 = NormalizeFea(X1,0);
                missIdx{iv} = find(ind_folds(:,iv) == 0);
                existIdx{iv} = find(ind_folds(:,iv) == 1);
                % ---------- 鍒濆KNN鍥炬瀯寤? ----------- %
                X1(:,missIdx{iv}) = [];
                
                options = []; 
                options.NeighborMode = 'KNN';
                options.k = 20;
                options.WeightMode = 'HeatKernel';
                P{iv} = full(constructW(X1',options));
            end
            
            
            max_iter = 120;
            miu =2;
            rho = 1.2;
            [Z,~,S,obj] = EMLIMC(P,existIdx,missIdx,lambda((idata-1)*3+idel,1),lambda((idata-1)*3+idel,2),lambda((idata-1)*3+idel,3),miu,rho,max_iter);
            Z_metric = zeros(rep,4);
            S_metric = zeros(rep,4);
            for ir = 1:rep
                Z_metric(ir,:) = CalMetric(Z,truthF,nv,numClust);
                S_metric(ir,:) = CalMetric(S,truthF,nv,numClust);
            end
            Z1(f,:) = mean(Z_metric);
            S1(f,:) = mean(S_metric);
            
            fprintf('For Z incomplete fold = %d,Data= %s,del = %.2f,ACC = %.4f,ARI = %.4f, NMI = %.4f, Purity=%.4f\n\n\n'...
            ,f,Dataname(idata),percentDel(idel),Z1(f,1),Z1(f,2),Z1(f,3),Z1(f,4));
        end
        Result{idata,idel} = mean(Z1);
    end
    
end

