function [Z,E,S,obj] = EMLIMC(B,existIdx,missIdx,lambda1,lambda2,lambda3,mu,rho,max_iter)
V = length(B);
N = size(existIdx{1},1)+size(missIdx{1},1);
obj = zeros(120,1);
for iv = 1:V
    E{iv} = zeros(N,N);
    S{iv} = zeros(N,N);
    A{iv} = zeros(N,N);
    Z{iv} = zeros(N,N);
    Q{iv} = zeros(N,N);
end
check = true;
for iter = 1:max_iter
    Z_pre = Z;
    E_pre = E;
    S_pre = S;
    %% Update S
    for iv = 1:V 
        S{iv}(missIdx{iv},missIdx{iv}) = 1/(mu) * (mu*(Z{iv}(missIdx{iv},missIdx{iv})+E{iv}(missIdx{iv},missIdx{iv}))-A{iv}(missIdx{iv},missIdx{iv}) );
        S{iv}(existIdx{iv},missIdx{iv}) = 1/(mu) * (mu*(Z{iv}(existIdx{iv},missIdx{iv})+E{iv}(existIdx{iv},missIdx{iv}))-A{iv}(existIdx{iv},missIdx{iv}));
        S{iv}(missIdx{iv},existIdx{iv}) = 1/(mu) * (mu*(Z{iv}(missIdx{iv},existIdx{iv})+E{iv}(missIdx{iv},existIdx{iv}))-A{iv}(missIdx{iv},existIdx{iv}));
        S{iv}(existIdx{iv},existIdx{iv}) = 1/(2*lambda2+mu) * (mu*(Z{iv}(existIdx{iv},existIdx{iv})+E{iv}(existIdx{iv},existIdx{iv}))-A{iv}(existIdx{iv},existIdx{iv}) +2*lambda2*B{iv});
        for ii = 1:N
            S{iv}(ii,:) = EProjSimplex_new(S{iv}(ii,:));
        end
    end 
    %% update Z
    
    Sum_Q = zeros(N,N);
    for iv = 1:V
       Sum_Q = Sum_Q + Z{iv};
    end
    for iv = 1:V
       Q{iv} = (Sum_Q - Z{iv})/(V-1);
    end
    S_tensor = cat(3, S{:,:});
    A_tensor = cat(3, A{:,:});
    E_tensor = cat(3, E{:,:});
    Q_tensor = cat(3, Q{:,:});
    Sv = S_tensor(:);
    Av = A_tensor(:);
    Ev = E_tensor(:);
    Qv = Q_tensor(:);
    Fv = (mu/(mu+2*lambda3))*(Sv-Ev) +(1/(mu+2*lambda3))*Av + (2*lambda3/(mu+2*lambda3))*Qv;
    tau = 2/(mu/2 + lambda3);
    [Zv] = wshrinkObj(Fv,tau,[N,N,V],0,3);
    Z_tensor = reshape(Zv, [N,N,V]);
    %% update E
    D_tensor = S_tensor - Z_tensor + 1/mu*A_tensor;
    
    alpha = lambda1/mu;
    ttemp = tenmat(D_tensor,3);
    D_norm = ttemp.data;
    D_norm = sqrt(sum(D_norm.^2));
    D_norm = max((D_norm-alpha)./D_norm,0);
    D_norm = reshape(D_norm,[N,N]);
    for iv = 1:V
    E_tensor(:,:,iv) = D_norm;
    end
    E_tensor = E_tensor.*D_tensor;
    

    
    %% upload A
    A_tensor = A_tensor + mu*(S_tensor-Z_tensor-E_tensor);
    for iv = 1:V
        Z{iv} = Z_tensor(:,:,iv);
        E{iv} = E_tensor(:,:,iv);
        A{iv} = A_tensor(:,:,iv);
    end
    mu = min(mu*rho, 1e10);
    %% convergent check
    diff_Z = 0;
    diff_E = 0;
    diff_S = 0;
    err1  = 0;
    for iv = 1:V
        tempNum = S{iv} - Z{iv}-E{iv};
        err1 = max(err1,max(abs(tempNum(:))));
        tempNum = (S_pre{iv} - S{iv});
        diff_S = max(diff_S,max(abs(tempNum(:))));
        tempNum = Z_pre{iv} - Z{iv};
        diff_Z = max(diff_Z,max(abs(tempNum(:))));
        tempNum = E_pre{iv} - E{iv};
        diff_E = max(diff_E,max(abs(tempNum(:))));
    end

    error = max(err1,diff_Z);
       fprintf('iter = %d, miu = %.3f, difS = %.d,difZ = %.d,difE = %.d, err = %.d\n'...
            , iter,mu,diff_S,diff_Z,diff_E,error);
    obj(iter) = error;
    if error < 1e-6
        break;
    end
end

end