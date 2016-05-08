function [W, b] = hybrid(data, labels, valid_ratio)
    dim = size(data,2);
    
    t_posI = find(labels == 1);
    t_posI = t_posI(randperm(length(t_posI)));
    t_negI = find(labels == 0);
    t_negI = t_negI(randperm(length(t_negI)));

    split_p = floor(length(t_posI) * valid_ratio);
    split_n = floor(length(t_negI) * valid_ratio);
    validI = [t_posI((split_p + 1):end); t_negI((split_n + 1):end)];
    %trainI = [t_posI(1:split_p); t_negI(1:split_n)];

    valid_data = data(validI,:);
    valid_label = labels(validI);
    train_data = data(t_posI(1:split_p),:);
    mn = mean(data(t_negI(1:split_n), :));
    covMat = cov(data(t_negI(1:split_n), :));


    if sum(isnan(covMat))
        W =[];
        b =[];
        display('Failed computing covariance matrix');
        return
    end
    [Q,S]=eig(covMat);
    S=diag(S);
    S(abs(S)<0.00001)=0;
    S=sqrt(S);
    S=diag(S);
    M=S*Q';
    
    [C, beta, W, b] = runSeqTraining(train_data, M, mn, valid_data, valid_label);
    if isnan(C)
        W =[];
        b =[];
        display('Failed computing hybrid');
        return
    end
    
    mn = mean(data(labels == 0, :));
    covMat = cov(data(labels == 0, :));
    [Q,S] = eig(covMat);
    S = diag(S);
    S(abs(S) < 0.00001) = 0;
    S = sqrt(S);
    S = diag(S);
    M = S*Q';
    [W_f, ~] = hybrid_solve(data(labels == 1, :), M, mn, C, beta);
    W = W_f(1:dim);
    b = W_f(dim+1);
end

function [C_best, beta_best, W_best, b_best]=runSeqTraining(data, ...
                                        M, mn, valid_data, valid_label)
    C = logspace(-3, 3, 4);
    beta = linspace(0.1, 1.5, 5); 
    alpha = 0.5; % Pseudo EER
    
    C_best = NaN;
    beta_best = NaN;
    W_best = [];
    b_best = [];
    rec = 0;
    
    dim = size(data, 2);
    validNeg = find(valid_label == 0);
    validPos = find(valid_label == 1);

    for k=1:length(C)
        for j=1:length(beta)
			[W,~] = hybrid_solve(data, M, mn, C(k), beta(j));

			% calc recall as average between positive and negatives
			f = valid_data * W(1:dim);
			th = W(dim + 1);
			tn = sum(f(validNeg) < th) / length(validNeg);
			tp = sum(f(validPos) >= th) / length(validPos);
			cur_rec = (1 - alpha) * tp + alpha * tn;

            if (norm(W(1:dim)) > 0.0001 && cur_rec > rec)
				rec = cur_rec;
				W_best = W(1:dim);
                b_best= th;
				C_best = C(k);
				beta_best = beta(j);
            end
        end
    end
end

