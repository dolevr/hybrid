function [W, done] = hybrid_solve(data, M, mn, C, beta)
 
done=1;
numim = size(data,1);
numBins = size(data,2);
cvx_begin quiet
    variables w(numBins) b e(numim);
    minimize (w'*w+C*sum(e))
    subject to     
        data*w-b>=1-e;
        e>=0;
        beta*norm(M*w)+mn*w-b<=0;
cvx_end      

switch(cvx_status) 
    case 'Infeasible'
        display('hybrid failed');
        done=0;
end

W=[w;b;e];