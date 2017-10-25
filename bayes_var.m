clc
clear

% Load the data from csv file
fileName = 'Recent_toBeUsed.csv';
% Exclude first row of column Names and first 2 columns.
Data = csvread(fileName, 1, 3);
Data = Data(1:end-1, 1:end);
% 4 columns which are: _, _, _, _.
Yraw = [Data(:, 1), Data(:, 5), Data(:, 9), Data(:, 12)];

% Specification of Model:
constant = 1;        % 1: if you desire intercepts, 0: otherwise 
p = 3;               % Number of lags on dependent variables

% Let's have loop that takes 10 years of data and predicts
% for the eleventh year.

rmse_prior = zeros(4+1, 12);

for prior_value=1:4
prior = prior_value ; 

Yraw_pred_final = zeros(120, 4);

for l=0:9

% Now lets's take first 10 years of data and predict 
% the next years(12 months) params:

Y2 = Yraw(1+12*l:12*(l+10), 1:end);

% initial dimensions of dependent variable.
[T2, M] = size(Y2);

% Generate lagged Y matrix. This will be part of the X matrix
Ylag = mlag2(Y2, p); % Y is [T x M]. ylag is [T x (Mp)]
Ylag_comp = mlag2(Yraw, p);

% Now define matrix X which has all the R.H.S. variables (constant, lags of
% the dependent variable and exogenous regressors/dummies)
if constant
    X1 = [ones(T2-p,1) Ylag(p+1:T2,:)];
else
    X1 = Ylag(p+1:T2,:);
end

% Get size of final matrix X
[T11, K] = size(X1);

% Create the block diagonal matrix Z
Z1 = kron(eye(M),X1);  %% why did he do this!

Y1 = Y2(p+1:T2,:); % This is the final Y matrix used for the VAR

T = T2 - p;

% Change the Variable Names:
X = X1;
Y = Y1;
Z = Z1;

% From here There is No change in the Code:

%--------------------------------PRIORS------------------------------------
% First get Ordinary Least Squares (OLS) estimators
A_OLS = inv(X'*X)*(X'*Y); % This is the matrix of regression coefficients
a_OLS = A_OLS(:);         % This is the vector of coefficients, i.e. it holds
                          % that a_OLS = vec(A_OLS)
SSE = (Y - X*A_OLS)'*(Y - X*A_OLS);
SIGMA_OLS = SSE./(T-K);

%-----------------Prior hyperparameters for bvar model
% Define hyperparameters
if prior == 1 % Noninformtive
    % I guess there is nothing to specify in this case!
    % Posteriors depend on OLS quantities
elseif prior == 2 % Minnesota
    A_prior = 0*ones(K,M);
    
    % we need to have 1 at t-1 time coefficient for like variables.
    A_prior(2, 1) = 1;
    A_prior(3, 2) = 1;
    A_prior(4, 3) = 1;
    A_prior(5, 4) = 1;
    
    a_prior = A_prior(:);
    
    % Hyperparameters on the Minnesota variance of alpha
%     a_bar_1 = 0.5;
%     a_bar_2 = 0.5;
%     a_bar_3 = 10^2;
    
    a_bar_1 = 0.2;
    a_bar_2 = 0.14;
    a_bar_3 = 10^4;
    
    % Now get residual variances of univariate p-lag autoregressions. Here
    % we just run the AR(p) model on each equation, ignoring the constant
    % and exogenous variables (if they have been specified for the original
    % VAR model)
    sigma_sq = zeros(M,1); % vector to store residual variances
    for i = 1:M
        % Create lags of dependent variable in i-th equation
        Ylag_i = mlag2(Yraw(:,i),p);
        Ylag_i = Ylag_i(p+1:T2,:);
        % Dependent variable in i-th equation
        Y_i = Yraw(p+1:T2,i);
        % OLS estimates of i-th equation
        alpha_i = inv(Ylag_i'*Ylag_i)*(Ylag_i'*Y_i);
        sigma_sq(i,1) = (1./(T-p+1))*(Y_i - Ylag_i*alpha_i)'*(Y_i - Ylag_i*alpha_i);
    end
    
    % Now define prior hyperparameters.
    % Create an array of dimensions K x M, which will contain the K diagonal
    % elements of the covariance matrix, in each of the M equations.
    V_i = zeros(K, M);
    
    % index in each equation which are the own lags
    ind = zeros(M, p);
    for i=1:M
        ind(i,:) = constant+i:M:K;
    end
   for i = 1:M  % for each i-th equation
        for j = 1:K   % for each j-th RHS variable
            if constant==1
                if j==1
                    V_i(j,i) = a_bar_3*sigma_sq(i,1); % variance on constant                
                elseif find(j==ind(i,:))>0
                    V_i(j,i) = a_bar_1./(ceil((j-1)/M)^2); % variance on own lags           
                else
                    for kj=1:M
                        if find(j==ind(kj,:))>0
                            ll = kj;                   
                        end
                    end
                    V_i(j,i) = (a_bar_2*sigma_sq(i,1))./((ceil((j-1)/M)^2)*sigma_sq(ll,1));           
                end
            else
                if find(j==ind(i,:))>0
                    V_i(j,i) = a_bar_1./(ceil((j-1)/M)^2); % variance on own lags
                else
                    for kj=1:M
                        if find(j==ind(kj,:))>0
                            ll = kj;
                        end                        
                    end
                    V_i(j,i) = (a_bar_2*sigma_sq(i,1))./((ceil((j-1)/M)^2)*sigma_sq(ll,1));            
                end
            end
        end
    end

    
    % Now V is a diagonal matrix with diagonal elements the V_i
    V_prior = diag(V_i(:));  % this is the prior variance of the vector a  
    
    % SIGMA is equal to the OLS quantity
    SIGMA = SIGMA_OLS;
    
elseif prior == 3 % Normal-Wishart (nat conj)
    % Hyperparameters on a ~ N(a_prior, SIGMA x V_prior)
    A_prior = 0*ones(K,M);
    
     % we need to have 1 at t-1 time coefficient for like variables.
    A_prior(2, 1) = 1;
    A_prior(3, 2) = 1;
    A_prior(4, 3) = 1;
    A_prior(5, 4) = 1;
    
    a_prior = A_prior(:);
    V_prior = 10*eye(K);
    % Hyperparameters on inv(SIGMA) ~ W(v_prior,inv(S_prior))
    v_prior = M;
    S_prior = eye(M);
    inv_S_prior = inv(S_prior);
end
    
%============================ POSTERIORS ==================================
%==========================================================================
    
%--------- Posterior hyperparameters of ALPHA and SIGMA with Diffuse Prior
if prior == 1
    % Posterior of alpha|Data ~ Multi-T(kron(SSE,inv(X'X)),alpha_OLS,T-K)
    V_post = inv(X'*X);
    a_post = a_OLS;
    A_post = reshape(a_post,K,M);
    
    % posterior of SIGMA|Data ~ inv-Wishart(SSE,T-K)
    S_post = SSE;
    v_post = T-K;
    
    % Now get the mean and variance of the Multi-t marginal posterior of alpha
    alpha_mean = a_post;
    alpha_var = (1/(v_post - M - 1))*kron(S_post,V_post);
    
%--------- Posterior hyperparameters of ALPHA and SIGMA with Minnesota Prior
elseif prior == 2
    % ******Get all the required quantities for the posteriors       
    V_post = inv( inv(V_prior) + kron(inv(SIGMA),X'*X) );
    a_post = V_post * ( inv(V_prior)*a_prior + kron(inv(SIGMA),X'*X)*a_OLS );
    A_post = reshape(a_post,K,M);
     
    % In this case, the mean is a_post and the variance is V_post
   alpha_mean=a_post;
%--------- Posterior hyperparameters of ALPHA and SIGMA with Normal-Wishart Prior
elseif prior == 3
    % ******Get all the required quantities for the posteriors       
    % For alpha
    V_post = inv( inv(V_prior) + X'*X );
    A_post = V_post * ( inv(V_prior)*A_prior + X'*X*A_OLS );
    a_post = A_post(:);
    
    % For SIGMA
    S_post = SSE + S_prior + A_OLS'*X'*X*A_OLS + A_prior'*inv(V_prior)*A_prior - A_post'*( inv(V_prior) + X'*X )*A_post;
    v_post = T + v_prior;
    
    % Now get the mean and variance of the Multi-t marginal posterior of alpha
    alpha_mean = a_post;
    alpha_var = (1/(v_post - M - 1))*kron(S_post,V_post); 

elseif prior == 4
    Apost = A_OLS;
end

%======================= PREDICTIVE INFERENCE =============================
%==========================================================================

% Now let's predict the Next Month economic indices.

Pred_mean = zeros(12, 4);

last = Y(T, :);
Yraw_pred = zeros(size(Yraw));
Yraw_pred(1+12*l:12*(l+10), 1:end) = Yraw(1+12*l:12*(l+10), 1:end);

for i=0:11

X_tplus1 = [1 reshape(transpose(Yraw_pred(12*(l+10)+i:-1:12*(l+10)+i-p+1, 1:end)), 1, M*p)];

% As a point forecast use predictive mean 

Pred_mean = X_tplus1*A_post;

Yraw_pred(12*(l+10)+i+1, 1:end) = Pred_mean;

end

Yraw_pred_final(12*l+1:12*(l+1), 1:end) = Yraw_pred(12*(l+10)+1:12*(l+11), 1:end);

end

est_Result = Yraw_pred_final;
actual_Result = Yraw(121:end, 1:end);

% plot them
% plot(est_Result(1:end, 1))
% hold on
% plot(actual_Result(1:end, 1))
% legend('Predicted', 'True Value')
% title('Log e')

% Now let's convert log_e to e.
est_Result_exp = exp(est_Result);
actual_Result_exp = exp(actual_Result);

% let's find RMSE b/w actual and predicted values.
rmse = zeros(1, 12);
for m=1:12
    for n=1:10
        rmse(1, m) = rmse(1, m) + power(est_Result_exp(12*(n-1)+m, 1)-actual_Result_exp(12*(n-1)+m, 1), 2);
    end
    rmse(1, m) = sqrt(rmse(1, m)/10);
end

%figure
%plot(rmse)
%title('rmse with months')

rmse_prior(prior_value, 1:end) = rmse;

end

% Random walk rmse:
rmse = zeros(1, 12);
for m=1:12
    for n=1:10
        rmse(1, m) = rmse(1, m) + power(exp(Yraw(12*(n-1)+120, 1))-actual_Result_exp(12*(n-1)+m, 1), 2);
    end
    rmse(1, m) = sqrt(rmse(1, m)/10);
end
rmse_prior(5, 1:end) = rmse;

figure

for k=1:5
%subplot(5,1,k)  
plot(rmse_prior(k, 1:end))
hold on
end
legend('NonInfo','Minnesota','Normal-Wishart Prior','OLS','RandWalk')

% Print some results
disp('The mean of alpha is in the vector alpha_mean')
disp('Its variance is in the vector alpha_var')
disp('Point forecast is Pred_mean')