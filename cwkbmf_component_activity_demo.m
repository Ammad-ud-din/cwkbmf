
function cwkbmf_component_activity_demo()

rand('state', 1606); %#ok<RAND>
randn('state', 1606); %#ok<RAND>
Px = 10;
Nx = 100; 
Pz = 3;
R = Pz;
Nz = 100; 

activityMat = zeros(Px,R);
seq = floor(linspace(2,Px,5));
activityMat(1,:) = 1;
activityMat(seq(1):seq(2),1) = 1;
activityMat((seq(2)+1):seq(3),2) = 1;
activityMat((seq(3)+1):seq(4),3) = 1;           


X = randn(Px, Nx);
Z = randn(Pz, Nz);

% generalized activations
Y = 0;
for pxi=1:Px
    for pzi=1:Pz
            if activityMat(pxi,pzi) == 1
                Y = Y + X(pxi, :)' * Z(pzi, :);
            end
    end
end


% add random noise
Y = Y  + std2(Y) * 0.05 * randn(Nx, Nz);
Y = Y/std2(Y);


Kx = zeros(Nx, Nx, Px);
for m = 1:Px
    Kx(:, :, m) = X(m, :)' * X(m, :);
end

Kz = zeros(Nz, Nz, Pz); %eye(Nz, Nz);
for n = 1:Pz
    Kz(:, :, n) = Z(n, :)' * Z(n, :);
end


parameters.alpha_lambda = 1;
parameters.beta_lambda = 1;            
if Px > 1 || Pz > 1
    parameters.alpha_eta = 1e-3;
    parameters.beta_eta = 1e+3;
end
parameters.iteration = 1000;
parameters.progress = 1;
parameters.R = R;
parameters.seed = 3030;
parameters.sigmag = 0.1;
if Px > 1 || Pz > 1
    parameters.sigmah = 0.1;
end
parameters.sigmay = 1;


state = cwkbmf_1mkl1mkl_regression_variational_train(Kx,Kz,Y,parameters);
prediction = cwkbmf_1mkl1mkl_regression_variational_test(Kx,Kz,state);

thresh = 0.67; 

state.ex.act = state.ex.mean;
for k=1:R
    mas.mea=mean(state.ex.act(:,k));
    mas.std=std(state.ex.act(:,k));
    state.ex.act(:,k)=(state.ex.act(:,k)-mas.mea)./mas.std;
    state.ex.act(:,k) = ((state.ex.act(:,k)) > thresh); 
end


state.ez.act = state.ez.mean;
for k=1:R
    mas.mea=mean(state.ez.act(:,k));
    mas.std=std(state.ez.act(:,k));
    state.ez.act(:,k)=(state.ez.act(:,k)-mas.mea)./mas.std;
    state.ez.act(:,k) = ((state.ez.act(:,k)) > thresh); 
end

end