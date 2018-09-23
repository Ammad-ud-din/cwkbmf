%Muhammad Ammad-ud-din
%muhammad.ammad-ud-din@aalto.fi

function state = cwkbmf_1mkl1mkl_regression_variational_train(Kx, Kz, Y, parameters)
    rand('state', parameters.seed); %#ok<RAND>
    randn('state', parameters.seed); %#ok<RAND>

    Dx = size(Kx, 1);
    Nx = size(Kx, 2);
    Px = size(Kx, 3);
    Dz = size(Kz, 1);
    Nz = size(Kz, 2);
    Pz = size(Kz, 3);
    R = parameters.R;
    sigmag = parameters.sigmag;
    sigmah = parameters.sigmah;
    sigmay = parameters.sigmay;

    Lambdax.shape = (parameters.alpha_lambda + 0.5) * ones(Dx, R);
    Lambdax.scale = parameters.beta_lambda * ones(Dx, R);
    Ax.mean = randn(Dx, R);
    Ax.covariance = repmat(eye(Dx, Dx), [1, 1, R]);
    Gx.mean = randn(R, Nx, Px);
    Gx.covariance = repmat(eye(R, R), [1, 1, Px]);
    etax.shape = (parameters.alpha_eta + 0.5) * ones(Px, 1);
    etax.scale = parameters.beta_eta * ones(Px, R);
    ex.mean = ones(Px, R); 
    ex.covariance = repmat(eye(Px, Px), [1, 1, R]); 
    Hx.mean = randn(R, Nx);
    Hx.covariance = repmat(eye(R, R), [1, 1, Nx]);

    Lambdaz.shape = (parameters.alpha_lambda + 0.5) * ones(Dz, R);
    Lambdaz.scale = parameters.beta_lambda * ones(Dz, R);
    Az.mean = randn(Dz, R);
    Az.covariance = repmat(eye(Dz, Dz), [1, 1, R]);
    Gz.mean = randn(R, Nz, Pz);
    Gz.covariance = repmat(eye(R, R), [1, 1, Pz]);
    etaz.shape = (parameters.alpha_eta + 0.5) * ones(Pz, 1);
    
     
    etaz.scale = parameters.beta_eta * ones(Pz, R); 
    ez.mean = ones(Pz, R); 
    ez.covariance = repmat(eye(Pz, Pz), [1, 1, R]); 
    
    Hz.mean = randn(R, Nz); 
    Hz.covariance = repmat(eye(R, R), [1, 1, Nz]);

    KxKx = zeros(Dx, Dx);
    for m = 1:Px
        KxKx = KxKx + Kx(:, :, m) * Kx(:, :, m)';
    end
    Kx = reshape(Kx, [Dx, Nx * Px]);

    KzKz = zeros(Dz, Dz);
    for n = 1:Pz
        KzKz = KzKz + Kz(:, :, n) * Kz(:, :, n)';
    end
    Kz = reshape(Kz, [Dz, Nz * Pz]);

    lambdax_indices = repmat(logical(eye(Dx, Dx)), [1, 1, R]);
    lambdaz_indices = repmat(logical(eye(Dz, Dz)), [1, 1, R]);

    for iter = 1:parameters.iteration
        if parameters.progress==1
            if mod(iter, 1) == 0
                fprintf(1, '.');
            end
            if mod(iter, 10) == 0
                fprintf(1, ' %5d\n', iter);
            end
        end

        %%%% update Lambdax
        Lambdax.scale = 1 ./ (1 / parameters.beta_lambda + 0.5 * (Ax.mean.^2 + reshape(Ax.covariance(lambdax_indices), Dx, R)));
        %%%% update Ax
        for s = 1:R
            Ax.covariance(:, :, s) = (diag(Lambdax.shape(:, s) .* Lambdax.scale(:, s)) + KxKx / sigmag^2) \ eye(Dx, Dx);
            Ax.mean(:, s) = Ax.covariance(:, :, s) * (Kx * reshape(squeeze(Gx.mean(s, :, :)), Nx * Px, 1) / sigmag^2);
        end
        %%%% update Gx
        for m = 1:Px
            Gx.covariance(:, :, m) = (eye(R, R) / sigmag^2 + diag( ex.mean(m,:) .* ex.mean(m,:) + permute(ex.covariance(m, m,:),[3 2 1])' ) / sigmah^2) \ eye(R, R); %DONE %DEBUG 
            Gx.mean(:, :, m) = Ax.mean' * Kx(:, (m - 1) * Nx + 1:m * Nx) / sigmag^2 + (repmat(ex.mean(m,:)', [1 Nx]) .* Hx.mean) / sigmah^2; %DONE
            for o = [1:m - 1, m + 1:Px]
                Gx.mean(:, :, m) = Gx.mean(:, :, m) - (repmat((ex.mean(m,:) .* ex.mean(o,:) + permute(ex.covariance(m, o,:),[3 2 1])')',[1 Nx]) .* Gx.mean(:, :, o)) / sigmah^2; %DONE
            end
            Gx.mean(:, :, m) = Gx.covariance(:, :, m) * Gx.mean(:, :, m);
        end
        %%%% update etax
        for r = 1:R
        	etax.scale(:,r) = 1 ./ (1 / parameters.beta_eta + 0.5 * (ex.mean(:,r).^2 + diag(ex.covariance(:,:,r)))); %DONE
        end
        %%%% update ex
        for r = 1:R
        	ex.covariance(:,:,r) = diag(etax.shape .* etax.scale(:,r)); %DONE %DEBUG 
        end
        for r = 1:R
		for m = 1:Px
		    for o = 1:Px
		        ex.covariance(m, o, r) = ex.covariance(m, o, r) + (sum(sum(Gx.mean(r, :, m) .* Gx.mean(r, :, o))) + (m == o) * Nx * Gx.covariance(r, r, m)) / sigmah^2; %DONE
		    end
		end
        	ex.covariance(:, :, r) = ex.covariance(:, :, r) \ eye(Px, Px); %DONE
        end
        for r = 1:R                	
		for m = 1:Px
		    ex.mean(m,r) = sum(Gx.mean(r, :, m) .* Hx.mean(r,:) ) / sigmah^2; %DONE
		end
        end
        for r = 1:R
        	ex.mean(:,r) = ex.covariance(:, :, r) * ex.mean(:,r); %DONE
        end
        
        %%move sign from e to G
        for r = 1:R
             for m = 1:Px
                sn = sign(ex.mean(m,r)); 
                if sn == -1
                    Gx.mean(r, :, m) = Gx.mean(r, :, m)*sn;
                    ex.mean(m,r) = ex.mean(m,r)*sn;
                end
             end
        end
        %%%% update Hx
        for i = 1:Nx
            indices = ~isnan(Y(i, :));
            Hx.covariance(:, :, i) = (eye(R, R) / sigmah^2 + (Hz.mean(:, indices) * Hz.mean(:, indices)' + sum(Hz.covariance(:, :, indices), 3)) / sigmay^2) \ eye(R, R);
            Hx.mean(:, i) = Hz.mean(:, indices) * Y(i, indices)' / sigmay^2;
            for m = 1:Px
                Hx.mean(:, i) = Hx.mean(:, i) + (ex.mean(m,:)' .* Gx.mean(:, i, m)) / sigmah^2;  %DONE
            end
            Hx.mean(:, i) = Hx.covariance(:, :, i) * Hx.mean(:, i);
        end

        %%%% update Lambdaz 
        Lambdaz.scale = 1 ./ (1 / parameters.beta_lambda + 0.5 * (Az.mean.^2 + reshape(Az.covariance(lambdaz_indices), Dz, R)));
        %%%% update Az 
        for s = 1:R
            Az.covariance(:, :, s) = (diag(Lambdaz.shape(:, s) .* Lambdaz.scale(:, s)) + KzKz / sigmag^2) \ eye(Dz, Dz);
            Az.mean(:, s) = Az.covariance(:, :, s) * (Kz * reshape(squeeze(Gz.mean(s, :, :)), Nz * Pz, 1) / sigmag^2);
        end
        %%%% update Gz 
        for m = 1:Pz
            Gz.covariance(:, :, m) = (eye(R, R) / sigmag^2 + diag( ez.mean(m,:) .* ez.mean(m,:) + permute(ez.covariance(m, m,:),[3 2 1])' ) / sigmah^2) \ eye(R, R); %DONE %DEBUG 
            Gz.mean(:, :, m) = Az.mean' * Kz(:, (m - 1) * Nz + 1:m * Nz) / sigmag^2 + (repmat(ez.mean(m,:)', [1 Nz]) .* Hz.mean) / sigmah^2; %DONE
            for o = [1:m - 1, m + 1:Pz]
                Gz.mean(:, :, m) = Gz.mean(:, :, m) - (repmat((ez.mean(m,:) .* ez.mean(o,:) + permute(ez.covariance(m, o,:),[3 2 1])')',[1 Nz]) .* Gz.mean(:, :, o)) / sigmah^2; %DONE
            end
            Gz.mean(:, :, m) = Gz.covariance(:, :, m) * Gz.mean(:, :, m);
        end
        %%%% update etaz DONE
        %etaz.scale = 1 ./ (1 / parameters.beta_eta + 0.5 * (ez.mean.^2 + diag(ez.covariance)));
        for r = 1:R
        	etaz.scale(:,r) = 1 ./ (1 / parameters.beta_eta + 0.5 * (ez.mean(:,r).^2 + diag(ez.covariance(:,:,r)))); %DONE
        end
        for r = 1:R
        	ez.covariance(:,:,r) = diag(etaz.shape .* etaz.scale(:,r)); %DONE %DEBUG 
        end
        for r = 1:R
		for m = 1:Pz
		    for o = 1:Pz
		        ez.covariance(m, o, r) = ez.covariance(m, o, r) + (sum(sum(Gz.mean(r, :, m) .* Gz.mean(r, :, o))) + (m == o) * Nz * Gz.covariance(r, r, m)) / sigmah^2; %DONE
		    end
		end
        	ez.covariance(:, :, r) = ez.covariance(:, :, r) \ eye(Pz, Pz); %DONE
        end
        for r = 1:R                	
		for m = 1:Pz
		    ez.mean(m,r) = sum(Gz.mean(r, :, m) .* Hz.mean(r,:) ) / sigmah^2; %DONE
		end
        end
        for r = 1:R
        	ez.mean(:,r) = ez.covariance(:, :, r) * ez.mean(:,r); %DONE
        end
        
        %%move sign from e to G
        for r = 1:R
             for m = 1:Pz
                sn = sign(ez.mean(m,r)); 
                if sn == -1
                    Gz.mean(r, :, m) = Gz.mean(r, :, m)*sn;
                    ez.mean(m,r) = ez.mean(m,r)*sn;
                end
             end
        end

        for j = 1:Nz
            indices = ~isnan(Y(:, j));
            Hz.covariance(:, :, j) = (eye(R, R) / sigmah^2 + (Hx.mean(:, indices) * Hx.mean(:, indices)' + sum(Hx.covariance(:, :, indices), 3)) / sigmay^2) \ eye(R, R);
            Hz.mean(:, j) = Hx.mean(:, indices) * Y(indices,j) / sigmay^2;
            for m = 1:Pz
                Hz.mean(:, j) = Hz.mean(:, j) + (ez.mean(m,:)' .* Gz.mean(:, j, m)) / sigmah^2;  %DONE
            end
            Hz.mean(:, j) = Hz.covariance(:, :, j) * Hz.mean(:, j);
        end
        
    end

    state.Lambdax = Lambdax;
    state.Ax = Ax;
    state.etax = etax;
    state.ex = ex;
    state.Lambdaz = Lambdaz;
    state.Az = Az;
    state.etaz = etaz;
    state.ez = ez;
    state.Hx=Hx;
    state.Hz=Hz;
    state.Gx=Gx;
    state.Gz=Gz;
    state.parameters = parameters;
end
