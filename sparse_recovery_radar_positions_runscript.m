%% Run

%Given a simulated graph of received power vs. time, use sparse coding to
%find the likely locations of each pulse. 

scaling_flag = 0;

%Sample sinc and dictionary
d_max = 100;
fs = 1e10;
num_samples_base = 1000;
num_periods = 8;
t = -num_samples_base/2:num_samples_base/2-1;
t = (t / num_samples_base) * num_periods * 2;

%Create an example scene and test LASSO for recovery
%For now, consider returned power as a function of distance
%To simulate real life, downsample the returned pulse from a much higher
%resolution signal

%Test Parameters
str = 'sinc';
solver_flag = 4;
up_factor = 1000;
num_iterations = 100;
num_targets = 20;

x = input_select(str, t);
[dict, num_samples_dict] = generate_dict(x, fs, d_max);

%Find coefficients after sampling from test signal
%[error_src, error_c] = run_src_one_target(x, dict, num_iterations, up_factor, num_samples_base, num_samples_dict, num_periods, str, solver_flag)

%Find coefficients with multiple random targets
error_src = run_src_multiple_target(x, dict, num_targets, num_iterations, up_factor, num_samples_base, num_samples_dict, num_periods, scaling_flag, fs, str, solver_flag)

%% functions
%Run SRC on the test signal with random test bed
function error_src = run_src_multiple_target(x, dict, num_targets, num_iterations, up_factor, num_samples_base, num_samples_dict, num_periods, scaling_flag, fs, str, solver_flag)
    
    %Configure the upsampled test signal
    num_samples_up = num_samples_base * up_factor;
    t_up = -num_samples_up/2:num_samples_up/2-1;
    t_up = (t_up / num_samples_up) * num_periods * 2;
    x_up = input_select(str, t_up);
    x_up_appended = [x_up, zeros(1, up_factor*num_samples_dict - num_samples_up + up_factor * num_samples_base)];
    
    %Generate random targets
    d = randi(length(x_up_appended) - up_factor*num_samples_base, [num_iterations, num_targets]);
    received_sig = zeros(num_iterations, num_samples_dict + num_samples_base);

    %Downsample to get the received signals from those targets
    for i = 1:num_iterations
        for j = 1:num_targets
            delayed_upsampled_sig = circshift(x_up_appended, d(i, j));
            if scaling_flag
                received_sig(i, :) = received_sig(i, :) + downsample(delayed_upsampled_sig, up_factor) / (d(i, j) ./ (fs*up_factor) * 3E8).^2;
            else
                received_sig(i, :) = received_sig(i, :) + downsample(delayed_upsampled_sig, up_factor);
            end
        end
    end

    %Run LASSO for reconstruction
    coeff_src = src_eval(dict, received_sig, d, num_targets, solver_flag, 0);
    
    %Compare the accuracy of the correlation to the SRC
    [vals, ind] = sort(coeff_src, 'descend');
    inds_trunc = ind(1:2*num_targets, :) - 1;
    vals_trunc = vals(1:2*num_targets, :);
    
    for i = 1:num_iterations
        [c(:, i), ~] = xcorr(received_sig(i, :), x);
    end
    [vals_c, ind_c] = sort(c, 'descend');
    inds_trunc_c = ind_c(1:2*num_targets, :) - length(received_sig(1, :));
    vals_trunc_c = vals_c(1:2*num_targets, :);

    pred_samp_src = inds_trunc(1:num_targets, :);
    pred_samp_c = inds_trunc_c(1:num_targets, :);

    d_down = round(d/up_factor);
    for i = 1:num_iterations
        [~, m_ind(i, :)] = min(abs(pred_samp_src(:, i) - d_down(i, :)));
    end
    
    error_points = [];
    for i = 1:num_iterations
        error(i) = mean(abs(d_down(i, :) - pred_samp_src(m_ind(i, :), i)'));
        error_points = [error_points, abs(d_down(i, :) - pred_samp_src(m_ind(i, :), i)')];
    end
    error_src = mean(error);
    %error_c = mean(abs(d_down - pred_samp_c'));

    %pairs_src = sort_pairs(inds_trunc);
    %pairs_c = sort_pairs(inds_trunc_c);

end

function pairs = sort_pairs(inds)
k = 1;
pairs = [];
    for i = 1:size(inds, 1)
        ind = inds(i);
        if ismember(i, pairs) == 0
            for j = 1:size(inds, 1)
                if (ind == inds(j) + 1 || ind == inds(j) - 1) && i ~= j
                    pairs(k, :) = [i, j];
                    k = k + 1;
                    break;
                end
                if j == size(inds, 1)
                    pairs(k, :) = [i, 0];
                    k = k + 1;
                end
            end
        end
    end
end

%Run SRC on the test signal with random test bed
function [error_src, error_c] = run_src_one_target(x, dict, num_iterations, up_factor, num_samples_base, num_samples_dict, num_periods, str, solver_flag)
    
    %Configure the upsampled test signal
    num_samples_up = num_samples_base * up_factor;
    t_up = -num_samples_up/2:num_samples_up/2-1;
    t_up = (t_up / num_samples_up) * num_periods * 2;
    x_up = input_select(str, t_up);
    x_up_appended = [x_up, zeros(1, up_factor*num_samples_dict - num_samples_up + up_factor * num_samples_base)];
    
    %Generate random targets
    d = randi(length(x_up_appended) - up_factor*num_samples_base, [num_iterations, 1]);
    received_sig = zeros(num_iterations, num_samples_dict + num_samples_base);

    %Downsample to get the received signals from those targets
    for i = 1:num_iterations
        delayed_upsampled_sig(i, :) = circshift(x_up_appended, d(i));
        received_sig(i, :) = downsample(delayed_upsampled_sig(i, :), up_factor);
    end

    %Run LASSO for reconstruction
    coeff_src = src_eval(dict, received_sig, d, 2, solver_flag, 0);
    
    %Compare the accuracy of the correlation to the SRC
    [vals, ind] = sort(coeff_src, 'descend');
    inds_trunc = ind(1:5, :) - 1;
    vals_trunc = vals(1:5, :);
    
    for i = 1:num_iterations
        [c(:, i), ~] = xcorr(received_sig(i, :), x);
    end
    [vals_c, ind_c] = sort(c, 'descend');
    inds_trunc_c = ind_c(1:5, :) - length(received_sig(1, :));
    vals_trunc_c = vals_c(1:5, :);

    ref = xcorr(x);
    ref_max = max(ref);
    ref_list = sort(ref, 'descend');
    ref_step = ref_list(1) - ref_list(2);
    
    pred_samp_src = round(up_factor .* (inds_trunc(2, :) - inds_trunc(1, :)) .* vals_trunc(2, :)) + inds_trunc(1, :)*up_factor;
    pred_samp_c = round(up_factor .* (inds_trunc_c(2, :) - inds_trunc_c(1, :)) .* abs(vals_trunc_c(1, :) - ref_max) / ref_step) + inds_trunc_c(1, :)*up_factor;
 
    error_src = mean(abs(d - pred_samp_src'));
    error_c = mean(abs(d - pred_samp_c'));

end

%Given a signal, fs, and maximum distance (meters), make a dictionary
%If max distance is synchronized to pulse rate, set wrap_flag
function [dict, num_samples] = generate_dict(x, fs, d_max)
    max_time = 1/ (3E8 / (2*d_max));
    num_samples = floor(fs * max_time);
    sig_length = length(x);
    dict = zeros(num_samples, num_samples+sig_length);
    nonshifted_sig = zeros(1, num_samples+sig_length);
    nonshifted_sig(1:sig_length) = x;

    for i=0:num_samples-1
        dict(i+1, :) = circshift(nonshifted_sig,i);
    end
end

function x_hat = src_eval(dictionary, test_data, test_labels, num_targets, solver_flag, robust_flag)
    S = num_targets;

    dictionary = dictionary ./ vecnorm(dictionary, 2, 2);
    test_data = test_data ./ vecnorm(test_data, 2, 2);

    if robust_flag
        m = size(dictionary, 1);
        n = size(dictionary, 2);
        dictionary(m+1, :) = reshape(eye(96, 84), [1, n]);
        train_labels(m+1) = num_classes+1;
    end

    for i = 1:size(test_labels, 1)  
        switch solver_flag
            case 1
                B = lasso(dictionary', test_data(i, :), 'NumLambda', 10);
                x_hat(:, i) = B(:, 4);
            case 2
                [x_hat(:, i), ~] = SP_solver(dictionary', test_data(i,:)', S);
            case 3
                [x_hat(:, i), ~] = CoSaMP_solver(dictionary', test_data(i,:)', S);
            case 4
                [x_hat(:, i), ~] = OMP_solver(dictionary', test_data(i,:)', S);
        end
    end
end

%% Solvers
function [x, i] = OMP_solver(A, y, S)
    r = y;
    S_i = [];
    max_iterations = S;
    eps = 1E-6;
    x = zeros(size(A, 2), 1);

    for i = 1:max_iterations
        [~, I] = max(dot(repmat(r, 1, size(A,2)), A));
        max_ind = I(1);
        S_i = [S_i, max_ind];
        A_s = A(:, S_i);
        [x_hat, flag] = lsqr(A_s, y);
        r = y - A_s*x_hat;

        if norm(A_s*x_hat - y) < eps
            x(S_i) = x_hat;
            break
        end
    end
    x(S_i) = x_hat;
end

function [x, i] = SP_solver(A, y, S)
    r = y;
    max_iterations = 10*S;
    eps = 1E-6;
    A_s = [];
    S_i = [];
    x = zeros(size(A, 2), 1);

    for i = 1:max_iterations
        [~, ind] = sort(A'*r, 'descend');
        ind = ind(1:S);
        S_union_supp = sort([S_i; ind]);
        [x_hat, ind_p] = sort(pinv(A(:, S_union_supp))*y, 'descend');
        ind = S_union_supp(ind_p);
        x_hat = x_hat(1:S);
        S_i = ind(1:S);
        A_s = A(:, S_i);

        r = y - A_s*x_hat;

        if norm(A_s*x_hat - y) < eps
            x(ind(1:S)) = x_hat;
            break
        end
    end
    x(ind(1:S)) = x_hat;
end

function [x, i] = CoSaMP_solver(A, y, S)
    r = y;
    max_iterations = 10*S;
    eps = 1E-6;
    A_s = [];
    S_i = [];
    x = zeros(size(A, 2), 1);

    for i = 1:max_iterations
        [~, ind] = sort(A'*r, 'descend');
        ind = ind(1:2*S);
        S_union_supp = [S_i; ind];
        [x_hat, ind_p] = sort(pinv(A(:, S_union_supp))*y, 'descend');
        ind = S_union_supp(ind_p);
        x_hat = x_hat(1:S);
        S_i = ind(1:S);
        A_s = A(:, S_i);

        r = y - A_s*x_hat;

        if norm(A_s*x_hat - y) < eps
            x(ind(1:S)) = x_hat;
            break
        end
    end
    x(ind(1:S)) = x_hat;
end