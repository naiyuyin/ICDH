clear;clc;close all;
variances = readmatrix('results/hetero/ER2_d20_n1000/var_est_2.txt');
Var_gt = readmatrix('data/hetero/ER2_d20/graph/Var_2.txt');
lb = readmatrix('data/hetero/ER2_d20/graph/lb_2.txt');
ub = readmatrix('data/hetero/ER2_d20/graph/ub_2.txt');
lb_est = readmatrix('results/hetero/ER2_d20_n1000/A_nll_2.txt');
interval_est = readmatrix('results/hetero/ER2_d20_n1000/B_nll_2.txt');
ub_est = lb_est + interval_est;
% ub = lb + interval;


% for ind = 1:10
%     subplot(2, 5, ind)
%     histogram(Var_gt(:,ind), 25)
%     hold on
%     histogram(variances(:, ind), 25)
%     plot([lb(ind), lb(ind)], [0, 120], 'r','LineWidth', 2)
%     plot([ub(ind), ub(ind)], [0, 120], 'r','LineWidth', 2)
%     xlim([0, 5])
%     title("Node "+num2str(ind))
%     
% end

for ind = 1:20
    subplot(4,5,ind)
    plot(1:1000, Var_gt(1:1000,ind),'LineWidth', 2)
    hold on;
    plot(1:1000, variances(1:1000, ind), 'LineWidth',2)
    plot([1, 1000], [lb(ind), lb(ind)], 'r','LineWidth', 2)
%     hold on;
    plot([1, 1000], [ub(ind), ub(ind)], 'r','LineWidth', 2)
    plot([1, 1000], [lb_est(ind), lb_est(ind)], 'b','LineWidth', 2)
    plot([1, 1000], [ub_est(ind), ub_est(ind)], 'b','LineWidth', 2)
    ylim([0, 6])
    title("Node "+num2str(ind))
%     disp(corrcoef(Var_gt(:, ind), variances(:,ind)))
end
sgtitle("distribution parameters (upper bound and lower bound) versus samples")
% legend(["ground-truth var", "estimated var", "lower bound", "upper bound"])

%% check B
clear; clc; close all
B_1 = readmatrix('results/homo/ER2_d10_n1000/B_formulation1_nll_1.txt');
B0_1 = readmatrix('results/homo/ER2_d10_n1000/B0_formulation1_nll_1.txt');
variances = readmatrix('results/homo/ER2_d10_n1000/var_est_1.txt');

for ind = 1:10
    subplot(2,5,ind)
    plot(1:1000, variances(1:1000,ind),'LineWidth', 2)
    hold on;
    plot([1, 1000], [1, 1], 'r','LineWidth', 2)
    ylim([0, 2])
    title("Node "+num2str(ind))
end
sgtitle("Variances versus Samples")
legend(["estimated var", "ground-truth var"])

%% check B hetero
clear; clc; close all;
ind = 1;
B_1 = readmatrix("results/hetero_old/ER2_d10_n1000/B_formulation1_nll_"+num2str(ind)+".txt");
B0_1 = readmatrix("results/hetero_old/ER2_d10_n1000/B0_formulation1_nll_"+num2str(ind)+".txt");
variances = readmatrix("results/hetero_old/ER2_d10_n1000/var_est_"+num2str(ind)+".txt");
B_1_gt = readmatrix("data/hetero_old/ER2_d10/graph/B_"+num2str(ind)+".txt");
B0_1_gt = readmatrix("data/hetero_old/ER2_d10/graph/B0_"+num2str(ind)+".txt");
X = readmatrix("data/hetero_old/ER2_d10/data/X_"+num2str(ind)+".txt");
X = X(1:1000, :);
X_n = (X - mean(X, 1)) ./ std(X, 1);
Var_gt = exp(X_n * B_1_gt + ones([1000, 10]) .* B0_1_gt');
% Var_gt = readmatrix("data/hetero_old/ER2_d10/graph/Var_"+num2str(ind)+".txt");
% lb = readmatrix("data/hetero_old/ER2_d10/graph/lb_"+num2str(ind)+".txt");
% ub = readmatrix("data/hetero_old/ER2_d10/graph/ub_"+num2str(ind)+".txt");

for ind = 1:10
    subplot(2, 5, ind)
%     histogram(Var_gt(:,ind), 20)
    plot(1:1000, variances(1:1000,ind),'LineWidth', 2)
    hold on
    plot(1:1000, Var_gt(1:1000,ind),'LineWidth', 2)
%     histogram(variances(:, ind), 20)
%     plot([lb(ind), lb(ind)], [0, 120], 'r','LineWidth', 2)
%     plot([ub(ind), ub(ind)], [0, 120], 'r','LineWidth', 2)
%     xlim([0, 10])
    title("Node "+num2str(ind))   
end

disp(mean(B_1 .* B_1, "all"))
disp(mean(B0_1 .* B0_1, "all"))

%%

for ind = 1:10
    B_1 = readmatrix("results/hetero/ER2_d10_n1000/B_formulation1_nll_"+num2str(ind)+".txt");
    B0_1 = readmatrix("results/hetero/ER2_d10_n1000/B0_formulation1_nll_"+num2str(ind)+".txt");
    variances = readmatrix("results/hetero/ER2_d10_n1000/var_est_"+num2str(ind)+".txt");
%     Var_gt = readmatrix("data/hetero/ER2_d10/graph/Var_"+num2str(ind)+".txt");
%     lb = readmatrix("data/hetero/ER2_d10/graph/lb_"+num2str(ind)+".txt");
%     ub = readmatrix("data/hetero/ER2_d10/graph/ub_"+num2str(ind)+".txt");
    B_1_gt = readmatrix("data/hetero_old/ER2_d10/graph/B_"+num2str(ind)+".txt");
    B0_1_gt = readmatrix("data/hetero_old/ER2_d10/graph/B0_"+num2str(ind)+".txt");
    X = readmatrix("data/hetero_old/ER2_d10/data/X_"+num2str(ind)+".txt");
    X = X(1:1000, :);
    X_n = (X - mean(X, 1)) ./ std(X, 1);
    Var_gt = exp(X_n * B_1_gt + ones([1000, 10]) .* B0_1_gt');
    disp(ind)
   disp(mean(B_1 .* B_1, "all"))
   disp(mean(B0_1 .* B0_1, "all"))
end