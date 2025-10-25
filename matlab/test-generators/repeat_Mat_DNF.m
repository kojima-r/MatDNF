%-*-Text-*-
function repeat_Mat_DNF(root_dir, i)
% Repeat deterministic Mat_DNF using saved CSVs. For debugging Python implementation.

addpath("../")

I1 = csvread(sprintf("%s/i1_%d.csv", root_dir, i));
I1_dr = csvread(sprintf("%s/i1_dr_%d.csv", root_dir, i));
I2_k = csvread(sprintf("%s/i2_k_%d.csv", root_dir, i));
I2_k_dr = csvread(sprintf("%s/i2_k_dr_%d.csv", root_dir, i));
h = csvread(sprintf("%s/h_%d.csv", root_dir, i));
Er_max = csvread(sprintf("%s/er_max_%d.csv", root_dir, i));
alpha = csvread(sprintf("%s/alpha_%d.csv", root_dir, i));
max_itr = csvread(sprintf("%s/max_itr_%d.csv", root_dir, i));
max_try = csvread(sprintf("%s/max_try_%d.csv", root_dir, i));
C_init = csvread(sprintf("%s/c_init_%d.csv", root_dir, i));
D_k_init = csvread(sprintf("%s/d_k_init_%d.csv", root_dir, i));

l2 = size(I1,2);

[C D_k V_k_th learned_DNF] = Mat_DNF_deterministic(fold=i,I1_dr,I2_k_dr,h,Er_max,alpha,max_itr,max_try,C_init,D_k_init);

xV_k = D_k*(1-min(C*[1-I1;I1],1));  I2_k_learned = (xV_k>=V_k_th);
exact_acc_classi = 1.0 - sum(abs(I2_k - I2_k_learned))/l2;          % l2 = size(I1,2) = 2^n unless (double noise)

learned_DNFs = simp_DNF_annotated(learned_DNF); n2 = size(learned_DNFs,2);
learned_DNFn = learned_DNFs(:,1:n2/2)-learned_DNFs(:,n2/2+1:n2);

xx = (learned_DNFs*[I1;1-I1]==sum(learned_DNFs,2));  %<={exact_acc_DNF,conseq,equive} may be incorrect (use learned_DNF)
I2_k_learned_B = (sum(xx,1)>=1);
zz = I2_k - I2_k_learned_B;
exact_acc_DNF = 1 - sum(abs(zz))/l2;                                % l2 = size(I1,2) = domain size

cnsq = logi_conseq(learned_DNFs,I2_k,I1);  eqv = logi_equiv(learned_DNFs,I2_k,I1);
end