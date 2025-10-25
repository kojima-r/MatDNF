%-*-Text-*-
function generate_test_arrays(save_dir, i)
% Generate array CSVs for testing Python Mat_DNF implementations.

addpath("../")

%%- Default constants
n = 5; alpha = 0.1;   max_try = 20; max_itr = 500; h = 1000;  Er_max = 0;
dr = 0.5;

I0 = all_bit_seq(n); l = 2^n;   %I0(n x 2^n)  = vectors of all possible bit strings of length n
I0 = I0(:,randperm(l,l));

%%--(random DNF) (no noise)
d_size = 3; h_gen = 10; c_max = 5; %|disjunct|=3, |conj|=<5
I1 = I0(:,randperm(l,l));     %I1(n x l=2^n)
[D0 C0] = gen_DNF(n,h_gen,d_size,c_max);
DNF0s = simp_DNF(C0(find(D0),:));
csvwrite(sprintf("%s/c0_%d.csv", save_dir, i), C0);
csvwrite(sprintf("%s/d0_%d.csv", save_dir, i), D0);
csvwrite(sprintf("%s/dnf_%d.csv", save_dir, i), DNF0s);
I2_k = eval_DNF(D0,C0,I1);

%% Initializer for Mat_DNF
aa = 4;
C_init = (sqrt(aa/(h*2*n)))*randn(h,2*n) + 0.5;  %(h x 2*n)
D_k_init = (sqrt(aa/h))*randn(1,h)       + 0.5;  %(1 x h)

l2 = size(I1,2);
x = floor(l2*dr);
I1_dr = I1(:,1:x);
I2_k_dr = I2_k(:,1:x);

csvwrite(sprintf("%s/i1_%d.csv", save_dir, i), I1)
csvwrite(sprintf("%s/i2_k_%d.csv", save_dir, i), I2_k)
csvwrite(sprintf("%s/i1_dr_%d.csv", save_dir, i), I1_dr)
csvwrite(sprintf("%s/i2_k_dr_%d.csv", save_dir, i), I2_k_dr)
csvwrite(sprintf("%s/h_%d.csv", save_dir, i), h)
csvwrite(sprintf("%s/er_max_%d.csv", save_dir, i), Er_max)
csvwrite(sprintf("%s/alpha_%d.csv", save_dir, i), alpha)
csvwrite(sprintf("%s/max_itr_%d.csv", save_dir, i), max_itr)
csvwrite(sprintf("%s/max_try_%d.csv", save_dir, i), max_try)
csvwrite(sprintf("%s/c_init_%d.csv", save_dir, i), C_init)
csvwrite(sprintf("%s/d_k_init_%d.csv", save_dir, i), D_k_init)

[C D_k V_k_th learned_DNF] = Mat_DNF_deterministic(fold=i,I1_dr,I2_k_dr,h,Er_max,alpha,max_itr,max_try,C_init,D_k_init);

csvwrite(sprintf("%s/c_%d.csv", save_dir, i), C)
csvwrite(sprintf("%s/d_k_%d.csv", save_dir, i), D_k)
csvwrite(sprintf("%s/v_k_th_%d.csv", save_dir, i), V_k_th)
csvwrite(sprintf("%s/learned_dnf_%d.csv", save_dir, i), learned_DNF)

xV_k = D_k*(1-min(C*[1-I1;I1],1));  I2_k_learned = (xV_k>=V_k_th);
exact_acc_classi = 1.0 - sum(abs(I2_k - I2_k_learned))/l2;          % l2 = size(I1,2) = 2^n unless (double noise)
csvwrite(sprintf("%s/exact_acc_classi_%d.csv", save_dir, i), exact_acc_classi);

learned_DNFs = simp_DNF(learned_DNF); n2 = size(learned_DNFs,2);
learned_DNFn = learned_DNFs(:,1:n2/2)-learned_DNFs(:,n2/2+1:n2);
csvwrite(sprintf("%s/learned_dnf_n_%d.csv", save_dir, i), learned_DNFn);

xx = (learned_DNFs*[I1;1-I1]==sum(learned_DNFs,2));  %<={exact_acc_DNF,conseq,equive} may be incorrect (use learned_DNF)
I2_k_learned_B = (sum(xx,1)>=1);
zz = I2_k - I2_k_learned_B;
exact_acc_DNF = 1 - sum(abs(zz))/l2;                                % l2 = size(I1,2) = domain size
csvwrite(sprintf("%s/exact_acc_dnf_%d.csv", save_dir, i), exact_acc_DNF);

cnsq = logi_conseq(learned_DNFs,I2_k,I1);  eqv = logi_equiv(learned_DNFs,I2_k,I1);
csvwrite(sprintf("%s/cnsq_%d.csv", save_dir, i), cnsq);
csvwrite(sprintf("%s/eqv_%d.csv", save_dir, i), eqv);
end