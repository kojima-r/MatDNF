%-*-Text-*-
%% Modified readme_for_test.m which reads E-MTAB-1908 data

addpath("../")

%---------------------------
% Setting learning parameters

% n = 5; alpha = 0.1;   max_try = 20; max_itr = 500; h = 1000;  Er_max = 0;
n = 5; alpha = 0.1;   max_try = 4; max_itr = 50; h = 1000;  Er_max = 0;

dr = 1.0;            % dr = domain_ratio, (dr x 100)% of (I1,I2_k) used for learning

% i_max = 10;
i_max = 4;


arr_acc_classi = zeros(1,i_max); arr_acc_DNF = zeros(1,i_max);
arr_test_size = zeros(1,i_max);
arr_time = zeros(1,i_max);  arr_conseq = zeros(1,i_max); arr_equiv = zeros(1,i_max);

acc_unk = -1;

%---------------------------
% Repeat learning i_max times and measure exact_acc_DNF, exact_acc_class, prob(conseq), prob(equive)

dnf_i = 1
dnfs = csvread("../../data/E-MTAB-1908/01_T.bin.csv");

I1 = dnfs(:, 1:end-1);
I2_k = dnfs(dnf_i, 2:end);

fprintf('%s\n', num2str(size(I1)));
fprintf('%s\n', num2str(size(I2_k)));


for i=1:i_max
   d_size = 3; h_gen = 10; c_max = 5; %|disjunct|=3, |conj|=<5
   [D0 C0] = gen_DNF(n,h_gen,d_size,c_max);
   DNF0s = simp_DNF(C0(find(D0),:));
   DNF0n = DNF0s(:,1:n)-DNF0s(:,n+1:2*n);

   l2 = size(I1,2);         % l2 = whole domain size = 2^n except (double noise)
   x = floor(l2*dr);        % x = learning data size
   I1_dr = I1(:,1:x);
   I2_k_dr = I2_k(:,1:x);
   I1_test = I1(:,x+1:l2);
   I2_k_test = I2_k(x+1:l2);
   arr_test_size(i) = size(I1_test,2);

   tic(); [C D_k V_k_th learned_DNF] = Mat_DNF(fold=i,I1_dr,I2_k_dr,h,Er_max,alpha,max_itr,max_try); elapsed_time = toc();
   arr_time(i) = elapsed_time;

   xV_k = D_k*(1-min(C*[1-I1;I1],1));  I2_k_learned = (xV_k>=V_k_th);
   exact_acc_classi = 1.0 - sum(abs(I2_k - I2_k_learned))/l2;          % l2 = size(I1,2) = 2^n unless (double noise)
   arr_acc_classi(i) = exact_acc_classi;


   learned_DNFs = simp_DNF(learned_DNF); n2 = size(learned_DNFs,2);
   learned_DNFn = learned_DNFs(:,1:n2/2)-learned_DNFs(:,n2/2+1:n2);

   xx = (learned_DNFs*[I1;1-I1]==sum(learned_DNFs,2));  %<={exact_acc_DNF,conseq,equive} may be incorrect (use learned_DNF)
   I2_k_learned_B = (sum(xx,1)>=1);
   zz = I2_k - I2_k_learned_B;
   exact_acc_DNF = 1 - sum(abs(zz))/l2;                                % l2 = size(I1,2) = domain size
   arr_acc_DNF(i) = exact_acc_DNF;

   cnsq = logi_conseq(learned_DNFs,I2_k,I1);  eqv = logi_equiv(learned_DNFs,I2_k,I1);
   arr_conseq(i) = cnsq; arr_equiv(i) = eqv;
endfor

printf("dr = %d, average over %d trials\n",dr,i_max);
printf("exact_acc_DNF   exact_acc_classi   |test_size|/|I1|   conseq    equiv    time(s)\n");
printf("%0.3f           %0.3f              %0.1f/%d            %0.3f     %0.3f    %0.6f ± %0.6f\n",
   mean(arr_acc_DNF), mean(arr_acc_classi),  mean(arr_test_size),size(I1,2),
   mean(arr_conseq), mean(arr_equiv), mean(arr_time), std(arr_time));fflush(stdout);
