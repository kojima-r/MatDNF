%-*-Text-*-
% file = readme_for_test.m

%% How to use:
%(1) Move to (Choose learning data) in this file and choose a target fucntion: {random function, parity, random-DNF}
%(2) Open Mat_DNF.m and choose one of {(DNF)<-default, (over-itr), (classi)} in the file.
%(3)
% > octave
% > clear all
% > readme_for_test
% > ...
% > learned_DNFn

%% When (DNF) is chosen in Mat_DNF.m:
% Mat_DNF(I1,I2_k,h,alpha,max_itr,max_try) learns a DNF=(D_th,C_th)
% from learning data {I1(n x l),I2_k(1 x l)} and returns [learned_DNF C D_k V_k_th]
% by minizing J2 w.r.t. D_k and C
%
%  J2 = (I2_k <> 1-min_1(V)) + (1-I2_k <> max_0(V)) + (1/2)*|| C.*(1-C) ||^2 + (1/2)*|| D.*(1-D) ||^2 => O(n*l)+O(h*2n)
%   V = D_k*M           (1 x h)*(h x l) = (1 x l): continuous truth values of n DNF=(C,D) by I1:(1 x l) => O(h*l)
%   M = 1-min_1(N)      (h x l): conti. truth values of h1 conjunctions by I1 => O(h*l)
%   N = C*[1-I1;I1]     (h x 2n)*(2n x l): continuous #false literal in h1 conjunctinons by I1 => O(h*2n*l)
%
% while optimally thresholding (C,D_k) to binary learned_DNF_th = (C_th,D_k_th) to minimize
% learning_error = |I2_k - I2_k_learned| where I2_k_learned = (D_k_th*(1-min_1(C_th*[1-I1;I1])))>=1
% learned_DNF = non_zero rows of C_th --> I2_k_learned = (sum(1-min_1(learned_DNF*[1-I1;I1])))>=1

%% When (classi) is chosen in Mat_DNF.m:
%   neither (C_th,D_k_th) nor learned_DNF is computed and
%   {C,D_k,V_k_th} is returned. As a classifier, for given input vectors I1, compute
%   I2_k_learned = V>=V_k_th where V = D_k*(1-min_1(C*[1-I1;I1]))

%%
% exact_acc_DNF    = prediction accuracy of the learned DNF (learned_DNF) measured by |whole domain data of target func| = 2^n
% exact_acc_class  = prediction accuracy of the learned classifier measured by |whole domain data of target func| = 2^n


clear;

%---------------------------
% Setting learning parameters

n = 5; alpha = 0.1;   max_try = 20; max_itr = 500; h = 1000;  Er_max = 0;
%n = 5; alpha = 0.1;   max_try = 20; max_itr = 500; h = 20;  Er_max = 0;

%dr = 1.0;            % dr = domain_ratio, (dr x 100)% of (I1,I2_k) used for learning
dr = 0.5;

i_max = 10;
%i_max = 100;



%---------------------------
% Leanring starts
I0 = all_bit_seq(n); l = 2^n;   %I0(n x 2^n)  = vectors of all possible bit strings of length n
I0 = I0(:,randperm(l,l));

arr_acc_classi = zeros(1,i_max); arr_acc_DNF = zeros(1,i_max);
arr_test_size = zeros(1,i_max);
arr_time = zeros(1,i_max);  arr_conseq = zeros(1,i_max); arr_equiv = zeros(1,i_max);

acc_unk = -1;

%---------------------------
% Repeat learning i_max times and measure exact_acc_DNF, exact_acc_class, prob(conseq), prob(equive)

for i=1:i_max

%% (Choose learning data)
%%-(random func.)
%--(no noise)(over-itr)
% I1 = I0(:,randperm(l,l)); I2_k = (rand(1,l)<0.5)*1.0;
%--(noise bits appended to input I1)
% I1 = [I0(:,randperm(l,l)); (rand(n,l)<0.5)*1.0];
% I2_k = (rand(1,l)<0.5)*1.0;
%--

%%--(n-parity func.)
%--(no noise)(over-itr)
%  I1 = I0(:,randperm(l,l)); I2_k = rem(sum(I1,1),2);
%--(noise bits appended to input I1)
%  I1 = [I0; rand(n,l)<0.5]; l = size(I1,2); I1 = I1(:,randperm(l,l));   %I1(2n x l)
%  I2_k = rem(sum(I1(1:n,:),1),2);
%--

%%--(random DNF)
   d_size = 3; h_gen = 10; c_max = 5; %|disjunct|=3, |conj|=<5
%--(no noise)
   I1 = I0(:,randperm(l,l));     %I1(n x l=2^n)
   [D0 C0] = gen_DNF(n,h_gen,d_size,c_max);  DNF0s = simp_DNF(C0(find(D0),:)); DNF0n = DNF0s(:,1:n)-DNF0s(:,n+1:2*n);
   I2_k = eval_DNF(D0,C0,I1);
%--(noise bits appended to input I1)
%  I1 = [I0; rand(n,l)<0.5]; l = size(I1,2); I1 = I1(:,randperm(l,l));   %I1(2n x l=2^n)
%  [D0 C0] = gen_DNF(n,h_gen,d_size,c_max);                              %random DNF0=(D0,C0)
%  I2_k = eval_DNF(D0,C0,I1(1:n,:));                                     %I2_k = DNF(I0)
%--

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
printf("%0.3f           %0.3f              %0.1f/%d            %0.3f     %0.3f    %0.3f ± %0.3f\n",
   mean(arr_acc_DNF), mean(arr_acc_classi),  mean(arr_test_size),size(I1,2),
   mean(arr_conseq), mean(arr_equiv), mean(arr_time), std(arr_time));fflush(stdout);
