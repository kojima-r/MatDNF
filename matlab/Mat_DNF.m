%-*-Text-*-
% file = Mat_DNF.m  (= learn_DNF2.m)

function [C D_k V_k_th learned_DNF] = Mat_DNF(fold,I1,I2_k,h,Er_max,alpha,max_itr,max_try)
% Mat_DNF learns a DNF or classifier that maps I1 as closely as possible to I2_k and
% returns a DNF formula "learned_DNF" or classifier (C,D) with V_k_th
% from which the predicted output I2_k_learned for I1 is computed.

% Input:
%   I1(n x l) : 0-1 matrix of l data points in n variables to be classified
%   I2_k(1 x l) : 0-1 row vector representing target truth values corresponding to I1
%   h: the maximum number of conjunctions in a DNF
%   alpha: learning rate
%   max_itr: maximum number of iteration
%   max_try: maximum number of trials to learn a DNF, each with perturbation
% Output:
%   learned_DNF(h' x 2n):  0-1 matrix (h'=<h) representing a DNF that approximately gives I2_k when evaluated by I1
%                          I2_k_learned = ([1..1](1-min_1(learned_DNF*[1-I1;I1])))>=1  in {0,1}
%   C(h x 2n), D_k(1 x h): learned real matrices (see below)
%   V_k_th:                threshold for classification s.t.
%                          I2_k_learned = (V_k>=V_k_th) where V_k = D_k*(1-min_1(C*[1-I1;I1]))  in {0,1}

% How to use:
% First choose (DNF) or (classifier) in the program
%  (DNF) -> comment out program code between (classifier)s
%           set extra_itr for (over-itr), extra_itr = 0 by default
%  (classifier) -> comment out program code between (DNF)s
%
%--(DNF):
% We learn a DNF=(C:conjunctions,D_k:disjuntion) from (I1,I2_k) by first minimizing J2:
%
%   J2 = (I2_k <> 1-min_1(V_k)) + (1-I2_k <> max_0(V_k)) + (l2/2)*|| C.*(1-C) ||^2 + (l2/2)*|| D.*(1-D) ||^2 => O(n*l)+O(h*2n)
%
%   N = C*[1-I1;I1]     (h x 2n)*(2n x l) = (h x l): continuous #false literal in h1 conjunctinons by I1 => O(h*2n*l)
%   M = 1-min_1(N)      (h x l): continuous truth values of h conjunctions by I1 => O(h*l)
%   V_k = D_k*M         (1 x h)*(h x l) = (1 x l): continuous truth values of DNF=(C,D_k) by I1:(n x l) => O(1*h*l)
%
% and then thresholding (C,D_k) to leaned_DNF giving a miminum classification error Er_k_th = |I2_k-I2_k_learned| where
%   V_k = sum(1-min_1(leaned_DNF*[1-I1;I1]),1)
%   I2_k_learned = (V_k>=1)
%       where learned_DNF is a matrix [C1;..;Cm] representing an m-conjunction DNF = C1 v..v Cm
%
%--(classifier):
% We learn a classifier (C,D_k) from (I1,I2_k) by minimizing J2 while computing threshold V_k_th
% giving miminum classification error Er_k = |I2_k-I2_k_learned| where
%
%   V_k = D_k*(1-min(C*[1-I1;I1],1))
%   I2_k_learned = (V_k>=V_k_th)


%%[Start learning]
[n l] = size(I1);
[m l] = size(I2_k);     % m=1
dI1 = [I1;1-I1];
I1_d = [1-I1;I1];

J = 10000;
extra_itr =  0;         % (over-itr)
c_extra_itr = 0;        % count how many times E_kr_th=0 happens


%--------------------------
%[Initialize]:
   aa = 4;
   C = (sqrt(aa/(h*2*n)))*randn(h,2*n) + 0.5;  %(h x 2*n)
   D_k = (sqrt(aa/h))*randn(1,h)       + 0.5;  %(1 x h)

%--------------------------
%[loop and retry]
for i=1:max_try

%--------------------------
%[GD+Adam]:initialize
   m_adam_C = zeros(h,2*n);
   v_adam_C = zeros(h,2*n);
   m_adam_D_k = zeros(1,h);
   v_adam_D_k = zeros(1,h);

   alpha_adam = alpha;

   beta1_adam = 0.9;
   beta2_adam = 0.999;
   epsilon_adam = 1e-8;
   t_adam = 1;              % clock for GD+Adam


   for j=1:max_itr
%--------------------------
%[Compute layers]
      N = C*I1_d;          %(h x l), O(h*2n*l)   I1_d=[1-I1;I1]
      M = 1-min(N,1);      %(h x l), O(h*l)
      V_k = D_k*M;         %(1 x l), O(h*l) analogue disjunction output

      X = I2_k-min(V_k,1);
      Y = C.*(1-C);        %(h x l), O(h*2n*l)
      Z = D_k.*(1-D_k);

%--------------------------
%[Compute minimum classification error Er_k=|I2-(V_k>=V_k_th)|]
      split_V_k = 20;             % 20 notch thresholds for disjunction V_k=D_k*M

      error_V_k = zeros(1,split_V_k);
      ls_V_k = linspace(min(V_k),max(V_k),split_V_k);
      for s=1:split_V_k
         d = (V_k>=ls_V_k(s));   % 1 x l
         error_V_k(s) = sum(abs(I2_k-d));
      endfor
      [Er_k y] = min(error_V_k); % Er_k: minimum error by predicted I2_k(= V_k>=V_k_th)

      V_k_th = ls_V_k(y);
      Er_k_th = -1;


%if 0  %(DNF), skip when (classifier)
%--------------------------
%[Compute minimum approx. error Er_k_th by {D_k_th,C_th}]
      split_C   = 10;  % 10 notch thresholds for h conjuntions C
      split_D_k = 10;  % 10 notch thresholds for disjunction D_k

      if (J > 10) split_C = split_D_k = 2; endif

      error_CD = zeros(split_C,split_D_k);
      ls_C = linspace(min(min(C)),max(max(C)),split_C);
      ls_D_k = linspace(min(D_k),max(D_k),split_D_k);
      for s=1:split_D_k
         d = (D_k>=ls_D_k(s));         % 1 x h
         for t=1:split_C
            c = (C>=ls_C(t));          % h x 2n
            c_sum = sum(c,2);
            b = ((c*dI1)==c_sum);      % h x l
            e = ((d*b)>=1);            % 1 x l
            error_CD(t,s) = sum(abs(I2_k-e));
         endfor
      endfor
      [x y] = min(error_CD);  % error_CD(y(s),s) = minimum of error_CD(:,s)
      [Er_k_th w] = min(x);   % Er_k_th = x(w) = min(x) = min(min(error_CD)) = error_CD(y(w),w)

      C_th = (C>=ls_C(y(w)));        % (h x 2n) 0-1 Mat for conjuntion
      D_k_th = (D_k>=ls_D_k(w));     % (1 x h) 0-1 Mat for disjunction
%endif  %(DNF)(classifier)


if ( (Er_k_th <= Er_max) || (c_extra_itr > 0) ) c_extra_itr++; endif   %(over-itr)


%--------------------------
%[Compute J and Jacobian]

      l2 = 0.1;
%     l3 = 0;

      W = V_k;           %V_k = D_k*(1-min(C*[1-I1;I1],1))
      f = sum(dot(I2_k,1-min(W,1))) + sum(dot(1-I2_k,max(W,0)));
      r =  l2*0.5*(sum(dot(Y,Y)) + sum(dot(Z,Z)));
%     r2 = l3*0.5*(sum(dot(C,C)) + dot(D_k,D_k));

      J = f + r;    % O(h*2n*l)

      if (f == 0) V_k_th = 1; endif

     printf("trial=%d i=%d j=%d: (f=%0.3f  r=%0.3f)  Er_k=%d  Er_k_th=%d/%d  |V_k|=%0.2f  D_k:[%0.2f..%0.2f] C:[%0.2f .. %0.2f] c_extra_itr=%d\n",
         fold,i,j, f,r, Er_k, Er_k_th,length(I2_k),sum(abs(V_k)),max(D_k),min(D_k),max(max(C)),min(min(C)), c_extra_itr);fflush(stdout);
     % J:loss, Er_k:error by (classi), Er_k_th:error by (DNF)(over-itr), V_k:continuous truth values, c_extra_itr:count after erro=0


      if (Er_k_th <= Er_max)  break; endif                                    %(DNF)
%     if ( (Er_k_th <= Er_max) && (c_extra_itr >= extra_itr) ) break; endif   %(over-itr)
%     if (f == 0 || Er_k <= Er_max)  break; endif                             %(classifier)

      X = -(W<=1).*I2_k + (W>=0).*(1-I2_k);                                   %<- X(1 x l) is an integer matrix
      Ja_C = ( (- (N<=1)).*(D_k'*X) )*[1-I1;I1]' + l2*(1-2*C).*Y;    %N(h x l)=C*I1_d, D_k=continuous disjunction
      Ja_D_k = X*M'                              + l2*(1-2*D_k).*Z;


if 1 %SAM  (add steps for SAM)
%---------------------
% (https://qiita.com/omiita/items/f24e4f06ae89115d248e)
% "Sharpness-Aware Minimization for Efficiently Improving Generalization", Foret, P., Kleiner, A., Mobahi, H., Neyshabur, B. (2020)
% J = (I2_k <> 1-min_1(D*M)) + (1-I2_k <> max_0(D*M)) + r
% J_SAM = max_{e_C:|| e_C ||<rho, e_D_k:|| e_D_k ||<rho} J(C+e_C, D_k+e_D_k)
%       = J(C+e^C, D_k+e^D_k)  where  e^C = rho*Ja(C)/|| Ja(C) ||, e^D_k = rho*Ja(D_k)/|| Ja(D_k) ||
%       = f^ + r^
%          f^ = sum(dot(I2_k,1-min(V_k^,1))) + sum(dot(1-I2_k,max(V_k^,0)))
%          r^ = 0.5*l2*(|| C_hat.*(1-C_hat) ||^2 + || D_k_hat.*(1-D_k_hat) ||^2)
% Ja_SAM_C = Ja_C(C+e^_C,D_k+e^D_k)
% Ja_SAM_D_k = Ja_D_k(C+e^C,D_k+e^D_k)
% C_new = C - alpha*Ja_SAM_C
% D_k_new = D_k - alpha*Ja_SAM_D_k

      rho = 0.001;
      e_hat_C = ( rho/sqrt(sum(sumsq(Ja_C))) )*Ja_C;   e_hat_D_k = ( rho/sqrt(sum(sumsq(Ja_D_k))) )*Ja_D_k;
      C_hat = C + e_hat_C;                             D_k_hat = D_k + e_hat_D_k;
      N_hat = C_hat*[1-I1;I1];
      M_hat = 1 - min(N_hat,1);
      W_hat = V_k_hat = D_k_hat*M_hat;
      Y_hat = C_hat.*(1-C_hat);                        Z_hat = D_k_hat.*(1-D_k_hat);
%     f_hat = sum(dot(I2_k,1-min(V_k_hat,1))) + sum(dot(1-I2_k,max(V_k_hat,0)));
%     r_hat = 0.5*l2*(sum(dot(Y_hat,Y_hat)) + sum(dot(Z_hat,Z_hat)));
%     J_SAM = f_hat + r_hat;
      Ja_SAM_C =  ((- (N_hat<=1)).*(D_k_hat'*((- (W_hat<=1).*I2_k) + (W_hat>=0).*(1-I2_k)) ))*[1-I1;I1]'  + l2*(1-2*C_hat).*Y_hat;    %+ l3*C_hat;
      Ja_SAM_D_k = ((-(W_hat<=1).*I2_k) + (W_hat>=0).*(1-I2_k))*M_hat'                                    + l2*(1-2*D_k_hat).*Z_hat; %+ l3*D_k_hat;
      Ja_C = Ja_SAM_C;
      Ja_D_k = Ja_SAM_D_k;
%---------------------
endif %SAM


%--------------------------
%[Update by subgradient]
     [C   m_adam_C   v_adam_C  ] = adam_update(C,Ja_C,m_adam_C,v_adam_C,alpha_adam,beta1_adam,beta2_adam,epsilon_adam,t_adam);
     [D_k m_adam_D_k v_adam_D_k] = adam_update(D_k,Ja_D_k,m_adam_D_k,v_adam_D_k,alpha_adam,beta1_adam,beta2_adam,epsilon_adam,t_adam);
     t_adam += 1;

   endfor  %j=1:max_itr


   if (Er_k_th <= Er_max) break; endif                                    %(DNF)
%  if ((Er_k_th <= Er_max) && (c_extra_itr >= extra_itr)) break; endif    %(over-itr)
%  if (f == 0 || Er_k <= Er_max)  break; endif                            %(classifier)

   if (i == max_try) break; endif

%--------------------------
%[Perturbate]
   aa = 4;
   C0 = (sqrt(aa/(h*2*n)))*randn(h,2*n) + 0.5;     %(h x 2*n)
   C = 0.5*(C + C0);
   D0_k = (sqrt(aa/h))*randn(1,h) + 0.5;           %(1 x h)
   D_k = 0.5*(D_k + D0_k);

endfor  %i=1:max_try

%--------------------------
%[compute learned_DNF]

learned_DNF = 0;

%if 0   %(DNF)(over-itr), skip when (classifier)
   DNF_th = C_th(find(D_k_th),:);      %(h' x 2n): h'(=<h) disjuncts in n variables
   x = (sum((DNF_th(:,1:n)+DNF_th(:,n+1:2*n))==2, 2) == 0);
   no_A_not_A = find(x);               %rows of  non_A_notA in DNF2_th
   learned_DNF = DNF_th(no_A_not_A,:);
%endif  %(DNF)

endfunction
