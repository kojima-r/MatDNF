%-*-Text-*-
% file = simp_DNF

%-- simplification:
%%[ Logical case ]
%  ()v(a1 & a2)                           => true                  empty elimi.
%  (..a3 & ~a3..)v(a1 & a2)               => (a1 & a2)             anti-tautology elimi.
%  a1 v (~a1 & a2 & a4) v ...             => a1 v (a2 & a4) v ...  unit propa.
%  (a1 & a2 & a4) v (~a1 & a2 & a4) v ... => (a2 & a4) v ...       resolution
%  (a1 & a2 & a4) v (a2 & a4) v ...       => (a2 & a4) v ...       subsumption

%% [ Continuous case ]
%   V2_k = sum(1 - min_1(DNF*[1-V1;V1]))   : anti-tautology elimi. applicable
%   I2_k_learned_DNF = (V2_k>=1)


function DNFs = simp_DNF_annotated(DNF)
% DNF(r x 2n): r conjunctions

if (size(DNF,1) == 1) DNFs = DNF; return; endif

%DNF =  DNF(find(sum(DNF,2)>0),:);   % Don't remove zero-rows
fprintf('Starting shape: %s\n', num2str(size(DNF)))

% remove ..A&~A.. conjunction
n = floor(size(DNF,2)/2);
bb = DNF(:,1:n) + DNF(:,n+1:2*n);
no_A_notA = find(!any(bb'==2));     % list of rows in DNF w.o. 2
DNF = DNF(no_A_notA,:);
fprintf('Conjunction: %s\n', num2str(size(DNF)))

aa = DNF(:,1:n) - DNF(:,n+1:2*n);

%if 0
% unit propagation
do
  found = 0;
  for k=1:size(aa,1)
     b_k = aa(k,:);                % k-th monomial
     if (sum(abs(b_k)) != 1) continue; endif
     b1 = find(b_k != 0);          % unit clause b_k's {1,-1}-position (var = 1 or -1)
     c1 = (aa(:,b1) == -b_k(b1));  % c1(i,:) indicates if aa(i,:)'s {1,-1}-positons = -(b_k's {1,-1}-position)
     d1 = find(c1);
     if (length(d1) > 0)
        aa(d1,b1) = 0;             % resolve |d1| monomials with ~b_k

%       printf(" unit propagation: k=%d  |propagated|=%d\n",k,length(d1)); fflush(stdout);

        found = 1; break;
     endif
  endfor
until (found == 0)
DNFs = [(aa == 1) (aa == -1)];
%endif
fprintf('Unit propagation: %s\n', num2str(size(DNFs)))


%if 0
% resolution
% disp(aa)
do
  found = 0;
  for k=1:size(aa,1)
     bb = aa(k,:);        % monomial
     b0 = find(bb==0);    % bb's 0-positions
     b1 = find(bb!=0);    % bb's {1,-1}-positions
     c0 = (aa(:,b0)==bb(b0));  % c0(i,:) = [1..1] => aa(i,:)'s 0-positons = bb's 0-positions
     c1 = (aa(:,b1)==bb(b1));  % c1(i,:) indicates if aa(i,:)'s {1,-1}-positons = bb's {1,-1}-positions
     c2 = abs(aa(:,b1));       % c2(i,:) indicates bb's {1,-1}-positions by {0,1}
     nb1 = length(b1)-1;       % bb's complementary monomial differs from bb just by 1 lieteral
     % This is a bug? If c0 is only one column it reduces to just scalar
     %   d0 = (all(c0')' & all(c2')' & (sum(c1,2)==nb1));
     d0 = (all(c0, 2) & all(c2, 2) & (sum(c1,2)==nb1));  % CORRECTED
         % (all(c0')' & all(c2')')(i)=1 <=> bb's {1,0,-1}-positions = i-th row's {1,0,-1}-positions
     disp(k)
     disp(c1)
     fprintf("--\n")
   %   disp(b0)
   %   disp(b1)
   %   disp(all(c0')')
   disp(c0)
   %   disp(c2)
   %   disp(nb1)
   %   sum(c1,2) == nb1 & all(c2')' & all(c0')'
     fprintf("==\n")
     d1 = find(d0);
     if (length(d1) > 0)
        fprintf('+++')
        disp(k)
        disp(d1)
        del_r = d1(1); del_c = find(abs(aa(k,:)-aa(del_r,:)));
        aa(k,del_c) = 0;   % k-th row and del_r-th row are resolved upon the variable del_c
        fprintf('Delete [R]: %s\n', num2str(aa(del_r,:)))
        aa(del_r,:) = [];  % delete r-th row from aa and keep k-th row

%       printf(" resolution k=%d  del_r=%d  del_c=%d\n",k,del_r,del_c); fflush(stdout);

        found = 1; break;
     endif
  endfor
until (found == 0)
DNFs = [(aa == 1) (aa == -1)];
%endif
fprintf('Resolution: %s\n', num2str(size(DNFs)))


%if 0
% remove [0..0] row as it behaves as a false disjunct in a DNF
aa_2 = [(aa == 1) (aa == -1)];
zero_rows = find((sum(abs(aa_2),2) == 0));
len_0_row = length(zero_rows);
if ( len_0_row > 0 )
   aa_2(zero_rows,:) = [];

%  printf("   %d all_zero rows deleted\n",len_0_row); fflush(stdout);

endif
DNFs = aa_2 = unique(aa_2,"rows");
fprintf('Remove false disjunct: %s\n', num2str(size(DNFs)))
%endif


%if 0
% subsumption   <- very inefficient
do
  found = 0;
  m1 = size(aa_2,1);
  for k=1:m1
     b_k = aa_2(k,:);
     zz = all(((aa_2 - b_k)>=0)')';  % indicates rows of aa_2 subsumed by b_k
     if (sum(zz) > 1)                % b_k subsumes other conj.s  Av(A&B) <=> A
        zz(k) = 0; xx = find(zz);
        aa_2(xx,:) = [];             % remove subsumed rows
        found = 1;

%       printf("   subsumption: b_%d subsumed %d rows\n",k,length(xx)); fflush(stdout);

        break;
     endif
  endfor
until (found == 0)
DNFs = unique(aa_2,"rows");
%endif
fprintf('Subsumption: %s\n', num2str(size(DNFs)))

endfunction
