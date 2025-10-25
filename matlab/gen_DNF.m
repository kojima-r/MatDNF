%-*-Text-*-
% file = gen_DNF.m

function [rand_D rand_C] = gen_DNF(n,h_gen,d_size,c_max)
% REQUIRED: n >= c_max, h_gen >= d_size
% [D C] = gen_DNF(n,h_gen=10,d_size=10,c_max=5); x = C(find(D),:); simp_DNF(x)
% [D C] = gen_DNF(n,h_gen=10,d_size=3,c_max=5); x = C(find(D),:); simp_DNF(x)

% Generate a random DNF formula F in {a1...an} = (rand_D(1 x h_gen),rand_C(h_gen x 2n))
% with d_size disjuncts where each disjunct contains at most c_max literals
% a half of which is negative on average

rand_C = zeros(h_gen,2*n);
rand_D = zeros(1,h_gen);

for i=1:h_gen
   c_max0 = min(c_max,n);
   c_size = randi(c_max0,1); % conjuntion_size =< c_max0
   y = randperm(n,c_size);   % y = [y_1..y_c_size] in {1..n}
   w = rand(1,c_size)>0.5;
   p_y = find(w==1);         % postive literal
   n_y = find(w==0);         % negative literal
   x = [y(p_y) y(n_y)+n];
   rand_C(i,x) = 1;          % rand_C(i,:) = y_1&..&~y_c_size
endfor
z = randperm(h_gen,d_size);  % z = [z1..z_{d_size}] in {1..h_gen}
rand_D(1,z) = 1;             % rand_D(1,:) = z_1v..&z_{d_size}

endfunction
