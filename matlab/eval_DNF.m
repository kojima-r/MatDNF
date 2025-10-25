%-*-Text-*-
% file = eval_DNF.m

function I_out = eval_DNF(D,C,I_in)

% D(1 x h),C(h x 2n): binary {0,1}
% I_in(n x l): l assignments over n variables
% I_out(1 x l): truth values of (D,C) by I_in

M = 1-min(C*[1-I_in;I_in],1);
I_out = ((D*M)>=1)*1.0;

endfunction
