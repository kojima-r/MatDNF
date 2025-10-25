%-*-Text-*-
% file = logi_conseq.m

function [conseq counter_examples] = logi_conseq(learned_DNF,I_out,I_in)

% I_in(n x l): l=2^n complete assignments over n variables
% I_out(1 x l): truth values of the original DNF against I_in
% => (I_in I_out) = complete spec. of Boolean func., DNF

% If I_in_true |= learned_DNF, conseq = 1.
% O.w. conseq = 0 and couner_example = I_in_counter

counter_examples = [];
I_in_true = I_in(:,find(I_out == 1));
M = 1-min(learned_DNF*[1-I_in_true;I_in_true],1);
%D = ones(1,size(learned_DNF,1));
%I_out_true = (D*M)>=1;
I_out_true = (sum(M,1))>=1;
conseq = all(I_out_true);     %conseq = 1 iff I_in_true |= learned_DNF
if ( !conseq ) counter_examples = I_in_true(:,find(I_out_true == 0)); endif

endfunction
