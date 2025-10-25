%-*-Text-*-
% file = logi_equiv.m

function [equiv counter_examples] = logi_equiv(learned_DNF,I2_k,I1)

counter_examples = [];
xx = (learned_DNF*[I1;1-I1]==sum(learned_DNF,2));
yy = (sum(xx,1)>=1);
equiv = all(I2_k==yy);             %equive = 1 iff (I_in|= learned_DNF)=I_out
if ( !equiv ) zz = I2_k - yy; counter_examples = I1(:,find(zz)); endif

endfunction
