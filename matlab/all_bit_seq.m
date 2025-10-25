function y = all_bit_seq(x)
  if (x==1) y=[0 1];
  else
    z = all_bit_seq(x-1); [s1 s2] = size(z);
    u = zeros(1,s2); v = ones(1,s2);
    y1 = [u;z]; y2 = [v;z];
    y = [y1 y2];
  endif
endfunction
