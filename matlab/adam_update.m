%-*-Text-*-
% file = adam_update.m

function [New_X  New_m_X  New_v_X] = adam_update(X,Ja_X,m_X,v_X,alpha_adam,beta1_adam,beta2_adam,epsilon_adam,t_adam)

   New_m_X = beta1_adam*m_X + (1-beta1_adam)*Ja_X;
   New_v_X = beta2_adam*v_X + (1-beta2_adam)*(Ja_X.*Ja_X);
   m2_X = New_m_X/(1 - exp(t_adam*log(beta1_adam)));
   v2_X = New_v_X/(1 - exp(t_adam*log(beta2_adam)));
   New_X = X - alpha_adam*(m2_X./(sqrt(v2_X)+epsilon_adam));

endfunction
