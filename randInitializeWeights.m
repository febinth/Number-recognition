function W = randInitializeWeights(L_in, L_out)

W = zeros(L_out, 1 + L_in);
init_epsilon=0.12;
W = rand(L_out,L_in+1)*2*init_epsilon-init_epsilon;

end
