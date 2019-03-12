function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


X = [ones(size(X,1),1),X];
m = size(X, 1);
         

J = 0;
regz=0; 	%regularization term
theta1mod=Theta1;
theta2mod=Theta2; 
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
b=1:10;

for i=1:m,
	a2=sigmoid(Theta1*X(i,:)');
	a2=[1;a2];
	a3=sigmoid(Theta2*a2);
	for k=1:num_labels,
		J = J+(-1.*(b==y(i))(k))*log(a3(k)) - (1.-(b==y(i))(k))*log(1.-(a3(k))); 
		%if y(1)=5, then y(1)(k=1,2,3,4,6,7,8,9)=0 and y(1)(k=5)=1;
	end;
end;

J=J/m;

theta1mod(:,1)=zeros(size(Theta1,1),1);
theta2mod(:,1)=zeros(size(Theta2,1),1);


for j=1:size(Theta1,1),
	for k=1:size(Theta1,2),
		regz=regz+((theta1mod(j,k))^2);
	end;
end;

for j=1:size(Theta2,1),
	for k=1:size(Theta2,2),
		regz=regz+((theta2mod(j,k))^2);
	end;
end;

regz=(lambda*regz)/(2*m);

J=J+regz;

DELTA1=zeros(size(Theta1));
D1=DELTA1;

DELTA2=zeros(size(Theta2));
D2=DELTA2;
c=1:10;
for t=1:m,
	A1=X(t,:)';
	z2=Theta1*A1;
	A2=sigmoid(z2);
	A2=[1;A2];
	z3=Theta2*A2;
	A3=sigmoid(z3);
	for k=1:num_labels,
		d3(k,1)=A3(k)-(c==y(t))(k);
		%d3=d3';
	end;
	%disp(size(Theta2(:,2:end)'));
	%disp(size(d3));
	d2=((Theta2(:,2:end)')*(d3)).*sigmoidGradient(z2);
	DELTA1=DELTA1+(d2*(A1'));
	D1=(1/m)*DELTA1+((lambda/m)*theta1mod);
	DELTA2=DELTA2+(d3*(A2)');
	D2=(1/m)*DELTA2+((lambda/m)*theta2mod);
end;
	
Theta1_grad=D1;
Theta2_grad=D2;
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
