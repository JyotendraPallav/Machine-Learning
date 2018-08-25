function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

param1 = -y.* log(sigmoid(X * theta));
param2 = (1 - y).* log(1 - sigmoid(X * theta));
param3 = (lambda/(2*m)) * (theta(2:size(theta),:).^2);
J = (sum(param1 - param2)/m) + sum(param3);

for i=1:size(theta,1)
  if i ==1 grad(i) = ((sigmoid(X * theta) - y) ' * X(:,i))/m;
  else grad(i) = (((sigmoid(X * theta) - y) ' * X(:,i))/m) + (lambda/m) * theta(i);
  endif
endfor


% =============================================================

end
