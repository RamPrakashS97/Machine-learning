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
z=X*theta;
for i=1:m
    hypothesis(i)=1/(1+exp(-(z(i))));
    c(i)=(y(i)*log(hypothesis(i))+(1-y(i))*log(1-hypothesis(i)));
end
J=(-1/m)*sum(c)+(lambda/(2*m))*sum(theta(2:length(theta),:).^2);

errorh=hypothesis'-y;
for j=1:size(theta)
    if j==1
        Xelem=X(:,j)';
        grad(j)=Xelem*errorh;
        grad(j)=grad(j)/m;
    else
        Xelem=X(:,j)';
        grad(j)=(Xelem*errorh)+(lambda*theta(j));
        grad(j)=grad(j)/m;
    end
end

% =============================================================

end
