N = 100; a = -7; b = 7;
x_ = linspace(a,b,N);
[x1,x2] = meshgrid(x_,x_);
X = [x1(:) x2(:)];
x_new = [2 2];
mu = [0 0]; 
sigma1 = [1 0 ; 0 1]; sigma2 = [2 0; 0 1]; sigma3 = [1 0; 0 2];

class1 = mvnpdf(X,mu,sigma1);
class1 = reshape(class1,length(x_),length(x_));

class2 = mvnpdf(X,mu,sigma2);
class3 = mvnpdf(X,mu,sigma3);


figure(1); contour(x_,x_,class1); grid on;