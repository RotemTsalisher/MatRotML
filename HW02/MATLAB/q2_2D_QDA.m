slope =@(p1,p2) (p1(2)-p2(2))./(p1(1)-p2(1));
myLine =@(x,p,m) m.*x - m.*p(1) + p(2);

N = 1000; a = -8; b = 21;
x_ = linspace(a,b,N);
[x1,x2] = meshgrid(x_,x_);
X = [x1(:) x2(:)];
x_new = [2 2];
mu = [0 0;1 1;-1 1]; sigma1 = [0.7 0 ; 0 0.7]; sigma23 = [0.8, 0.2; 0.2, 0.8];

class1 = mvnpdf(X,mu(1,:),sigma1);
class1 = reshape(class1,length(x_),length(x_));

class2 = mvnpdf(X,mu(2,:),sigma23);
class2 = reshape(class2,length(x_),length(x_));

class3 = mvnpdf(X,mu(3,:),sigma23);
class3 = reshape(class3,length(x_),length(x_));

figure(1); contour(x_,x_,class1); hold on; contour(x_,x_,class2); contour(x_,x_,class3); grid on; axis tight;

m12 = slope(mu(1,:),mu(2,:)); m23 = slope(mu(2,:),mu(3,:)); m13 = slope(mu(2,:),mu(3,:));
m12_ = -(1/m12); m23_ = -(1/m23); m13_ = -1/(m13); p12 = [(mu(1,1) + mu(2,1))/2, (mu(1,2) + mu(2,2))/2]; p23 = [(mu(2,1) + mu(3,1))/2, (mu(2,2) + mu(3,2))/2];
p13 = [(mu(1,1) + mu(3,1))/2, (mu(1,2) + mu(3,2))/2]; 
y12 = myLine(x_,p12,m12_); y23 = myLine(x_,p23,m23_); y13 = myLine(x_,p13,m13_); 

xlabel("x_1"); ylabel("x_2"); 
plot([mu(1,1),mu(2,1),mu(3,1)],[mu(1,2),mu(2,2),mu(3,2)],"r.");

title("LDA Decision Criteria Lines","By Matan and ROT");
plot(.5,.5,marker=".",MarkerSize=12,MarkerEdgeColor="black");
plot(-.5,.5,marker=".",MarkerSize=12,MarkerEdgeColor="black");
text(-.5,.5,"(-0.5,0.5)"); text(0.5,0.5,"(0.5,0.5)"); text(0,-1.7,"Class 1"); text(2.5,0,"Class 2"); text(-2.75,0,"Class 3");
hold off; axis([-3 3 -2 3]);
