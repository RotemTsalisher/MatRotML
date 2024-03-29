slope =@(p1,p2) (p1(2)-p2(2))./(p1(1)-p2(1));
myLine =@(x,p,m) m.*x - m.*p(1) + p(2);

N = 100; a = -8; b = 21;
x_ = linspace(a,b,N);
[x1,x2] = meshgrid(x_,x_);
X = [x1(:) x2(:)];
x_new = [2 2];
mu = [1 1;4 5;13 2]; sigma = [2 1 ; 1 2];

class1 = mvnpdf(X,mu(1,:),sigma);
class1 = reshape(class1,length(x_),length(x_));

class2 = mvnpdf(X,mu(2,:),sigma);
class2 = reshape(class2,length(x_),length(x_));

class3 = mvnpdf(X,mu(3,:),sigma);
class3 = reshape(class3,length(x_),length(x_));

figure(1); contour(x_,x_,class1); hold on; contour(x_,x_,class2); contour(x_,x_,class3); grid on; axis tight;

m12 = slope(mu(1,:),mu(2,:)); m23 = slope(mu(2,:),mu(3,:)); m13 = slope(mu(2,:),mu(3,:));
m12_ = -(1/m12); m23_ = -(1/m23); m13_ = -1/(m13); p12 = [(mu(1,1) + mu(2,1))/2, (mu(1,2) + mu(2,2))/2]; p23 = [(mu(2,1) + mu(3,1))/2, (mu(2,2) + mu(3,2))/2];
p13 = [(mu(1,1) + mu(3,1))/2, (mu(1,2) + mu(3,2))/2]; 
y12 = myLine(x_,p12,m12_); y23 = myLine(x_,p23,m23_); y13 = myLine(x_,p13,m13_); 

plot(x_,y12,"black:",x_,y23,"black:",x_,y13, "black:");
axis([a b a/2 b/2]);
xlabel("x_1"); ylabel("x_2"); title("LDA Decision Criteria Lines","By Matan and ROT");
plot([mu(1,1),mu(2,1),mu(3,1)],[mu(1,2),mu(2,2),mu(3,2)],"r.");
text(mu(1,1) + 10^(-1),mu(1,2),["(" + num2str(mu(1,1)) + "," + num2str(mu(1,2)) + ")"],fontsize=7);
text(mu(2,1) + 10^(-1),mu(2,2),["(" + num2str(mu(2,1)) + "," + num2str(mu(2,2)) + ")"],fontsize=7);
text(mu(3,1) + 10^(-1),mu(3,2),["(" + num2str(mu(3,1)) + "," + num2str(mu(3,2)) + ")"],fontsize=7);

text(mu(1,1) -1.7*sigma(1,1),mu(1,2) - 1.7*sigma(2,2),"Class 1",fontsize=8.5);
text(mu(2,1) +1.7*sigma(2,1),mu(2,2) + 1.7*sigma(2,2),"Class 2",fontsize=8.5);
text(mu(3,1) +1.7*sigma(2,1),mu(3,2) + 1.7*sigma(2,2),"Class 3",fontsize=8.5);

text(-6,10.25,"(Class 1, Class 2) Seperation",fontsize=10);
text(-6,9.75,["x2 = " + num2str(m12_) + "*x1 + " + num2str(-m12_*p12(1)+p12(2))],fontsize=8.5);

text(11,10.25,"(Class 2, Class 3) Seperation",fontsize=10);
text(11,9.75,["x2 = " + num2str(m23_) + "*x1 + " + num2str(-m23_*p23(1)+p23(2))],fontsize=8.5);

text(0.5,-3,"(Class 1, Class 3) Seperation",fontsize=10);
text(2.5,-3.5,["x2 = " + num2str(m13_) + "*x1 + " + num2str(-m13_*p13(1)+p13(2))],fontsize=8.5);
hold off; 

