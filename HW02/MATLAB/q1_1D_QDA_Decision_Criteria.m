opengl('software');
opengl('save','software'); % solve graph legend issues

sigma0 = 1;
sigma1 = sqrt(10^6);
mu0 = 0; mu1 = 1;
pi00 = 1/2; pi10 = 1-pi00;
pi01 = 1/8; pi11 = 1-pi01;
den = sigma0.^2 - sigma1.^2;
a = 1;
b = 2*(mu0*sigma1.^2-mu1*sigma0.^2)/den;
c_ =@(pi0,pi1) mu1.^2*sigma0.^2-mu0.^2*sigma1.^2+(sigma0*sigma1).^2*(log((sigma1/sigma0).^2)+log((pi0/pi1).^2));
c0 = c_(pi00,pi10)/den;
c1 = c_(pi01,pi11)/den;

x = linspace(-10,10,10000);
y00 = polyval([a b c0], x);
y01 = polyval([a b c1], x);
plot(x,y00,"black",x,y01,"red"); grid on; xline(0,"black--"); yline(0,"black--");
legend("{\pi}_0 = 1/2","{\pi}_0 = 1/8"); title(["Decision Criteria for 1D QDA" "y <= 0 is the x range where we classify y_c = 0" "Class_0 ~ N({\mu}_0 = 0,{\sigma}_0^2 = 1), Class_1 ~ N({\mu}_1 = 1,{\sigma}_1^2 = 10^6)"]);
xlabel("x"); text(0,0,"(0,0)");