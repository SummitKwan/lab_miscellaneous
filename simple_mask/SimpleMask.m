npxs = 256;     % number of pixels
edge = 0.2;     % size of eage, proportion of 1.0
squareness= 5;  % 1: diamond, 2:circle, 5:round-conner square; 100:square


[X,Y] = meshgrid( linspace(-1,1,npxs), linspace(-1,1,npxs) );
scale = 1/edge;

% circle
% img= scale-scale*sqrt(X.^2+Y.^2);
% square
% img= scale-scale*max(abs(X),abs(Y));

% soft
img= scale-scale.*(abs(X).^squareness+abs(Y).^squareness).^(1/squareness);

img=img.*(img>=0 & img<=1)+(img>1);
imshow(img);


