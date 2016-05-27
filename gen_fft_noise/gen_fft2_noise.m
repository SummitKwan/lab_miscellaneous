function matrix_noise = gen_fft2_noise(Or,Sc,Li)
% function to generate 2D Noise (W*H*1) using inverse fft
% Shabo Guan, 2016-0520, Brown University
% input:
%   Or: orientation: [ 0: horizontal,  1: vertical ]
%   Sc: scale      : [ 1: high freq , 20: low freq ]
%   Li: linearirty : [ 1: no stripes,  5: conlinear stripe pattern ]
% Output: image_noise
%   matrix of complex values, size (W*H*1)
% Usage :
%   matrix_noise = gen_fft2_img(90,10,5); 
%   imshow(mat2gray(real(matrix_noise)));


% ----- default parameters -----

% size of noise patch
W = 512;
H = 512;

% method of generating noise in fft domain
noise_gen = 'laplace';

% ----- generating matrix in the fft domain -----
[X,Y] = meshgrid(linspace(-1,1,H), linspace(-1,1,W));

% calculate rotation
compl_for_rotation = (X+1i*Y)*exp(1i* (Or/180*pi) );
X = real(compl_for_rotation);
Y = imag(compl_for_rotation);

% calculate scale 
sigmaY = 1/Sc;
sigmaX = sigmaY/Li;

% generate power in fft domain
if strcmp(noise_gen, 'gaussian')
    fftPower = exp(-1/2* (X.^2/sigmaX.^2 + Y.^2/sigmaY.^2) );
elseif strcmp(noise_gen, 'laplace')
    fftPower = exp(- (abs(X)/sigmaX + abs(Y)/sigmaY) );
end
% generate phase in fft domain
fftPhase = (rand([W,H])-0.5)*2*pi ;

% get the complex values in fft domain
fftA = abs( fftshift(fftPower) ) .* exp(1i* angle( fftPhase ));

% ----- get the noise patthern by inverse fft -----
matrix_noise = ifft2(fftA);

end
