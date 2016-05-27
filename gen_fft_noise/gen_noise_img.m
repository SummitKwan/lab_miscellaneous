function image_noise = gen_noise_img(Or,Sc,Li)
% function to generate 2D Noise image (W*H*4), based on gen_fft2_noise()
% Shabo Guan, 2016-0520, Brown University
% input:
%   Or: orientation: [ 0: horizontal,  1: vertical ]
%   Sc: scale      : [ 1: high freq , 20: low freq ]
%   Li: linearirty : [ 1: no stripes,  5: conlinear stripe pattern ]
% Output: image_noise
%   image matrix, size (W*H*3), r,g,b, every value in [0,1]
% Usage :
%   image_noise = gen_noise_img(90,10,5);


coef_saturation = 0.5;

% generate two noise patterns
Noise1 = gen_fft2_noise(Or,Sc,Li);  % for hue and saturation
Noise2 = gen_fft2_noise(Or,Sc,Li);  % for v (brighness related)

% calulate image in hsv space (hue, saturation and value)
h = mat2gray(angle(Noise1));                  % use phase of Noise1
s = mat2gray(  abs(Noise1)) *coef_saturation; % use   abs of Noise1
v = real(Noise2)/max(abs(Noise2(:)))/2+0.5;   % use real of Noise2, normalized

% convert to rgb space
image_noise = hsv2rgb(cat(3, h,s,v));

% if plot image
if true  
    imshow(image_noise);

end