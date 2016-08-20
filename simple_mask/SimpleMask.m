tf_save = true;
npxs = 512;     % number of pixels
edge_ratio = 5; % proportion of edge, "5" means edge ocupies 0.20 of radias
squareness= 5;  % 1: diamond, 2:circle, 5:round-conner square; 100:square

[X,Y] = meshgrid( linspace(-1,1,npxs), linspace(-1,1,npxs) );

% circle
% img= scale-scale*sqrt(X.^2+Y.^2);
% square
% img= scale-scale*max(abs(X),abs(Y));

% genearte 2D mask with ajustable squareness
img= edge_ratio-edge_ratio.*(abs(X).^squareness+abs(Y).^squareness).^(1/squareness);

img=img.*(img>=0 & img<=1)+(img>1);
imshow(img);

% temp, for trasparent mask:
img_rgb = ones(npxs,npxs,3)*0.5;

outputfolder = './simple_mask_temp';
if tf_save
    if exist(outputfolder, 'dir')
        delete( fullfile(outputfolder, '*'));
    else
        mkdir(outputfolder);
    end
    filename = sprintf('simplemask_%03d_%03d.png',edge_ratio,squareness);
    imwrite(img_rgb, fullfile(outputfolder, filename),'Alpha', (1-img));
    
    
end

