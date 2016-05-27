% script to generate many images and save to disk

% range of orientation, scale and liniearity of noise
% refer to funciton gen_noise_img() for details

% Setting 1 for Shaobo MTS
% lOr = [0,45,90,135];
% lSc = [5, 10, 20];
% lLi = [2, 5];
% % number of noise for every condition
% N = 20;
% tf_gray = false;
% tf_alpha= true;

% Setting 2 for Ruobing
lOr = [0, 30, 60, 90, 120, 150];
lSc = [5, 10, 15];
lLi = [15];
N = 1;
tf_gray = true;
tf_alpha= false;

% specify the folder to store images
outputfolder = './noise_image_temp';
if exist(outputfolder, 'dir')
    delete( fullfile(outputfolder, '*'));
else
    mkdir(outputfolder);
end

% generate image and save as png file with alpha channel
for Or = lOr
    for Sc = lSc
        for Li = lLi
            for i = 1:N
                image_noise = gen_noise_img(Or,Sc,Li);
                if tf_gray
                    image_gray = rgb2gray(image_noise);
                    image_noise = cat(3, image_gray,image_gray,image_gray);
                end
                
                filename = sprintf('fftnoise_%03d_%03d_%03d_%03d.png',Or,Sc,Li,i);
                if tf_alpha
                    imwrite(image_noise, fullfile(outputfolder, filename), ...
                    'Alpha', image_noise(:,:,1)*0+1);
                else
                    imwrite(image_noise, fullfile(outputfolder, filename));
                end
            end
        end
    end
end
