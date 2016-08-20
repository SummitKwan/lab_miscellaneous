function dataman_TDT(varargin)

% aim:         convert all TDT blocks of the same day to mat file
% requires:    my_TDT2mat.m, OpenDeveloper from TDT
% example:
%     dataman_TDT()
%       --  convert today's all blocks
%     dataman_TDT('2014/07/30')
%       --  convert all blocks recorded from 2014/07/30
%     dataman_TDT('dat')
%       --  convert all blocks indicated by the dat file
% ---------- Shaobo Guan, 2014-0730, WED ----------
% Sheinberg lab, Brown University, USA, Shaobo_Guan@brown.edu


% default date to convert
date_convert = date;
if length(varargin)>=1
    date_convert = varargin{1};
end

% determine which files to upload based on input
tf_upload_plx = false;
tf_upload_tdt = false;
tf_upload_mat = false;
for i=1:length(varargin)
    switch lower(varargin{i})
        case 'plx'
            tf_upload_plx = true;
        case 'tdt'
            tf_upload_tdt = true;
        case 'mat'
            tf_upload_mat = true;
    end
end

set_default_data_path;
% default remote disk location to upload converted date
dir_store = DEFAULT_MAT_PATH_STORE;

if strcmp(date_convert, 'dat')
    [datfilename, datfilepath] = uigetfile('D:\PLX_combined\*.dat', 'Select the .dat file');
    datfilename = fullfile(datfilepath, datfilename);
    fprintf('the dat file selected is: %s \n', datfilename);    
    
    fid=fopen(datfilename);
    name_tank_blocks=textscan(fid, '%s');
    fclose(fid);

    tank = name_tank_blocks{1}{1};
    blocks = name_tank_blocks{1}(2:end);
    name_block_cell = blocks;
    
    % copy .dat and .plx file to shared disk
    [~, datfilename_no_ext]=fileparts(datfilename);
    
    copyfile([fullfile(datfilepath, datfilename_no_ext) ,'*'], dir_store);
    display('file copied: ');
    dir([fullfile(datfilepath, datfilename_no_ext) ,'*']);
else

    % location of data tank
    tank = uigetdir('T:\tdt_tanks\');
    % tank = 'T:\tdt_tanks\PowerPac_32C';

    % translate the date to posivle strings contained in the file name    
    str_date = {datestr(date_convert, 'mmddyy'), datestr(date_convert, 'yyyy-mmdd')};

    % get the block names to convert
    name_block_cell = {};
    for i=1:length(str_date)
        name_block_strc = dir([tank, '/*', str_date{i} ,'*']);
        name_block_cell = [name_block_cell, {name_block_strc.name}];
    end

end

% display block names to convert
display([10, 'the blocks to be converted are: ', 10, '----------']);
for i=1:length(name_block_cell)
    display(name_block_cell{i});
end
display(['----------', 10]);

%% convert using my_TDT2mat
name_converted = {};
for i=1:length(name_block_cell)
    display(['converting: ',name_block_cell{i}]);
    
    [~, name_save]= my_TDT2mat(tank,...
         name_block_cell{i}, 'EXCLUDE', {'raws'}, 'NODATA', false,...
         'SORTNAME', 'PLX', 'SAVE', true, 'VERBOSE', false);

    name_converted = [name_converted, {name_save}];
    display(['generated : ', name_converted{i}, 10]); 
end

% upload covnerted file
display([10, 'the blocks successfully uploaded are: ', 10, '----------']);
for i=1:length(name_converted)
    % upload covnerted file
    copyfile([name_converted{i},'.mat'], dir_store);
    % display block names to upload
    display(name_converted{i});
end
display(['----------', 10]);

display(['data converting and uploading finished']);

end