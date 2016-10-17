cd('L:/tdt/export_mat/TDT2PLX2TDT');
set_default_data_path;
addpath(genpath('Matlab Offline Files SDK'))

%% TDT2PLX
TDT2PLX('', {}, 'PLXDIR', DEFAULT_PLX_PATH);

%% PLX2TDT
PLX2TDT('');

%% upload TDT tank
dataman_TDT('dat','tdt','plx');

%% convert to matlab format and upload to shared disk
dataman_TDT('dat','mat');

%% Get computer name
getenv('COMPUTERNAME')

%% if no dat file
dataman_TDT('2016/09/17')
