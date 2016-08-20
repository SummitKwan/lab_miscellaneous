cd('L:/tdt/export_mat/TDT2PLX2TDT');
set_default_data_path;

%% TDT2PLX
TDT2PLX('', {}, 'PLXDIR', DEFAULT_PLX_PATH);

%% PLX2TDT
PLX2TDT('');

%% convert to matlab format and upload to shared disk
dataman_TDT('dat');

%% Get computer name
getenv('COMPUTERNAME')