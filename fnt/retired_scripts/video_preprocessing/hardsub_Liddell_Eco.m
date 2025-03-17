%% For liddell videos, the optimal video is set at -q 3 and -r 15. Setting the quality at 1 greatly bumps the file size unneccessarily. 


%%%%%%%%%%%%%

system('cd /d I:')
system('cd I:\3_Liddell_2018_RAW\Field\T001\convert')
 
fp='I:\3_Liddell_2018_RAW\Field\T001\convert';
fl=dir(fullfile(fp,'*.avi'));
out_fp='I:\3_Liddell_2018_RAW\Field\T001\convert\proc';
 
old_fp=pwd;  %pwd is the current folder! 
  
cd(fp)
for i = 1:numel(fl)
      cmdstr=['ffmpeg -i "' fl(i).name '" -vf "subtitles=' fl(i).name(1:end-3) 'smi:force_style=''FontSize=10,Alignment=1,BorderStyle=3,Outline=1,Shadow=0,MarginV=20''" -q 3 -r 10 -max_muxing_queue_size 1000000  "' fullfile(out_fp,fl(i).name) '"'];
       %disp(cmdstr) %display in matlab command window
        system(cmdstr) %send to windows
end


cd(old_fp)


%%%%%%%%%%%%%
% 
% system('cd /d I:')
% system('cd I:\3_Liddell_Ecology\Field\T004')
%  
% fp='I:\3_Liddell_Ecology\Field\T004';
% fl=dir(fullfile(fp,'*.avi'));
% out_fp='I:\3_Liddell_Ecology\Field\T004\proc';
%  
% old_fp=pwd;  %pwd is the current folder! 
%   
% cd(fp)
% for i = 1:numel(fl)
%       cmdstr=['ffmpeg -i "' fl(i).name '" -vf "subtitles=' fl(i).name(1:end-3) 'smi:force_style=''FontSize=10,Alignment=1,BorderStyle=3,Outline=1,Shadow=0,MarginV=20''" -q 3 -r 10 -max_muxing_queue_size 1000000  "' fullfile(out_fp,fl(i).name) '"'];
%        %disp(cmdstr) %display in matlab command window
%         system(cmdstr) %send to windows
% end
% 
% 
% cd(old_fp)
% 
% 
% %%%%%%%%%%%%
% 
% system('cd /d I:')
% system('cd I:\3_Liddell_Ecology\Field\T001')
%  
% fp='I:\3_Liddell_Ecology\Field\T001';
% fl=dir(fullfile(fp,'*.avi'));
% out_fp='I:\3_Liddell_Ecology\Field\T001\proc';
%  
% old_fp=pwd;  %pwd is the current folder! 
%   
% cd(fp)
% for i = 1:numel(fl)
%       cmdstr=['ffmpeg -i "' fl(i).name '" -vf "subtitles=' fl(i).name(1:end-3) 'smi:force_style=''FontSize=10,Alignment=1,BorderStyle=3,Outline=1,Shadow=0,MarginV=20''" -q 3 -r 10 -max_muxing_queue_size 1000000  "' fullfile(out_fp,fl(i).name) '"'];
%        %disp(cmdstr) %display in matlab command window
%         system(cmdstr) %send to windows
% end
% 
% 
% cd(old_fp)
% 
% 
% 
% %%%%%%%%%%%%%
% 
% system('cd /d I:')
% system('cd I:\3_Liddell_Ecology\Field\T002')
%  
% fp='I:\3_Liddell_Ecology\Field\T002';
% fl=dir(fullfile(fp,'*.avi'));
% out_fp='I:\3_Liddell_Ecology\Field\T002\proc';
%  
% old_fp=pwd;  %pwd is the current folder! 
%   
% cd(fp)
% for i = 1:numel(fl)
%       cmdstr=['ffmpeg -i "' fl(i).name '" -vf "subtitles=' fl(i).name(1:end-3) 'smi:force_style=''FontSize=10,Alignment=1,BorderStyle=3,Outline=1,Shadow=0,MarginV=20''" -q 3 -r 10 -max_muxing_queue_size 1000000  "' fullfile(out_fp,fl(i).name) '"'];
%        %disp(cmdstr) %display in matlab command window
%         system(cmdstr) %send to windows
% end
% 
% 
% cd(old_fp)


