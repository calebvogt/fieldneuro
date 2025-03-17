% %ffmpeg -i sample.avi -vf subtitles=sample.smi:force_style=''Fontsize=16,Alignment=1" -q 1 -r 10 -max_muxing_queue_size 100000 outq1.avi
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

system('cd /d F:')
%%%%system('cd F:\test\Pre-SPP')

fp='F:\2_8x8_KW\KW_T008_SPI\Pre_SPP';
fl=dir(fullfile(fp,'*.avi'));
out_fp='F:\2_8x8_KW\KW_T008_SPI\Pre_SPP\proc';

old_fp=pwd;  %pwd is the current folder! 
 
cd(fp)
for i = 1:numel(fl)
     cmdstr=['ffmpeg -i "' fl(i).name '" -vf "subtitles=' fl(i).name(1:end-3) 'smi:force_style=''FontSize=10,Alignment=1''" -q 1 -r 15 -max_muxing_queue_size 100000  "' fullfile(out_fp,fl(i).name) '"'];
      %disp(cmdstr) %display in matlab command window
       system(cmdstr) %send to windows
 end
   cd(old_fp)

   
%    
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% system('cd /d F:')
% %%%%system('cd F:\test\Pre-SPP')
% 
% fp='F:\2_8x8_KW\KW_T008_SPI\Post_SPP';
% fl=dir(fullfile(fp,'*.avi'));
% out_fp='F:\2_8x8_KW\KW_T008_SPI\Post_SPP\proc';
% 
% old_fp=pwd;  %pwd is the current folder! 
%  
% cd(fp)
% for i = 1:numel(fl)
%      cmdstr=['ffmpeg -i "' fl(i).name '" -vf "subtitles=' fl(i).name(1:end-3) 'smi:force_style=''FontSize=10,Alignment=1''" -q 1 -r 15 -max_muxing_queue_size 100000  "' fullfile(out_fp,fl(i).name) '"'];
%       %disp(cmdstr) %display in matlab command window
%        system(cmdstr) %send to windows
%  end
%    cd(old_fp)
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% system('cd /d F:')
% %%%%system('cd F:\test\Pre-SPP')
% 
% fp='F:\2_8x8_KW\KW_T007_ZBN\Pre_SPP';
% fl=dir(fullfile(fp,'*.avi'));
% out_fp='F:\2_8x8_KW\KW_T007_ZBN\Pre_SPP\proc';
% 
% old_fp=pwd;  %pwd is the current folder! 
%  
% cd(fp)
% for i = 1:numel(fl)
%      cmdstr=['ffmpeg -i "' fl(i).name '" -vf "subtitles=' fl(i).name(1:end-3) 'smi:force_style=''FontSize=10,Alignment=1''" -q 1 -r 15 -max_muxing_queue_size 100000  "' fullfile(out_fp,fl(i).name) '"'];
%       %disp(cmdstr) %display in matlab command window
%        system(cmdstr) %send to windows
%  end
%    cd(old_fp)
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% system('cd /d F:')
% %%%%system('cd F:\test\Pre-SPP')
% 
% fp='F:\2_8x8_KW\KW_T007_ZBN\Post_SPP';
% fl=dir(fullfile(fp,'*.avi'));
% out_fp='F:\2_8x8_KW\KW_T007_ZBN\Post_SPP\proc';
% 
% old_fp=pwd;  %pwd is the current folder! 
%  
% cd(fp)
% for i = 1:numel(fl)
%      cmdstr=['ffmpeg -i "' fl(i).name '" -vf "subtitles=' fl(i).name(1:end-3) 'smi:force_style=''FontSize=10,Alignment=1''" -q 1 -r 15 -max_muxing_queue_size 100000  "' fullfile(out_fp,fl(i).name) '"'];
%       %disp(cmdstr) %display in matlab command window
%        system(cmdstr) %send to windows
%  end
%    cd(old_fp)
% 
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% system('cd /d F:')
% %%%%system('cd F:\test\Pre-SPP')
% 
% fp='F:\2_8x8_KW\KW_T006_C57\Pre_SPP';
% fl=dir(fullfile(fp,'*.avi'));
% out_fp='F:\2_8x8_KW\KW_T006_C57\Pre_SPP\proc';
% 
% old_fp=pwd;  %pwd is the current folder! 
%  
% cd(fp)
% for i = 1:numel(fl)
%      cmdstr=['ffmpeg -i "' fl(i).name '" -vf "subtitles=' fl(i).name(1:end-3) 'smi:force_style=''FontSize=10,Alignment=1''" -q 1 -r 15 -max_muxing_queue_size 100000  "' fullfile(out_fp,fl(i).name) '"'];
%       %disp(cmdstr) %display in matlab command window
%        system(cmdstr) %send to windows
%  end
%    cd(old_fp)
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% system('cd /d F:')
% %%%%system('cd F:\test\Pre-SPP')
% 
% fp='F:\2_8x8_KW\KW_T006_C57\Post_SPP';
% fl=dir(fullfile(fp,'*.avi'));
% out_fp='F:\2_8x8_KW\KW_T006_C57\Post_SPP\proc';
% 
% old_fp=pwd;  %pwd is the current folder! 
%  
% cd(fp)
% for i = 1:numel(fl)
%      cmdstr=['ffmpeg -i "' fl(i).name '" -vf "subtitles=' fl(i).name(1:end-3) 'smi:force_style=''FontSize=10,Alignment=1''" -q 1 -r 15 -max_muxing_queue_size 100000  "' fullfile(out_fp,fl(i).name) '"'];
%       %disp(cmdstr) %display in matlab command window
%        system(cmdstr) %send to windows
%  end
%    cd(old_fp)
% 
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% system('cd /d F:')
% %%%%system('cd F:\test\Pre-SPP')
% 
% fp='F:\2_8x8_KW\KW_T005_PAH\Pre_SPP';
% fl=dir(fullfile(fp,'*.avi'));
% out_fp='F:\2_8x8_KW\KW_T005_PAH\Pre_SPP\proc';
% 
% old_fp=pwd;  %pwd is the current folder! 
%  
% cd(fp)
% for i = 1:numel(fl)
%      cmdstr=['ffmpeg -i "' fl(i).name '" -vf "subtitles=' fl(i).name(1:end-3) 'smi:force_style=''FontSize=10,Alignment=1''" -q 1 -r 15 -max_muxing_queue_size 100000  "' fullfile(out_fp,fl(i).name) '"'];
%       %disp(cmdstr) %display in matlab command window
%        system(cmdstr) %send to windows
%  end
%    cd(old_fp)
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% system('cd /d F:')
% %%%%system('cd F:\test\Pre-SPP')
% 
% fp='F:\2_8x8_KW\KW_T005_PAH\Post_SPP';
% fl=dir(fullfile(fp,'*.avi'));
% out_fp='F:\2_8x8_KW\KW_T005_PAH\Post_SPP\proc';
% 
% old_fp=pwd;  %pwd is the current folder! 
%  
% cd(fp)
% for i = 1:numel(fl)
%      cmdstr=['ffmpeg -i "' fl(i).name '" -vf "subtitles=' fl(i).name(1:end-3) 'smi:force_style=''FontSize=10,Alignment=1''" -q 1 -r 15 -max_muxing_queue_size 100000  "' fullfile(out_fp,fl(i).name) '"'];
%       %disp(cmdstr) %display in matlab command window
%        system(cmdstr) %send to windows
%  end
%    cd(old_fp)
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% system('cd /d F:')
% %%%%system('cd F:\test\Pre-SPP')
% 
% fp='F:\2_8x8_KW\KW_T004_SPI\Pre_SPP';
% fl=dir(fullfile(fp,'*.avi'));
% out_fp='F:\2_8x8_KW\KW_T004_SPI\Pre_SPP\proc';
% 
% old_fp=pwd;  %pwd is the current folder! 
%  
% cd(fp)
% for i = 1:numel(fl)
%      cmdstr=['ffmpeg -i "' fl(i).name '" -vf "subtitles=' fl(i).name(1:end-3) 'smi:force_style=''FontSize=10,Alignment=1''" -q 1 -r 15 -max_muxing_queue_size 100000  "' fullfile(out_fp,fl(i).name) '"'];
%       %disp(cmdstr) %display in matlab command window
%        system(cmdstr) %send to windows
%  end
%    cd(old_fp)
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% system('cd /d F:')
% %%%%system('cd F:\test\Pre-SPP')
% 
% fp='F:\2_8x8_KW\KW_T004_SPI\Post_SPP';
% fl=dir(fullfile(fp,'*.avi'));
% out_fp='F:\2_8x8_KW\KW_T004_SPI\Post_SPP\proc';
% 
% old_fp=pwd;  %pwd is the current folder! 
%  
% cd(fp)
% for i = 1:numel(fl)
%      cmdstr=['ffmpeg -i "' fl(i).name '" -vf "subtitles=' fl(i).name(1:end-3) 'smi:force_style=''FontSize=10,Alignment=1''" -q 1 -r 15 -max_muxing_queue_size 100000  "' fullfile(out_fp,fl(i).name) '"'];
%       %disp(cmdstr) %display in matlab command window
%        system(cmdstr) %send to windows
%  end
%    cd(old_fp)
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% system('cd /d F:')
% %%%%system('cd F:\test\Pre-SPP')
% 
% fp='F:\2_8x8_KW\KW_T003_C57\Pre_SPP';
% fl=dir(fullfile(fp,'*.avi'));
% out_fp='F:\2_8x8_KW\KW_T003_C57\Pre_SPP\proc';
% 
% old_fp=pwd;  %pwd is the current folder! 
%  
% cd(fp)
% for i = 1:numel(fl)
%      cmdstr=['ffmpeg -i "' fl(i).name '" -vf "subtitles=' fl(i).name(1:end-3) 'smi:force_style=''FontSize=10,Alignment=1''" -q 1 -r 15 -max_muxing_queue_size 100000  "' fullfile(out_fp,fl(i).name) '"'];
%       %disp(cmdstr) %display in matlab command window
%        system(cmdstr) %send to windows
%  end
%    cd(old_fp)
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% system('cd /d F:')
% %%%%system('cd F:\test\Pre-SPP')
% 
% fp='F:\2_8x8_KW\KW_T003_C57\Post_SPP';
% fl=dir(fullfile(fp,'*.avi'));
% out_fp='F:\2_8x8_KW\KW_T003_C57\Post_SPP\proc';
% 
% old_fp=pwd;  %pwd is the current folder! 
%  
% cd(fp)
% for i = 1:numel(fl)
%      cmdstr=['ffmpeg -i "' fl(i).name '" -vf "subtitles=' fl(i).name(1:end-3) 'smi:force_style=''FontSize=10,Alignment=1''" -q 1 -r 15 -max_muxing_queue_size 100000  "' fullfile(out_fp,fl(i).name) '"'];
%       %disp(cmdstr) %display in matlab command window
%        system(cmdstr) %send to windows
%  end
%    cd(old_fp)
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% system('cd /d F:')
% %%%%system('cd F:\test\Pre-SPP')
% 
% fp='F:\2_8x8_KW\KW_T002_C57\Pre_SPP';
% fl=dir(fullfile(fp,'*.avi'));
% out_fp='F:\2_8x8_KW\KW_T002_C57\Pre_SPP\proc';
% 
% old_fp=pwd;  %pwd is the current folder! 
%  
% cd(fp)
% for i = 1:numel(fl)
%      cmdstr=['ffmpeg -i "' fl(i).name '" -vf "subtitles=' fl(i).name(1:end-3) 'smi:force_style=''FontSize=10,Alignment=1''" -q 1 -r 15 -max_muxing_queue_size 100000  "' fullfile(out_fp,fl(i).name) '"'];
%       %disp(cmdstr) %display in matlab command window
%        system(cmdstr) %send to windows
%  end
%    cd(old_fp)
%  
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% system('cd /d F:')
% %%%%system('cd F:\test\Pre-SPP')
% 
% fp='F:\2_8x8_KW\KW_T002_C57\Post_SPP';
% fl=dir(fullfile(fp,'*.avi'));
% out_fp='F:\2_8x8_KW\KW_T002_C57\Post_SPP\proc';
% 
% old_fp=pwd;  %pwd is the current folder! 
%  
% cd(fp)
% for i = 1:numel(fl)
%      cmdstr=['ffmpeg -i "' fl(i).name '" -vf "subtitles=' fl(i).name(1:end-3) 'smi:force_style=''FontSize=10,Alignment=1''" -q 1 -r 15 -max_muxing_queue_size 100000  "' fullfile(out_fp,fl(i).name) '"'];
%       %disp(cmdstr) %display in matlab command window
%        system(cmdstr) %send to windows
%  end
%  
%  
%  cd(old_fp)
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% system('cd /d F:')
% %%%%system('cd F:\test\Pre-SPP')
% 
% fp='F:\2_8x8_KW\KW_T001_ZBN\Pre_SPP';
% fl=dir(fullfile(fp,'*.avi'));
% out_fp='F:\2_8x8_KW\KW_T001_ZBN\Pre_SPP\proc';
% 
% old_fp=pwd;  %pwd is the current folder! 
%  
% cd(fp)
% for i = 1:numel(fl)
%      cmdstr=['ffmpeg -i "' fl(i).name '" -vf "subtitles=' fl(i).name(1:end-3) 'smi:force_style=''FontSize=10,Alignment=1''" -q 1 -r 15 -max_muxing_queue_size 100000  "' fullfile(out_fp,fl(i).name) '"'];
%       %disp(cmdstr) %display in matlab command window
%        system(cmdstr) %send to windows
%  end
%  
%  
%  cd(old_fp)
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% 
% 
% % 
% % system('cd /d F:')
% % %%%%system('cd F:\test\Pre-SPP')
% % 
% % fp='F:\2_8x8_KW\KW_T001_ZBN\Post_SPP';
% % fl=dir(fullfile(fp,'*.avi'));
% % out_fp='F:\2_8x8_KW\KW_T001_ZBN\Post_SPP\proc';
% % 
% % old_fp=pwd;  %pwd is the current folder! 
% %  
% % cd(fp)
% % for i = 1:numel(fl)
% %      cmdstr=['ffmpeg -i "' fl(i).name '" -vf "subtitles=' fl(i).name(1:end-3) 'smi:force_style=''FontSize=10,Alignment=1''" -q 1 -r 15 -max_muxing_queue_size 100000  "' fullfile(out_fp,fl(i).name) '"'];
% %       %disp(cmdstr) %display in matlab command window
% %        system(cmdstr) %send to windows
% %  end
% %  
% %  
% %  cd(old_fp)
% % 
% 
% 
% 
% 
% % fp='F:\2_8x8_KW\KW_T003_C57\8x8';
% % fl=dir(fullfile(fp,'*.avi'));
% % out_fp='F:\2_8x8_KW\KW_T003_C57\8x8\proc';
% % 
% % old_fp=pwd;  %pwd is the current folder! 
% % 
% % cd(fp)
% % for i = 1:numel(fl)
% %     cmdstr=['ffmpeg -i "' fl(i).name '" -vf "subtitles=' fl(i).name(1:end-3) 'smi:force_style=''FontSize=10,Alignment=1''" -q 1 -r 15 -max_muxing_queue_size 100000  "' fullfile(out_fp,fl(i).name) '"'];
% %      %disp(cmdstr) %display in matlab command window
% %       system(cmdstr) %send to windows
% % end
% % 
% % 
% % cd(old_fp)
% %   
% % 
% % fp='F:\2_8x8_KW\KW_T004_SPITUA\8x8';
% % fl=dir(fullfile(fp,'*.avi'));
% % out_fp='F:\2_8x8_KW\KW_T004_SPITUA\8x8\proc';
% % 
% % old_fp=pwd;  %pwd is the current folder! 
% % 
% % cd(fp)
% % for i = 1:numel(fl)
% %     cmdstr=['ffmpeg -i "' fl(i).name '" -vf "subtitles=' fl(i).name(1:end-3) 'smi:force_style=''FontSize=10,Alignment=1''" -q 1 -r 15 -max_muxing_queue_size 100000  "' fullfile(out_fp,fl(i).name) '"'];
% %      %disp(cmdstr) %display in matlab command window
% %       system(cmdstr) %send to windows
% % end
% % 
% % 
% % cd(old_fp)
% %   
% % 
% % 
% % fp='F:\2_8x8_KW\KW_T005_PAH\8x8';
% % fl=dir(fullfile(fp,'*.avi'));
% % out_fp='F:\2_8x8_KW\KW_T005_PAH\8x8\proc';
% % 
% % old_fp=pwd;  %pwd is the current folder! 
% % 
% % cd(fp)
% % for i = 1:numel(fl)
% %     cmdstr=['ffmpeg -i "' fl(i).name '" -vf "subtitles=' fl(i).name(1:end-3) 'smi:force_style=''FontSize=10,Alignment=1''" -q 1 -r 15 -max_muxing_queue_size 100000  "' fullfile(out_fp,fl(i).name) '"'];
% %      %disp(cmdstr) %display in matlab command window
% %       system(cmdstr) %send to windows
% % end
% % 
% % 
% % cd(old_fp)
% %   
% % 
% % 
% % 
% % 
% % fp='F:\2_8x8_KW\KW_T006_C57\8x8';
% % fl=dir(fullfile(fp,'*.avi'));
% % out_fp='F:\2_8x8_KW\KW_T006_C57\8x8\proc';
% % 
% % old_fp=pwd;  %pwd is the current folder! 
% % 
% % cd(fp)
% % for i = 1:numel(fl)
% %     cmdstr=['ffmpeg -i "' fl(i).name '" -vf "subtitles=' fl(i).name(1:end-3) 'smi:force_style=''FontSize=10,Alignment=1''" -q 1 -r 15 -max_muxing_queue_size 100000  "' fullfile(out_fp,fl(i).name) '"'];
% %      %disp(cmdstr) %display in matlab command window
% %       system(cmdstr) %send to windows
% % end
% % 
% % 
% % cd(old_fp)
% % 
% % 
% % 
% % fp='F:\2_8x8_KW\KW_T007_ZBN\8x8';
% % fl=dir(fullfile(fp,'*.avi'));
% % out_fp='F:\2_8x8_KW\KW_T007_ZBN\8x8\proc';
% % 
% % old_fp=pwd;  %pwd is the current folder! 
% % 
% % cd(fp)
% % for i = 1:numel(fl)
% %     cmdstr=['ffmpeg -i "' fl(i).name '" -vf "subtitles=' fl(i).name(1:end-3) 'smi:force_style=''FontSize=10,Alignment=1''" -q 1 -r 15 -max_muxing_queue_size 100000  "' fullfile(out_fp,fl(i).name) '"'];
% %      %disp(cmdstr) %display in matlab command window
% %       system(cmdstr) %send to windows
% % end
% % 
% % 
% % cd(old_fp)
% % 
% % 
% % 
% % fp='F:\1_8x8_Single_Quad\8x8\T011_NY3\CAM5';
% % fl=dir(fullfile(fp,'*.avi'));
% % out_fp='F:\1_8x8_Single_Quad\8x8\T011_NY3\CAM5\proc';
% % 
% % old_fp=pwd;  %pwd is the current folder! 
% % 
% % cd(fp)
% % for i = 1:numel(fl)
% %     cmdstr=['ffmpeg -i "' fl(i).name '" -vf "subtitles=' fl(i).name(1:end-3) 'smi:force_style=''FontSize=10,Alignment=1''" -q 1 -r 15 -max_muxing_queue_size 100000  "' fullfile(out_fp,fl(i).name) '"'];
% %      %disp(cmdstr) %display in matlab command window
% %       system(cmdstr) %send to windows
% % end
% % 
% % 
% % cd(old_fp)