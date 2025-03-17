% %ffmpeg -i sample.avi -vf subtitles=sample.smi:force_style=''Fontsize=16,Alignment=1" -q 1 -r 10 -max_muxing_queue_size 100000 outq1.avi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

system('cd /d I:')
system('cd I:\3_Liddell_Ecology\Field\T003_Alpha\CAM1')
 
fp='I:\3_Liddell_Ecology\Field\T003_Alpha\CAM1';
fl=dir(fullfile(fp,'*.avi'));
out_fp='C:\Users\Caleb Vogt\Desktop\New folder\proc';
 
old_fp=pwd;  %pwd is the current folder! 
  
cd(fp)
for i = 1:numel(fl)
      cmdstr=['ffmpeg -i "' fl(i).name '" -vf "subtitles=' fl(i).name(1:end-3) 'smi:force_style=''FontSize=10,Alignment=1''" -q 1 -r 15 -max_muxing_queue_size 1000000  "' fullfile(out_fp,fl(i).name) '"'];
       %disp(cmdstr) %display in matlab command window
        system(cmdstr) %send to windows
end


cd(old_fp)
 