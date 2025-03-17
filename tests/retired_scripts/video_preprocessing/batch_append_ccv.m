%Batch append videos without re-encoding. CCV 7.16.2018
% Modified from - https://www.youtube.com/watch?v=1Xf1DfvVJbs
%               - https://stackoverflow.com/questions/38996925/ffmpeg-concat-unsafe-file-name

% Ffmpeg Command line code... turn this into code calling ffmpeg
% cd c:
% cd C:\Users\Caleb Vogt\Videos\batch test\proc
% (for %i in (*.avi) do @echo file '%i') > mylist.txt
% ffmpeg -f concat -safe 0 -i mylist.txt -c copy output.avi

% trim video with ffmpeg... doesnt appear to be a way to splice out
% multiple clips with ffmpeg: https://superuser.com/questions/377343/cut-part-from-video-file-from-start-position-to-end-position-with-ffmpeg

