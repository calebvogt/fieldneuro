## batch_sleap_rename.py
import os

#wd = os.chdir(r"C:\Users\sheehan-lab\Desktop\16_AG_MUS_temp\2_MESH") ## CHANGE THIS
wd = os.getcwd()

for filename in os.listdir(wd):
   os.rename(filename, filename.replace('USV_DATA_', ''))
