import os
import sys

# mydir = os.getenv("HOME") + "/ARC"
# sys.path.insert(0, mydir)
# from polsimulate.gotasks.polsimulate import polsimulate

model_image = [mydir + '/polsimulate/SFits.im',
               mydir + '/polsimulate/QFits.im',
               mydir + '/polsimulate/UFits.im',
               mydir + '/polsimulate/Zero.im']
I = [0.2]
Q = [0.03]
U = [0.01]
V = [0.0]
RM = [1.e7]
spec_index = [0.0]
spectrum_file = ''
LO = 100.e9
BBs = [0.]
spw_width = 2.e9
nchan = 16
onsource_time = 0.5
observe_time = 3.0
visib_time = '6s'
nscan = 10
corrupt = False
polsimulate()
