#!/usr/bin/python

import json
import time
import datetime
import sys
import argparse
import os
import re
import string
import signal
import subprocess

import numpy as np

OUT_DIR = "/projectnb/snoplus/sphere_data/KamLAND_Smeared_Center/"
ZDAB_DIR = "/projectnb/snoplus/sphere_data/input/"
MACRO_DIR = "/projectnb/snoplus/machine_learning/data/shell"
TIME = "12:00:00"
# SIG = 'Xe136_dVrndVtx_3p0mSphere'
# BKG = 'C10_dVrndVtx_3p0mSphere'

SIG = 'Xe136_center'
BKG = 'C10_center'

#filter = [1, 6, 7, 10, 11, 13, 17, 18, 23, 25, 26, 28, 29, 32, 34, 39, 47, 49, 50, 53, 54, 57, 65, 67, 69, 71, 74, 75, 80, 84, 85, 88, 89, 91, 92, 94, 97, 102, 105, 107, 108, 110, 120, 122, 124, 128, 131, 134, 138, 139, 147, 149, 156, 157, 158, 161, 162, 163, 165, 168, 180, 186, 189, 191, 193, 194, 197, 200]
filter = [1, 4, 5, 13, 14, 16, 17, 19, 21, 23, 26, 27, 32, 34, 40, 43, 45, 47, 58]
filter = ["sph_out_" + SIG + "_1k_" + str(f) + ".root" for f in filter]


def main(argv):

   # Setting the output Directory if it does not exist.
   if not os.path.exists(OUT_DIR):
       os.mkdir(OUT_DIR )

   #Looking for the input file in given input directory.
   inputfiles_sig = [(ifile) for ifile in os.listdir(ZDAB_DIR) if SIG in ifile]
   inputfiles_bkg = [(ifile) for ifile in os.listdir(ZDAB_DIR) if BKG in ifile]

   print len(inputfiles_sig)
   inputfiles_sig = [(ifile) for ifile in inputfiles_sig if not (ifile in filter)]
   print len(inputfiles_sig)
   #time.sleep(100)
   inputfiles_sig = inputfiles_sig[0:50]
   # inputfiles_bkg = inputfiles_bkg[0:200]

   # checkfiles = [ifile.replace("0.1000", "root") for ifile in os.listdir(OUT_DIR) if "indptr" in ifile]

   # #checkfile2 = [ifile.replace("0.1000.json", "root") for ifile in os.listdir(CHECK_DIR) if "indptr" in ifile]

   # checkfiles = checkfiles + checkfile2

   # checkfiles = [ifile.replace("indptr_", "") for ifile in checkfiles]

   # zdabfiles = np.setdiff1d(inputfiles,checkfiles)

   zdabfiles = inputfiles_sig
   print zdabfiles
   now = time.time()
   run_time = os.path.getmtime(ZDAB_DIR + '/' + zdabfiles[0])

   print len(zdabfiles)

   for rootfile in zdabfiles:
    process_count=0
    process_upper_lim=1
    event_number=1000
    while (process_count < process_upper_lim):
      start_ev = process_count * event_number
      end_ev = (process_count + 1) * event_number
      with cd(MACRO_DIR):
        macrotemplate = string.Template(open('/projectnb/snoplus/machine_learning/data/process_data.sh', 'r').read())
        outputstring = str(OUT_DIR)
        timestring = str(TIME)
        inputstring = str(ZDAB_DIR + rootfile)
        macrostring = macrotemplate.substitute(TIME=timestring, INPUT=inputstring, OUTPUT=outputstring, START=int(start_ev), END=int(end_ev), ELOW=2.2, EHI=2.7)
        macrofilename = 'shell_%s_range_%d_%d.sh' % (str(rootfile), start_ev, end_ev)
        macro = file(macrofilename,'w')
        macro.write(macrostring)
        macro.close()
        process_count += 1
        try:
          command = ['qsub', macrofilename]
          process = subprocess.call(command)
          #time.sleep(1)
        except Exception as error:
          return 0
   return 1


class cd:
   '''
   Context manager for changing the current working directory
   '''
   def __init__(self, newPath):
      self.newPath = newPath

   def __enter__(self):
      self.savedPath = os.getcwd()
      os.chdir(self.newPath)

   def __exit__(self, etype, value, traceback):
      os.chdir(self.savedPath)

if __name__=="__main__":

   print sys.exit(main(sys.argv[1:]))
