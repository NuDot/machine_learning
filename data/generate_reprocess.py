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

OUT_DIR = "/projectnb/snoplus/sphere_data/json_new/"
ZDAB_DIR = "/projectnb/snoplus/sphere_data/input/"
MACRO_DIR = "/projectnb/snoplus/machine_learning/data/shell"
TIME = "8:00:00"



def main(argv):

   # Setting the output Directory if it does not exist.
   if not os.path.exists(OUT_DIR):
       os.mkdir(OUT_DIR )

   # Looking for the input file in given input directory.
   #inputfiles = [(ifile) for ifile in os.listdir(ZDAB_DIR) if 'C10_center' in ifile]

   checkfiles = []

   for idir in os.listdir(OUT_DIR):
      if 'C10_center' in idir:
        if "data_" in idir:
          with cd(OUT_DIR + idir):
            if not os.path.isfile('8_9.json'):
                checkfiles.append(idir)


   checkfiles = [ifile.replace("0.1000", "root") for ifile in checkfiles]

   # checkfile2 = [ifile.replace("0.1000.json", "root") for ifile in os.listdir(CHECK_DIR) if "indptr" in ifile]

   # checkfiles = checkfiles + checkfile2

   zdabfiles = [ifile.replace("data_", "") for ifile in checkfiles]

   for rootfile in zdabfiles:
    process_count=0
    process_upper_lim=1
    event_number=1000
    while (process_count < process_upper_lim):
      start_ev = process_count * event_number
      end_ev = (process_count + 1) * event_number
      with cd(MACRO_DIR):
        macrotemplate = string.Template(open('/projectnb/snoplus/machine_learning/data/reprocess_data.sh', 'r').read())
        outputstring = str(OUT_DIR)
        timestring = str(TIME)
        inputstring = str(ZDAB_DIR + rootfile)
        macrostring = macrotemplate.substitute(TIME=timestring, INPUT=inputstring, OUTPUT=outputstring, START=int(start_ev), END=int(end_ev), ELOW=2.2, EHI=2.7)
        macrofilename = 'shell_%s_range_%d_%d_r.sh' % (str(rootfile), start_ev, end_ev)
        macro = file(macrofilename,'w')
        macro.write(macrostring)
        macro.close()
        process_count += 1
        try:
          command = ['qsub', macrofilename]
          process = subprocess.call(command)
          time.sleep(30)
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
