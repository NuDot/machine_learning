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
TIME = "16:00:00"



def main(argv):
   time.sleep(9000)
   # Setting the output Directory if it does not exist.
   # if not os.path.exists(OUT_DIR):
   #     os.mkdir(OUT_DIR )

   # Looking for the input file in given input directory.
  inputfiles = [("%s.%d.%d" % (os.path.basename(ifile).split('.')[0], 0, 1000)) for ifile in os.listdir(ZDAB_DIR)]
  # completefile = []
  # for prefix in ["data_", "indices_", "indptr_"]:
  #   completefile += [prefix + ifile for ifile in inputfiles]
  #   print len(completefile)
  outputfiles = [(ifile) for ifile in os.listdir(OUT_DIR)]

  zdabfiles = []
  for ifile in inputfiles:
    filecount = 0
    for cfile in outputfiles:
      if ifile in cfile:
        filecount += 1
    if filecount == 0:
      zdabfiles.append(ifile)



   # checkfiles = []

   # for idir in os.listdir(OUT_DIR):
   #    if "indptr" in idir:
   #      with cd(OUT_DIR + idir):
   #        if not os.path.isfile('8_9.json'):
   #          checkfiles.append(idir)


   # checkfiles = [ifile.replace("0.1000", "root") for ifile in checkfiles]

   # # checkfile2 = [ifile.replace("0.1000.json", "root") for ifile in os.listdir(CHECK_DIR) if "indptr" in ifile]

   # # checkfiles = checkfiles + checkfile2

   # zdabfiles = [ifile.replace("indptr_", "") for ifile in checkfiles]



   # # zdabfiles = np.setdiff1d(inputfiles,checkfiles)


   # print len(zdabfiles)

  for rootfile in zdabfiles:
    rootfile = rootfile.split('.')[0] + '.root'
    print rootfile
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
        macrofilename = 'shell_%s_range_%d_%d_r.sh' % (str(rootfile), start_ev, end_ev)
        macro = file(macrofilename,'w')
        macro.write(macrostring)
        macro.close()
        process_count += 1
        try:
          command = ['qsub', macrofilename]
          process = subprocess.call(command)
          time.sleep(60)
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
