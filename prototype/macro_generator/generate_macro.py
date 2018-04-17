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

import settings



def main(argv):

   # Setting the output Directory if it does not exist.
   if not os.path.exists(settings.OUT_DIR):
       os.mkdir(settings.OUT_DIR )

   # Looking for the input file in given input directory.
   zdabfiles = [(ifile) for ifile in os.listdir(settings.ZDAB_DIR)]
   
   now = time.time()
   run_time = os.path.getmtime(settings.ZDAB_DIR + '/' + zdabfiles[0])

   for macrofile in zdabfiles:
    production_name = get_run_number_fromfile(macrofile)
    if not (production_name == 0):
      for i in range(20,25):
        inputstring = str(settings.ZDAB_DIR + macrofile)
        outputstring = str(settings.OUT_DIR + str(production_name) + '_' + str(i) + '.root')
        timestring = str(settings.TIME)
        macrotemplate = string.Template(open('template_macro.mac', 'r').read())
        if not os.path.exists(settings.MACRO_DIR):
          os.mkdir(settings.MACRO_DIR )
        with cd(settings.MACRO_DIR):
          macrostring = macrotemplate.substitute(LOADINPUT=inputstring, OUTPUTDIRECTORY=outputstring, TIME=settings.TIME)
          macrofilename = 'theia_mc_%s_%s.sh' % (str(production_name), str(i))
          macro = file(macrofilename,'w')
          macro.write(macrostring)
          macro.close()
          try:
            command = ['qsub', macrofilename]
            process = subprocess.call(command)
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


def get_run_number_fromfile(file):
   '''
   Retrieve run number from file name
   '''
   zdab_regex = re.compile(r"^theia_(\S+)\.mac$")
   matches = zdab_regex.match(file)
   if matches:
      return str(matches.group(1))
   else:
      return 0


if __name__=="__main__":

   print sys.exit(main(sys.argv[1:]))
