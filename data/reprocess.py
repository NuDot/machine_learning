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

OUT_DIR = "/projectnb/snoplus/sphere_data/training_output/"
ZDAB_DIR = "/projectnb/snoplus/sphere_data/andrey_npy/"
MACRO_DIR = "/projectnb/snoplus/machine_learning/data/networktrain/"
TIME = "40:00:00"
# ;
PROCESSOR = "/projectnb/snoplus/machine_learning/prototype/elagin_classifier_in_keras.py"



def main(argv):

    # Setting the output Directory if it does not exist.
    if not os.path.exists(OUT_DIR):
       os.mkdir(OUT_DIR )

    if not os.path.exists(MACRO_DIR):
       os.mkdir(MACRO_DIR )

    zdabfiles = [(ifile) for ifile in os.listdir(ZDAB_DIR) if '.npy' in ifile]

    filename_array = {}
    for npyfile in zdabfiles:
      filetype, randdir, evtnum = get_name(npyfile)
      key = str(filetype + randdir)
      if key not in filename_array.keys():
        filename_array[key] = []
      filename_array[key].append(str(ZDAB_DIR + npyfile))

    training_combo = {}
    for key in filename_array.keys():
      training_combo[key] = {}
      with cd(MACRO_DIR):
        writefile = open(str(key+'.dat'),"w")
        for filename in filename_array[key]:
          writefile.write(filename + '\n')
        writefile.close()

    training_combo['Te130']['1el_2p529MeVrndDir'] = []
    training_combo['Te130']['C10'] = []

    for ifile in os.listdir(OUT_DIR):
      dir_regex = re.compile(r"^Te130(.+)?time(\d+)_qe(\d+)$")
      matches = dir_regex.match(ifile)
      if matches:
        if (os.listdir(OUT_DIR + ifile) == []):
          training_combo['Te130'][matches.group(1)].append([matches.group(2), matches.group(3)])

    print training_combo

        

    #Looking for the input file in given input directory.

    with cd(MACRO_DIR):
      for signal, backgrounds in training_combo.iteritems():
        if (len(backgrounds) == 0):
          continue
        for background in backgrounds:
          data_list = training_combo[signal][background]
          if (len(data_list) == 0):
            continue
          # time_list = [int(data[0]) for data in data_list]
          # qe_list = [int(data[1]) for data in data_list]
          macrotemplate = string.Template(open('/projectnb/snoplus/machine_learning/data/train.sh', 'r').read())
          signal_list = str(signal + '.dat')
          background_list = str(background + '.dat')
          for data in data_list:
              time_index = int(data[0])
              qe_index = int(data[1])
              outputstring = OUT_DIR + signal + background + 'time' + str(time_index) + '_' + 'qe' + str(qe_index)
              namestring = 'time' + str(time_index) + '_' + 'qe' + str(qe_index)
              if not os.path.exists(outputstring):
                os.mkdir(outputstring)
              macrostring = macrotemplate.substitute(TIME=str(TIME),
                                                    NAME=namestring,
                                                    PROCESSOR=PROCESSOR,
                                                    SGL=signal_list, 
                                                    BGL=background_list,
                                                    SG=signal,
                                                    BG=background,
                                                    OUTDIR=outputstring,
                                                    TIME_PARA=time_index,
                                                    QE_PARA=qe_index)
              macrofilename = 'train_signal_%s_background_%s_time_%d_qe_%d.qsub' % (str(signal), str(background), time_index, qe_index)
              #print macrostring
              macro = file(macrofilename,'w')
              macro.write(macrostring)
              macro.close()
              try:
                command = ['qsub', str('/projectnb/snoplus/machine_learning/data/networktrain/' + macrofilename)]
                process = subprocess.call(command)
                time.sleep(600)
              except Exception as error:
                return 0
      return 1

def get_name(file):
   '''
   Retrieve run number from file name
   '''
   zdab_regex = re.compile(r"^feature_map_collections\.sph_out_(.+)_center_?(.*)_1k_\d+\.0\.(\d+)\.npy$")
   matches = zdab_regex.match(file)
   if matches:
      return str(matches.group(1)), str(matches.group(2)), str(matches.group(3))
   else:
      return 0

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
