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

#OUT_DIR = "/projectnb/snoplus/sphere_data/Xe136_C10/"
OUT_DIR = "/projectnb/snoplus/sphere_data/Xe136_C10_balloon/"
ZDAB_DIR = "/projectnb/snoplus/sphere_data/json_new/"
C10_DIR = "/projectnb/snoplus/sphere_data/c10_2MeV"
MACRO_DIR = "/projectnb/snoplus/machine_learning/data/networktrain_v2/"
TIME = "12:00:00"
TIME_PRESSURE = 9
QE_PRESSSURE = 11
PROCESSOR = "/projectnb/snoplus/machine_learning/prototype/network.py"
#KEYWORD = "center"
KEYWORD = "dVrndVtx_3p0mSphere"



def main(argv):
    # Setting the output Directory if it does not exist.
    if not os.path.exists(OUT_DIR):
       os.mkdir(OUT_DIR )

    if not os.path.exists(MACRO_DIR):
       os.mkdir(MACRO_DIR )
    ######################################################
    #Looking for the input file in given input directory.
    # zdabfiles = [(ifile) for ifile in sorted(os.listdir(ZDAB_DIR), key = lambda f:(int(get_name(f)[2]) , get_name(f)[0])) if KEYWORD in ifile]

    # print len(zdabfiles)

    # zdabfiles_pure = []
    # for target_run in zdabfiles:
    #   if get_name(target_run) == 0:
    #     continue
    #   if 'Te130' in target_run:
    #     continue
    #   if not len(os.listdir(ZDAB_DIR + target_run)) == 99:
    #     continue
    #   target_run_number = '_' + get_name(target_run)[2] + '.'
    #   if len([f for f in zdabfiles if target_run_number in f]) % 3 == 0:
    #     zdabfiles_pure.append(target_run)
    # zdabfiles = zdabfiles_pure
    # print len(zdabfiles)

    # filename_array = {}
    # for npyfile in zdabfiles:
    #   filetype, eventname, filenumber = get_name(npyfile)
    #   key = str(eventname)
    #   if key not in filename_array.keys():
    #     filename_array[key] = []
    #   filename_array[key].append(str(ZDAB_DIR + npyfile))
    #########################################################

    training_combo = {}
    # for key in filename_array.keys():
    #   training_combo[key] = []
    #   with cd(MACRO_DIR):
    #     writefile = open(str(key + KEYWORD +'.dat'),"w")
    #     for filename in filename_array[key]:
    #       writefile.write(filename + '\n')
    #     writefile.close()

    #training_combo['Te130'].append('1el_2p529MeVrndDir')
    training_combo['Xe136'] = ['C10']

    with cd(MACRO_DIR):
      for signal, backgrounds in training_combo.iteritems():
        if (len(backgrounds) == 0):
          continue
        for background in backgrounds:
          macrotemplate = string.Template(open('/projectnb/snoplus/machine_learning/data/train.sh', 'r').read())
          signal_list = str(MACRO_DIR + signal + KEYWORD + '.dat')
          background_list = str(MACRO_DIR + background + KEYWORD + '.dat')
          for time_index in range(0, TIME_PRESSURE):
            for qe_index in range(0, QE_PRESSSURE):
          # for time_index in range(6, 7):
          #   for qe_index in range(6, 7):
              outputstring = OUT_DIR + signal + background + 'time' + str(time_index) + '_' + 'qe' + str(qe_index)
              namestring = 'T' + str(time_index) + '_' +  str(qe_index)
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
              #print macrostring
              macrofilename = 'train_signal_%s_background_%s_time_%d_qe_%d.qsub' % (str(signal), str(background), time_index, qe_index)
              macro = file(macrofilename,'w')
              macro.write(macrostring)
              macro.close()
              try:
                command = ['qsub', str('/projectnb/snoplus/machine_learning/data/networktrain_v2/' + macrofilename)]
                process = subprocess.call(command)
                #time.sleep(30)
              except Exception as error:
                return 0
      return 1



def get_name(file):
   '''
   Retrieve run number from file name
   '''
   zdab_regex = re.compile(r"^([a-z]+)_sph_out_(.+)_[a-z].*_1k_(\d+)\.0\.\d+$")
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
