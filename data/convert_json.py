import argparse
import os
import sys
import time
import json

from scipy import sparse
import numpy as np
from pandas import read_json
from datetime import datetime

OUT_DIR = "/projectnb/snoplus/sphere_data/json_new/"
INPUT_DIR = "/projectnb/snoplus/sphere_data/npy_new/"

def main(argv):

    # Setting the output Directory if it does not exist.
	if not os.path.exists(OUT_DIR):
		os.mkdir(OUT_DIR )

	with cd(INPUT_DIR):
		inputfiles = [(ifile) for ifile in sorted(os.listdir(INPUT_DIR), key=os.path.getmtime) if '.json' in ifile]

	for json_file in inputfiles:
		print json_file
		inputfilepath = INPUT_DIR + json_file
		outputdirpath = OUT_DIR + json_file.replace(".json",'')
		if not os.path.exists(outputdirpath):
   			os.mkdir(outputdirpath)
   		with cd(outputdirpath):
			if (split_files(inputfilepath)):
				os.remove(inputfilepath)

def split_files(npy_filename):
	try:
		with open(npy_filename) as json_data:
			data = read_json(json_data, orient='records')
			for pc in range(len(data)):
				for qe in range(len(data[0])):
					filename = str(pc) + '_' + str(qe) + '.json'
					with open(filename,'w') as datafile:
						json.dump(data[pc][qe], datafile)
		return True
	except:
		return False




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