import argparse
import math
import os
from random import *
import numpy as np
import time
from ROOT import TFile


#Global Variables
RUN_TIMESTAMP = time.time()
INITIAL_TIME = 32
FINAL_TIME = 38
TIME_STEP = 0.5
MAX_PRESSURE = 10
N_PMTS = 5000
COLS = int(math.sqrt(N_PMTS/2))
ROWS = COLS *2


def xyz_to_phi_theta(x, y, z):
   phi = math.atan2(y, x)
   r = (x**2 + y**2 + z**2)**.5
   theta = math.acos(z / r)
   return phi, theta


def phi_theta_to_row_col(phi, theta, rows=ROWS, cols=COLS):
   # phi is in [-pi, pi], theta is in [0, pi]
   row = min(rows/2 + (round((rows/2)*phi/math.pi)), rows-1)
   row = max(row, 0)
   col = min(round(cols*theta/math.pi), cols-1);
   col = max(col, 0)
   return int(row), int(col)


def xyz_to_row_col(x, y, z, rows=ROWS, cols=COLS):
   return phi_theta_to_row_col(*xyz_to_phi_theta(x, y, z), rows=rows, cols=cols)

def drange2(start, stop, step):
    numelements = int((stop-start)/float(step))
    for i in range(numelements+1):
            yield start + i*step

def random_decision(pressure):
    decision = False
    if (pressure > MAX_PRESSURE) or (pressure == 0):
      return False
    frac_pressure = float(pressure)/float(MAX_PRESSURE)
    return (random()<=frac_pressure)


def rotated(feature_map, theta, phi):
  row_rotation = int(math.fmod(theta, (2 * math.pi)) / (2 * math.pi) * ROWS)
  col_rotation = int(math.fmod(phi, math.pi) / math.pi * COLS)
  if not ((row_rotation == 0) or (row_rotation == 1)):
    top, bottom = np.split(feature_map, [row_rotation], axis=0)
    feature_map = np.concatenate((bottom, top), axis=0)
  if not ((col_rotation == 0) or (col_rotation == 1)):
    left, right = np.split(feature_map, [col_rotation], axis=1)
    feature_map = np.concatenate((right, left), axis = 1)
  return feature_map


def transcribe_hits(input, theta, phi, outputdir, start_evt, end_evt, elow, ehi):
  f1 = TFile(input)
  tree = f1.Get("epgTree")
  n_evts = tree.GetEntries()
  end_evt = min(n_evts, end_evt)
  n_time_cuts = int((FINAL_TIME - INITIAL_TIME) / TIME_STEP) + 1
  n_qe_values = MAX_PRESSURE + 1
  feature_map_collections = np.zeros((((n_time_cuts, n_qe_values, (end_evt-start_evt), ROWS, COLS))))
  for evt_index in range(start_evt, end_evt):
    #print evt_index
    tree.GetEntry(evt_index)
    if (tree.edep >= elow) and (tree.edep <= ehi):
      for i in range(tree.N_phot):
        row, col = xyz_to_row_col(tree.x_hit[i], tree.y_hit[i], tree.z_hit[i])
        for pressure_time in drange2(INITIAL_TIME, FINAL_TIME, TIME_STEP):
          if (tree.PE_time[i] >= INITIAL_TIME) and (tree.PE_time[i]  <= pressure_time):
            time_index = int((pressure_time - 32) / 0.5)
            for pressure_pe in range (0, 11):
              if (tree.PE_creation[i]) or random_decision(MAX_PRESSURE-pressure_pe):
                feature_map_collections[time_index][pressure_pe][evt_index - start_evt][row][col] += 1

  input_name = os.path.basename(input).split('.')[0]
  #print feature_map_collections.shape
  np.save(os.path.join(outputdir, "feature_map_collections.%s.%d.%d.npy" % (input_name, start_evt, end_evt)),
      feature_map_collections)




def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", default="/projectnb/snoplus/sphere_data/sph_out_1el_2p53_MeV_15k.root")
  parser.add_argument("--outputdir", default="/projectnb/snoplus/sphere_data")
  parser.add_argument("--type", "-t", help="Type of MC files, 1 ring or 2 ring", default = 1)
  parser.add_argument("--theta","-th", help="Rotate Camera with given theta(0 - 2pi)",type = float, default = 0)
  parser.add_argument("--phi","-ph", help="Rotate Camera with given phi(0 - pi)",type = float, default = 0)
  parser.add_argument("--start", help="start event",type = int, default = 0)
  parser.add_argument("--end", help="end event",type = int, default = 1000000000)
  parser.add_argument("--elow", help="lower energy cut",type = float, default = 0.0)
  parser.add_argument("--ehi", help="upper energy cut",type = float, default = 10000000.0)
  args = parser.parse_args()


  transcribe_hits(input=args.input, theta=args.theta, phi=args.phi, outputdir=args.outputdir, start_evt=args.start, end_evt=args.end, elow=args.elow, ehi=args.ehi)





main()
