import argparse
import math
from random import *
import numpy as np
import time
from ROOT import TFile


#Global Variables
INITIAL_TIME = 32
FINAL_TIME = 38
TIME_STEP =0.5
MAX_PRESSURE=10
N_PMTS=5000
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


def transcribe_hits(tree, theta, phi):
  n_evts = tree.GetEntries()
  n_time_cuts = int((FINAL_TIME - INITIAL_TIME) / TIME_STEP) + 1
  n_qe_values = MAX_PRESSURE + 1
  feature_map_collections = np.zeros((((n_evts, n_time_cuts,n_qe_values,ROWS,COLS))))
  for evt_index in range(n_evts):
    tree.GetEntry(evt_index)

    for i in range(tree.N_phot):
      row, col = xyz_to_row_col(tree.x_hit[i], tree.y_hit[i], tree.z_hit[i])
      for pressure_time in drange2(INITIAL_TIME, FINAL_TIME, TIME_STEP):
        if (tree.PE_time[i] >= INITIAL_TIME) and (tree.PE_time[i]  <= pressure_time):
          time_index = int((pressure_time - 32) / 0.5)
          for pressure_pe in range (0, 11):
            if (tree.PE_creation[i]) or random_decision(MAX_PRESSURE-pressure_pe):
              feature_map_collections[evt_index][time_index][pressure_pe][row][col] += 100

    for index_f, first_layer in enumerate(feature_map_collections[evt_index]):
      for index_s, target_map in enumerate(first_layer):
        target_map = rotated(target_map, theta=theta, phi=phi)
        feature_map_collections[evt_index][index_f][index_s] = target_map


  np.save("feature_map_collections_%d.npy" % time.time(),
      feature_map_collections)



def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", default="sph_out_topology180_center_NoMultScat_100.root")
  parser.add_argument("--type", "-t", help="Type of MC files, 1 ring or 2 ring", default = 1)
  parser.add_argument("--theta","-th", help="Rotate Camera with given theta(0 - 2pi)",type = float, default = 0)
  parser.add_argument("--phi","-ph", help="Rotate Camera with given phi(0 - pi)",type = float, default = 0)
  args = parser.parse_args()


  f1 = TFile(args.input)

  tree = f1.Get("epgTree")

  transcribe_hits(tree, theta=args.theta, phi=args.phi)





main()
