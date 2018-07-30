import argparse
import math
import os
import json
from scipy import sparse
from random import *
import numpy as np
import time
from ROOT import TFile

#Global Variables
PMT_POSITION = []
for pmt in np.loadtxt("/projectnb/snoplus/machine_learning/prototype/pmt.txt").tolist():
  if (len(pmt)):
    if (pmt[-1] == 17.0):
      PMT_POSITION.append([pmt[-4], pmt[-3], pmt[-2]])
N_PMTS = len(PMT_POSITION)
COLS = int(math.sqrt(N_PMTS/2))
ROWS = COLS *2
RUN_TIMESTAMP = time.time()
MAX_PRESSURE = 10
FIRST_PHOTON = False

class clock:
  tick_interval = 1.5
  final_time = 45
  initiated=False
  clock_array = np.arange(0, final_time, tick_interval)

  def __init__(self, initial_time):
    clock.initiated=True
    self.clock_array = self.clock_array + initial_time - 0.5

  def tick(self, time):
    return self.clock_array[self.clock_array < time].argmax()

  def clock_size(self):
    return len(self.clock_array)

def xyz_to_phi_theta(x, y, z):
   phi = math.atan2(y, x)
   r = (x**2 + y**2 + z**2)**.5
   theta = math.acos(z / r)
   return phi, theta


def phi_theta_to_row_col(phi, theta, rows=ROWS, cols=COLS):
   # phi is in [-pi, pi], theta is in [0, pi]
   row = min(rows/2 + (math.floor((rows/2)*phi/math.pi)), rows-1)
   row = max(row, 0)
   col = min(math.floor(cols*theta/math.pi), cols-1);
   col = max(col, 0)
   return int(row), int(col)

def pmt_allocator(x,y,z, photocoverage):

   vec1 = [x,y,z]
   pmt_index = np.array([calculate_angle(vec1, vec2) for vec2 in PMT_POSITION]).argmin()
   x2,y2,z2 = PMT_POSITION[pmt_index]
   detector_radius = (x2**2 + y2**2 + z2**2)**0.5
   single_coverage = 4 * math.pi * detector_radius**2 * photocoverage / float(N_PMTS)
   coverage_radius = (single_coverage / float(math.pi))**0.5
   coverage_angle = math.tan(coverage_radius/float(detector_radius))
   if ((calculate_angle(vec1, PMT_POSITION[pmt_index])) > coverage_angle):
    pmt_index = -1
   return pmt_index


def calculate_angle(vec1, vec2):
  x1,y1,z1 = vec1
  x2,y2,z2 = vec2
  inner_product = x1*x2 + y1*y2 + z1*z2
  len1 = (x1**2 + y1**2 + z1**2)**0.5
  len2 = (x2**2 + y2**2 + z2**2)**0.5
  return math.acos(float(inner_product)/float(len1*len2))




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

def set_clock(tree, evt):
  tree.GetEntry(evt)
  time_array = []
  for i in range(tree.N_phot):
    time_array.append(tree.PE_time[i])
  return clock(np.array(time_array).min())



def transcribe_hits(input, theta, phi, outputdir, start_evt, end_evt, elow, ehi):
  current_clock = clock(0)
  f1 = TFile(input)
  tree = f1.Get("epgTree")
  n_evts = tree.GetEntries()
  end_evt = min(n_evts, end_evt)
  n_qe_values = MAX_PRESSURE + 1
  photocoverage_scale = list(np.linspace(1.0,0.2,9))
  shrink_list = []
  feature_map_collections = np.zeros(((((len(photocoverage_scale), n_qe_values, (end_evt-start_evt), current_clock.clock_size(), ROWS, COLS)))))
  for evt_index in range(start_evt, end_evt):
    tree.GetEntry(evt_index)
    if (tree.edep < elow) or (tree.edep > ehi):
      shrink_list.append(evt_index)
      continue
    current_clock = set_clock(tree, evt_index)
    for i in range(tree.N_phot):
      for pressure_pc in photocoverage_scale:
        pmt_index = pmt_allocator(tree.x_hit[i],tree.y_hit[i],tree.z_hit[i], pressure_pc)
        if (pmt_index == -1):
          continue
        row, col = xyz_to_row_col(PMT_POSITION[pmt_index][0], PMT_POSITION[pmt_index][1], PMT_POSITION[pmt_index][2])
        for pressure_pe in range (0, n_qe_values):
          if (tree.PE_creation[i]) or random_decision(MAX_PRESSURE-pressure_pe):
            time_index = current_clock.tick(tree.PE_time[i])
            feature_map_collections[photocoverage_scale.index(pressure_pc)][pressure_pe][evt_index - start_evt][time_index][row][col] += 1.0
  feature_map_collections = np.delete(feature_map_collections, shrink_list ,2)
  for qcindex, qc in enumerate(feature_map_collections):
    for qeindex, qe in enumerate(qc):
      feature_map_collections[qcindex][qeindex] = np.divide(qe, (1.2 * qe.max()))
  
  print feature_map_collections.shape
  input_name = os.path.basename(input).split('.')[0]
  np.save(os.path.join(outputdir, "feature_map_collections.%s.%d.%d.npy" % (input_name, start_evt, end_evt)),
      feature_map_collections)
  return feature_map_collections

# def transcribe_hits(input, theta, phi, outputdir, start_evt, end_evt, elow, ehi):
#   current_clock = clock(0)
#   f1 = TFile(input)
#   tree = f1.Get("epgTree")
#   n_evts = tree.GetEntries()
#   end_evt = min(n_evts, end_evt)
#   n_qe_values = MAX_PRESSURE + 1
#   photocoverage_scale = list(np.linspace(1.0,0.2,9))
#   shrink_list = []
#   feature_map_collections = np.zeros(((((len(photocoverage_scale), n_qe_values, (end_evt-start_evt), current_clock.clock_size(), ROWS, COLS)))))
#   for evt_index in range(start_evt, end_evt):
#     tree.GetEntry(evt_index)
#     if (tree.edep < elow) or (tree.edep > ehi):
#       shrink_list.append(evt_index)
#       continue
#     current_clock = set_clock(tree, evt_index)
#     #for i in range(tree.N_phot):
#     for i in range(10): 
#       for pressure_pc in photocoverage_scale:
#         pmt_index = pmt_allocator(tree.x_hit[i],tree.y_hit[i],tree.z_hit[i], pressure_pc)
#         if (pmt_index == -1):
#           continue
#         row, col = xyz_to_row_col(PMT_POSITION[pmt_index][0], PMT_POSITION[pmt_index][1], PMT_POSITION[pmt_index][2])
#         for pressure_pe in range (0, n_qe_values):
#           if (tree.PE_creation[i]) or random_decision(MAX_PRESSURE-pressure_pe):
#             time_index = current_clock.tick(tree.PE_time[i])
#             feature_map_collections[photocoverage_scale.index(pressure_pc)][pressure_pe][evt_index - start_evt][time_index][row][col] += 1.0
#   feature_map_collections = np.delete(feature_map_collections, shrink_list ,2)
#   #print feature_map_collections.shape
#   dim1, dim2, dim3, dim4, dim5, dim6 = feature_map_collections.shape
#   lst = [[[[[] for i4 in range(dim4)] for i3 in range(dim3)] for i2 in range(dim2)] for i1 in range(dim1)]
#   #print lst
#   data = lst
#   indices = lst
#   indptr = lst
#   for qcindex, qc in enumerate(feature_map_collections):
#     for qeindex, qe in enumerate(qc): 
#       currentEntry = qe
#       if (qe.max() != 0):
#         currentEntry = np.divide(qe, (1.2 * qe.max()))
#       for evt_index, evt in enumerate(currentEntry):
#         for time_index, maps in enumerate(evt):
#           sparse_map = sparse.csr_matrix(maps)
#           print sparse_map.data
#           data[qcindex][qeindex][evt_index][time_index] = list(sparse_map.data)
#           indices[qcindex][qeindex][evt_index][time_index] = list(sparse_map.indices)
#           indices[qcindex][qeindex][evt_index][time_index] = list(sparse_map.indptr)
#   print indices[0][0][0]
#   input_name = os.path.basename(input).split('.')[0]
#   np.save(os.path.join(outputdir, "feature_map_collections.%s.%d.%d.npy" % (input_name, start_evt, end_evt)),
#       feature_map_collections)
#   return feature_map_collections




def main():
  #python /projectnb/snoplus/machine_learning/prototype/process_root_files.py --input /projectnb/snoplus/sphere_data/andrey_data/sph_out_1el_2p529MeV_center_rndDir_1k_20.root --outputdir /projectnb/snoplus/sphere_data/c10_2MeV/ --start 0 --end 1
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


  fmc = transcribe_hits(input=args.input, theta=args.theta, phi=args.phi, outputdir=args.outputdir, start_evt=args.start, end_evt=args.end, elow=args.elow, ehi=args.ehi)




main()
