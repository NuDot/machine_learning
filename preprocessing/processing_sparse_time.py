###########################
# Author: Aobo Li
############################
# History:
# Sep.19, 2018 - First Version
#################################
# Purpose:
# This code is used to convert MC simulated .root file into a 2D grid,
# then it saves the code as a CSR sparse matrix.
#############################################################
import argparse
import math
import os
import json
import pickle
from scipy import sparse
from scipy import constants as const
from random import *
import numpy as np
import time
from ROOT import TFile
from datetime import datetime
from tqdm import tqdm

#Global Variables
PMT_POSITION = []
for pmt in np.loadtxt("/projectnb/snoplus/machine_learning/prototype/pmt.txt").tolist():
  if (len(pmt)):
    if (pmt[-1] == 17.0):
      PMT_POSITION.append([pmt[-4], pmt[-3], pmt[-2]])
N_PMTS = len(PMT_POSITION)
COLS = math.sqrt(N_PMTS/2)
COLS = int(math.ceil(COLS / 2) * 2)
print COLS
ROWS = COLS
RUN_TIMESTAMP = time.time()
MAX_PRESSURE = 10
QE_FACTOR = 0.5
FIRST_PHOTON = False

#Clock class: dealing with the input time of the photon hit.
class clock:
  tick_interval = 1.5
  final_time = 45
  initiated=False
  clock_array = np.arange(-6, final_time, tick_interval)

  def __init__(self, initial_time):
    clock.initiated=True
    self.clock_array = self.clock_array + initial_time - 0.5

  def tick(self, time):
    if (time < self.clock_array[0]):
      return 0
    return self.clock_array[self.clock_array < time].argmax()

  def clock_size(self):
    return len(self.clock_array)

def xyz_to_phi_theta(x, y, z):
   phi = math.atan2(y, x)
   r = (x**2 + y**2 + z**2)**.5
   theta = math.acos(z / r)
   return phi, theta

#change directory
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


# Convert the phi theta information to row and column index in 2D grid
def phi_theta_to_row_col(phi, theta, rows=ROWS, cols=COLS):
   # phi is in [-pi, pi], theta is in [0, pi]
   row = min(rows/2 + (math.floor((rows/2)*phi/math.pi)), rows-1)
   row = max(row, 0)
   col = min(math.floor(cols*theta/math.pi), cols-1);
   col = max(col, 0)
   return int(row), int(col)

# Set up the current photon hit position with the PMT closest to it
def pmt_setup(vec1):
    pmt_index = np.array([calculate_angle(vec1, vec2) for vec2 in PMT_POSITION]).argmin()
    x2,y2,z2 = PMT_POSITION[pmt_index]
    detector_radius = (x2**2 + y2**2 + z2**2)**0.5
    pmt_angle = calculate_angle(vec1, PMT_POSITION[pmt_index])
    return pmt_index, detector_radius, pmt_angle

#Attempting to allocate the photon to the PMT, return whether the photon hit falls in 
#the grey disk range of the setup pmt
def pmt_allocator(pmt_angle, detector_radius, pmt_radius):
   coverage_angle = math.asin(pmt_radius/float(detector_radius))
   return (pmt_angle <= coverage_angle)

# Calculating the angle between two input vectors
def calculate_angle(vec1, vec2):
  x1,y1,z1 = vec1
  x2,y2,z2 = vec2
  inner_product = x1*x2 + y1*y2 + z1*z2
  len1 = (x1**2 + y1**2 + z1**2)**0.5
  len2 = (x2**2 + y2**2 + z2**2)**0.5
  return math.acos(float(inner_product)/float(len1*len2))

# Converting Cartesian position to 2D Grid
def xyz_to_row_col(x, y, z, rows=ROWS, cols=COLS):
   return phi_theta_to_row_col(*xyz_to_phi_theta(x, y, z), rows=rows, cols=cols)

# Making a random decision based on the input pressure, used to vary quantum efficiency.
def random_decision(pressure):
    decision = False
    if (pressure > MAX_PRESSURE) or (pressure == 0):
      return False
    frac_pressure = float(pressure)/float(MAX_PRESSURE)
    return (random()<=(frac_pressure * QE_FACTOR))

# Rotate the input image(renewal needed)
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

# Set up the clocl to start ticking on the first incoming photon of a given events.
def set_clock(tree, evt):
  tree.GetEntry(evt)
  time_array = []
  for i in range(tree.N_phot):
    time_array.append(tree.PE_time[i])
  return clock(np.array(time_array).min())

# Save input file as a .json file.
def savefile(saved_file, appendix, filename, pathname):
    if not os.path.exists(pathname):
     os.mkdir(pathname)
    with cd(pathname):
        with open(filename, 'w') as datafile:
          json.dump(saved_file, datafile)

#Extend the time of photon track through the buffer oil region.
def time_extend(time, wavelength):
  return time + (2.5 / (const.c/(1.44983 + 12860.6/(wavelength**2)))) * 1E9

# Smear the input photon time as a gaussian with given TTS, for KamLAND it's 1.0s
def smearing_time(time):
  return np.random.normal(loc=time, scale=1.0)

# Transcribe hits to a 6D pressure maps.
def transcribe_hits(input, theta, phi, outputdir, start_evt, end_evt, elow, ehi):
  current_clock = clock(0)
  f1 = TFile(input)
  tree = f1.Get("epgTree")
  n_evts = tree.GetEntries()
  end_evt = min(n_evts, end_evt)
  n_qe_values = MAX_PRESSURE + 1
  photocoverage_scale = list(np.linspace(0.3173, 0.2159, 9))
  shrink_list = []
  vertex_list = []
  direction_list = []
  feature_map_collections = np.zeros(((((len(photocoverage_scale), n_qe_values, (end_evt-start_evt), current_clock.clock_size(), ROWS, COLS)))))
  input_name = os.path.basename(input).split('.')[0]
  with open(os.path.join(outputdir, "eventfile_%s.%d.%d.pickle" % (input_name, start_evt, end_evt)), 'wb') as handle:
    for evt_index in tqdm(range(start_evt, end_evt)):
      event_dict = {}
      tree.GetEntry(evt_index)
      rlength = (tree.trueVtxX**2 + tree.trueVtxY**2 + tree.trueVtxZ**2)**0.5
      if (rlength > 150.4):
        shrink_list.append(evt_index)
        continue
      event_dict['id'] = evt_index
      event_dict['vertex'] = [tree.trueVtxX,tree.trueVtxY,tree.trueVtxZ]
      event_dict['direction'] = [tree.trueDirX,tree.trueDirY,tree.trueDirZ]
      #for i in range(tree.N_phot):
      current_clock = set_clock(tree, evt_index)
      #vertex_list.append([tree.trueVtxX,tree.trueVtxY,tree.trueVtxZ])
      #direction_list.append([tree.trueDirX,tree.trueDirY,tree.trueDirZ])
      for i in range(tree.N_phot):
        if (tree.process[i] == 0):
          continue
        ###############################################
        pmt_index, detector_radius, pmt_angle = pmt_setup([tree.x_hit[i],tree.y_hit[i],tree.z_hit[i]])
        ###############################################
        for pressure_pc in photocoverage_scale:
          if (pmt_allocator(pmt_angle, detector_radius, pressure_pc)):
            row, col = xyz_to_row_col(PMT_POSITION[pmt_index][0], PMT_POSITION[pmt_index][1], PMT_POSITION[pmt_index][2])
            for pressure_pe in range (0, n_qe_values):
              if (tree.PE_creation[i]) or random_decision(MAX_PRESSURE-pressure_pe):
                true_time = time_extend(tree.true_time[i], tree.photon_wavelength[i])
                time_index = current_clock.tick(smearing_time(true_time))
                feature_map_collections[photocoverage_scale.index(pressure_pc)][pressure_pe][evt_index - start_evt][time_index][row][col] += 1.0
      #event_map = np.transpose(feature_map_collections, (2,0,1,3,4,5))[evt_index - start_evt]
      for qcindex, qc in enumerate(feature_map_collections):
        for qeindex, qe in enumerate(qc):
            time_sequence = []
            for time_index, maps in enumerate(qe[evt_index - start_evt]):
               time_sequence.append(sparse.csr_matrix(maps))
            event_dict[str(qcindex) + '_' + str(qeindex)] = time_sequence
      #         data[evt_index][time_index] = map(float, sparse_map.data)
      #         indices[evt_index][time_index] = map(int, sparse_map.indices)
      #         indptr[evt_index][time_index] = map(int, sparse_map.indptr)
      # sparse_map = sparse.csr_matrix(event_map)
      # event_dict['data'] = map(float, sparse_map.data)
      # event_dict['indices'] = map(int, sparse_map.indices)
      # event_dict['indptr'] = map(int, sparse_map.indptr)
      pickle.dump(event_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # print(vertex_list, direction_list)
    # feature_map_collections = np.delete(feature_map_collections, shrink_list ,2)
    # dim1, dim2, dim3, dim4, dim5, dim6 = feature_map_collections.shape
    # lst = np.zeros((dim3,dim4,1))
    # input_name = os.path.basename(input).split('.')[0]
    # data_path = os.path.join(outputdir, "data_%s.%d.%d" % (input_name, start_evt, end_evt))
    # indices_path = os.path.join(outputdir, "indices_%s.%d.%d" % (input_name, start_evt, end_evt))
    # indptr_path = os.path.join(outputdir, "indptr_%s.%d.%d" % (input_name, start_evt, end_evt))
    # vertex_path = os.path.join(outputdir, "vertex_%s.%d.%d" % (input_name, start_evt, end_evt))
    # direction_path = os.path.join(outputdir, "direction_%s.%d.%d" % (input_name, start_evt, end_evt))
    # data = lst.tolist()
    # indices = lst.tolist()
    # indptr = lst.tolist()
    # for qcindex, qc in enumerate(feature_map_collections):
    #   for qeindex, qe in enumerate(qc): 
    #     currentEntry = qe
    #     if (qe.max() != 0):
    #       currentEntry = np.divide(qe, (1.2 * qe.max()))
    #     for evt_index, evt in enumerate(currentEntry):
    #       for time_index, maps in enumerate(evt):
    #         sparse_map = sparse.csr_matrix(maps)
    #         data[evt_index][time_index] = map(float, sparse_map.data)
    #         indices[evt_index][time_index] = map(int, sparse_map.indices)
    #         indptr[evt_index][time_index] = map(int, sparse_map.indptr)
    #     qcqename = str(qcindex) + '_' + str(qeindex) + '.json'
    #     savefile(data, 'data', qcqename, data_path)
    #     savefile(indices, 'indices', qcqename, indices_path)
    #     savefile(indptr, 'indptr', qcqename, indptr_path)
    #     savefile(vertex_list, 'vertex', qcqename, vertex_path)
    #     savefile(direction_list, 'direction', qcqename, direction_path)
  return feature_map_collections




def main():
  #python /projectnb/snoplus/machine_learning/prototype/processing_sparse.py --input /projectnb/snoplus/sphere_data/input/sph_out_C10_dVrndVtx_3p0mSphere_1k_16.root --outputdir /projectnb/snoplus/sphere_data/c10_2MeV/ --start 2 --end 3
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
