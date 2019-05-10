###########################
# Author: Aobo Li
############################
# History:
# Jan.22, 2019 - First Version
#################################
# Purpose:
# This code is used to convert MC simulated .root file into a 2D square grid,
# then it saves the code as a CSR sparse matrix in .pickle format.
#############################################################
import argparse
import math
import os
import json
import pickle
from scipy import sparse
from scipy import constants as const
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from random import *
import numpy as np
import time
from ROOT import TFile
from datetime import datetime
from tqdm import tqdm
from random import randint

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


# #Global Variables
# PMT_POSITION = {}
# 20_inch_PMT_position
# for pmt in np.loadtxt("/projectnb/snoplus/machine_learning/prototype/pmt.txt").tolist():
#   if (len(pmt)):
#     if (pmt[-1] == 17.0):
#       PMT_POSITION.append([pmt[-4], pmt[-3], pmt[-2]])
# for pmt in np.loadtxt("/project/snoplus/KamLAND-Zen/base-root-analysis/pmt_xyz.dat"):
#   if (np.linalg.norm(pmt[1:])>= 900.0):
#     continue
#   PMT_POSITION[int(pmt[0])] = pmt[1:] / 100.0

PMT_POSITION = []
for pmt in np.loadtxt("/projectnb/snoplus/machine_learning/prototype/pmt.txt").tolist():
  if (len(pmt)):
    if (pmt[-1] == 17.0):
      PMT_POSITION.append([pmt[-4], pmt[-3], pmt[-2]])

# N_PMTS = len(PMT_POSITION)
COLS = 38
ROWS = COLS
# RUN_TIMESTAMP = time.time()
# MAX_PRESSURE = 10
# QE_FACTOR = 0.5
# FIRST_PHOTON = False

# def PMT_setup(pmt_file_with_index, pmt_file_with_size):
#   PMT_POSITION = {}
#   large_PMT_position = []
#   # for pmt in np.loadtxt("/projectnb/snoplus/machine_learning/prototype/pmt.txt").tolist():
#   #   if (len(pmt)):
#   #     if (pmt[-1] == 20.0):
#   #       large_PMT_position.append([pmt[-4], pmt[-3], pmt[-2]])
#   large_PMT_position = np.fromfile(pmt_file_with_size, sep=' ').reshape(-1,7)
#   twenty_inch_index = np.argwhere(large_PMT_position[:,-1].flatten() == 20)
#   large_PMT_position = large_PMT_position[twenty_inch_index].reshape(-1,7)[:,3:6]
#   twenty_inch_id = []
#   for pmt in np.loadtxt(pmt_file_with_index):
#     current_pmt_pos = pmt[1:] / 100.0
#     if (np.linalg.norm(current_pmt_pos)>= 900.0):
#       continue
#     if np.linalg.norm(large_PMT_position - current_pmt_pos, axis=1).min() < 0.55:
#       print(current_pmt_pos, large_PMT_position[np.linalg.norm(large_PMT_position - current_pmt_pos, axis=1).argmin()])
#       twenty_inch_id.append(pmt[0])
#     PMT_POSITION[int(pmt[0])] = current_pmt_pos
#   # plt.hist(min_dis, bins=50)
#   # plt.show()

#   # N_PMTS = len(PMT_POSITION)
#   # COLS = 25
#   # print(COLS)
#   # ROWS = COLS
#   return PMT_POSITION, np.array(twenty_inch_id)


# def PMT_setup(pmt_file_with_index):
#   PMT_POSITION = {}
#   for pmt in np.loadtxt(pmt_file_with_index):
#     current_pmt_pos = pmt[1:] / 100.0
#     if (np.linalg.norm(current_pmt_pos)>= 900.0):
#       continue
#     PMT_POSITION[int(pmt[0])] = current_pmt_pos
#   return PMT_POSITION

#Clock class: dealing with the input time of the photon hit.
class clock:
  tick_interval = 20
  final_time = 200
  initiated=False
  clock_array = np.arange(-500, final_time, tick_interval)

  def __init__(self, initial_time):
    clock.initiated=True
    self.clock_array = self.clock_array + initial_time

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

def pmt_setup(vec1):
    pmt_index = np.array([calculate_angle(vec1, vec2) for vec2 in PMT_POSITION]).argmin()
    x2,y2,z2 = PMT_POSITION[pmt_index]
    detector_radius = (x2**2 + y2**2 + z2**2)**0.5
    pmt_angle = calculate_angle(vec1, PMT_POSITION[pmt_index])
    return pmt_index, detector_radius, pmt_angle

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

def pmt_allocator(pmt_angle, detector_radius, pmt_radius):
   coverage_angle = math.asin(pmt_radius/float(detector_radius))
   return (pmt_angle <= coverage_angle)

# Convert the phi theta information to row and column index in 2D grid
def phi_theta_to_row_col(phi, theta, rows=ROWS, cols=COLS):
   # phi is in [-pi, pi], theta is in [0, pi]
   row = min(rows/2 + (math.floor((rows/2)*phi/math.pi)), rows-1)
   row = max(row, 0)
   col = min(math.floor(cols*theta/math.pi), cols-1);
   col = max(col, 0)
   return int(row), int(col)

# Calculating the angle between two input vectors
def calculate_angle(vec1, vec2):
  x1,y1,z1 = vec1
  x2,y2,z2 = vec2
  inner_product = x1*x2 + y1*y2 + z1*z2
  len1 = (x1**2 + y1**2 + z1**2)**0.5
  len2 = (x2**2 + y2**2 + z2**2)**0.5
  return math.acos(float(inner_product)/float(len1*len2))

# Converting Cartesian position to 2D Grid
def xyz_to_row_col(pmt_index, PMT_POSITION,rows=ROWS, cols=COLS):
   x, y, z = tuple(PMT_POSITION[pmt_index])
   return phi_theta_to_row_col(*xyz_to_phi_theta(x, y, z), rows=rows, cols=cols)


# Set up the clock to start ticking on the first incoming photon of a given events.
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

# Transcribe hits to a 6D pressure maps.
def transcribe_hits(input, ainput, title):
  f1 = TFile(input)
  tree = f1.Get("gvf_tree")
  f2 = TFile(ainput)
  tree2 = f2.Get("epgTree")
  start_evt = 0
  end_evt = int(min(tree.GetEntries(), tree2.GetEntries()))
  input_name = os.path.basename(input).split('.')[0]
  timej = []
  timea = []
  for evt_index in tqdm(range(0,30)):
    ind_temp = randint(start_evt,end_evt)
    tree.GetEntry(ind_temp)
    tree2.GetEntry(ind_temp)
    # print(ind_temp)
    # #gettting PMT information for a event
    # pmt_map = np.array(tree.PMT_hit_list)
    timej += np.array(tree.PMT_time).tolist()
    for i in range(tree2.N_phot):
      if (tree2.PE_creation[i]):
        pmt_index, detector_radius, pmt_angle = pmt_setup([tree2.x_hit[i],tree2.y_hit[i],tree2.z_hit[i]])
        if (pmt_allocator(pmt_angle, detector_radius, 0.2159)):
          timea.append(tree2.PE_time[i])
  plt.figure(figsize=(10, 10))
  plt.subplot(211)
  plt.hist(timej, histtype='step', label='John\'s MC', normed=True, bins=np.linspace(-500, 200, 70))
  plt.hist(timea, histtype='step', label='Andrey\'s MC(Greydisc Corrected)', normed=True, bins=np.linspace(-500, 200, 70))
  plt.xlabel('Hit Time')
  plt.ylabel('Normalized Counts/10ns')
  plt.legend()
  plt.title(title)
  plt.subplot(212)
  timej = np.array(timej) - np.average(timej)
  timea = np.array(timea) - np.average(timea)
  plt.hist(timej, histtype='step', label='John\'s MC', normed=True, bins=np.linspace(-100, 100, 20))
  plt.hist(timea, histtype='step', label='Andrey\'s MC(Greydisc Corrected)', normed=True,  bins=np.linspace(-100, 100, 50))
  plt.xlabel('Hit Time')
  plt.ylabel('Normalized Counts/4ns')
  plt.legend()
  plt.tight_layout()
  plt.savefig(str(title) + '_hittime_dist.png')
  plt.show()
    #charge_signal = np.array(tree.PMT_charge)

    # good_pmt_map = pmt_map[np.where(np.absolute(time_signal) < 1000.0)]
    # print(pmt_map.shape, good_pmt_map[np.where(np.absolute(time_signal) > 0.5)].shape)
    # # pmt_separator = np.isin(pmt_map, twenty_inch_id)
    # twenty_inch_pmt = np.argwhere(np.absolute(time_signal) < 0.5)
    # seventeen_inch_pmt = np.argwhere(np.absolute(time_signal) > 0.5)
    # #print(np.where(charge_signal == 0.0))
    # # print(np.where(pmt_separator), np.where(np.invert(pmt_separator)))
    # good_pmt_list = good_pmt_map[seventeen_inch_pmt]
    # good_pmt_time_list = time_signal[seventeen_inch_pmt]
    # good_list_2 = time_signal[twenty_inch_pmt]

    # plt.hist2d(time_signal, charge_signal, norm=mpl.colors.LogNorm())
    # plt.xlabel('time')
    # plt.ylabel('charge')
    # plt.show()




  #   #Applying time cut:
  #   #  PMT hit time > 1000.0: some crazy time stamp
  #   #  Others: good events
  #   #=========================================================
  #   good_time_pmt = np.where(np.absolute(time_signal) < 1000.0)
  #   good_pmt_list = pmt_map[good_time_pmt]
  #   good_pmt_time_list = time_signal[good_time_pmt]
  #   good_pmt_charge_list = charge_signal[good_time_pmt]
  #   #row, col = xyz_to_row_col(PMT_POSITION[pmt_index][0], PMT_POSITION[pmt_index][1], PMT_POSITION[pmt_index][2])
  #   stacked_pmt_info = np.dstack((good_pmt_list, good_pmt_time_list, good_pmt_charge_list))[0]
  #   event = np.ndarray((current_clock.clock_size(),ROWS,COLS))
  #   for pmtinfo in stacked_pmt_info:
  #     if pmtinfo[-1] == 0.0:
  #       continue
  #     col, row = xyz_to_row_col(pmtinfo[0], PMT_POSITION)
  #     time = current_clock.tick(pmtinfo[1])
  #     event_array[evt_index-start_evt][time][row][col] += pmtinfo[-1]
  # #=======================================================
  # #Normalization-not implemented yet
  # #=======================================================
  # event_array = np.transpose(event_array, (1,0,2,3))
  # #np.set_printoptions(formatter={'float_kind':'{:f}'.format})
  # for time_indx, time_slice in enumerate(event_array):
  #   if np.count_nonzero(time_slice) == 0:
  #     continue
  #   # allocation_dic = {}
  #   normalization_vector = []
  #   for indx, event in enumerate(time_slice):
  #     if np.count_nonzero(event) == 0:
  #       continue
  #     normalization_vector += list(event.flatten()[np.flatnonzero(event)])
  #   if len(normalization_vector) >= 2:
  #     mean = np.mean(normalization_vector)
  #     std = np.std(normalization_vector)
  #     mean_matrix = np.zeros(time_slice.shape)
  #     std_matrix = np.ones(time_slice.shape)
  #     mean_matrix[np.where(time_slice != 0.0)] = mean
  #     std_matrix[np.where(time_slice != 0.0)] = std
  #     #print(mean_matrix,std_matrix)
  #     event_array[time_indx] = (time_slice - mean_matrix)/std_matrix
  # event_array = np.transpose(event_array, (1,0,2,3))
  # #=============================================================
  # with open(os.path.join(outputdir, "eventfile_%s.%d.%d.pickle" % (input_name, start_evt, end_evt)), 'wb') as handle:
  #   for id, evnt in enumerate(event_array):
  #     event_dict = {}
  #     event_dict['id'] = id
  #     time_sequence = []
  #     for time_index, maps in enumerate(evnt):
  #        time_sequence.append(sparse.csr_matrix(maps))
  #     event_dict['event'] = time_sequence
  #     pickle.dump(event_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
  return 0

def event_level(input, ainput, title):
  f1 = TFile(input)
  tree = f1.Get("gvf_tree")
  f2 = TFile(ainput)
  tree2 = f2.Get("epgTree")
  start_evt = 0
  end_evt = int(min(tree.GetEntries(), tree2.GetEntries()))
  edepj = []
  edepa = []
  for evt_index in tqdm(range(start_evt,end_evt)):
    tree.GetEntry(evt_index)
    tree2.GetEntry(evt_index)
    edepj.append(tree.energy)
    edepa.append(tree2.edep)
  plt.subplot(211)
  plt.hist(edepj, histtype='step', label='John\'s MC', bins=np.linspace(1.5,4,50))
  plt.hist(edepa, histtype='step', label='Andrey\'s MC', bins=np.linspace(1.5,4,50))
  plt.xlabel('Energy')
  plt.ylabel('Counts per 0.05 MeV')
  plt.legend()
  plt.title(title)
  plt.subplot(212)
  edepj = np.array(edepj) - np.average(edepj)
  edepa = np.array(edepa) - np.average(edepa)
  plt.hist(edepj, histtype='step', label='John\'s MC', bins=np.linspace(-1.25,1.25,50))
  plt.hist(edepa, histtype='step', label='Andrey\'s MC', bins=np.linspace(-1.25,1.25,50))
  plt.xlabel('Energy Spread')
  plt.ylabel('Counts per 0.05 MeV')
  plt.legend()
  plt.tight_layout()
  plt.savefig(str(title) + '_Energy_dist.png')
  plt.show()

def event_level2(input, ainput, title):
  f1 = TFile(input)
  tree = f1.Get("gvf_tree")
  f2 = TFile(ainput)
  tree2 = f2.Get("epgTree")
  start_evt = 0
  end_evt = int(min(tree.GetEntries(), tree2.GetEntries()))
  edepj = []
  edepa = []
  edepgd = []
  for evt_index in tqdm(range(start_evt,end_evt)):
    tree.GetEntry(evt_index)
    tree2.GetEntry(evt_index)
    edepj.append(tree.num_PMT_hit)
    if (tree2.edep > 2.3) and (tree2.edep<2.6):
      edepa.append(np.sum(np.array(tree2.PE_creation)))
      gd_pmt_count = 0
      for i in range(tree2.N_phot):
        if (tree2.PE_creation[i]):
          pmt_index, detector_radius, pmt_angle = pmt_setup([tree2.x_hit[i],tree2.y_hit[i],tree2.z_hit[i]])
          if (pmt_allocator(pmt_angle, detector_radius, 0.2159)):
            gd_pmt_count += 1
      edepgd.append(gd_pmt_count)
  plt.figure(figsize=(7, 7))
  plt.subplot(211)
  plt.title(title)
  plt.hist(edepj, histtype='step', label='John\'s MC', bins=30, normed=True)
  plt.hist(edepa, histtype='step', label='Andrey\'s MC', bins=30, normed=True)
  plt.hist(edepgd, histtype='step', label='Andrey\'s MC Greydisc Corrected', bins=30, normed=True)
  plt.xlabel('Nhit')
  plt.legend()
  plt.subplot(212)
  edepj = np.array(edepj) - np.average(edepj)
  edepa = np.array(edepa) - np.average(edepa)
  edepgd = np.array(edepgd) - np.average(edepgd)
  plt.hist(edepj, histtype='step', label='John\'s MC', bins=30, normed=True)
  plt.hist(edepa, histtype='step', label='Andrey\'s MC', bins=30, normed=True)
  plt.hist(edepgd, histtype='step', label='Andrey\'s MC Greydisc Corrected', bins=30, normed=True)
  plt.xlabel('Nhit Spread')
  plt.legend()
  plt.tight_layout()
  plt.savefig(str(title) + '_PMT_dist.png')

def john_time(input, ainput, title):
  type = 'Hittime'
  f1 = TFile(input)
  tree = f1.Get("gvf_tree")
  f2 = TFile(ainput)
  tree2 = f2.Get("gvf_tree")
  start_evt = 0
  end_evt = int(min(tree.GetEntries(), tree2.GetEntries()))
  input_name = os.path.basename(input).split('.')[0]
  timej = []
  timea = []
  rg = np.linspace(-300,-100,50)
  for evt_index in tqdm(range(0,5000)):
    ind_temp = randint(start_evt,end_evt)
    tree.GetEntry(ind_temp)
    tree2.GetEntry(ind_temp)
    # print(ind_temp)
    # #gettting PMT information for a event
    # pmt_map = np.array(tree.PMT_hit_list)
    timej += np.array(tree.PMT_time).tolist()
    timea += np.array(tree2.PMT_time).tolist()

  #print(np.unique(timej,return_counts=True), np.unique(timea,return_counts=True))
  plt.figure(figsize=(10, 10))
  plt.subplot(211)
  plt.hist(timej, histtype='step', label='John\'s MC Background', normed=True, bins=rg)
  plt.hist(timea, histtype='step', label='John\'s MC Signal',  bins=rg, normed=True)
  plt.xlabel('PMT ' + type)
  plt.ylabel('Counts')
  plt.legend()
  plt.title(title)
  plt.subplot(212)
  # timej = np.array(timej) - np.average(timej)
  # timea = np.array(timea) - np.average(timea)
  countj, rangej = np.histogram(timej, bins=rg, normed=True)
  counta, rangea = np.histogram(timea, bins=rg, normed=True)
  ratio = counta/countj
  # plt.hist(timej, histtype='step', label='John\'s MC Background', normed=True, bins=100)
  # plt.hist(timea, histtype='step', label='John\'s MC Signal', bins=100, normed=True)
  plt.scatter(rangej[:-1], ratio)
  plt.axhline(1.0, color = '#A9A9A9', linestyle='--')
  plt.xlabel('PMT ' + type + ' ratio')
  plt.ylabel('Signal/Background per 4 ns ' + type )
  plt.legend()
  plt.tight_layout()
  plt.savefig(str(title) + type + '_dist.png')
  plt.show()

def john_pe(input, ainput, title):
  type = 'Charge'
  f1 = TFile(input)
  tree = f1.Get("gvf_tree")
  f2 = TFile(ainput)
  tree2 = f2.Get("gvf_tree")
  start_evt = 0
  end_evt = int(min(tree.GetEntries(), tree2.GetEntries()))
  input_name = os.path.basename(input).split('.')[0]
  timej = []
  timea = []
  rg = np.linspace(0,6,60)
  for evt_index in tqdm(range(0,5000)):
    ind_temp = randint(start_evt,end_evt)
    tree.GetEntry(ind_temp)
    tree2.GetEntry(ind_temp)
    # print(ind_temp)
    # #gettting PMT information for a event
    # pmt_map = np.array(tree.PMT_hit_list)
    timej += np.array(tree.PMT_charge).tolist()
    timea += np.array(tree2.PMT_charge).tolist()

  #print(np.unique(timej,return_counts=True), np.unique(timea,return_counts=True))
  plt.figure(figsize=(10, 10))
  plt.subplot(211)
  plt.hist(timej, histtype='step', label='John\'s MC Background', normed=True, bins=rg)
  plt.hist(timea, histtype='step', label='John\'s MC Signal',  bins=rg, normed=True)
  plt.xlabel('PMT ' + type)
  plt.ylabel('Counts')
  plt.legend()
  plt.title(title)
  plt.subplot(212)
  # timej = np.array(timej) - np.average(timej)
  # timea = np.array(timea) - np.average(timea)
  countj, rangej = np.histogram(timej, bins=rg, normed=True)
  counta, rangea = np.histogram(timea, bins=rg, normed=True)
  ratio = counta/countj
  # plt.hist(timej, histtype='step', label='John\'s MC Background', normed=True, bins=100)
  # plt.hist(timea, histtype='step', label='John\'s MC Signal', bins=100, normed=True)
  plt.scatter(rangej[:-1], ratio)
  plt.axhline(1.0, color = '#A9A9A9', linestyle='--')
  plt.xlabel('PMT ' + type + ' ratio')
  plt.ylabel('Signal/Background per 0.1 ' + type )
  plt.legend()
  plt.tight_layout()
  plt.savefig(str(title) + type + '_dist.png')
  plt.show()

def main():
  #python /projectnb/snoplus/machine_learning/prototype/processing_sparse.py --input /projectnb/snoplus/sphere_data/input/sph_out_C10_dVrndVtx_3p0mSphere_1k_16.root --outputdir /projectnb/snoplus/sphere_data/c10_2MeV/ --start 2 --end 3
  parser = argparse.ArgumentParser()
  #####################
  # parser.add_argument("--input", default="/project/snoplus/KamLAND-Zen/base-root-analysis/fetch_take2_c10_roi.root")
  # parser.add_argument("--ainput", default="/projectnb/snoplus/sphere_data/input/sph_out_C10_dVrndVtx_3p0mSphere_1k_123.root")
  # parser.add_argument("--title", default="C10 MC Comparison")
  # ######################
  # parser.add_argument("--input", default="/project/snoplus/KamLAND-Zen/base-root-analysis/fetch_take2_doublebeta_roi.root")
  # parser.add_argument("--ainput", default="/projectnb/snoplus/sphere_data/input/sph_out_Xe136_dVrndVtx_3p0mSphere_1k_163.root")
  # parser.add_argument("--title", default="Xe136 MC Comparison")
  ######################
  parser.add_argument("--input", default="/project/snoplus/KamLAND-Zen/base-root-analysis/fetch_take2_c10_roi.root")
  parser.add_argument("--ainput", default="/project/snoplus/KamLAND-Zen/base-root-analysis/fetch_take2_doublebeta_roi.root")
  parser.add_argument("--title", default="Signal Background Comparison")
  ########################################
  #parser.add_argument("--input", default="/project/snoplus/KamLAND-Zen/base-root-analysis/fetch.root")
  parser.add_argument("--outputdir", default="/projectnb/snoplus/sphere_data/c10_2MeV")
  parser.add_argument("--pmt_file_index", default="/project/snoplus/KamLAND-Zen/base-root-analysis/pmt_xyz.dat")
  # parser.add_argument("--type", "-t", help="Type of MC files, 1 ring or 2 ring", default = 1)
  # parser.add_argument("--theta","-th", help="Rotate Camera with given theta(0 - 2pi)",type = float, default = 0)
  # parser.add_argument("--phi","-ph", help="Rotate Camera with given phi(0 - pi)",type = float, default = 0)
  # parser.add_argument("--start", help="start event",type = int, default = 0)
  # parser.add_argument("--end", help="end event",type = int, default = 1000000000)
  # parser.add_argument("--elow", help="lower energy cut",type = float, default = 0.0)
  # parser.add_argument("--ehi", help="upper energy cut",type = float, default = 10000000.0)
  args = parser.parse_args()
  #event_level2(args.input, args.ainput, args.title)

  #fmc = transcribe_hits(args.input, args.ainput, args.title)

  fmc = john_pe(args.input, args.ainput, args.title)





main()
