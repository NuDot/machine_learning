import argparse
import math
import os
import json
from scipy import sparse
from random import *
import numpy as np
from ROOT import TFile
from datetime import datetime
import rat
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm
from matplotlib.colors import LogNorm, Normalize
#Global Variables
PMT_POSITION = zip(x,y,z)
PMT_POSITION = [[x/1000.0,y/1000.0,z/1000.0] for x,y,z in PMT_POSITION if (x**2 + y**2 + z**2)**0.5 <= 9000.0]
del x 
del y 
del z
N_PMTS = len(PMT_POSITION)
COLS = int(math.sqrt(N_PMTS/2))
MIN_PMT_RADIUS = 8323.85127186
ROWS = COLS *2
MAX_PRESSURE = 10
QE_FACTOR = 1.0
FIRST_PHOTON = False
INPUT_DIR = "/projectnb/snoplus/axion_data/tl208_tracked/"

def pmt_allocator(pmt_angle, detector_radius, pmt_radius):
   coverage_angle = math.asin(pmt_radius/float(detector_radius))
   return (pmt_angle <= coverage_angle)

def calculate_angle(vec1, vec2):
  x1,y1,z1 = vec1
  x2,y2,z2 = vec2
  inner_product = x1*x2 + y1*y2 + z1*z2
  len1 = (x1**2 + y1**2 + z1**2)**0.5
  len2 = (x2**2 + y2**2 + z2**2)**0.5
  return math.acos(float(inner_product)/float(len1*len2))

class clock:
  tick_interval = 20.0
  final_time = 700.0
  initiated=False
  clock_array = np.arange(-100.0, final_time, tick_interval)

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


def phi_theta_to_row_col(phi, theta, rows=ROWS, cols=COLS):
   # phi is in [-pi, pi], theta is in [0, pi]
   row = min(rows/2 + (math.floor((rows/2)*phi/math.pi)), rows-1)
   row = max(row, 0)
   col = min(math.floor(cols*theta/math.pi), cols-1);
   col = max(col, 0)
   return int(row), int(col)



def xyz_to_row_col(x, y, z, rows=ROWS, cols=COLS):
   return phi_theta_to_row_col(*xyz_to_phi_theta(x, y, z), rows=rows, cols=cols)

def random_decision(pressure):
    decision = False
    if (pressure > MAX_PRESSURE) or (pressure == 0):
      return False
    frac_pressure = float(pressure)/float(MAX_PRESSURE)
    return (random()<=(frac_pressure * QE_FACTOR))


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

def set_clock(time_array):
  time_array = np.array(time_array)
  clock_time = 0.0
  try:
    if len(time_array != 0):
      if len(time_array) >1:
        time_array = np.delete(time_array, np.argmin(time_array))
      clock_time = time_array.min()
  except:
    print time_array
  return clock(clock_time)

def savefile(saved_file, appendix, filename, pathname):
    if not os.path.exists(pathname):
     os.mkdir(pathname)
    with cd(pathname):
        with open(filename, 'w') as datafile:
          json.dump(saved_file, datafile)

def smearing_time(time):
  return np.random.normal(loc=time, scale=1.5)

def pmt_setup(vec):
  index_array = []
  vec1 = [vec.x()/1000.0, vec.y()/1000.0, vec.z()/1000.0]
  for pmt_grid in PMT_POSITION:
    abs_distance = (pmt_grid[0] - vec1[0])**2 + (pmt_grid[1] - vec1[1])**2 + (pmt_grid[2] - vec1[2])**2
    index_array.append(abs_distance)
  pmt_arg = np.array(index_array).argmin()
  x2,y2,z2 = PMT_POSITION[pmt_arg]
  detector_radius = vec.Mag()/1000.0
  pmt_angle = calculate_angle(vec1, PMT_POSITION[pmt_arg])
  return pmt_arg, detector_radius, pmt_angle

def transcribe_hits(input, start_evt):
  evt_count = 0
  current_clock = clock(0)
  n_qe_values = MAX_PRESSURE + 1
  photocoverage_scale = list(np.linspace(0.1413, 0.1016, 9))
  feature_map_collections = np.zeros(((((len(photocoverage_scale), n_qe_values, 1, current_clock.clock_size(), ROWS, COLS)))))
  pmt_info = rat.utility().GetPMTInfo()
  for ds, run in tqdm(rat.dsreader(str(input))):
    if evt_count != start_evt:
      evt_count += 1
      continue
    rmc= ds.GetMC()
    particleNum = rmc.GetMCParticleCount()
    nevC = ds.GetEVCount()

    #Remove Retriggers
    if nevC > 1:
      nevC = 1

    # #Read PMT DAQ information
    # for iev in range(0, nevC):
    #   calibrated_pmts = ds.GetMCEV(iev).GetMCHits()
    #   for ipmt in range(0, calibrated_pmts.GetNormalCount()):
    #     pmt_cal = calibrated_pmts.GetNormalPMT(ipmt)
    #     pmt_pos = pmt_info.GetPosition(pmt_cal.GetID())
    #     if not (pmt_cal.GetCrossTalkFlag()):
    #       pmt_hit.append([pmt_pos.x(), pmt_pos.y(), pmt_pos.z()])
    #       hit_time.append(pmt_cal.GetTime())
    # pmt_hit = np.array(pmt_hit)
    # hit_time = np.array(hit_time)
    # current_clock = set_clock(hit_time)

    #Start Optical Photon Tracking
    for iev in range(0, nevC):
      trackIDs = rmc.GetMCTrackIDs()
      pmt_hit = []
      hit_time = []
      #Read MC PMT Info
      pmts = rmc.GetMCPMTCount()
      for ipmt in range(pmts):
        mcpmt = rmc.GetMCPMT(ipmt)
        mcpes = mcpmt.GetMCPECount()
        mcpmt_pos = pmt_info.GetPosition(mcpmt.GetID())
        pos_vec = [mcpmt_pos.x(), mcpmt_pos.y(), mcpmt_pos.z()]
        for ipe in range(mcpes):
          mcpe = mcpmt.GetMCPE(ipe)
          pmt_hit.append(pos_vec)
          hit_time.append(mcpe.GetCreationTime())
      pmt_hit = np.array(pmt_hit)
      hit_time = np.array(hit_time)
      current_clock = set_clock(hit_time)

        #Start Tracking Individual Photon
      for trackID in trackIDs:
        mctrack = rmc.GetMCTrack(trackID)
        detected_photon = False
        if (mctrack.GetParticleName() == "opticalphoton"):
          pos = mctrack.GetLastMCTrackStep().GetPosition()
          if pos.Mag() < MIN_PMT_RADIUS:
            continue
          input_time = mctrack.GetLastMCTrackStep().GetGlobalTime()
          pmt_index, detector_radius, pmt_angle = pmt_setup(pos)
          pmt_radius = (PMT_POSITION[pmt_index][0]**2 + PMT_POSITION[pmt_index][1]**2 + PMT_POSITION[pmt_index][2]**2)**0.5
          if detector_radius < pmt_radius:
            continue
          else:
            # #Matching Photon Tracking information with detected PMT DAQ information based on Position and Time
            # try:
            #     hit_index = np.where(pmt_hit == PMT_POSITION[pmt_index])[0]
            #     current_pmt_index = hit_index
            #     if not len(hit_index) == 0:
            #       detected_photon = True
            #       if len(hit_index) > 1:
            #         abs_time_diff = np.array([abs(time - input_time) for time in hit_time[hit_index]])
            #         current_pmt_index = hit_index(np.argmin(abs_time_diff))
            #     input_time = hit_time(current_pmt_index)
            # except:
            #   print "PMT Allocation Failed! Using Truth Position of photon hit to Allocate PMT"
            #Writing Photon Hit into Pressure Map with various pressures
            #print(detected_photon)
            for pressure_pc in photocoverage_scale:
              if (pmt_allocator(pmt_angle, detector_radius, pressure_pc)):
                row, col = xyz_to_row_col(PMT_POSITION[pmt_index][0], PMT_POSITION[pmt_index][1], PMT_POSITION[pmt_index][2])
                for pressure_pe in range (0, n_qe_values):
                  if (detected_photon) or random_decision(MAX_PRESSURE-pressure_pe):
                    time_index = current_clock.tick(smearing_time(input_time))
                    feature_map_collections[photocoverage_scale.index(pressure_pc)][pressure_pe][0][time_index][row][col] += 1.0
    evt_count += 1
  # dim1, dim2, dim3, dim4, dim5, dim6 = feature_map_collections.shape
  # lst = np.zeros((dim3,dim4,1))
  # input_name = os.path.basename(input).split('.')[0]
  # data_path = os.path.join(outputdir, "data_%s" % (input_name))
  # indices_path = os.path.join(outputdir, "indices_%s" % (input_name))
  # indptr_path = os.path.join(outputdir, "indptr_%s" % (input_name))
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
  return feature_map_collections



def sum_plot(maps, upper_bound):
  output = np.zeros((50,25))
  for i in range(upper_bound):
    output+=maps[i]
  return output


def main():
  #python /projectnb/snoplus/machine_learning/prototype/processing_sparse.py --input /projectnb/snoplus/sphere_data/input/sph_out_C10_dVrndVtx_3p0mSphere_1k_66.root --outputdir /projectnb/snoplus/sphere_data/c10_2MeV/ --start 72 --end 82 --elow 2.2 --ehi 2.7
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

  fmc_xe136 = transcribe_hits(input='/projectnb/snoplus/axion_data/axion_tracked/axion_macro_ae_39.root', start_evt=35)
  fmc_c10 = transcribe_hits(input='/projectnb/snoplus/axion_data/tl208_tracked/Tl208_4.root', start_evt=66)
  #print(fmc_xe136.shape, fmc_c10.shape)

  # fig = plt.figure(figsize=(15, 15))
  # gs = gridspec.GridSpec(8, 7)
  # for i in range(50):
  #   fig.add_subplot(gs[i])
  #   plt.imshow(fmc_xe136[0][0][i][0], norm=LogNorm(vmin=1, vmax=30))
  #   #plt.imshow(sum_plot(fmc[0][0][i], 1))
  # plt.show()

  print np.unique(np.nonzero(fmc_xe136[0][0][0])[0], return_index=True)

  plot_slice = [5,6,10,20]
  ###################################################################
  fig = plt.figure(figsize=(22, 15))
  gs = gridspec.GridSpec(2, 6, width_ratios=[2, 3, 3, 3, 3, 1])
  ax1 = fig.add_subplot(gs[0,0])
  ax1.patch.set_visible(False)
  ax1.axis('off')
  plt.text(x=0.0, y=0.5, s=r'AE', fontsize = 35)
  ax2 = fig.add_subplot(gs[1,0])
  ax2.patch.set_visible(False)
  ax2.axis('off')
  plt.text(x=0.0, y=0.5, s=r'$^{208}$Tl', fontsize = 35)

  for i, timeslice in enumerate(plot_slice):
    fig.add_subplot(gs[0,(i + 1)])
    plt.imshow(fmc_xe136[0][0][0][timeslice] ,cmap = 'viridis')
    plt.title(str((timeslice - 5) * 20.0) + 'ns - ' + str((timeslice-4) * 20.0) + 'ns' )
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\phi$')
  for i, timeslice in enumerate(plot_slice):
    fig.add_subplot(gs[1,(i + 1)])
    plt.imshow(fmc_c10[0][0][0][timeslice] , cmap = 'viridis')
    plt.title(str((timeslice - 5) * 20.0) + 'ns - ' + str((timeslice-4) * 20.0) + 'ns' )
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\phi$')

  loss_ax_scale = fig.add_subplot(gs[:,5])
  loss_scale = np.linspace(5, 0, 100)
  loss_scale = np.transpose(loss_scale.reshape(loss_scale.shape+(1,)))
  loss_ax_scale.set_xticks([])
  ticklabel_loss = np.linspace(5, 0, 6.5)
  ticklabel_loss =   ['%.3f' % i for i in ticklabel_loss]
  ticklabel_loss.insert(0,'0')
  loss_ax_scale.set_yticklabels(ticklabel_loss)
  for i in range(0,2):
    loss_scale = np.vstack((loss_scale,loss_scale))
  loss_scale = loss_ax_scale.imshow(np.transpose(loss_scale),cmap='viridis', interpolation='nearest')

  plt.savefig('a.pdf', format='pdf', dpi=1000)
  plt.tight_layout()
  plt.show()
  #################################################################################






main()
