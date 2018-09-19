import argparse
import math
import os
import json
from random import *
import numpy as np
import matplotlib
import time
from ROOT import TFile
from datetime import datetime
import matplotlib.pyplot as plt

from tqdm import tqdm

#Global Variables
# PMT_POSITION = []
# for pmt in np.loadtxt("/projectnb/snoplus/machine_learning/prototype/pmt.txt").tolist():
#   if (len(pmt)):
#     if (pmt[-1] == 17.0):
#       PMT_POSITION.append([pmt[-4], pmt[-3], pmt[-2]])
# N_PMTS = len(PMT_POSITION)
# COLS = int(math.sqrt(N_PMTS/2))
# ROWS = COLS *2
# RUN_TIMESTAMP = time.time()
# MIN_PRESSURE = 0
# MAX_PRESSURE = 10
# FIRST_PHOTON = False

OUT_DIR = "/projectnb/snoplus/sphere_data/json_new/"
ZDAB_DIR = "/projectnb/snoplus/sphere_data/input/"
MACRO_DIR = "/projectnb/snoplus/machine_learning/data/shell"
#KEYWORD = "center"
KEYWORD = "dVrndVtx_3p0mSphere"
PLOT_DIR = "/projectnb/snoplus/machine_learning/plots/"

class clock:
  tick_interval = 1.5
  final_time = 45
  initiated=False
  initial_time = 0.0
  clock_array = np.arange(0, final_time, tick_interval)

  def __init__(self, initial_time):
    clock.initiated=True
    self.initial_time = initial_time
    self.clock_array = self.clock_array + initial_time - 0.5

  def tick(self, time):
    return self.clock_array[self.clock_array < time].argmax()

  def clock_size(self):
    return len(self.clock_array)

  def exact_time(self, time):
    return time - self.initial_time

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

def drange2(start, stop, step):
    numelements = int((stop-start)/float(step))
    for i in range(numelements+1):
            yield start + i*step

def set_clock(tree, evt):
  tree.GetEntry(evt)
  time_array = []
  for i in range(tree.N_phot):
    time_array.append(tree.PE_time[i])
  return clock(np.array(time_array).min())

def savefile(saved_file, appendix, filename, pathname):
    if not os.path.exists(pathname):
     os.mkdir(pathname)
    with cd(pathname):
        with open(filename, 'w') as datafile:
          json.dump(saved_file, datafile)


def transcribe_hits(input):
  hit_time = []
  current_clock = clock(0)
  f1 = TFile(input)
  tree = f1.Get("epgTree")
  n_evts = tree.GetEntries()
  for evt_index in tqdm(range(n_evts)):
    tree.GetEntry(evt_index)
    current_clock = set_clock(tree, evt_index)
    for i in range(tree.N_phot):
      hit_time.append(current_clock.exact_time(tree.PE_time[i]))
  return hit_time

def transcribe_energy(input, upperbound=99999):
  ev_energy = []
  f1 = TFile(input)
  tree = f1.Get("epgTree")
  n_evts = min(tree.GetEntries(), upperbound)
  for evt_index in range(n_evts):
    tree.GetEntry(evt_index)
    ev_energy.append(tree.edep)
  return ev_energy

def transcribe_pe(input):
  hit_time = []
  f1 = TFile(input)
  tree = f1.Get("epgTree")
  n_evts = tree.GetEntries()
  for evt_index in range(n_evts):
    pe_count = 0
    tree.GetEntry(evt_index)
    for i in range(tree.N_phot):
      if (tree.detector_coverage_included[i]):
        pe_count += 1
    hit_time.append(pe_count)
  return hit_time



def main():
  if not os.path.exists(PLOT_DIR):
   os.mkdir(PLOT_DIR)


  file_collection = ['sph_out_Xe136_center_1k_30.root', 'sph_out_C10_center_1k_29.root']
  sig_input, bkg_input = [transcribe_hits(ZDAB_DIR + ifile) for ifile in file_collection]
  file_collection_b = ['sph_out_Xe136_dVrndVtx_3p0mSphere_1k_109.root', 'sph_out_C10_dVrndVtx_3p0mSphere_1k_305.root']
  sig_input_b, bkg_input_b = [transcribe_hits(ZDAB_DIR + ifile) for ifile in file_collection_b]
  # plt.hist(sig_input,bins=np.linspace(0, 140, 140), histtype='step',normed=True, color=matplotlib.cm.get_cmap('winter')(0.72), label=r'$^{136}$Xe-0$\nu\beta\beta$ Center')
  # plt.hist(bkg_input,bins=np.linspace(0, 140, 140), histtype='step',normed=True, color=matplotlib.cm.get_cmap('Blues_r')(0.3),label=r'$^{10}$C Center')
  # plt.hist(sig_input_b,bins=np.linspace(0, 140, 140), histtype='step',normed=True, color = matplotlib.cm.get_cmap('winter')(0.72),linestyle='--', label=r'$^{136}$Xe-0$\nu\beta\beta$ Balloon')
  # plt.hist(bkg_input_b,bins=np.linspace(0, 140, 140), histtype='step',normed=True, color = matplotlib.cm.get_cmap('Blues_r')(0.3), linestyle='--', label=r'$^{10}$C Balloon')
  plt.hist(sig_input,bins=np.linspace(0, 140, 140), histtype='step',normed=True, color='red', label=r'$^{136}$Xe-0$\nu\beta\beta$ Center')
  plt.hist(bkg_input,bins=np.linspace(0, 140, 140), histtype='step',normed=True, color=matplotlib.cm.get_cmap('Blues_r')(0.3),label=r'$^{10}$C Center')
  plt.hist(sig_input_b,bins=np.linspace(0, 140, 140), histtype='step',normed=True, color = 'red',linestyle='--', label=r'$^{136}$Xe-0$\nu\beta\beta$ Balloon')
  plt.hist(bkg_input_b,bins=np.linspace(0, 140, 140), histtype='step',normed=True, color = matplotlib.cm.get_cmap('Blues_r')(0.3), linestyle='--', label=r'$^{10}$C Balloon')
  plt.axvspan(0, 5, color='gray', alpha=0.3)
  plt.xlabel('Time Since First Photon(s)',fontsize=15)
  plt.ylabel('Normalized Count',fontsize=15)
  #plt.yscale('log')
  plt.legend(fontsize=12,fancybox=True)
  plt.savefig( PLOT_DIR  + 'Timing_Profile_Center_Linear.pdf', format='pdf')
  plt.show()

  # plt.hist(sig_input,bins=np.linspace(0, 70, 140), histtype='step',normed=True, label=r'$^{136}$Xe-0$\nu\beta\beta$')
  # plt.hist(bkg_input,bins=np.linspace(0, 70, 140), histtype='step',normed=True, label=r'$^{10}$C')
  # plt.xlabel('Time Since First Photon(s)',fontsize=15)
  # plt.ylabel('Normalized Count',fontsize=15)
  # plt.legend(title="3m Sphere Events", fontsize=15,fancybox=True)
  # plt.savefig( PLOT_DIR  + 'Timing_Profile_3m_Sphere.pdf', format='pdf', dpi=1000)
  # plt.show()

  # signal_collection = ['sph_out_Xe136_dVrndVtx_3p0mSphere_1k_109.root']
  # bkg_collection = ['sph_out_C10_dVrndVtx_3p0mSphere_1k_323.root','sph_out_C10_dVrndVtx_3p0mSphere_1k_5.root','sph_out_C10_dVrndVtx_3p0mSphere_1k_6.root','sph_out_C10_dVrndVtx_3p0mSphere_1k_7.root','sph_out_C10_dVrndVtx_3p0mSphere_1k_8.root','sph_out_C10_dVrndVtx_3p0mSphere_1k_9.root','sph_out_C10_dVrndVtx_3p0mSphere_1k_10.root','sph_out_C10_dVrndVtx_3p0mSphere_1k_11.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_334.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_285.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_340.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_323.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_305.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_282.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_41.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_327.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_399.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_324.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_303.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_400.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_316.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_394.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_278.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_397.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_333.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_283.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_12.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_13.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_14.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_15.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_16.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_17.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_18.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_19.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_20.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_21.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_22.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_23.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_24.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_25.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_26.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_27.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_28.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_29.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_30.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_31.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_32.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_33.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_34.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_35.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_36.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_37.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_38.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_39.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_40.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_42.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_43.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_44.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_45.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_46.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_47.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_48.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_49.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_50.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_51.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_52.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_53.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_54.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_55.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_56.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_57.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_58.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_59.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_60.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_61.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_62.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_63.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_64.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_65.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_66.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_67.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_68.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_69.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_70.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_71.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_72.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_73.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_74.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_75.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_76.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_77.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_78.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_79.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_80.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_81.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_82.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_83.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_84.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_85.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_86.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_87.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_88.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_89.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_90.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_91.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_92.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_93.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_94.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_95.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_96.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_97.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_98.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_99.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_100.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_101.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_102.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_103.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_104.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_105.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_106.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_107.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_108.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_109.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_110.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_111.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_112.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_113.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_114.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_115.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_116.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_117.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_118.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_119.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_120.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_121.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_122.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_123.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_124.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_125.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_126.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_127.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_128.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_129.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_130.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_131.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_132.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_133.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_134.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_135.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_136.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_137.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_138.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_139.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_140.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_141.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_142.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_143.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_144.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_145.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_146.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_147.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_148.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_149.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_150.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_151.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_152.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_153.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_154.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_155.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_156.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_157.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_158.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_159.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_160.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_161.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_162.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_163.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_164.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_165.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_166.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_167.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_168.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_169.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_170.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_275.root']
  # sig_input = [transcribe_energy(ZDAB_DIR + ifile, 100) for ifile in signal_collection]
  # bkgrd_input = [transcribe_energy(ZDAB_DIR + ifile) for ifile in bkg_collection]
  # bkg_input = []
  # for bkgrd in bkgrd_input:
  #   bkg_input += bkgrd
  # plt.hist(sig_input,bins=np.linspace(1.5, 4.0, 300), histtype='step',label=r'$^{136}$Xe-0$\nu\beta\beta$')
  # plt.hist(bkg_input,bins=np.linspace(1.5, 4.0, 300), histtype='step',label=r'$^{10}$C')
  # plt.axvline(x=2.2, color='b', linestyle=':',label = 'Fiducial Energy Cut')
  # plt.axvline(x=2.7, color='b', linestyle=':')
  # plt.xlabel('MC Energy(MeV)', fontsize = 15)
  # plt.legend(title="3m Sphere Events", fontsize=10,fancybox=True)
  # plt.savefig( PLOT_DIR  + 'Energy_Spectrum_3m_Sphere.pdf', format='pdf', dpi=1000)
  # plt.show()

  # signal_collection = ['sph_out_Xe136_dVrndVtx_3p0mSphere_1k_109.root']
  # bkg_collection = ['sph_out_C10_dVrndVtx_3p0mSphere_1k_323.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_334.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_285.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_340.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_323.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_305.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_282.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_41.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_327.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_399.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_324.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_303.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_400.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_316.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_394.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_278.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_397.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_333.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_283.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_12.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_13.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_14.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_15.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_16.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_17.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_18.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_19.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_20.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_21.root',
  #                 'sph_out_C10_dVrndVtx_3p0mSphere_1k_22.root']
  # sig_input = [transcribe_pe(ZDAB_DIR + ifile) for ifile in signal_collection]
  # bkgrd_input = [transcribe_pe(ZDAB_DIR + ifile) for ifile in bkg_collection]
  # bkg_input = []
  # for bkgrd in bkgrd_input:
  #   bkg_input += bkgrd
  # plt.hist(sig_input, bins=np.linspace(1500, 4000, 250), histtype='step',label=r'$^{136}$Xe-0$\nu\beta\beta$')
  # plt.hist(bkg_input, bins=np.linspace(1500, 4000, 250), histtype='step',label=r'$^{10}$C')
  # plt.xlabel('Detected PE(MeV)', fontsize = 15)
  # plt.legend(title="3m Sphere Events", fontsize=10,fancybox=True)
  # plt.savefig( PLOT_DIR  + 'detected_pe_3m_Sphere.pdf', format='pdf', dpi=1000)
  # plt.show()





main()
