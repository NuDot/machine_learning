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
from matplotlib.ticker import FormatStrFormatter
plt.rcParams['font.size'] = 18
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams['legend.handlelength'] = 2
# plt.rcParams["legend.title_fontsize"] = 15
plt.rcParams["legend.fontsize"] = 20
from tqdm import tqdm

# tajima=[-12.03678212,1293.862458,
# 0.0,50850.32188,
# 9.052364958,78542.53154,
# 19.77864655,32921.72004,
# 28.21981168,17288.696,
# 37.52546666,11374.80751,
# 46.86415946,8242.973736,
# 57.00677275,6269.065352,
# 65.59660812,5085.032188,
# 77.34155608,4191.578677,
# 84.35658829,3399.918695,
# 94.55426463,3037.508364,
# 104.7794725,2941.246806,
# 113.407852,2670.382213,
# 123.5504653,2030.917621]

tajima = [
-13.6842, 0.0133,
-3.1579, 0.5264,
5.2632, 0.8239,
15.7895, 0.3490,
25.2632, 0.1832,
35.7895, 0.1166,
45.2632, 0.0862,
55.7895, 0.0637,
65.2632, 0.0524,
74.7368, 0.0432,
84.2105, 0.0355,
94.7368, 0.0299,
105.2632, 0.0292,
114.7368, 0.0267
]

tajima = np.array(tajima).reshape(-1,2)

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
    rlength = (tree.trueVtxX**2 + tree.trueVtxY**2 + tree.trueVtxZ**2)**0.5
    if (tree.edep < 2.2) or (tree.edep > 2.7) or (rlength > 150.4):
      continue
    current_clock = set_clock(tree, evt_index)
    #print(tree.N_phot)
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
  events = 0
  for evt_index in range(n_evts):
    pe_count = 0
    tree.GetEntry(evt_index)
    rlength = (tree.trueVtxX**2 + tree.trueVtxY**2 + tree.trueVtxZ**2)**0.5
    if (tree.edep < 2.2) or (tree.edep > 2.7) or (rlength > 150.4):
      continue
    for i in range(tree.N_phot):
      if (tree.detector_coverage_included[i]):
        pe_count += 1
    hit_time.append(pe_count)
    events += 1
    if events > 300:
      break
  return hit_time

def plot_numpy(sig_input, bkg_input, c, l):
  sig_input = squeeze_lastbin(sig_input)
  bkg_input = squeeze_lastbin(bkg_input)
  shist, bin_edges = np.histogram(sig_input,bins=np.linspace(0, 45, 30), density=True)
  bhist, bin_edges = np.histogram(bkg_input,bins=np.linspace(0, 45, 30), density=True)

  hist = (shist*1.0)/(bhist*1.0)
  bincentres = [(bin_edges[i]+bin_edges[i+1])/2. for i in range(len(bin_edges)-1)]
  #plt.step(bincentres,hist,where='mid',color=c, label = l)
  plt.errorbar(x=bincentres, y=hist, yerr=error_propagation(shist * 11000 * 1000.0,bhist * 11000 * 1000.0), fmt='', color=c, label=l, capsize=2, elinewidth=1)

def error_propagation(shist, bhist):
  sigma_s = shist**0.5
  sigma_b = bhist**0.5
  print((shist*1.0/bhist), ((sigma_s/shist)**2 + (sigma_b/bhist)**2)**0.5)
  sigma_f = (shist*1.0/bhist) * ((sigma_s/shist)**2 + (sigma_b/bhist)**2)**0.5
  return sigma_f

def squeeze_lastbin(input):
  for i in range(len(input)):
    if input[i] > 45:
      input[i] = 44.5
  return input



def main():
  if not os.path.exists(PLOT_DIR):
   os.mkdir(PLOT_DIR)

  plt.figure(figsize=(8,6))
  # file_collection = ['sph_out_Xe136_center_1k_30.root', 'sph_out_C10_center_1k_29.root']
  # sig_input, bkg_input = [transcribe_hits(ZDAB_DIR + ifile) for ifile in file_collection]
  file_collection_b = ['sph_out_Xe136_dVrndVtx_3p0mSphere_1k_109.root', 'sph_out_C10_dVrndVtx_3p0mSphere_1k_305.root']
  sig_input_b, bkg_input_b = [transcribe_hits(ZDAB_DIR + ifile) for ifile in file_collection_b]
  # plt.hist(sig_input,bins=np.linspace(0, 150, 100), histtype='step',normed=True, color=matplotlib.cm.get_cmap('winter')(0.72), label=r'$^{136}$Xe-0$\nu\beta\beta$ Center')
  # plt.hist(bkg_input,bins=np.linspace(0, 150, 100), histtype='step',normed=True, color=matplotlib.cm.get_cmap('Blues_r')(0.3),label=r'$^{10}$C Center')
  heights, bins = (tajima[:,1], tajima[:,0])
  bins = list(bins - 5.0)
  bins.append(bins[-1] + 5.0)
  bins = np.array(bins)
  bin_widths = bins[1:] - bins[:-1]
  normed_heights = heights / bin_widths / heights.sum()
  print(normed_heights)
  plt.hist(sig_input_b,bins=np.linspace(0, 105, 70), histtype='step',normed=True, color = matplotlib.cm.get_cmap('winter')(0.72),linestyle='--', label=r'$^{136}$Xe-0$\nu\beta\beta$ Balloon')
  plt.hist(bkg_input_b,bins=np.linspace(0, 105, 70), histtype='step',normed=True, color = matplotlib.cm.get_cmap('Blues_r')(0.3), linestyle='--', label=r'$^{10}$C Balloon')
  plt.axvspan(xmin=0.0, xmax=24.0, alpha=0.1)
  plt.scatter(x=tajima[:,0] + 8.0, y=normed_heights, s=3, c='red', label = 'Digitized Tajima\'s Thesis')
  #######################################################
  #plot_numpy(sig_input, bkg_input, c=matplotlib.cm.get_cmap('winter')(0.85), l=r'$^{136}$Xe-0$\nu\beta\beta$/$^{10}$C Center')
  #plot_numpy(sig_input_b, bkg_input_b, c=matplotlib.cm.get_cmap('winter')(0.2), l=r'$^{136}$Xe-0$\nu\beta\beta$/$^{10}$C Balloon')
  #######################################################
  # plt.hist(sig_input_hist, histtype='step',normed=True, color='red', label=r'$^{136}$Xe-0$\nu\beta\beta$ Center')
  # plt.hist(bkg_input_hist,bins=np.linspace(0, 140, 140), histtype='step',normed=True, color=matplotlib.cm.get_cmap('Blues_r')(0.3),label=r'$^{10}$C Center')
  # plt.hist(sig_input_b_hist,bins=np.linspace(0, 140, 140), histtype='step',normed=True, color = 'red',linestyle='--', label=r'$^{136}$Xe-0$\nu\beta\beta$ Balloon')
  # plt.hist(bkg_input_b_hist,bins=np.linspace(0, 140, 140), histtype='step',normed=True, color = matplotlib.cm.get_cmap('Blues_r')(0.3), linestyle='--', label=r'$^{10}$C Balloon')
  #######################################################
  #plt.axvspan(0, 5, color='gray', alpha=0.3)
  plt.xlabel('Time Since First Photon',fontsize=15)
  plt.ylabel('Normalized Count',fontsize=15)
  plt.yscale('log')
  plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d ns'))
  plt.rcParams.update({'font.size': 20})
  plt.legend(fontsize=12,fancybox=True)
  #plt.savefig( PLOT_DIR  + 'Timing_Hist_3m_Sphere_L.pdf', format='pdf', dpi=1000)
  plt.savefig('tajima_compare_shift.png')
  plt.show()

  # plt.hist(sig_input,bins=np.linspace(0, 70, 140), histtype='step',normed=True, label=r'$^{136}$Xe-0$\nu\beta\beta$')
  # plt.hist(bkg_input,bins=np.linspace(0, 70, 140), histtype='step',normed=True, label=r'$^{10}$C')
  # plt.xlabel('Time Since First Photon(s)',fontsize=15)
  # plt.ylabel('Normalized Count',fontsize=15)
  # plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d ns'))
  # plt.legend(title="3m Sphere Events", fontsize=15,fancybox=True)
  # plt.savefig( PLOT_DIR  + 'Timing_Profile_3m_Sphere.pdf', format='pdf', dpi=1000)
  # plt.show()
  ######################################################################################################################################################################






main()