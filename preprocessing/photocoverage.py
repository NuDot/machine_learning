import argparse
import math
import os
import json
from scipy import sparse
from random import *
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

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

def pmt_setup(vec1):
    pmt_index = np.array([calculate_angle(vec1, vec2) for vec2 in PMT_POSITION]).argmin()
    x2,y2,z2 = PMT_POSITION[pmt_index]
    detector_radius = (x2**2 + y2**2 + z2**2)**0.5
    pmt_angle = calculate_angle(vec1, PMT_POSITION[pmt_index])
    return pmt_index, detector_radius, pmt_angle

def pmt_allocator(pmt_angle, detector_radius, photocoverage):
   single_coverage = 4 * math.pi * detector_radius**2 * photocoverage / float(N_PMTS)
   coverage_radius = (single_coverage / float(math.pi))**0.5
   coverage_angle = math.tan(coverage_radius/float(detector_radius))
   return (pmt_angle <= coverage_angle)


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

def cross(a, b):
    c = [a[1]*b[2] - a[2]*b[1],
         a[2]*b[0] - a[0]*b[2],
         a[0]*b[1] - a[1]*b[0]]

    return c

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

def sample_pmt_rim(pmt_position, radii):
  x,y,z = pmt_position
  sphere_rad = float((x**2 + y**2 + z**2)**0.5)
  angle = math.asin(radii / sphere_rad)
  output_vect = []
  for i in range(3):
    x_u = random()
    y_u = random()
    z_u = -(x*x_u + y*y_u)/z
    unit_vec = np.array([x_u,y_u,z_u])
    unit_vec = unit_vec/np.linalg.norm(unit_vec)
    v1 = np.array(pmt_position)
    v2 = v1*math.cos(angle) + np.array(cross(unit_vec, v1)) * math.sin(angle)
    output_vect.append(v2)
  return output_vect

def transcribe_hits():
  radius = np.linspace(0.2159, 0.4, 50);
  overlap_count = []
  for radii in tqdm(radius): 
    count = 0
    for pmt in PMT_POSITION:
      datapoint_set = sample_pmt_rim(pmt, radii)
      for data_point in datapoint_set:
        pmt_index, detector_radius, pmt_angle = pmt_setup(data_point)
        if not (PMT_POSITION[pmt_index] == pmt):
          count += 1;
    overlap_count.append(count)
  overlap_count = np.array(overlap_count)
  max_untouch = radius[np.nonzero(overlap_count)[0][0] - 1]
  plt.axvline(x=max_untouch, label = "Maximum PMT radius:" + str(max_untouch), linestyle='--', color="b")
  plt.plot(radius, overlap_count)
  plt.xlabel("PMT Radius")
  plt.ylabel("Overlapped Rim Counts")
  plt.legend()
  plt.show()



transcribe_hits()
