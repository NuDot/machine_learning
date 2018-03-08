import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
from ROOT import TFile


def xyz_to_phi_theta(x, y, z):
   phi = math.atan2(y, x)
   r = (x**2 + y**2 + z**2)**.5
   theta = math.acos(z / r)
   return phi, theta


def xyz_to_phi_cos_z(x, y, z):
   phi = math.atan2(y, x)
   r = (x**2 + y**2 + z**2)**.5
   cos_z = z / r
   return phi, cos_z


def phi_cos_z_to_row_col(phi, cos_z, rows=100, cols=100):
   # phi is in [-pi, pi], z is in [-1, 1]
   row = min(rows/2 + (round((rows/2)*phi/math.pi)), rows-1)
   row = max(row, 0)
   col = min(cols/2 + (round((cols/2)*cos_z/1)), cols-1);
   col = max(col, 0)
   return row, col


def phi_theta_to_row_col(phi, theta, rows=100, cols=50):
   # phi is in [-pi, pi], theta is in [0, pi]
   row = min(rows/2 + (round((rows/2)*phi/math.pi)), rows-1)
   row = max(row, 0)
   col = min(cols*theta/math.pi, cols-1);
   col = max(col, 0)
   return row, col


def xyz_to_row_col(x, y, z):
   #return phi_cos_z_to_row_col(*xyz_to_phi_cos_z(x, y, z))
   return phi_theta_to_row_col(*xyz_to_phi_theta(x, y, z))

def drange2(start, stop, step):
    numelements = int((stop-start)/float(step))
    for i in range(numelements+1):
            yield start + i*step

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="sph_out_topology180_center_NoMultScat_100.root")
parser.add_argument("--type", "-t", help="Type of MC files, 1 ring or 2 ring", default = 1)
args = parser.parse_args()

feature_map_collections = np.zeros((((13,11,100,50))))
initial_time = 32
final_time = 38
time_interval =0.5

f1 = TFile(args.input)

tree = f1.Get("epgTree")

counter = 0
for event in tree:
  counter += 1
  if counter > 1:
    break

  feature_map = np.zeros((100,50))
  for i in range(tree.N_phot):
    row, col = xyz_to_row_col(tree.x_hit[i], tree.y_hit[i], tree.z_hit[i])
    for pressure_time in drange2(initial_time, final_time, time_interval):
      if (tree.PE_time[i] >= initial_time) and (tree.PE_time[i]  <= pressure_time):
        time_index = int((pressure_time - 32) / 0.5)
        for pressure_pe in range (0, 11):
              if (tree.PE_creation[i]):
                feature_map_collections[time_index][pressure_pe][row][col] += 10
              else :
                feature_map_collections[time_index][pressure_pe][row][col] += pressure_pe

plt.imshow(feature_map_collections[5][7])
plt.show()

