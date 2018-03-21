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


def xyz_to_row_col(x, y, z, rows=100, cols=50):
   #return phi_cos_z_to_row_col(*xyz_to_phi_cos_z(x, y, z))
   return phi_theta_to_row_col(*xyz_to_phi_theta(x, y, z), rows=rows, cols=cols)


parser = argparse.ArgumentParser()
parser.add_argument("--input", default="../sph_out_topology180_center_NoMultScat_100.root")
args = parser.parse_args()


f1 = TFile(args.input)

tree = f1.Get("epgTree")

print tree

counter = 0
for event in tree:
  counter += 1
  if counter > 1:
    break
  ROWS, COLS = 100, 50
  z = np.zeros((ROWS, COLS))
    if tree.PE_time[i] > 32.5:# or (not tree.PE_creation[i]):
      continue
    row, col = xyz_to_row_col(tree.x_hit[i], tree.y_hit[i], tree.z_hit[i], ROWS, COLS)
    z[row][col] += 100
    if i < 10:
      print row, col

print z
plt.imshow(z)
plt.show()

