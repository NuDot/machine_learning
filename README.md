# machine_learning  



This file contains tools used for conducting machine learning tasks in Liquid Scintillator Detector. The purpose of 4 folders are listed below:



Preprocessing: Converting Monte Carlo Simulation .root file to a 2D Grid (theta, phi), and store them as a sparse matrix.



Network: Storing the Convolutional Neural Network Used to conduct the classification task, also include the hyperparameter search algorithm.


Analyzer: data analysis tool for making the pressure map and roc curves.


data: Data processing script to be use on BU SCC cluster.

Adding Spherical CNN feature modified from https://github.com/jonas-koehler/s2cnn

A full study contains:




Step 1: Given Monte Carlo file, preprocess it using preprocessing/processing_sparse_time.py to convert it to a pmt hitmap.
  
  
  Output: eventfile.pickles for each .root file
  
  
  
Step 2: Collecting all eventfile.pickle into a list, each entry of the list should contains the address to the pickle files
  
  
  Example(content in the .dat file):
  /projectnb/snoplus/sphere_data/kamland_38/eventfile_sph_out_Xe136_dVrndVtx_3p0mSphere_1k_129.0.1000.pickle
  /projectnb/snoplus/sphere_data/kamland_38/eventfile_sph_out_Xe136_dVrndVtx_3p0mSphere_1k_122.0.1000.pickle
  /projectnb/snoplus/sphere_data/kamland_38/eventfile_sph_out_Xe136_dVrndVtx_3p0mSphere_1k_112.0.1000.pickle
  /projectnb/snoplus/sphere_data/kamland_38/eventfile_sph_out_Xe136_dVrndVtx_3p0mSphere_1k_1.0.1000.pickle
  
  
  Output: There should be two .dat file in total:
    Xe136.dat
    C10.dat
    
    
    
    
    
Step 3: Feeding .dat file into network/network_spherical.py. A set of pressure parameters(photocoverage, QE) will also need to                 be given in argument(by default 8,10, highest pressure and current KamLAND parameter)
  
  
  Output: 
    roc.png(plot of the ROC curve in validation data)
    sigmoid.png(plot of sigmoid output in validation data)
    roc_curve.npy(contain various numpy array of the output classification: (fpr, tpr, thr, sigmoid_s, sigmoid_b))
    
    
    
    
    
Step 4: Making various plots. Plotting code is shown in the analyzer folder, e.g. roc_pm plots the pressure map.

