#!/usr/bin/env python
# coding: utf-8

# # Figures for Cruz-Arce et. al 202X

# This notebook will take in real spectral data as well as PSG models to produce plots for Cruz-Arce et al 202X which will attempt to takle the question, "are exoplanet surfaces detectable?"

# ## First we read in all of our data and models

# In[1]:


# All the modules you may or may not need
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
from pandas import DataFrame as df
import os
import shutil
import matplotlib.gridspec as gridspec
from collections import OrderedDict
from matplotlib import cm as cm


# In[2]:


# Read in Data and Models

# Mars, Earth, and Some Icy Moon .pickle file models
Mars = pd.read_pickle('/Users/ccruzarc/Desktop/PSG_GPC_framework_master/PSG_Spectra/ColdRegimeF3D4Mars.pickle')
Earth = pd.read_pickle('/Users/ccruzarc/Desktop/PSG_GPC_framework_master/PSG_Spectra/TempRegimeF3D3Earth.pickle')
EarthSR = pd.read_pickle('/Users/ccruzarc/Desktop/PSG_GPC_framework_master/PSG_Spectra/TempRegimeF3D3EarthSR.pickle')
## All Need to have the same PARAMS! 
## P0 = [-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0] # Surface pressure in log(bars)
## Aslist = [0.0]
## gasMixRat = [-7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, -0.01]
# Hot, Cold, & Temperate Multi Surface Regime Models

# Hot, Cold & Temperate Single Surface Regime Models 
Cold = pd.read_pickle("/Users/ccruzarc/Desktop/PSG_GPC_framework_master/PSG_Spectra/FinalColdSingleSurfaceModels.pickle")  # T =150
Temp = pd.read_pickle('/Users/ccruzarc/Desktop/PSG_GPC_framework_master/PSG_Spectra/FinalTempSingleSurfaceModels.pickle') # T = 300
Hot  = pd.read_pickle("/Users/ccruzarc/Desktop/PSG_GPC_framework_master/PSG_Spectra/FinalHotSingleSurfaceModels.pickle") # T= 450

#Hot, Cold & Temperate Albedo Only Models
ColdAlbedo = pd.read_pickle("/Users/ccruzarc/Desktop/PSG_GPC_framework_master/PSG_Spectra/ColdRegimeAlbedoFinalCorrectedName.pickle")
TempAlbedo = pd.read_pickle("/Users/ccruzarc/Desktop/PSG_GPC_framework_master/PSG_Spectra/TempRegimeFinalAlbedo.pickle")
HotAlbedo = pd.read_pickle("/Users/ccruzarc/Desktop/PSG_GPC_framework_master/PSG_Spectra/HotRegimeAlbedoFinalCorrectedName.pickle")

# All the Cold Surface Endmember Models - No Atmosphere, albedo = 0
ColdSand_Pure = np.genfromtxt('/Users/ccruzarc/Desktop/PSG_GPC_framework_master/Pure_Surfaces/ColdSand_Pure.txt',comments = '#')
ColdBasalt_Pure = np.genfromtxt('/Users/ccruzarc/Desktop/PSG_GPC_framework_master/Pure_Surfaces/ColdBasalt_Pure.txt', comments = '#')
ColdWaterIce_Pure = np.genfromtxt('/Users/ccruzarc/Desktop/PSG_GPC_framework_master/Pure_Surfaces/ColdWaterIce_Pure.txt',comments = '#')

# All the Temperate Surface Endmember Models - No Atmosphere, albedo = 0
TempSand_Pure = np.genfromtxt('/Users/ccruzarc/Desktop/PSG_GPC_framework_master/Pure_Surfaces/TempSand_Pure.txt',comments = '#')
TempBasalt_Pure = np.genfromtxt('/Users/ccruzarc/Desktop/PSG_GPC_framework_master/Pure_Surfaces/TempBasalt_Pure.txt',comments = '#')
TempWaterIce_Pure = np.genfromtxt('/Users/ccruzarc/Desktop/PSG_GPC_framework_master/Pure_Surfaces/TempWaterIce_Pure.txt', comments = '#')
TempGrass_Pure = np.genfromtxt('/Users/ccruzarc/Desktop/PSG_GPC_framework_master/Pure_Surfaces/TempGrass_Pure.txt',comments = '#')
TempSeawater_Pure = np.genfromtxt('/Users/ccruzarc/Desktop/PSG_GPC_framework_master/Pure_Surfaces/TempSeawater_Pure.txt', comments = '#')

# All the Hot Surface Endmember Models - No Atmosphere, albedo = 0
HotBasalt_Pure = np.genfromtxt('/Users/ccruzarc/Desktop/PSG_GPC_framework_master/Pure_Surfaces/HotBasalt_Pure.txt',comments = '#')
HotSand_Pure = np.genfromtxt('/Users/ccruzarc/Desktop/PSG_GPC_framework_master/Pure_Surfaces/HotSand_Pure.txt',comments = '#')
HotSilicates_Pure = np.genfromtxt('/Users/ccruzarc/Desktop/PSG_GPC_framework_master/Pure_Surfaces/HotSilicates_Pure.txt',comments = '#')

# Pure Mars & Earth Endmember Spectra
MarsBasalt = np.genfromtxt('/Users/ccruzarc/Desktop/PSG_GPC_framework_master/Pure_Surfaces/MarsPureBasalt.txt', comments = '#')
MarsSand = np.genfromtxt('/Users/ccruzarc/Desktop/PSG_GPC_framework_master/Pure_Surfaces/MarsPureSand.txt', comments = '#')

# PSG Simulated Generic Models - Smith 2020 uses these
MarsPSGSim = np.genfromtxt('/Users/ccruzarc/Desktop/PSG_GPC_framework_master/Pure_Surfaces/NewMarsPSGsim.txt', comments = '#')
EarthPSGSim = np.genfromtxt('/Users/ccruzarc/Desktop/PSG_GPC_framework_master/EarthPSGSimAlbPT3.txt', comments = '#')
EarthPSGSimSR = np.genfromtxt('/Users/ccruzarc/Desktop/PSG_GPC_framework_master/NewEarthPSGSP.txt')
EarthPSGSimSR1 = np.genfromtxt('/Users/ccruzarc/Desktop/PSG_GPC_framework_master/NewEarthPSGSP1.txt') #albedo 0f .1
EarthPSGSimSR15 = np.genfromtxt('/Users/ccruzarc/Desktop/PSG_GPC_framework_master/NewEarthPSGSP15.txt') #albedo of .15
EarthPSGSimSR2 = np.genfromtxt('/Users/ccruzarc/Desktop/PSG_GPC_framework_master/NewEarthPSGSP2.txt') #albedo of .15

# Geometric Albedo Spectra from Madden 2018 - Earth's Spectrum doesn't have very good data.
MarsMadden = np.genfromtxt('/Users/ccruzarc/Desktop/CatalogofSolarSystemObjects/Albedos/Mars_McCord1971_Albedo.txt')
EarthMadden = np.genfromtxt('/Users/ccruzarc/Desktop/CatalogofSolarSystemObjects/Albedos/Earth_Lundock081121_Albedo_reduced.txt')
EarthMadden2 = np.genfromtxt('/Users/ccruzarc/Desktop/CatalogofSolarSystemObjects/Albedos/Earth_Lundock081121_Albedo.txt') #original
EarthMadden3 = np.genfromtxt('/Users/ccruzarc/Desktop/CatalogofSolarSystemObjects/Spectra/R8resolution/Sun/Earth_Lundock081121_Spec_Sun_LoRes.txt') #From Spectra folder
# VPL Reduced Earth Spectrum & the Original below it - just in case
Reduced_VPLEarth = np.genfromtxt('/Users/ccruzarc/Desktop/PSG_GPC_framework_master/Pure_Surfaces/VPLEarthbinned.txt')
VPLEarth = np.genfromtxt('/Users/ccruzarc/Desktop/PSG_GPC_framework_master/Pure_Surfaces/earth_24hr_diskavg_03192008.dat',skip_header=10)
VPLHEllabinned = np.genfromtxt('/Users/ccruzarc/Desktop/PSG_GPC_framework_master/VPLEarthbinned_hella_100.txt')


# In[6]:


print (TempAlbedo)
quit()

# In[266]:


space = ' ';


# ## Now we can start working on our plots.

# ### Let's start with Figure 3 Mars & Earth Comparisons

# #### Mars & Earth Comparisons - best matching surface combos

# In[17]:


fig = plt.figure(figsize=(12, 8))
# Set common labels
#fig.text(0.5, 0.04, 'common xlabel', ha='center', va='center')
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
# 1
G = gridspec.GridSpec(5, 1)
# Earth Comparison
axes_11 = plt.subplot(G[0, :])
axes_11.plot(EarthPSGSim[:,0],EarthPSGSim[:,1], label = 'PSG Earth Model' , marker = ",", color = "tab:cyan")
EarthPSGmodel = df(Earth,columns=['','H2O-2.0Basalt75Grass0Seawater0Sand0WaterIce25P0_0.0.cfg.txt'])
axes_11.plot(EarthPSGmodel, label = '75% Basalt\n25% Frost', marker = ",", color = 'tab:red')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_11.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_11.title.set_text('Earth Grid Models')
axes_11.set_xticklabels([])
#axes_11.set_ylim(0,.6)
axes_11.text(0.09,.6,'Combo  #1', ha='left', va='top')
#axes_11.set_ylabel('Geometric Albedo(I/F)')
plt.grid(True)  # show the grid
plt.style.context('seaborn-bright')  
# 2
#EarthComparisonsUpdatedSnames_H2O-2.0Basalt0Grass0Seawater60Sand0WaterIce40P0_0.0.cfg.txt
axes_15 = plt.subplot(G[1, :])
axes_15.plot(EarthPSGSim[:,0],EarthPSGSim[:,1], label = 'PSG Earth Model' , marker = ",", color = "tab:cyan")
EarthPSGmodel = df(Earth,columns=['','H2O-2.0Basalt0Grass0Seawater60Sand0WaterIce40P0_0.0.cfg.txt'])
axes_15.plot(EarthPSGmodel, label = '60% Seawater\n40% Frost', marker = ",", color = 'springgreen')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_15.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
#axes_15.set_ylabel('Geometric Albedo(I/F)')
axes_15.set_xticklabels([])
axes_15.text(0.09,.6,'Combo  #2', ha='left', va='top')
plt.grid(True) 
# 3
#EarthComparisonsUpdatedSnames_H2O-2.0Basalt12Grass50Seawater37Sand0WaterIce0P0_0.0.cfg.txt.png
axes_17 = plt.subplot(G[2, :])
axes_17.plot(EarthPSGSim[:,0],EarthPSGSim[:,1], label = 'PSG Earth Model' , marker = ",", color = "tab:cyan")
EarthPSGmodel = df(Earth,columns=['','H2O-2.0Basalt12Grass50Seawater37Sand0WaterIce0P0_0.0.cfg.txt'])
axes_17.plot(EarthPSGmodel, label = '12% Basalt\n50% Grass\n37% Seawater', marker = ",", color = 'olive')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_17.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
#axes_17.set_ylabel('Geometric Albedo(I/F)')
axes_17.set_xticklabels([])
axes_17.text(0.09,.6,'Combo  #3', ha='left', va='top')
plt.grid(True)
#axes_18.set_xlabel('Wavelength (um)')
# 4
axes_19 = plt.subplot(G[3, :])
axes_19.plot(EarthPSGSim[:,0],EarthPSGSim[:,1], label = 'PSG Earth Model' , marker = ",", color = "tab:cyan")
EarthPSGmodel = df(Earth,columns=['','H2O-2.0Basalt23Grass15Seawater38Sand15WaterIce8P0_0.0.cfg.txt'])
axes_19.plot(EarthPSGmodel, label = '23% Basalt\n15% Grass\n38% Seawater\n15% Sand\n8% Frost', marker = ",", color = 'rebeccapurple')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_19.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
#axes_19.set_ylabel('Geometric Albedo(I/F)')
axes_19.text(0.09,.6,'Combo  #4', ha='left', va='top')
axes_19.set_xticklabels([])
plt.grid(True)
#axes_19.set_xlabel('Wavelength (um)')
# 5
axes_20 = plt.subplot(G[4, :])
axes_20.plot(EarthPSGSim[:,0],EarthPSGSim[:,1], label = 'PSG Earth Model' , marker = ",", color = "tab:cyan")
EarthPSGmodel = df(Earth,columns=['','H2O-2.0Basalt75Grass0Seawater0Sand0WaterIce25P0_0.0.cfg.txt'])
axes_20.plot(EarthPSGmodel, label = 'Combo #1', marker = ",", color = 'tab:red')
EarthPSGmodel = df(Earth,columns=['','H2O-2.0Basalt0Grass0Seawater60Sand0WaterIce40P0_0.0.cfg.txt'])
axes_20.plot(EarthPSGmodel, label = 'Combo #2', marker = ",", color = 'springgreen')
EarthPSGmodel = df(Earth,columns=['','H2O-2.0Basalt12Grass50Seawater37Sand0WaterIce0P0_0.0.cfg.txt'])
axes_20.plot(EarthPSGmodel, label = 'Combo #3', marker = ",", color = 'olive')
EarthPSGmodel = df(Earth,columns=['','H2O-2.0Basalt23Grass15Seawater38Sand15WaterIce8P0_0.0.cfg.txt'])
axes_20.plot(EarthPSGmodel, label = 'Combo #4', marker = ",", color = 'rebeccapurple')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_20.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2,framealpha=0.5 )#,bbox_to_anchor=(.7, -.09))
#axes_19.set_ylabel('Geometric Albedo(I/F)')
plt.grid(True)
axes_20.set_xlabel('Wavelength (um)')
fig.tight_layout()
plt.show()
print("Figure 1. Shown here is the PSG vetted Earth geometric albedo spectrum along with Earth analog Temperature regime grid models. Different combinations of surface end members yield similar morphologies. There are no clear indications of surface detection features.\n *A real Earth geometric albedo spectrum will also be included in these figures once the data has been sent over.")
plt.close(fig)
#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs"
if not os.path.exists(dst):
	os.makedirs(dst)
    
  

fig.savefig("CruzArceFigs//EarthCrestIIAPR.png")

###########################

fig = plt.figure(figsize=(12, 8))
# Set common labels
#fig.text(0.5, 0.04, 'common xlabel', ha='center', va='center')
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')

G = gridspec.GridSpec(5, 1)
# Mars Comparison
axes_11 = plt.subplot(G[0, :])
axes_11.plot(MarsPSGSim[:,0],MarsPSGSim[:,1], label = 'PSG Mars Model' , marker = ",", color = "tab:cyan")
axes_11.plot(MarsMadden[:,0],MarsMadden[:,1], label = "Mars' Geometric Albedo", marker = ",", color = "blue")
MarsPSGmodel = df(Mars,columns=['','CO2-0.01Basalt100Sand0WaterIce0P0_-2.2.cfg.txt'])
axes_11.plot(MarsPSGmodel, label = '100% Basalt', marker = ",", color = 'tab:red')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_11.legend(by_label.values(), by_label.keys(),loc='upper left', handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_11.title.set_text('Mars Grid Models')
axes_11.set_xticklabels([])
axes_11.text(2.6,.5,'Combo #1', ha='right', va='top')
axes_11.set_ylim(0,.55)
#axes_11.set_ylabel('Geometric Albedo(I/F)')
plt.grid(True)  # show the grid
plt.style.context('seaborn-bright')  

#MarsComparisonsUpdatedSnames_H2O-2.0Basalt0Grass0Seawater60Sand0WaterIce40P0_0.0.cfg.txt
axes_15 = plt.subplot(G[1, :])
axes_15.plot(MarsPSGSim[:,0],MarsPSGSim[:,1], label = 'PSG Mars Model' , marker = ",", color = "tab:cyan")
axes_15.plot(MarsMadden[:,0],MarsMadden[:,1], label = "Mars' Geometric Albedo", marker = ",", color = "blue")
MarsPSGmodel = df(Mars,columns=['','CO2-0.01Basalt75Sand25WaterIce0P0_-2.2.cfg.txt'])
axes_15.plot(MarsPSGmodel, label = '75% Basalt\n25% Sand', marker = ",", color = 'springgreen')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_15.legend(by_label.values(), by_label.keys(),loc='upper left', handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2)#,bbox_to_anchor=(.7, -.09))
#axes_15.set_ylabel('Geometric Albedo(I/F)')
axes_15.set_xticklabels([])
axes_15.set_ylim(0,.55)
axes_15.text(2.6,.5,'Combo #2', ha='right', va='top')
plt.grid(True) 
#MarsComparisonsUpdatedSnames_H2O-2.0Basalt12Grass50Seawater37Sand0WaterIce0P0_0.0.cfg.txt.png
axes_17 = plt.subplot(G[2, :])
axes_17.plot(MarsPSGSim[:,0],MarsPSGSim[:,1], label = 'PSG Mars Model' , marker = ",", color = "tab:cyan")
axes_17.plot(MarsMadden[:,0],MarsMadden[:,1], label = "Mars' Geometric Albedo", marker = ",", color = "blue")
MarsPSGmodel = df(Mars,columns=['','CO2-0.01Basalt50Sand50WaterIce0P0_-2.2.cfg.txt'])
axes_17.plot(MarsPSGmodel, label = '50% Basalt\n50% Sand', marker = ",", color = 'olive')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_17.legend(by_label.values(), by_label.keys(),loc='upper left', handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
#axes_17.set_ylabel('Geometric Albedo(I/F)')
axes_17.set_xticklabels([])
axes_17.set_ylim(0,.55)
axes_17.text(2.6,.5,'Combo #3', ha='right', va='top')
plt.grid(True)

#axes_18.set_xlabel('Wavelength (um)')
#
axes_19 = plt.subplot(G[3, :])
axes_19.plot(MarsPSGSim[:,0],MarsPSGSim[:,1], label = 'PSG Mars Model' , marker = ",", color = "tab:cyan")
MarsPSGmodel = df(Mars,columns=['','CO2-0.01Basalt25Sand75WaterIce0P0_-2.2.cfg.txt'])
axes_19.plot(MarsMadden[:,0],MarsMadden[:,1], label = "Mars' Geometric Albedo", marker = ",", color = "blue")
axes_19.plot(MarsPSGmodel, label = '25% Basalt\n75% Sand', marker = ",", color = 'rebeccapurple')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_19.legend(by_label.values(), by_label.keys(),loc='upper left', handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09)) #framealpha=0.3
#axes_19.set_ylabel('Geometric Albedo(I/F)')
axes_19.text(2.6,.5,'Combo #4', ha='right', va='top')
axes_19.set_ylim(0,.55)
axes_19.set_xticklabels([])
plt.grid(True)
#axes_19.set_xlabel('Wavelength (um)')
#
axes_20 = plt.subplot(G[4,:])
axes_20.plot(MarsPSGSim[:,0],MarsPSGSim[:,1], label = 'PSG Mars Model' , marker = ",", color = "tab:cyan")
axes_20.plot(MarsMadden[:,0],MarsMadden[:,1], label = "Mars' Geometric Albedo", marker = ",", color = "blue")
MarsPSGmodel = df(Mars,columns=['','CO2-0.01Basalt100Sand0WaterIce0P0_-2.2.cfg.txt'])
axes_20.plot(MarsPSGmodel, label = 'Combo #1', marker = ",", color = 'tab:red')
MarsPSGmodel = df(Mars,columns=['','CO2-0.01Basalt75Sand25WaterIce0P0_-2.2.cfg.txt'])
axes_20.plot(MarsPSGmodel, label = 'Combo #2', marker = ",", color = 'springgreen')
MarsPSGmodel = df(Mars,columns=['','CO2-0.01Basalt50Sand50WaterIce0P0_-2.2.cfg.txt'])
axes_20.plot(MarsPSGmodel, label = 'Combo #3', marker = ",", color = 'olive')
MarsPSGmodel = df(Mars,columns=['','CO2-0.01Basalt25Sand75WaterIce0P0_-2.2.cfg.txt'])
axes_20.plot(MarsPSGmodel, label = 'Combo #4', marker = ",", color = 'rebeccapurple')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_20.legend(by_label.values(), by_label.keys(),loc='upper left', handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2,framealpha=0.5 )#,bbox_to_anchor=(.7, -.09)) #framealpha=0.3
#axes_19.set_ylabel('Geometric Albedo(I/F)')
plt.grid(True)
axes_20.set_ylim(0,.55)
axes_20.set_xlabel('Wavelength (um)')
fig.tight_layout()
plt.show()
print("Figure 2. Shown here are the PSG vetted Mars geometric albedo spectrum model, Mars' geometric albedo spectrum (Madden et al 2018) and Mars analog Cold regime grid models. Contrary to figure 1, the specific combination of 75% basalt and 25% sand yields the best match to the PSG model and geometric albedo data.")
plt.close(fig)


#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs"
if not os.path.exists(dst):
	os.makedirs(dst)
    
  

fig.savefig("CruzArceFigs//MarsCrestIIAPR.png")


# #### Mars & Earth Comparisons - Single Surface Models for each surface

# In[18]:


# Earth and Mars Grid Models for each Surface End-member
fig = plt.figure(figsize=(12, 8))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
# 1
G = gridspec.GridSpec(5, 1)
# Earth Comparison
axes_11 = plt.subplot(G[0, :])
axes_11.plot(EarthPSGSim[:,0],EarthPSGSim[:,1], label = 'PSG Earth Model' , marker = ",", color = "tab:cyan")
EarthPSGmodel = df(Earth,columns=['','H2O-2.0Basalt100Grass0Seawater0Sand0WaterIce0P0_0.0.cfg.txt'])
axes_11.plot(EarthPSGmodel, label = '100% Basalt', marker = ",", color = 'tab:red')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_11.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_11.title.set_text('Earth Grid Models')
axes_11.set_xticklabels([])
axes_11.set_ylim(0,.8)
#axes_11.set_ylim(0,.6)
#axes_11.text(0.09,.6,'Combo  #1', ha='left', va='top')
#axes_11.set_ylabel('Geometric Albedo(I/F)')
plt.grid(True)  # show the grid
plt.style.context('seaborn-bright')  
# 2
#EarthComparisonsUpdatedSnames_H2O-2.0Basalt0Grass0Seawater60Sand0WaterIce40P0_0.0.cfg.txt
axes_15 = plt.subplot(G[1, :])
axes_15.plot(EarthPSGSim[:,0],EarthPSGSim[:,1], label = 'PSG Earth Model' , marker = ",", color = "tab:cyan")
EarthPSGmodel = df(Earth,columns=['','H2O-2.0Basalt0Grass100Seawater0Sand0WaterIce0P0_0.0.cfg.txt'])
axes_15.plot(EarthPSGmodel, label = '100% Grass', marker = ",", color = 'springgreen')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_15.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
#axes_15.set_ylabel('Geometric Albedo(I/F)')
axes_15.set_xticklabels([])
axes_15.set_ylim(0,.8)
#axes_15.text(0.09,.6,'Combo  #2', ha='left', va='top')
plt.grid(True) 
# 3
#EarthComparisonsUpdatedSnames_H2O-2.0Basalt12Grass50Seawater37Sand0WaterIce0P0_0.0.cfg.txt.png
axes_17 = plt.subplot(G[2, :])
axes_17.plot(EarthPSGSim[:,0],EarthPSGSim[:,1], label = 'PSG Earth Model' , marker = ",", color = "tab:cyan")
EarthPSGmodel = df(Earth,columns=['','H2O-2.0Basalt0Grass0Seawater100Sand0WaterIce0P0_0.0.cfg.txt'])
axes_17.plot(EarthPSGmodel, label = '100% Seawater', marker = ",", color = 'olive')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_17.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
#axes_17.set_ylabel('Geometric Albedo(I/F)')
axes_17.set_xticklabels([])
axes_17.set_ylim(0,.8)
#axes_17.text(0.09,.6,'Combo  #3', ha='left', va='top')
plt.grid(True)
#axes_18.set_xlabel('Wavelength (um)')
# 4
axes_19 = plt.subplot(G[3, :])
axes_19.plot(EarthPSGSim[:,0],EarthPSGSim[:,1], label = 'PSG Earth Model' , marker = ",", color = "tab:cyan")
EarthPSGmodel = df(Earth,columns=['','H2O-2.0Basalt0Grass0Seawater0Sand100WaterIce0P0_0.0.cfg.txt'])
axes_19.plot(EarthPSGmodel, label = '100% Sand', marker = ",", color = 'rebeccapurple')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_19.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
#axes_19.set_ylabel('Geometric Albedo(I/F)')
#axes_19.text(0.09,.6,'Combo  #4', ha='left', va='top')
axes_19.set_xticklabels([])
axes_19.set_ylim(0,.8)
plt.grid(True)
#axes_19.set_xlabel('Wavelength (um)')
# 5
axes_20 = plt.subplot(G[4, :])
axes_20.plot(EarthPSGSim[:,0],EarthPSGSim[:,1], label = 'PSG Earth Model' , marker = ",", color = "tab:cyan")
EarthPSGmodel = df(Earth,columns=['','H2O-2.0Basalt0Grass0Seawater0Sand0WaterIce100P0_0.0.cfg.txt'])
axes_20.plot(EarthPSGmodel, label = '100% Frost', marker = ",", color = 'slategrey')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_20.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2,framealpha=0.5 )#,bbox_to_anchor=(.7, -.09))
#axes_19.set_ylabel('Geometric Albedo(I/F)')
axes_20.set_ylim(0,.8)
plt.grid(True)
axes_20.set_xlabel('Wavelength (um)')
fig.tight_layout()
plt.show()
#print("Figure 1. Shown here is the PSG vetted Earth geometric albedo spectrum along with Earth analog Temperature regime grid models. Different combinations of surface end members yield similar morphologies. There are no clear indications of surface detection features.\n *A real Earth geometric albedo spectrum will also be included in these figures once the data has been sent over.")
plt.close(fig)
#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs"
if not os.path.exists(dst):
	os.makedirs(dst)
    
  

fig.savefig("CruzArceFigs//EarthIndividualSurfaces.png")

###########################

fig = plt.figure(figsize=(12, 8))
# Set common labels
#fig.text(0.5, 0.04, 'common xlabel', ha='center', va='center')
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')

G = gridspec.GridSpec(3, 1)
# Mars Comparison
axes_11 = plt.subplot(G[0, :])
axes_11.plot(MarsPSGSim[:,0],MarsPSGSim[:,1], label = 'PSG Mars Model' , marker = ",", color = "tab:cyan")
axes_11.plot(MarsMadden[:,0],MarsMadden[:,1], label = "Mars' Geometric Albedo", marker = ",", color = "blue")
MarsPSGmodel = df(Mars,columns=['','CO2-0.01Basalt100Sand0WaterIce0P0_-2.2.cfg.txt'])
axes_11.plot(MarsPSGmodel, label = '100% Basalt', marker = ",", color = 'tab:red')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_11.legend(by_label.values(), by_label.keys(),loc='upper left', handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_11.title.set_text('Mars Grid Models')
axes_11.set_xticklabels([])
#axes_11.text(2.6,.5,'Combo #1', ha='right', va='top')
axes_11.set_ylim(0,.8)
#axes_11.set_ylabel('Geometric Albedo(I/F)')
plt.grid(True)  # show the grid
plt.style.context('seaborn-bright')  

#MarsComparisonsUpdatedSnames_H2O-2.0Basalt0Grass0Seawater60Sand0WaterIce40P0_0.0.cfg.txt
axes_15 = plt.subplot(G[1, :])
axes_15.plot(MarsPSGSim[:,0],MarsPSGSim[:,1], label = 'PSG Mars Model' , marker = ",", color = "tab:cyan")
axes_15.plot(MarsMadden[:,0],MarsMadden[:,1], label = "Mars' Geometric Albedo", marker = ",", color = "blue")
MarsPSGmodel = df(Mars,columns=['','CO2-0.01Basalt0Sand100WaterIce0P0_-2.2.cfg.txt'])
axes_15.plot(MarsPSGmodel, label = '100% Sand', marker = ",", color = 'springgreen')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_15.legend(by_label.values(), by_label.keys(),loc='upper left', handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2)#,bbox_to_anchor=(.7, -.09))
#axes_15.set_ylabel('Geometric Albedo(I/F)')
axes_15.set_xticklabels([])
axes_15.set_ylim(0,.8)
#axes_15.text(2.6,.5,'Combo #2', ha='right', va='top')
plt.grid(True) 
#MarsComparisonsUpdatedSnames_H2O-2.0Basalt12Grass50Seawater37Sand0WaterIce0P0_0.0.cfg.txt.png
axes_17 = plt.subplot(G[2, :])
axes_17.plot(MarsPSGSim[:,0],MarsPSGSim[:,1], label = 'PSG Mars Model' , marker = ",", color = "tab:cyan")
axes_17.plot(MarsMadden[:,0],MarsMadden[:,1], label = "Mars' Geometric Albedo", marker = ",", color = "blue")
MarsPSGmodel = df(Mars,columns=['','CO2-0.01Basalt0Sand0WaterIce100P0_-2.2.cfg.txt'])
axes_17.plot(MarsPSGmodel, label = '100% Sand', marker = ",", color = 'olive')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_17.legend(by_label.values(), by_label.keys(),loc='upper left', handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
#axes_17.set_ylabel('Geometric Albedo(I/F)')
#axes_17.set_xticklabels([])
axes_17.set_ylim(0,.8)
#axes_17.text(2.6,.5,'Combo #3', ha='right', va='top')
plt.grid(True)
axes_17.set_xlabel('Wavelength (um)')
fig.tight_layout()
plt.show()
#print("Figure 2. Shown here are the PSG vetted Mars geometric albedo spectrum model, Mars' geometric albedo spectrum (Madden et al 2018) and Mars analog Cold regime grid models. Contrary to figure 1, the specific combination of 75% basalt and 25% sand yields the best match to the PSG model and geometric albedo data.")
plt.close(fig)


#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs"
if not os.path.exists(dst):
	os.makedirs(dst)
    
  

fig.savefig("CruzArceFigs//MarsIndividualSurfaces.png")

print("No Matter what, the molecules in the atmosphere always show up")


# ### Now recreate figure two for albedos. What Params allow for the surface to be detectable?

# In[19]:


# Figure 2 edited and cleaned for Albedos. -Almost perfect plot design. Still want to incorporate mini splot withint aka, Magnifiying Glass view type deal-eo 
fig = plt.figure(figsize=(12, 8))
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0 , 'Wavelength', ha='center', va='center', rotation='horizontal')

G = gridspec.GridSpec(8, 2)
G.update( wspace = 0 ) # set the spacing between axes. wspace=0.025,hspace=0.0

# first quadrant will be CH4 in the Cold Regime
# gasMixRat = [-7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, -0.01]#,-6.0, -5.0, -4.0]#, -3.0, -2.0, -1.0, -0.01 ] # log(mixing ratio)
axes_00 = plt.subplot(G[0:2,0 ])
ColdAlbedoMabsCH4_7 = df(ColdAlbedo,columns=['','ColdCH4Mabs_-7.0P0_1.0.cfg.txt'])
axes_00.plot(ColdAlbedoMabsCH4_7, label = '-7.0', marker = ',', color = 'tab:red', linestyle= 'solid')


# ColdAlbedoMabsCH4_6 = df(ColdAlbedo,columns=['','ColdCH4Mabs_-6.0P0_1.0.cfg.txt'])
# axes_00.plot(ColdAlbedoMabsCH4_6, label = '-6.0', marker = ',', color = 'springgreen',linestyle='dashed')

# ColdAlbedoMabsCH4_5 = df(ColdAlbedo,columns=['','ColdCH4Mabs_-5.0P0_1.0.cfg.txt'])
# axes_00.plot(ColdAlbedoMabsCH4_5, label = '-5.0', marker = ',', color = 'olive')#,linestyle='dashed')

ColdAlbedoMabsCH4_4 = df(ColdAlbedo,columns=['','ColdCH4Mabs_-4.0P0_1.0.cfg.txt'])
axes_00.plot(ColdAlbedoMabsCH4_4, label = '-4.0', marker = ',', color = 'rebeccapurple',linestyle='dotted')

# ColdAlbedoMabsCH4_3 = df(ColdAlbedo,columns=['','ColdCH4Mabs_-3.0P0_1.0.cfg.txt'])
# axes_00.plot(ColdAlbedoMabsCH4_3, label = '-3.0', marker = ',', color = 'slategrey')#,linestyle='dashed')

ColdAlbedoMabsCH4_2 = df(ColdAlbedo,columns=['','ColdCH4Mabs_-2.0P0_1.0.cfg.txt'])
axes_00.plot(ColdAlbedoMabsCH4_2, label = '-2.0', marker = ',', color = 'tab:blue',linestyle='dashed')

# ColdAlbedoMabsCH4_1 = df(ColdAlbedo,columns=['','ColdCH4Mabs_-1.0P0_1.0.cfg.txt'])
# axes_00.plot(ColdAlbedoMabsCH4_1, label = '-1.0', marker = ',', color = 'm',linestyle='dotted')

ColdAlbedoMabsCH4_01 = df(ColdAlbedo,columns=['','ColdCH4Mabs_-0.01P0_1.0.cfg.txt'])
axes_00.plot(ColdAlbedoMabsCH4_01, label = '-0.01', marker = ',', color = 'darkorange',linestyle='dotted')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.title.set_text('Cold CH4')
axes_00.text(0.25,0.93,'log($M_{abs}$)', ha='left', va='top',fontweight = 'roman',clip_box='bbox')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off#axes_17.set_ylim(0,.6)
#axes_17.set_ylabel('Geometric Albedo(I/F)')
plt.grid(True)  # show the grid
axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.8)
# plt.tight_layout()


axes_10 = plt.subplot(G[2:4,0 ])
# ColdAlbedoCH4_P0N3 = df(ColdAlbedo,columns=['','ColdCH4Mabs_-2.0P0_0.001.cfg.txt'])
# axes_10.plot(ColdAlbedoCH4_P0N3, label = '- 3.0', marker = ',', color = 'tab:red')

ColdAlbedoCH4_P0N2 = df(ColdAlbedo,columns=['','ColdCH4Mabs_-2.0P0_0.01.cfg.txt'])
axes_10.plot(ColdAlbedoCH4_P0N2, label = '-2.0', marker = ',', color = 'springgreen', linestyle = 'dotted')

# ColdAlbedoCH4_P0N1 = df(ColdAlbedo,columns=['','ColdCH4Mabs_-2.0P0_0.1.cfg.txt'])
# axes_10.plot(ColdAlbedoCH4_P0N1, label = '-1.0', marker = ',', color = 'olive')

ColdAlbedoCH4_P00 = df(ColdAlbedo,columns=['','ColdCH4Mabs_-2.0P0_1.0.cfg.txt'])
axes_10.plot(ColdAlbedoCH4_P00, label = '0.0', marker = ',', color = 'rebeccapurple', linestyle = 'dotted')

# ColdAlbedoCH4_P0P1 = df(ColdAlbedo,columns=['','ColdCH4Mabs_-2.0P0_10.0.cfg.txt'])
# axes_10.plot(ColdAlbedoCH4_P0P1, label = '1.0', marker = ',', color = 'slategrey')#,linestyle='dashed')

ColdAlbedoCH4_P0P2 = df(ColdAlbedo,columns=['','ColdCH4Mabs_-2.0P0_100.0.cfg.txt'])
axes_10.plot(ColdAlbedoCH4_P0P2, label = '2.0', marker = ',', color = 'tab:blue',linestyle='dotted')

# ColdAlbedoCH4_P0P3 = df(ColdAlbedo,columns=['','ColdCH4Mabs_-2.0P0_1000.0.cfg.txt'])
# axes_10.plot(ColdAlbedoCH4_P0P3, label = '3.0', marker = ',', color = 'm')#,linestyle='dashed')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_10.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_10.text(0.25,0.93,'log($P_{0}$)', ha='left', va='top',fontweight = 'roman',clip_box='bbox')
plt.grid(True)
axes_10.set_ylim(0,1)
axes_10.set_xlim(0.2,2.8)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
# plt.tight_layout()

# Next quadrant to the right will be Temperate H2O

axes_01 = plt.subplot(G[0:2,1 ])
TempAlbedoMabsH2O_7 = df(TempAlbedo,columns=['','TempH2OMabs_-7.0P0_1.0.cfg.txt'])
axes_01.plot(TempAlbedoMabsH2O_7, label = '-7.0', marker = ',', color = 'tab:red',linestyle='solid')

# TempAlbedoMabsH2O_6 = df(TempAlbedo,columns=['','TempH2OMabs_-6.0P0_1.0.cfg.txt'])
# axes_01.plot(TempAlbedoMabsH2O_6, label = '-6.0', marker = ',', color = 'springgreen',linestyle='dashed')

# TempAlbedoMabsH2O_5 = df(TempAlbedo,columns=['','TempH2OMabs_-5.0P0_1.0.cfg.txt'])
# axes_01.plot(TempAlbedoMabsH2O_5, label = '-5.0', marker = ',', color = 'olive',linestyle='dashed')

TempAlbedoMabsH2O_4 = df(TempAlbedo,columns=['','TempH2OMabs_-4.0P0_1.0.cfg.txt'])
axes_01.plot(TempAlbedoMabsH2O_4, label = '-4.0', marker = ',', color = 'rebeccapurple',linestyle='dotted')

# TempAlbedoMabsH2O_3 = df(TempAlbedo,columns=['','TempH2OMabs_-3.0P0_1.0.cfg.txt'])
# axes_01.plot(TempAlbedoMabsH2O_3, label = '-3.0', marker = ',', color = 'slategrey',linestyle='dashed')

TempAlbedoMabsH2O_2 = df(TempAlbedo,columns=['','TempH2OMabs_-2.0P0_1.0.cfg.txt'])
axes_01.plot(TempAlbedoMabsH2O_2, label = '-2.0', marker = ',', color = 'tab:blue',linestyle='dashed')

# TempAlbedoMabsH2O_1 = df(TempAlbedo,columns=['','TempH2OMabs_-1.0P0_1.0.cfg.txt'])
# axes_01.plot(TempAlbedoMabsH2O_1, label = '-1.0', marker = ',', color = 'm',linestyle='dotted')

TempAlbedoMabsH2O_01 = df(TempAlbedo,columns=['','TempH2OMabs_-0.01P0_1.0.cfg.txt'])
axes_01.plot(TempAlbedoMabsH2O_01, label = '-0.01', marker = ',', color = 'darkorange',linestyle='dotted')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.title.set_text('Temp H2O')
axes_01.text(0.25,0.93,'log($M_{abs}$)', ha='left', va='top',fontweight = 'roman',clip_box='bbox')
#axes_01.set_ylim(0,.6)
#axes_01.set_ylabel('Geometric Albedo(I/F)')
plt.grid(True)  # show the grid
axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.8)
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
# plt.tight_layout()

axes_11 = plt.subplot(G[2:4,1 ])
# TempAlbedoH2O_P0N3 = df(TempAlbedo,columns=['','TempH2OMabs_-1.0P0_0.001.cfg.txt'])
# axes_11.plot(TempAlbedoH2O_P0N3, label = '-3.0', marker = ',', color = 'tab:red')

TempAlbedoH2O_P0N2 = df(TempAlbedo,columns=['','TempH2OMabs_-1.0P0_0.01.cfg.txt'])
axes_11.plot(TempAlbedoH2O_P0N2, label = '-2.0', marker = ',', color = 'springgreen',linestyle='dotted')

# TempAlbedoH2O_P0N1 = df(TempAlbedo,columns=['','TempH2OMabs_-1.0P0_0.1.cfg.txt'])
# axes_11.plot(TempAlbedoH2O_P0N1, label = '-1.0', marker = ',', color = 'olive')

TempAlbedoH2O_P00 = df(TempAlbedo,columns=['','TempH2OMabs_-1.0P0_1.0.cfg.txt'])
axes_11.plot(TempAlbedoH2O_P00, label = '0.0', marker = ',', color = 'rebeccapurple',linestyle='dotted')

# TempAlbedoH2O_P0P1 = df(TempAlbedo,columns=['','TempH2OMabs_-1.0P0_10.0.cfg.txt'])
# axes_11.plot(TempAlbedoH2O_P0P1, label = '1.0', marker = ',', color = 'slategrey',linestyle='dashed')

TempAlbedoH2O_P0P2 = df(TempAlbedo,columns=['','TempH2OMabs_-1.0P0_100.0.cfg.txt'])
axes_11.plot(TempAlbedoH2O_P0P2, label = '2.0', marker = ',', color = 'tab:blue',linestyle='dotted')

# TempAlbedoH2O_P0P3 = df(TempAlbedo,columns=['','TempH2OMabs_-1.0P0_1000.0.cfg.txt'])
# axes_11.plot(TempAlbedoH2O_P0P3, label = '3.0', marker = ',', color = 'm',linestyle='dashed')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_11.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_11.text(0.25,0.93,'log($P_{0}$)', ha='left', va='top',fontweight = 'roman',clip_box='bbox')
plt.grid(True)
axes_11.set_ylim(0,1)
axes_11.set_xlim(0.2,2.8)
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
# plt.tight_layout()

# Dope. Next quadrant will be below Cold CH4. It is Hot SO2
# SO2 Mabs
axes_20 = plt.subplot(G[4:6,0 ])
HotAlbedoMabsSO2_7 = df(HotAlbedo,columns=['','HotSO2Mabs_-7.0P0_1.0.cfg.txt'])
axes_20.plot(HotAlbedoMabsSO2_7, label = '-7.0', marker = ',', color = 'tab:red',linestyle='solid')

# HotAlbedoMabsSO2_6 = df(HotAlbedo,columns=['','HotSO2Mabs_-6.0P0_1.0.cfg.txt'])
# axes_20.plot(HotAlbedoMabsSO2_6, label = '-6.0', marker = ',', color = 'springgreen',linestyle='dashed')

# HotAlbedoMabsSO2_5 = df(HotAlbedo,columns=['','HotSO2Mabs_-5.0P0_1.0.cfg.txt'])
# axes_20.plot(HotAlbedoMabsSO2_5, label = '-5.0', marker = ',', color = 'olive',linestyle='dashed')

HotAlbedoMabsSO2_4 = df(HotAlbedo,columns=['','HotSO2Mabs_-4.0P0_1.0.cfg.txt'])
axes_20.plot(HotAlbedoMabsSO2_4, label = '-4.0', marker = ',', color = 'rebeccapurple',linestyle='dotted')

# HotAlbedoMabsSO2_3 = df(HotAlbedo,columns=['','HotSO2Mabs_-3.0P0_1.0.cfg.txt'])
# axes_20.plot(HotAlbedoMabsSO2_3, label = '-3.0', marker = ',', color = 'slategrey',linestyle='dotted')

HotAlbedoMabsSO2_2 = df(HotAlbedo,columns=['','HotSO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_20.plot(HotAlbedoMabsSO2_2, label = '-2.0', marker = ',', color = 'tab:blue',linestyle='dotted')

# HotAlbedoMabsSO2_1 = df(HotAlbedo,columns=['','HotSO2Mabs_-1.0P0_1.0.cfg.txt'])
# axes_20.plot(HotAlbedoMabsSO2_1, label = '-1.0', marker = ',', color = 'm',linestyle='dotted')

HotAlbedoMabsSO2_01 = df(HotAlbedo,columns=['','HotSO2Mabs_-0.01P0_1.0.cfg.txt'])
axes_20.plot(HotAlbedoMabsSO2_01, label = '-0.01', marker = ',', color = 'darkorange',linestyle='dotted')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_20.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_20.title.set_text('Hot SO2')
axes_20.text(0.25,0.93,'log($M_{abs}$)', ha='left', va='top',fontweight = 'roman',clip_box='bbox')
#axes_20.set_ylim(0,.6)
#axes_20.set_ylabel('Geometric Albedo(I/F)')
plt.grid(True)  # show the grid
# plt.tight_layout()
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
axes_20.set_ylim(0,1)
axes_20.set_xlim(0.2,2.8)
# SO2 Surface Pressure
axes_30 = plt.subplot(G[6:8,0 ])
# HotAlbedoSO2_P0N3 = df(HotAlbedo,columns=['','HotSO2Mabs_-1.0P0_0.001.cfg.txt'])
# axes_30.plot(HotAlbedoSO2_P0N3, label = '-3.0', marker = ',', color = 'tab:red')

HotAlbedoSO2_P0N2 = df(HotAlbedo,columns=['','HotSO2Mabs_-1.0P0_0.01.cfg.txt'])
axes_30.plot(HotAlbedoSO2_P0N2, label = '-2.0', marker = ',', color = 'springgreen',linestyle='dotted')

# HotAlbedoSO2_P0N1 = df(HotAlbedo,columns=['','HotSO2Mabs_-1.0P0_0.1.cfg.txt'])
# axes_30.plot(HotAlbedoSO2_P0N1, label = '-1.0', marker = ',', color = 'olive')

HotAlbedoSO2_P00 = df(HotAlbedo,columns=['','HotSO2Mabs_-1.0P0_1.0.cfg.txt'])
axes_30.plot(HotAlbedoSO2_P00, label = '0.0', marker = ',', color = 'rebeccapurple',linestyle='dotted')

# HotAlbedoSO2_P0P1 = df(HotAlbedo,columns=['','HotSO2Mabs_-1.0P0_10.0.cfg.txt'])
# axes_30.plot(HotAlbedoSO2_P0P1, label = '1.0', marker = ',', color = 'slategrey',linestyle='dashed')

HotAlbedoSO2_P0P2 = df(HotAlbedo,columns=['','HotSO2Mabs_-1.0P0_100.0.cfg.txt'])
axes_30.plot(HotAlbedoSO2_P0P2, label = '2.0', marker = ',', color = 'tab:blue',linestyle='dotted')

# HotAlbedoSO2_P0P3 = df(HotAlbedo,columns=['','HotSO2Mabs_-1.0P0_1000.0.cfg.txt'])
# axes_30.plot(HotAlbedoSO2_P0P3, label = '3.0', marker = ',', color = 'm',linestyle='dashed')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_30.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_30.text(0.25,0.93,'log($P_{0}$)', ha='left', va='top',fontweight = 'roman',clip_box='bbox')
plt.grid(True)
axes_30.set_ylim(0,1)
axes_30.set_xlim(0.2,2.8)
# plt.tight_layout()

# Last quadrant will be for CO2 in Each Regime
# This first one will be for M_abs
axes_21 = plt.subplot(G[4:6,1])
# Here are the Cold ones
ColdAlbedoMabs2 = df(ColdAlbedo,columns=['','ColdCO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_21.plot(ColdAlbedoMabs2, label = 'Cold Regime', marker = ',', color = 'tab:red',linestyle='dotted')


# ColdAlbedoMabs4 = df(ColdAlbedo,columns=['','ColdCO2Mabs_-4.0P0_1.0.cfg.txt'])
# axes_21.plot(ColdAlbedoMabs4, label = '-4.0', marker = ',', color = 'rebeccapurple',linestyle='dashed')


# ColdAlbedoMabs1 = df(ColdAlbedo,columns=['','ColdCO2Mabs_-1.0P0_1.0.cfg.txt'])
# axes_21.plot(ColdAlbedoMabs1, label = '-1.0', marker = ',', color = 'm',linestyle='dashed')

# ColdAlbedoMabs01 = df(ColdAlbedo,columns=['','ColdCO2Mabs_-0.01P0_1.0.cfg.txt'])
# axes_21.plot(ColdAlbedoMabs01, label = '-0.01', marker = ',', color = 'darkorange',linestyle='dashed')
# these are the temperate ones
TempAlbedoMabs2 = df(TempAlbedo,columns=['','TempCO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_21.plot(TempAlbedoMabs2, label = 'T emperate Regime', marker = ',', color = 'tab:blue',linestyle='dotted')

# TempAlbedoMabs4 = df(TempAlbedo,columns=['','TempCO2Mabs_-4.0P0_1.0.cfg.txt'])
# axes_21.plot(TempAlbedoMabs4, label = '-4.0', marker = ',', color = 'rebeccapurple',linestyle='dashed')

# TempAlbedoMabs1 = df(TempAlbedo,columns=['','TempCO2Mabs_-1.0P0_1.0.cfg.txt'])
# axes_21.plot(TempAlbedoMabs1, label = '-1.0', marker = ',', color = 'm',linestyle='dashed')

# TempAlbedoMabs01 = df(TempAlbedo,columns=['','TempCO2Mabs_-0.01P0_1.0.cfg.txt'])
# axes_21.plot(TempAlbedoMabs01, label = '-0.01', marker = ',', color = 'darkorange',linestyle='dashed')
# and here are the hot ones
HotAlbedoMabs2 = df(HotAlbedo,columns=['','HotCO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_21.plot(HotAlbedoMabs2, label = 'Hot Regime', marker = ',', color = 'darkorange',linestyle='dotted')

# HotAlbedoMabs4 = df(HotAlbedo,columns=['','HotCO2Mabs_-4.0P0_1.0.cfg.txt'])
# axes_21.plot(HotAlbedoMabs4, label = '-4.0', marker = ',', color = 'rebeccapurple',linestyle='dashed')

# HotAlbedoMabs1 = df(HotAlbedo,columns=['','HotCO2Mabs_-1.0P0_1.0.cfg.txt'])
# axes_21.plot(HotAlbedoMabs1, label = '-1.0', marker = ',', color = 'm',linestyle='dashed')

# HotAlbedoMabs01 = df(HotAlbedo,columns=['','HotCO2Mabs_-0.01P0_1.0.cfg.txt'])
# axes_21.plot(HotAlbedoMabs01, label = '-0.01', marker = ',', color = 'darkorange',linestyle='dashed')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_21.title.set_text('CO2 in All Regimes')
axes_21.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_21.text(0.25,0.93,'-2.0 log($M_{abs}$)', ha='left', va='top',fontweight = 'roman',clip_box='bbox')
axes_21.set_ylim(0,1)
axes_21.set_xlim(0.2,2.8)
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
#axes_2111.set_ylim(0,.6)
#axes_2111.set_ylabel('Geometric Albedo(I/F)')
plt.grid(True)  # show the grid
# plt.tight_layout()

# This last one will be for CO2 in all temp regimes, Surface Pressure
axes_31 = plt.subplot(G[6:8,1])
# Here are the cold ones
# ColdAlbedoP0N2 = df(ColdAlbedo,columns=['','ColdCO2Mabs_-1.0P0_0.01.cfg.txt'])
# axes_31.plot(ColdAlbedoP0N2, label = '-2.0', marker = ',', color = 'springgreen')

ColdAlbedoP00 = df(ColdAlbedo,columns=['','ColdCO2Mabs_-1.0P0_1.0.cfg.txt'])
axes_31.plot(ColdAlbedoP00, label = 'Cold Regime', marker = ',',color = 'springgreen',linestyle='dotted')

# ColdAlbedoP0P2 = df(ColdAlbedo,columns=['','ColdCO2Mabs_-1.0P0_100.0.cfg.txt'])
# axes_31.plot(ColdAlbedoP0P2, label = '2.0', marker = ',', color = 'tab:blue',linestyle='dashed')

# Here goes the temperate ones
# TempAlbedoP0N2 = df(TempAlbedo,columns=['','TempCO2Mabs_-1.0P0_0.01.cfg.txt'])
# axes_31.plot(TempAlbedoP0N2, label = '-2.0', marker = ',', color = 'springgreen')

TempAlbedoP00 = df(TempAlbedo,columns=['','TempCO2Mabs_-1.0P0_1.0.cfg.txt'])
axes_31.plot(TempAlbedoP00, label = 'T emperate Regime', marker = ',', color = 'rebeccapurple',linestyle='dotted')

# TempAlbedoP0P2 = df(TempAlbedo,columns=['','TempCO2Mabs_-1.0P0_100.0.cfg.txt'])
# axes_31.plot(TempAlbedoP0P2, label = '2.0', marker = ',', color = 'tab:blue',linestyle='dashed')

# and here are the hot ones
# HotAlbedoP0N2 = df(HotAlbedo,columns=['','HotCO2Mabs_-1.0P0_0.01.cfg.txt'])
# axes_31.plot(HotAlbedoP0N2, label = '-2.0', marker = ',', color = 'springgreen')

HotAlbedoP00 = df(HotAlbedo,columns=['','HotCO2Mabs_-1.0P0_1.0.cfg.txt'])
axes_31.plot(HotAlbedoP00, label = 'Hot Regime', marker = ',', color = 'tab:blue',linestyle='dotted')

# HotAlbedoP0P2 = df(HotAlbedo,columns=['','HotCO2Mabs_-1.0P0_100.0.cfg.txt'])
# axes_31.plot(HotAlbedoP0P2, label = '2.0', marker = ',', color = 'tab:blue',linestyle='dashed')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_31.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_31.text(0.25,0.93,'0.0 log($P_{0}$)', ha='left', va='top',fontweight = 'roman',clip_box='bbox')
plt.grid(True)
axes_31.set_ylim(0,1)
axes_31.set_xlim(0.2,2.8) 
plt.tick_params(axis='y', 
	which='both',  
	right=False, 
	left=False, 
	labelleft=False)
# plt.tight_layout()
G.tight_layout(fig)
# Getting respective M_abs and P0 plots all together. I.e. no vertical spacing
# nbins = len(axes_10.get_xticklabels())
# axes_10.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='upper'))
# axes_00.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='upper'))

# # change axis location of ax5
# pos00 = axes_00.get_position()
# pos10 = axes_10.get_position()

# points00 = pos00.get_points()
# points10 = pos10.get_points()

# points10[1][0]=points00[0][0]

# pos10.set_points(points10)

# axes_10.set_position(pos10)


plt.show()



print('done')


# #### Well the answer isn't here. Above are albedo only models

# ### Still looking to answer; what params allow for a detectable surface

# #### Below is a 6x3 gridspec plot of surfaces under different regimes and different parameters. Work off this plot to answer the question above

# In[20]:


fig = plt.figure(figsize=(15, 8.5))
#fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
#fig.text(0.5,0 , 'Wavelength', ha='center', va='center', rotation='horizontal')
G = gridspec.GridSpec(6,3)
#G.update( wspace = 0 ) # set the spacing between axes. wspace=0.025,hspace=0.0
# Seawater Block - Left Top, M_abs
axes_010 = plt.subplot(G[0:1,0]) # Seawater Block - Left Top, M_abs
# M_abs first
# CO2
MabsSeawater7CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-7.0P0_1.0.cfg.txt'])
axes_010.plot(MabsSeawater7CO2, label = '-7.0 ($CO_{2}$)', marker = ',', color = 'tab:red', linestyle= 'dotted')
# MabsSeawater4CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-4.0P0_1.0.cfg.txt'])
# axes_010.plot(MabsSeawater4CO2, label = '-4.0 ($CO_{2}$)', marker = ',', color = 'rebeccapurple', linestyle= 'dotted')
MabsSeawater2CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_010.plot(MabsSeawater2CO2, label = '-2.0 ($CO_{2}$)', marker = ',', color = 'tab:blue', linestyle= 'dotted')
MabsSeawater01CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-0.01P0_1.0.cfg.txt'])
axes_010.plot(MabsSeawater01CO2, label = '-0.01 ($CO_{2}$)', marker = ',', color = 'darkorange', linestyle= 'dotted')
# H2O
MabsSeawater7H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-7.0P0_1.0.cfg.txt'])
axes_010.plot(MabsSeawater7H2O, label = '-7.0 ($H_{2}O$)', marker = ',', color = 'red', linestyle= 'dotted')
# MabsSeawater4H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-4.0P0_1.0.cfg.txt'])
# axes_010.plot(MabsSeawater4H2O, label = '-4.0 ($H_{2}O$)', marker = ',', color = 'rebeccapurple', linestyle= 'dotted')
MabsSeawater2H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-2.0P0_1.0.cfg.txt'])
axes_010.plot(MabsSeawater2H2O, label = '-2.0 ($H_{2}O$)', marker = ',', color = 'blue', linestyle= 'dotted')
MabsSeawater01H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-0.01P0_1.0.cfg.txt'])
axes_010.plot(MabsSeawater01H2O, label = '-0.01 ($H_{2}O$)', marker = ',', color = 'orange', linestyle= 'dotted')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
# axes_010.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_010.text(0.25,0.93,'log($M_{abs}$)', ha='left', va='top',fontweight = 'roman',clip_box='bbox')
plt.grid(True)
axes_010.set_ylim(0,1)
axes_010.set_xlim(0.2,2.8)
axes_010.title.set_text('Seawater: Temperate Regime')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
# boop
axes_120 = plt.subplot(G[1:2,0]) # Seawater Block - Left Top, P_0
# P_0 second
# CO2
P0Seawater7CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-2.0P0_0.01.cfg.txt'])
axes_120.plot(P0Seawater7CO2, label = '-2.0 ($CO_{2}$)', marker = ',', color = 'springgreen', linestyle= 'dotted')
P0Seawater4CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-2.0P0_1.0cfg.txt'])
axes_120.plot(P0Seawater4CO2, label = '0.0 ($CO_{2}$)', marker = ',', color = 'rebeccapurple', linestyle= 'dotted')
P0Seawater2CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-2.0P0_100.0.cfg.txt'])
axes_120.plot(P0Seawater2CO2, label = '2.0 ($CO_{2}$)', marker = ',', color = 'tab:blue', linestyle= 'dotted')
# H2O
P0Seawater7H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-2.0P0_0.01.cfg.txt'])
axes_120.plot(P0Seawater7H2O, label = '-2.0 ($H_{2}O$)', marker = ',', color = 'green', linestyle= 'dotted')
P0Seawater4H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-2.0P0_1.0.cfg.txt'])
axes_120.plot(P0Seawater4H2O, label = '0.0 ($H_{2}O$)', marker = ',', color = 'purple', linestyle= 'dotted')
P0Seawater2H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-2.0P0_100.0.cfg.txt'])
axes_120.plot(P0Seawater2H2O, label = '2.0 ($H_{2}O$)', marker = ',', color = 'blue', linestyle= 'dotted')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
# axes_120.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_120.text(0.25,0.93,'log($P_{0}$)', ha='left', va='top',fontweight = 'roman',clip_box='bbox')
plt.grid(True)
axes_120.set_ylim(0,1)
axes_120.set_xlim(0.2,2.8)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

# Grass Block - Left Middel
axes_230 = plt.subplot(G[2:3,0]) # Grass Block - Left Middel
MabsGrass7CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-7.0P0_1.0.cfg.txt'])
axes_230.plot(MabsGrass7CO2, label = '-7.0 ($CO_{2}$)', marker = ',', color = 'tab:red', linestyle= 'dotted')
# MabsGrass4CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-4.0P0_1.0.cfg.txt'])
# axes_230.plot(MabsGrass4CO2, label = '-4.0 ($CO_{2}$)', marker = ',', color = 'rebeccapurple', linestyle= 'dotted')
MabsGrass2CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_230.plot(MabsGrass2CO2, label = '-2.0 ($CO_{2}$)', marker = ',', color = 'tab:blue', linestyle= 'dotted')
MabsGrass01CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-0.01P0_1.0.cfg.txt'])
axes_230.plot(MabsGrass01CO2, label = '-0.01 ($CO_{2}$)', marker = ',', color = 'darkorange', linestyle= 'dotted')
# H2O
MabsGrass7H2O = df(Temp,columns=['','TempGrassH2OMabs_-7.0P0_1.0.cfg.txt'])
axes_230.plot(MabsGrass7H2O, label = '-7.0 ($H_{2}O$)', marker = ',', color = 'red', linestyle= 'dotted')
# MabsGrass4H2O = df(Temp,columns=['','TempGrassH2OMabs_-4.0P0_1.0.cfg.txt'])
# axes_230.plot(MabsGrass4H2O, label = '-4.0 ($H_{2}O$)', marker = ',', color = 'rebeccapurple', linestyle= 'dotted')
MabsGrass2H2O = df(Temp,columns=['','TempGrassH2OMabs_-2.0P0_1.0.cfg.txt'])
axes_230.plot(MabsGrass2H2O, label = '-2.0 ($H_{2}O$)', marker = ',', color = 'blue', linestyle= 'dotted')
MabsGrass01H2O = df(Temp,columns=['','TempGrassH2OMabs_-0.01P0_1.0.cfg.txt'])
axes_230.plot(MabsGrass01H2O, label = '-0.01 ($H_{2}O$)', marker = ',', color = 'orange', linestyle= 'dotted')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
# axes_230.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_230.text(0.25,0.93,'log($M_{abs}$)', ha='left', va='top',fontweight = 'roman',clip_box='bbox')
plt.grid(True)
axes_230.set_ylim(0,1)
axes_230.set_xlim(0.2,2.8)
axes_230.title.set_text('Grass: Temperate Regime')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
# boop
axes_340 = plt.subplot(G[3:4,0]) # Grass Block - Left Top, P_0
# P_0 second
# CO2
P0Grass7CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-2.0P0_0.01.cfg.txt'])
axes_340.plot(P0Grass7CO2, label = '-2.0 ($CO_{2}$)', marker = ',', color = 'springgreen', linestyle= 'dotted')
P0Grass4CO2 = df(Temp,columns=['','TempCO2Mabs_-2.0P0_1.0cfg.txt'])
axes_340.plot(P0Grass4CO2, label = '0.0 ($CO_{2}$)', marker = ',', color = 'rebeccapurple', linestyle= 'dotted')
P0Grass2CO2 = df(Temp,columns=['','TempCO2Mabs_-2.0P0_100.0.cfg.txt'])
axes_340.plot(P0Grass2CO2, label = '2.0 ($CO_{2}$)', marker = ',', color = 'tab:blue', linestyle= 'dotted')
# H2O
P0Grass7H2O = df(Temp,columns=['','TempGrassH2OMabs_-2.0P0_0.01.cfg.txt'])
axes_340.plot(P0Grass7H2O, label = '-2.0 ($H_{2}O$)', marker = ',', color = 'green', linestyle= 'dotted')
P0Grass4H2O = df(Temp,columns=['','TempGrassH2OMabs_-2.0P0_1.0.cfg.txt'])
axes_340.plot(P0Grass4H2O, label = '0.0 ($H_{2}O$)', marker = ',', color = 'purple', linestyle= 'dotted')
P0Grass2H2O = df(Temp,columns=['','TempGrassH2OMabs_-2.0P0_100.0.cfg.txt'])
axes_340.plot(P0Grass2H2O, label = '2.0 ($H_{2}O$)', marker = ',', color = 'blue', linestyle= 'dotted')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
# axes_340.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_340.text(0.25,0.93,'log($P_{0}$)', ha='left', va='top',fontweight = 'roman',clip_box='bbox')
plt.grid(True)
axes_340.set_ylim(0,1)
axes_340.set_xlim(0.2,2.8)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
# Glass Block - Left Bottom
# CO2
axes_450 = plt.subplot(G[4:5,0]) # Glass Block - Left Bottom
MabsSilicates7CO2 = df(Hot,columns=['','HotSilicatesCO2Mabs_-7.0P0_1.0.cfg.txt'])
axes_450.plot(MabsSilicates7CO2, label = '-7.0 ($CO_{2}$)', marker = ',', color = 'tab:red', linestyle= 'dotted')
# MabsSilicates4CO2 = df(Hot,columns=['','HotSilicatesCO2Mabs_-4.0P0_1.0.cfg.txt'])
# axes_450.plot(MabsSilicates4CO2, label = '-4.0 ($CO_{2}$)', marker = ',', color = 'rebeccapurple', linestyle= 'dotted')
MabsSilicates2CO2 = df(Hot,columns=['','HotSilicatesCO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_450.plot(MabsSilicates2CO2, label = '-2.0 ($CO_{2}$)', marker = ',', color = 'tab:blue', linestyle= 'dotted')
MabsSilicates01CO2 = df(Hot,columns=['','HotSilicatesCO2Mabs_-0.01P0_1.0.cfg.txt'])
axes_450.plot(MabsSilicates01CO2, label = '-0.01 ($CO_{2}$)', marker = ',', color = 'darkorange', linestyle= 'dotted')
# SO2
MabsSilicates7SO2 = df(Hot,columns=['','HotSilicatesSO2Mabs_-7.0P0_1.0.cfg.txt'])
axes_450.plot(MabsSilicates7SO2, label = '-7.0 ($SO_{2}$)', marker = ',', color = 'red', linestyle= 'dotted')
# MabsSilicates4SO2 = df(Hot,columns=['','HotSilicatesSO2Mabs_-4.0SilicatesP0_1.0.cfg.txt'])
# axes_450.plot(MabsSilicates4SO2, label = '-4.0 ($SO_{2}$)', marker = ',', color = 'rebeccapurple', linestyle= 'dotted')
MabsSilicates2SO2 = df(Hot,columns=['','HotSilicatesSO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_450.plot(MabsSilicates2SO2, label = '-2.0 ($SO_{2}$)', marker = ',', color = 'blue', linestyle= 'dotted')
MabsSilicates01SO2 = df(Hot,columns=['','HotSilicatesSO2Mabs_-0.01P0_1.0.cfg.txt'])
axes_450.plot(MabsSilicates01SO2, label = '-0.01 ($SO_{2}$)', marker = ',', color = 'orange', linestyle= 'dotted')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
# axes_450.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_450.text(0.25,0.93,'log($M_{abs}$)', ha='left', va='top',fontweight = 'roman',clip_box='bbox')
plt.grid(True)
axes_450.set_ylim(0,1)
axes_450.set_xlim(0.2,2.8)
axes_450.title.set_text('Glass: Hot Regime')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
# boop
axes_560 = plt.subplot(G[5:6,0]) 
# P_0 second
# CO2
P0Silicates7CO2 = df(Hot,columns=['','HotSilicatesCO2Mabs_-2.0P0_0.01.cfg.txt'])
axes_560.plot(P0Silicates7CO2, label = '-2.0 ($CO_{2}$)', marker = ',', color = 'springgreen', linestyle= 'dotted')
P0Silicates4CO2 = df(Hot,columns=['','HotSilicatesCO2Mabs_-2.0P0_1.0cfg.txt'])
axes_560.plot(P0Silicates4CO2, label = '0.0 ($CO_{2}$)', marker = ',', color = 'rebeccapurple', linestyle= 'dotted')
P0Silicates2CO2 = df(Hot,columns=['','HotCO2Mabs_-2.0P0_100.0.cfg.txt'])
axes_560.plot(P0Silicates2CO2, label = '2.0 ($CO_{2}$)', marker = ',', color = 'tab:blue', linestyle= 'dotted')
# SO2
P0Silicates7SO2 = df(Hot,columns=['','HotSilicatesSO2Mabs_-2.0P0_0.01.cfg.txt'])
axes_560.plot(P0Silicates7SO2, label = '-2.0 ($SO_{2}$)', marker = ',', color = 'green', linestyle= 'dotted')
P0Silicates4SO2 = df(Hot,columns=['','HotSilicatesSO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_560.plot(P0Silicates4SO2, label = '0.0 ($SO_{2}$)', marker = ',', color = 'purple', linestyle= 'dotted')
P0Silicates2SO2 = df(Hot,columns=['','HotSilicatesSO2Mabs_-2.0P0_100.0.cfg.txt'])
axes_560.plot(P0Silicates2SO2, label = '2.0 ($SO_{2}$)', marker = ',', color = 'blue', linestyle= 'dotted')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
# axes_560.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_560.text(0.25,0.93,'log($P_{0}$)', ha='left', va='top',fontweight = 'roman',clip_box='bbox')
plt.grid(True)
axes_560.set_ylim(0,1)
axes_560.set_xlim(0.2,2.8)

# Basalt Block - Middle Top - M_abs
axes_011 = plt.subplot(G[0:2,1]) # Basalt Block - Middle Top - M_abs
# M_abs first
# CO2 - TEMP
# MabsBasaltT7CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-7.0P0_1.0.cfg.txt'])
# axes_011.plot(MabsBasaltT7CO2, label = '-7.0 ($CO_{2}$)', marker = ',', color = 'tab:red', linestyle= 'dotted')
# MabsBasaltT4CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-4.0P0_1.0.cfg.txt'])
# axes_011.plot(MabsBasaltT4CO2, label = '-4.0 ($CO_{2}$)', marker = ',', color = 'rebeccapurple', linestyle= 'dotted')
MabsBasaltT2CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_011.plot(MabsBasaltT2CO2, label = '-2.0 ($CO_{2}$) Temp', marker = ',', color = 'tab:red', linestyle= 'dotted')
# MabsBasaltT01CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-0.01P0_1.0.cfg.txt'])
# axes_011.plot(MabsBasaltT01CO2, label = '-0.01 ($CO_{2}$)', marker = ',', color = 'darkorange', linestyle= 'dotted')
# H2O - TEMP
# MabsBasaltT7H2O = df(Temp,columns=['','TempBasaltH2OMabs_-7.0P0_1.0.cfg.txt'])
# axes_011.plot(MabsBasaltT7H2O, label = '-7.0 ($H_{2}O$)', marker = ',', color = 'tab:red', linestyle= 'dotted')
# MabsBasaltT4H2O = df(Temp,columns=['','TempBasaltH2OMabs_-4.0P0_1.0.cfg.txt'])
# axes_011.plot(MabsBasaltT4H2O, label = '-4.0 ($H_{2}O$)', marker = ',', color = 'rebeccapurple', linestyle= 'dotted')
MabsBasaltT2H2O = df(Temp,columns=['','TempBasaltH2OMabs_-2.0P0_1.0.cfg.txt'])
axes_011.plot(MabsBasaltT2H2O, label = '-2.0 ($H_{2}O$) Temp', marker = ',', color = 'rebeccapurple', linestyle= 'dotted')
# MabsBasaltT01H2O = df(Temp,columns=['','TempBasaltH2OMabs_-0.01P0_1.0.cfg.txt'])
# axes_011.plot(MabsBasaltT01H2O, label = '-0.01 ($H_{2}O$)', marker = ',', color = 'darkorange', linestyle= 'dotted')
# CO2 - Cold
# MabsBasaltC7CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-7.0P0_1.0.cfg.txt'])
# axes_011.plot(MabsBasaltC7CO2, label = '-7.0 ($CO_{2}$)', marker = ',', color = 'tab:red', linestyle= 'dotted')
# MabsBasaltC4CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-4.0P0_1.0.cfg.txt'])
# axes_011.plot(MabsBasaltC4CO2, label = '-4.0 ($CO_{2}$)', marker = ',', color = 'rebeccapurple', linestyle= 'dotted')
MabsBasaltC2CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_011.plot(MabsBasaltC2CO2, label = '-2.0 ($CO_{2}$) Cold', marker = ',', color = 'tab:blue', linestyle= 'dotted')
# MabsBasaltC01CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-0.01P0_1.0.cfg.txt'])
# axes_011.plot(MabsBasaltC01CO2, label = '-0.01 ($CO_{2}$)', marker = ',', color = 'darkorange', linestyle= 'dotted')
# CH4 - Cold
# MabsBasaltC7CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-7.0P0_1.0.cfg.txt'])
# axes_011.plot(MabsBasaltC7CH4, label = '-7.0 ($CH_{4}$)', marker = ',', color = 'tab:red', linestyle= 'dotted')
# MabsBasaltC4CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-4.0P0_1.0.cfg.txt'])
# axes_011.plot(MabsBasaltC4CH4, label = '-4.0 ($CH_{4}$)', marker = ',', color = 'rebeccapurple', linestyle= 'dotted')
MabsBasaltC2CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-2.0P0_1.0.cfg.txt'])
axes_011.plot(MabsBasaltC2CH4, label = '-2.0 ($CH_{4}$) Cold', marker = ',', color = 'darkorange', linestyle= 'dotted')
# MabsBasaltC01CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-0.01P0_1.0.cfg.txt'])
# axes_011.plot(MabsBasaltC01CH4, label = '-0.01 ($CH_{4}$)', marker = ',', color = 'darkorange', linestyle= 'dotted')
# CO2 - Hot
# MabsBasaltH7CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-7.0P0_1.0.cfg.txt'])
# axes_011.plot(MabsBasaltH7CO2, label = '-7.0 ($CO_{2}$)', marker = ',', color = 'tab:red', linestyle= 'dotted')
# MabsBasaltH4CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-4.0P0_1.0.cfg.txt'])
# axes_011.plot(MabsBasaltH4CO2, label = '-4.0 ($CO_{2}$)', marker = ',', color = 'rebeccapurple', linestyle= 'dotted')
MabsBasaltH2CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_011.plot(MabsBasaltH2CO2, label = '-2.0 ($CO_{2}$) Hot', marker = ',', color = 'springgreen', linestyle= 'dotted')
# MabsBasaltH01CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-0.01P0_1.0.cfg.txt'])
# axes_011.plot(MabsBasaltH01CO2, label = '-0.01 ($CO_{2}$)', marker = ',', color = 'darkorange', linestyle= 'dotted')
# SO2 - Hot
# MabsBasaltH7SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-7.0P0_1.0.cfg.txt'])
# axes_011.plot(MabsBasaltH7SO2, label = '-7.0 ($SO_{2}$)', marker = ',', color = 'tab:red', linestyle= 'dotted')
# MabsBasaltH4SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-4.0P0_1.0.cfg.txt'])
# axes_011.plot(MabsBasaltH4SO2, label = '-4.0 ($SO_{2}$)', marker = ',', color = 'rebeccapurple', linestyle= 'dotted')
MabsBasaltH2SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_011.plot(MabsBasaltH2SO2, label = '-2.0 ($SO_{2}$) Hot', marker = ',', color = 'olive', linestyle= 'dotted')
# MabsBasaltH01SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-0.01P0_1.0.cfg.txt'])
# axes_011.plot(MabsBasaltH01SO2, label = '-0.01 ($SO_{2}$)', marker = ',', color = 'darkorange', linestyle= 'dotted')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
# axes_011.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_011.text(0.25,0.93,'log($M_{abs}$)', ha='left', va='top',fontweight = 'roman',clip_box='bbox')
plt.grid(True)
axes_011.set_ylim(0,1)
axes_011.set_xlim(0.2,2.8)
axes_011.title.set_text('Basalt in All Regimes')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
# boop
# Basalt Block - Middle Top - P_0
axes_121 = plt.subplot(G[2:3,1]) # Basalt Block - Middle Top - P_0
# P_0 second
# P0BasaltT7CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-2.0P0_0.01.cfg.txt'])
# axes_121.plot(P0BasaltT7CO2, label = '-2.0 ($CO_{2}$)', marker = ',', color = 'springgreen', linestyle= 'dotted')
P0BasaltT4CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_121.plot(P0BasaltT4CO2, label = '0.0 ($CO_{2}$) Temp', marker = ',', color = 'tab:red', linestyle= 'dotted')
# P0BasaltT2CO2 = df(Temp,columns=['','TempBasaltCO2Mabs-2.0P0_100.0.cfg.txt'])
# axes_121.plot(P0BasaltT2CO2, label = '2.0 ($CO_{2}$)', marker = ',', color = 'tab:blue', linestyle= 'dotted')

# H2O - TEMP
# P0BasaltT7H2O = df(Temp,columns=['','TempBasaltH2OMabs_-2.0P0_0.01.cfg.txt'])
# axes_121.plot(P0BasaltT7H2O, label = '-2.0 ($H_{2}O$)', marker = ',', color = 'springgreen', linestyle= 'dotted')
P0BasaltT4H2O = df(Temp,columns=['','TempBasaltH2OMabs_-2.0P0_1.0.cfg.txt'])
axes_121.plot(P0BasaltT4H2O, label = '0.0 ($H_{2}O$) Temp', marker = ',', color = 'rebeccapurple', linestyle= 'dotted')
# P0BasaltT2H2O = df(Temp,columns=['','TempBasaltH2OMabs_-2.0P0_100.0.cfg.txt'])
# axes_121.plot(P0BasaltT2H2O, label = '2.0 ($H_{2}O$)', marker = ',', color = 'tab:blue', linestyle= 'dotted')

# CO2 - Cold
# P0BasaltC7CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-2.0P0_0.01.cfg.txt'])
# axes_121.plot(P0BasaltC7CO2, label = '-2.0 ($CO_{2}$)', marker = ',', color = 'springgreen', linestyle= 'dotted')
P0BasaltC4CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_121.plot(P0BasaltC4CO2, label = '0.0 ($CO_{2}$) Cold', marker = ',', color = 'tab:blue', linestyle= 'dotted')
# P0BasaltC2CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-2.0P0_100.0.cfg.txt'])
# axes_121.plot(P0BasaltC2CO2, label = '2.0 ($CO_{2}$)', marker = ',', color = 'tab:blue', linestyle= 'dotted')

# CH4 - Cold
# P0BasaltC7CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-2.0P0_0.01.cfg.txt'])
# axes_121.plot(P0BasaltC7CH4, label = '-2.0 ($CH_{4}$)', marker = ',', color = 'springgreen', linestyle= 'dotted')
P0BasaltC4CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-2.0P0_1.0.cfg.txt'])
axes_121.plot(P0BasaltC4CH4, label = '0.0 ($CH_{4}$) Cold', marker = ',', color = 'darkorange', linestyle= 'dotted')
# P0BasaltC2CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-2.0P0_100.0.cfg.txt'])
# axes_121.plot(P0BasaltC2CH4, label = '2.0 ($CH_{4}$)', marker = ',', color = 'tab:blue', linestyle= 'dotted')

# CO2 - Hot
# P0BasaltH7CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-2.0P0_0.01.cfg.txt'])
# axes_121.plot(P0BasaltH7CO2, label = '-2.0 ($CO_{2}$)', marker = ',', color = 'springgreen', linestyle= 'dotted')
P0BasaltH4CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_121.plot(P0BasaltH4CO2, label = '0.0 ($CO_{2}$) Hot', marker = ',', color = 'springgreen', linestyle= 'dotted')
# P0BasaltH2CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-2.0P0_100.0.cfg.txt'])
# axes_121.plot(P0BasaltH2CO2, label = '2.0 ($CO_{2}$)', marker = ',', color = 'tab:blue', linestyle= 'dotted')

# SO2 - Hot
# P0BasaltH7SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-2.0P0_0.01.cfg.txt'])
# axes_121.plot(P0BasaltH7SO2, label = '-2.0 ($SO_{2}$)', marker = ',', color = 'springgreen', linestyle= 'dotted')
P0BasaltH4SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_121.plot(P0BasaltH4SO2, label = '0.0 ($SO_{2}$) Hot', marker = ',', color = 'olive', linestyle= 'dotted')
# P0BasaltH2SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-2.0P0_100.0.cfg.txt'])
# axes_121.plot(P0BasaltH2SO2, label = '2.0 ($SO_{2}$)', marker = ',', color = 'tab:blue', linestyle= 'dotted')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
# axes_121.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_121.text(0.25,0.93,'log($P_{0}$)', ha='left', va='top',fontweight = 'roman',clip_box='bbox')
plt.grid(True)
axes_121.set_ylim(0,1)
axes_121.set_xlim(0.2,2.8)
# Purposely leave space between [2:4,1]. Testing this to see if the plot looks fine.

# Sand Block - Middle Botto
axes_451 = plt.subplot(G[3:4,1]) # Sand Block - Middle Bottom
# M_abs first
# CO2 - TEMP
# MabsSandT7CO2 = df(Temp,columns=['','TempSandCO2Mabs_-7.0P0_1.0.cfg.txt'])
# axes_451.plot(MabsSandT7CO2, label = '-7.0 ($CO_{2}$)', marker = ',', color = 'tab:red', linestyle= 'dotted')
# MabsSandT4CO2 = df(Temp,columns=['','TempSandCO2Mabs_-4.0P0_1.0.cfg.txt'])
# axes_451.plot(MabsSandT4CO2, label = '-4.0 ($CO_{2}$)', marker = ',', color = 'rebeccapurple', linestyle= 'dotted')
MabsSandT2CO2 = df(Temp,columns=['','TempSandCO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_451.plot(MabsSandT2CO2, label = '-2.0 ($CO_{2}$) Temmp', marker = ',', color = 'tab:red', linestyle= 'dotted')
# MabsSandT01CO2 = df(Temp,columns=['','TempSandCO2Mabs_-0.01P0_1.0.cfg.txt'])
# axes_451.plot(MabsSandT01CO2, label = '-0.01 ($CO_{2}$)', marker = ',', color = 'darkorange', linestyle= 'dotted')
# H2O - TEMP
# MabsSandT7H2O = df(Temp,columns=['','TempSandH2OMabs_-7.0P0_1.0.cfg.txt'])
# axes_451.plot(MabsSandT7H2O, label = '-7.0 ($H_{2}O$)', marker = ',', color = 'tab:red', linestyle= 'dotted')
# MabsSandT4H2O = df(Temp,columns=['','TempSandH2OMabs_-4.0P0_1.0.cfg.txt'])
# axes_451.plot(MabsSandT4H2O, label = '-4.0 ($H_{2}O$)', marker = ',', color = 'rebeccapurple', linestyle= 'dotted')
MabsSandT2H2O = df(Temp,columns=['','TempSandH2OMabs_-2.0P0_1.0.cfg.txt'])
axes_451.plot(MabsSandT2H2O, label = '-2.0 ($H_{2}O$) Temp', marker = ',', color = 'rebeccapurple', linestyle= 'dotted')
# MabsSandT01H2O = df(Temp,columns=['','TempSandH2OMabs_-0.01P0_1.0.cfg.txt'])
# axes_451.plot(MabsSandT01H2O, label = '-0.01 ($H_{2}O$)', marker = ',', color = 'darkorange', linestyle= 'dotted')
# CO2 - Cold
# MabsSandC7CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-7.0P0_1.0.cfg.txt'])
# axes_451.plot(MabsSandC7CO2, label = '-7.0 ($CO_{2}$)', marker = ',', color = 'tab:red', linestyle= 'dotted')
# MabsSandC4CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-4.0P0_1.0.cfg.txt'])
# axes_451.plot(MabsSandC4CO2, label = '-4.0 ($CO_{2}$)', marker = ',', color = 'rebeccapurple', linestyle= 'dotted')
MabsSandC2CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_451.plot(MabsSandC2CO2, label = '-2.0 ($CO_{2}$) Cold', marker = ',', color = 'tab:blue', linestyle= 'dotted')
# MabsSandC01CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-0.01P0_1.0.cfg.txt'])
# axes_451.plot(MabsSandC01CO2, label = '-0.01 ($CO_{2}$)', marker = ',', color = 'darkorange', linestyle= 'dotted')
# CH4 - Cold
# MabsSandC7CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-7.0P0_1.0.cfg.txt'])
# axes_451.plot(MabsSandC7CH4, label = '-7.0 ($CH_{4}$)', marker = ',', color = 'tab:red', linestyle= 'dotted')
# MabsSandC4CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-4.0P0_1.0.cfg.txt'])
# axes_451.plot(MabsSandC4CH4, label = '-4.0 ($CH_{4}$)', marker = ',', color = 'rebeccapurple', linestyle= 'dotted')
MabsSandC2CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-2.0P0_1.0.cfg.txt'])
axes_451.plot(MabsSandC2CH4, label = '-2.0 ($CH_{4}$) Cold', marker = ',', color = 'darkorange', linestyle= 'dotted')
# MabsSandC01CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-0.01P0_1.0.cfg.txt'])
# axes_451.plot(MabsSandC01CH4, label = '-0.01 ($CH_{4}$)', marker = ',', color = 'darkorange', linestyle= 'dotted')
# CO2 - Hot
# MabsSandH7CO2 = df(Hot,columns=['','HotSandCO2Mabs_-7.0P0_1.0.cfg.txt'])
# axes_451.plot(MabsSandH7CO2, label = '-7.0 ($CO_{2}$)', marker = ',', color = 'tab:red', linestyle= 'dotted')
# MabsSandH4CO2 = df(Hot,columns=['','HotSandCO2Mabs_-4.0P0_1.0.cfg.txt'])
# axes_451.plot(MabsSandH4CO2, label = '-4.0 ($CO_{2}$)', marker = ',', color = 'rebeccapurple', linestyle= 'dotted')
MabsSandH2CO2 = df(Hot,columns=['','HotSandCO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_451.plot(MabsSandH2CO2, label = '-2.0 ($CO_{2}$) Hot', marker = ',', color = 'springgreen', linestyle= 'dotted')
# MabsSandH01CO2 = df(Hot,columns=['','HotSandCO2Mabs_-0.01P0_1.0.cfg.txt'])
# axes_451.plot(MabsSandH01CO2, label = '-0.01 ($CO_{2}$)', marker = ',', color = 'darkorange', linestyle= 'dotted')
# # SO2 - Hot
# MabsSandH7SO2 = df(Hot,columns=['','HotSandSO2Mabs_-7.0P0_1.0.cfg.txt'])
# axes_451.plot(MabsSandH7SO2, label = '-7.0 ($SO_{2}$)', marker = ',', color = 'tab:red', linestyle= 'dotted')
# MabsSandH4SO2 = df(Hot,columns=['','HotSandSO2Mabs_-4.0P0_1.0.cfg.txt'])
# axes_451.plot(MabsSandH4SO2, label = '-4.0 ($SO_{2}$)', marker = ',', color = 'rebeccapurple', linestyle= 'dotted')
MabsSandH2SO2 = df(Hot,columns=['','HotSandSO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_451.plot(MabsSandH2SO2, label = '-2.0 ($SO_{2}$ Hot)', marker = ',', color = 'olive', linestyle= 'dotted')
# MabsSandH01SO2 = df(Hot,columns=['','HotSandSO2Mabs_-0.01P0_1.0.cfg.txt'])
# axes_451.plot(MabsSandH01SO2, label = '-0.01 ($SO_{2}$)', marker = ',', color = 'darkorange', linestyle= 'dotted')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
# axes_451.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_451.text(0.25,0.93,'log($M_{abs}$)', ha='left', va='top',fontweight = 'roman',clip_box='bbox')
plt.grid(True)
axes_451.set_ylim(0,1)
axes_451.set_xlim(0.2,2.8)
axes_451.title.set_text('Sand in All Regimes')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
# boop
# Sand Block - Middle Bottom
axes_561 = plt.subplot(G[4:6,1]) # Sand Block - Middle Bottom
# P_0 second
# P0SandT7CO2 = df(Temp,columns=['','TempSandCO2Mabs_-2.0P0_0.01.cfg.txt'])
# axes_561.plot(P0SandT7CO2, label = '-2.0 ($CO_{2}$)', marker = ',', color = 'springgreen', linestyle= 'dotted')
P0SandT4CO2 = df(Temp,columns=['','TempSandCO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_561.plot(P0SandT4CO2, label = '0.0 ($CO_{2}$) Temp', marker = ',', color = 'tab:red', linestyle= 'dotted')
# P0SandT2CO2 = df(Temp,columns=['','TempSandCO2Mabs_-2.0P0_100.0.cfg.txt'])
# axes_561.plot(P0SandT2CO2, label = '2.0 ($CO_{2}$)', marker = ',', color = 'tab:blue', linestyle= 'dotted')

# H2O - TEMP

P0SandT4H2O = df(Temp,columns=['','TempSandH2OMabs_-2.0P0_1.0.cfg.txt'])
axes_561.plot(P0SandT4H2O, label = '0.0 ($H_{2}O$) Temp', marker = ',', color = 'rebeccapurple', linestyle= 'dotted')
# P0SandT2H2O = df(Temp,columns=['','TempSandH2OMabs_-2.0P0_100.0.cfg.txt'])
# axes_561.plot(P0SandT2H2O, label = '2.0 ($H_{2}O$)', marker = ',', color = 'tab:blue', linestyle= 'dotted')

# CO2 - Cold
# P0SandC7CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-2.0P0_0.01.cfg.txt'])
# axes_561.plot(P0SandC7CO2, label = '-2.0 ($CO_{2}$)', marker = ',', color = 'springgreen', linestyle= 'dotted')
P0SandC4CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_561.plot(P0SandC4CO2, label = '0.0 ($CO_{2}$) Cold', marker = ',', color = 'tab:blue', linestyle= 'dotted')
# P0SandC2CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-2.0P0_100.0.cfg.txt'])
# axes_561.plot(P0SandC2CO2, label = '2.0 ($CO_{2}$)', marker = ',', color = 'tab:blue', linestyle= 'dotted')

# CH4 - Cold
# P0SandC7CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-2.0P0_0.01.cfg.txt'])
# axes_561.plot(P0SandC7CH4, label = '-2.0 ($CH_{4}$)', marker = ',', color = 'springgreen', linestyle= 'dotted')
P0SandC4CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-2.0P0_1.0.cfg.txt'])
axes_561.plot(P0SandC4CH4, label = '0.0 ($CH_{4}$) Cold', marker = ',', color = 'darkorange', linestyle= 'dotted')
# P0SandC2CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-2.0P0_100.0.cfg.txt'])
# axes_561.plot(P0SandC2CH4, label = '2.0 ($CH_{4}$)', marker = ',', color = 'tab:blue', linestyle= 'dotted')

# CO2 - Hot
# P0SandC7CO2 = df(Hot,columns=['','HotSandCO2Mabs_-2.0P0_0.01.cfg.txt'])
# axes_561.plot(P0SandC7CO2, label = '-2.0 ($CO_{2}$)', marker = ',', color = 'springgreen', linestyle= 'dotted')
P0SandC4CO2 = df(Hot,columns=['','HotSandCO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_561.plot(P0SandC4CO2, label = '0.0 ($CO_{2}$) Hot', marker = ',', color = 'springgreen', linestyle= 'dotted')
# P0SandC2CO2 = df(Hot,columns=['','HotSandCO2Mabs_-2.0P0_100.0.cfg.txt'])
# axes_561.plot(P0SandC2CO2, label = '2.0 ($CO_{2}$)', marker = ',', color = 'tab:blue', linestyle= 'dotted')

# SO2 - Hot
# P0SandC7SO2 = df(Hot,columns=['','HotSandSO2Mabs_-2.0P0_0.01.cfg.txt'])
# axes_561.plot(P0SandC7SO2, label = '-2.0 ($SO_{2}$)', marker = ',', color = 'springgreen', linestyle= 'dotted')
P0SandC4SO2 = df(Hot,columns=['','HotSandSO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_561.plot(P0SandC4SO2, label = '0.0 ($SO_{2}$) Hot', marker = ',', color = 'olive', linestyle= 'dotted')
# P0SandC2SO2 = df(Hot,columns=['','HotSandSO2Mabs_-2.0P0_100.0.cfg.txt'])
# axes_561.plot(P0SandC2SO2, label = '2.0 ($SO_{2}$)', marker = ',', color = 'tab:blue', linestyle= 'dotted')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
# axes_561.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_561.text(0.25,0.93,'log($P_{0}$)', ha='left', va='top',fontweight = 'roman',clip_box='bbox')
plt.grid(True)
axes_561.set_ylim(0,1)
axes_561.set_xlim(0.2,2.8)
# Frost Block - Right 
# boop
axes_032 = plt.subplot(G[0:3,2]) # Frost Block - Right 
# M_abs
# CO2 - Temp
MabsWaterIceT7CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-7.0P0_1.0.cfg.txt'])
axes_032.plot(MabsWaterIceT7CO2, label = '-7.0 ($CO_{2}$)', marker = ',', color = 'tab:red', linestyle= 'dotted')
MabsWaterIceT4CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-4.0P0_1.0.cfg.txt'])
axes_032.plot(MabsWaterIceT4CO2, label = '-4.0 ($CO_{2}$)', marker = ',', color = 'rebeccapurple', linestyle= 'dotted')
MabsWaterIceT2CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_032.plot(MabsWaterIceT2CO2, label = '-2.0 ($CO_{2}$)', marker = ',', color = 'tab:blue', linestyle= 'dotted')
MabsWaterIceT01CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-0.01P0_1.0.cfg.txt'])
axes_032.plot(MabsWaterIceT01CO2, label = '-0.01 ($CO_{2}$)', marker = ',', color = 'darkorange', linestyle= 'dotted')
# H2O - TEMP
MabsWaterIceT7H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-7.0P0_1.0.cfg.txt'])
axes_032.plot(MabsWaterIceT7H2O, label = '-7.0 ($H_{2}O$)', marker = ',', color = 'red', linestyle= 'dotted')
MabsWaterIceT4H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-4.0P0_1.0.cfg.txt'])
axes_032.plot(MabsWaterIceT4H2O, label = '-4.0 ($H_{2}O$)', marker = ',', color = 'purple', linestyle= 'dotted')
MabsWaterIceT2H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-2.0P0_1.0.cfg.txt'])
axes_032.plot(MabsWaterIceT2H2O, label = '-2.0 ($H_{2}O$)', marker = ',', color = 'blue', linestyle= 'dotted')
MabsWaterIceT01H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-0.01P0_1.0.cfg.txt'])
axes_032.plot(MabsWaterIceT01H2O, label = '-0.01 ($H_{2}O$)', marker = ',', color = 'orange', linestyle= 'dotted')
# CO2 - Cold
MabsWaterIceC7CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-7.0P0_1.0.cfg.txt'])
axes_032.plot(MabsWaterIceC7CO2, label = '-7.0 ($CO_{2}$)', marker = ',', color = 'r', linestyle= 'dotted')
MabsWaterIceC4CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-4.0P0_1.0.cfg.txt'])
axes_032.plot(MabsWaterIceC4CO2, label = '-4.0 ($CO_{2}$)', marker = ',', color = 'black', linestyle= 'dotted')
MabsWaterIceC2CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_032.plot(MabsWaterIceC2CO2, label = '-2.0 ($CO_{2}$)', marker = ',', color = 'b', linestyle= 'dotted')
MabsWaterIceC01CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-0.01P0_1.0.cfg.txt'])
axes_032.plot(MabsWaterIceC01CO2, label = '-0.01 ($CO_{2}$)', marker = ',', color = 'pink', linestyle= 'dotted')
# CH4 - Cold
MabsWaterIceC7CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-7.0P0_1.0.cfg.txt'])
axes_032.plot(MabsWaterIceC7CH4, label = '-7.0 ($CH_{4}$)', marker = ',', color = 'tab:cyan', linestyle= 'dotted')
MabsWaterIceC4CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-4.0P0_1.0.cfg.txt'])
axes_032.plot(MabsWaterIceC4CH4, label = '-4.0 ($CH_{4}$)', marker = ',', color = 'cyan', linestyle= 'dotted')
MabsWaterIceC2CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-2.0P0_1.0.cfg.txt'])
axes_032.plot(MabsWaterIceC2CH4, label = '-2.0 ($CH_{4}$)', marker = ',', color = 'yellow', linestyle= 'dotted')
MabsWaterIceC01CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-0.01P0_1.0.cfg.txt'])
axes_032.plot(MabsWaterIceC01CH4, label = '-0.01 ($CH_{4}$)', marker = ',', color = 'olive', linestyle= 'dotted')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
# axes_032.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_032.text(0.25,0.93,'log($M_{abs}$)', ha='left', va='top',fontweight = 'roman',clip_box='bbox')
plt.grid(True)
axes_032.set_ylim(0,1)
axes_032.set_xlim(0.2,2.8)
axes_032.title.set_text('Frost in Cold & Temperate Regimes')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
# boop

axes_362 = plt.subplot(G[3:6,2]) # Frost Block - Right 
# P_0 second
P0WaterIceT7CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-2.0P0_0.01.cfg.txt'])
axes_362.plot(P0WaterIceT7CO2, label = '-2.0 ($CO_{2}$)', marker = ',', color = 'springgreen', linestyle= 'dotted')
P0WaterIceT4CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_362.plot(P0WaterIceT4CO2, label = '0.0 ($CO_{2}$)', marker = ',', color = 'rebeccapurple', linestyle= 'dotted')
P0WaterIceT2CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-2.0P0_100.0.cfg.txt'])
axes_362.plot(P0WaterIceT2CO2, label = '2.0 ($CO_{2}$)', marker = ',', color = 'tab:blue', linestyle= 'dotted')

# H2O - TEMP
P0WaterIceT7H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-2.0P0_0.01.cfg.txt'])
axes_362.plot(P0WaterIceT7H2O, label = '-2.0 ($H_{2}O$)', marker = ',', color = 'green', linestyle= 'dotted')
P0WaterIceT4H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-2.0P0_1.0.cfg.txt'])
axes_362.plot(P0WaterIceT4H2O, label = '0.0 ($H_{2}O$)', marker = ',', color = 'purple', linestyle= 'dotted')
P0WaterIceT2H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-2.0P0_100.0.cfg.txt'])
axes_362.plot(P0WaterIceT2H2O, label = '2.0 ($H_{2}O$)', marker = ',', color = 'blue', linestyle= 'dotted')

# CO2 - Cold
P0WaterIceC7CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-2.0P0_0.01.cfg.txt'])
axes_362.plot(P0WaterIceC7CO2, label = '-2.0 ($CO_{2}$)', marker = ',', color = 'black', linestyle= 'dotted')
P0WaterIceC4CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_362.plot(P0WaterIceC4CO2, label = '0.0 ($CO_{2}$)', marker = ',', color = 'cyan', linestyle= 'dotted')
P0WaterIceC2CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-2.0P0_100.0.cfg.txt'])
axes_362.plot(P0WaterIceC2CO2, label = '2.0 ($CO_{2}$)', marker = ',', color = 'darkred', linestyle= 'dotted')

# CH4 - Cold
P0WaterIceC7CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-2.0P0_0.01.cfg.txt'])
axes_362.plot(P0WaterIceC7CH4, label = '-2.0 ($CH_{4}$)', marker = ',', color = 'pink', linestyle= 'dotted')
P0WaterIceC4CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-2.0P0_1.0.cfg.txt'])
axes_362.plot(P0WaterIceC4CH4, label = '0.0 ($CH_{4}$)', marker = ',', color = 'b', linestyle= 'dotted')
P0WaterIceC2CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-2.0P0_100.0.cfg.txt'])
axes_362.plot(P0WaterIceC2CH4, label = '2.0 ($CH_{4}$)', marker = ',', color = 'r', linestyle= 'dotted')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
# axes_032.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_362.text(0.25,0.93,'log($P_{0}$)', ha='left', va='top',fontweight = 'roman',clip_box='bbox')
plt.grid(True)
axes_362.set_ylim(0,1)
axes_362.set_xlim(0.2,2.8)
# boop

G.tight_layout(fig)
print('boop')
plt.show()


# ### ok now lets break down the plot above.

# #### Starting with Frost aka WaterIce

# ##### This is m_abs for frost under every surface pressure

# ###### starting with -3.0 = 0.001 bars

# In[185]:


fig = plt.figure(figsize=(12, 8))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0.0 , 'Wavelength (um)', ha='center', va='bottom', rotation='horizontal')
G = gridspec.GridSpec(2, 2)
# this is quadrant 1, [CO2 Temp]
axes_00 = plt.subplot(G[0,0]) # Frost Block - Right 
# M_abs
# CO2 - Temp 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_00.plot(TempWaterIce_Pure[:,0],TempWaterIce_Pure[:,1], label = 'Frost Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsWaterIceT7CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-7.0P0_0.001.cfg.txt'])
axes_00.plot(MabsWaterIceT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsWaterIceT6CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-6.0P0_0.001.cfg.txt'])
axes_00.plot(MabsWaterIceT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsWaterIceT5CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-5.0P0_0.001.cfg.txt'])
axes_00.plot(MabsWaterIceT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsWaterIceT4CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-4.0P0_0.001.cfg.txt'])
axes_00.plot(MabsWaterIceT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsWaterIceT3CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-3.0P0_0.001.cfg.txt'])
axes_00.plot(MabsWaterIceT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsWaterIceT2CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-2.0P0_0.001.cfg.txt'])
axes_00.plot(MabsWaterIceT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsWaterIceT1CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-1.0P0_0.001.cfg.txt'])
axes_00.plot(MabsWaterIceT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsWaterIceT01CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-0.01P0_0.001.cfg.txt'])
axes_00.plot(MabsWaterIceT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.text(0.25,0.5,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.5)
axes_00.set_yscale("log")
axes_00.title.set_text('log($M_{abs}$) for Frost in the Temperate Regime @ ($P_{0}$) = 0.001 bars')
#
# this is quadrant 2, [H2O Temp]
axes_01 = plt.subplot(G[1,0]) # Frost Block - Right 
# M_abs
# H2O - Temp 
axes_01.plot(TempWaterIce_Pure[:,0],TempWaterIce_Pure[:,1], label = 'Frost Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsWaterIceT7H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-7.0P0_0.001.cfg.txt'])
axes_01.plot(MabsWaterIceT7H2O, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
MabsWaterIceT6H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-6.0P0_0.001.cfg.txt'])
axes_01.plot(MabsWaterIceT6H2O, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsWaterIceT5H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-5.0P0_0.001.cfg.txt'])
axes_01.plot(MabsWaterIceT5H2O, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsWaterIceT4H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-4.0P0_0.001.cfg.txt'])
axes_01.plot(MabsWaterIceT4H2O, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsWaterIceT3H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-3.0P0_0.001.cfg.txt'])
axes_01.plot(MabsWaterIceT3H2O, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsWaterIceT2H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-2.0P0_0.001.cfg.txt'])
axes_01.plot(MabsWaterIceT2H2O, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsWaterIceT1H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-1.0P0_0.001.cfg.txt'])
axes_01.plot(MabsWaterIceT1H2O, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsWaterIceT01H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-0.01P0_0.001.cfg.txt'])
axes_01.plot(MabsWaterIceT01H2O, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.text(0.25,0.5,'($H_{2}O$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.5)
axes_01.set_yscale("log")
# boop
# this is quadrant 2, formally quadrant 3 [CO2 Cold]
axes_10 = plt.subplot(G[0,1])  
# M_abs
# CO2 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_10.plot(ColdWaterIce_Pure[:,0],ColdWaterIce_Pure[:,1], label = 'Frost Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsWaterIceT7CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-7.0P0_0.001.cfg.txt'])
axes_10.plot(MabsWaterIceT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsWaterIceT6CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-6.0P0_0.001.cfg.txt'])
axes_10.plot(MabsWaterIceT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsWaterIceT5CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-5.0P0_0.001.cfg.txt'])
axes_10.plot(MabsWaterIceT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsWaterIceT4CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-4.0P0_0.001.cfg.txt'])
axes_10.plot(MabsWaterIceT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsWaterIceT3CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-3.0P0_0.001.cfg.txt'])
axes_10.plot(MabsWaterIceT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsWaterIceT2CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-2.0P0_0.001.cfg.txt'])
axes_10.plot(MabsWaterIceT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsWaterIceT1CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-1.0P0_0.001.cfg.txt'])
axes_10.plot(MabsWaterIceT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsWaterIceT01CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-0.01P0_0.001.cfg.txt'])
axes_10.plot(MabsWaterIceT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_10.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_10.text(0.25,0.5,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_10.set_ylim(0,1)
axes_10.set_xlim(0.2,2.5)
axes_10.set_yscale("log")
axes_10.title.set_text('log($M_{abs}$) for Frost in the Cold Regime @ ($P_{0}$) = 0.001 bars')


#
# this is quadrant 4, [CH4 Cold]
axes_11 = plt.subplot(G[1,1]) # Frost Block - Right 
# M_abs
# CH4 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_11.plot(ColdWaterIce_Pure[:,0],ColdWaterIce_Pure[:,1], label = 'Frost Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsWaterIceT7CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-7.0P0_0.001.cfg.txt'])
axes_11.plot(MabsWaterIceT7CH4, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsWaterIceT6CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-6.0P0_0.001.cfg.txt'])
axes_11.plot(MabsWaterIceT6CH4, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsWaterIceT5CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-5.0P0_0.001.cfg.txt'])
axes_11.plot(MabsWaterIceT5CH4, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsWaterIceT4CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-4.0P0_0.001.cfg.txt'])
axes_11.plot(MabsWaterIceT4CH4, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsWaterIceT3CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-3.0P0_0.001.cfg.txt'])
axes_11.plot(MabsWaterIceT3CH4, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsWaterIceT2CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-2.0P0_0.001.cfg.txt'])
axes_11.plot(MabsWaterIceT2CH4, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsWaterIceT1CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-1.0P0_0.001.cfg.txt'])
axes_11.plot(MabsWaterIceT1CH4, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsWaterIceT01CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-0.01P0_0.001.cfg.txt'])
axes_11.plot(MabsWaterIceT01CH4, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_11.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_11.text(0.25,0.5,'($CH_{4}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_11.set_ylim(0,1)
axes_11.set_xlim(0.2,2.5)
axes_11.set_yscale("log")
G.tight_layout(fig)
print('boop')
plt.show()
#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs/ParamsComparisons/Frost"
if not os.path.exists(dst):
	os.makedirs(dst)
    
  

fig.savefig("CruzArceFigs/ParamsComparisons/Frost/Frost_M_abs_P0_0001")


# ###### Now -2.0 = 0.01 bars

# In[184]:


fig = plt.figure(figsize=(12, 8))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0.0 , 'Wavelength (um)', ha='center', va='bottom', rotation='horizontal')
G = gridspec.GridSpec(2, 2)
# this is quadrant 1, [CO2 Temp]
axes_00 = plt.subplot(G[0,0]) # Frost Block - Right 
# M_abs
# CO2 - Temp 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_00.plot(TempWaterIce_Pure[:,0],TempWaterIce_Pure[:,1], label = 'Frost Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsWaterIceT7CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-7.0P0_0.01.cfg.txt'])
axes_00.plot(MabsWaterIceT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsWaterIceT6CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-6.0P0_0.01.cfg.txt'])
axes_00.plot(MabsWaterIceT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsWaterIceT5CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-5.0P0_0.01.cfg.txt'])
axes_00.plot(MabsWaterIceT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsWaterIceT4CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-4.0P0_0.01.cfg.txt'])
axes_00.plot(MabsWaterIceT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsWaterIceT3CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-3.0P0_0.01.cfg.txt'])
axes_00.plot(MabsWaterIceT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsWaterIceT2CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-2.0P0_0.01.cfg.txt'])
axes_00.plot(MabsWaterIceT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsWaterIceT1CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-1.0P0_0.01.cfg.txt'])
axes_00.plot(MabsWaterIceT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsWaterIceT01CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-0.01P0_0.01.cfg.txt'])
axes_00.plot(MabsWaterIceT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.text(0.25,0.5,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.5)
axes_00.set_yscale("log")
axes_00.title.set_text('log($M_{abs}$) for Frost in the Temperate Regime @ ($P_{0}$) = 0.01 bars')
#
# this is quadrant 2, [H2O Temp]
axes_01 = plt.subplot(G[1,0]) # Frost Block - Right 
# M_abs
# H2O - Temp 
axes_01.plot(TempWaterIce_Pure[:,0],TempWaterIce_Pure[:,1], label = 'Frost Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsWaterIceT7H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-7.0P0_0.01.cfg.txt'])
axes_01.plot(MabsWaterIceT7H2O, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
MabsWaterIceT6H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-6.0P0_0.01.cfg.txt'])
axes_01.plot(MabsWaterIceT6H2O, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsWaterIceT5H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-5.0P0_0.01.cfg.txt'])
axes_01.plot(MabsWaterIceT5H2O, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsWaterIceT4H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-4.0P0_0.01.cfg.txt'])
axes_01.plot(MabsWaterIceT4H2O, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsWaterIceT3H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-3.0P0_0.01.cfg.txt'])
axes_01.plot(MabsWaterIceT3H2O, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsWaterIceT2H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-2.0P0_0.01.cfg.txt'])
axes_01.plot(MabsWaterIceT2H2O, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsWaterIceT1H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-1.0P0_0.01.cfg.txt'])
axes_01.plot(MabsWaterIceT1H2O, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsWaterIceT01H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-0.01P0_0.01.cfg.txt'])
axes_01.plot(MabsWaterIceT01H2O, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.text(0.25,0.1,'($H_{2}O$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.5)
axes_01.set_yscale("log")
# boop
# this is quadrant 2, formally quadrant 3 [CO2 Cold]
axes_10 = plt.subplot(G[0,1])  
# M_abs
# CO2 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_10.plot(ColdWaterIce_Pure[:,0],ColdWaterIce_Pure[:,1], label = 'Frost Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsWaterIceT7CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-7.0P0_0.01.cfg.txt'])
axes_10.plot(MabsWaterIceT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsWaterIceT6CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-6.0P0_0.01.cfg.txt'])
axes_10.plot(MabsWaterIceT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsWaterIceT5CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-5.0P0_0.01.cfg.txt'])
axes_10.plot(MabsWaterIceT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsWaterIceT4CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-4.0P0_0.01.cfg.txt'])
axes_10.plot(MabsWaterIceT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsWaterIceT3CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-3.0P0_0.01.cfg.txt'])
axes_10.plot(MabsWaterIceT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsWaterIceT2CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-2.0P0_0.01.cfg.txt'])
axes_10.plot(MabsWaterIceT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsWaterIceT1CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-1.0P0_0.01.cfg.txt'])
axes_10.plot(MabsWaterIceT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsWaterIceT01CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-0.01P0_0.01.cfg.txt'])
axes_10.plot(MabsWaterIceT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_10.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_10.text(0.25,0.5,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_10.set_ylim(0,1)
axes_10.set_xlim(0.2,2.5)
axes_10.set_yscale("log")
axes_10.title.set_text('log($M_{abs}$) for Frost in the Cold Regime @ ($P_{0}$) = 0.01 bars')


#
# this is quadrant 4, [CH4 Cold]
axes_11 = plt.subplot(G[1,1]) # Frost Block - Right 
# M_abs
# CH4 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_11.plot(ColdWaterIce_Pure[:,0],ColdWaterIce_Pure[:,1], label = 'Frost Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsWaterIceT7CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-7.0P0_0.01.cfg.txt'])
axes_11.plot(MabsWaterIceT7CH4, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsWaterIceT6CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-6.0P0_0.01.cfg.txt'])
axes_11.plot(MabsWaterIceT6CH4, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsWaterIceT5CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-5.0P0_0.01.cfg.txt'])
axes_11.plot(MabsWaterIceT5CH4, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsWaterIceT4CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-4.0P0_0.01.cfg.txt'])
axes_11.plot(MabsWaterIceT4CH4, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsWaterIceT3CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-3.0P0_0.01.cfg.txt'])
axes_11.plot(MabsWaterIceT3CH4, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsWaterIceT2CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-2.0P0_0.01.cfg.txt'])
axes_11.plot(MabsWaterIceT2CH4, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsWaterIceT1CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-1.0P0_0.01.cfg.txt'])
axes_11.plot(MabsWaterIceT1CH4, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsWaterIceT01CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-0.01P0_0.01.cfg.txt'])
axes_11.plot(MabsWaterIceT01CH4, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_11.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_11.text(0.25,0.1,'($CH_{4}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_11.set_ylim(0,1)
axes_11.set_xlim(0.2,2.5)
axes_11.set_yscale("log")
G.tight_layout(fig)
print('boop')
plt.show()
#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs/ParamsComparisons/Frost"
if not os.path.exists(dst):
	os.makedirs(dst)
    
  

fig.savefig("CruzArceFigs/ParamsComparisons/Frost/Frost_M_abs_P0_001")


# In[5]:


MabsWaterIceT01CH4 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-0.01P0_0.01.cfg.txt'])
plt.plot(MabsWaterIceT01CH4, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')


# ###### Now -1.0 = 0.1 bars

# In[183]:


fig = plt.figure(figsize=(12, 8))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0.0 , 'Wavelength (um)', ha='center', va='bottom', rotation='horizontal')
G = gridspec.GridSpec(2, 2)
# this is quadrant 1, [CO2 Temp]
axes_00 = plt.subplot(G[0,0]) # Frost Block - Right 
# M_abs
# CO2 - Temp 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_00.plot(TempWaterIce_Pure[:,0],TempWaterIce_Pure[:,1], label = 'Frost Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsWaterIceT7CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-7.0P0_0.1.cfg.txt'])
axes_00.plot(MabsWaterIceT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsWaterIceT6CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-6.0P0_0.1.cfg.txt'])
axes_00.plot(MabsWaterIceT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsWaterIceT5CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-5.0P0_0.1.cfg.txt'])
axes_00.plot(MabsWaterIceT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsWaterIceT4CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-4.0P0_0.1.cfg.txt'])
axes_00.plot(MabsWaterIceT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsWaterIceT3CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-3.0P0_0.1.cfg.txt'])
axes_00.plot(MabsWaterIceT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsWaterIceT2CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-2.0P0_0.1.cfg.txt'])
axes_00.plot(MabsWaterIceT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsWaterIceT1CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-1.0P0_0.1.cfg.txt'])
axes_00.plot(MabsWaterIceT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsWaterIceT01CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-0.01P0_0.1.cfg.txt'])
axes_00.plot(MabsWaterIceT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.text(0.25,0.5,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.5)
axes_00.set_yscale("log")
axes_00.title.set_text('log($M_{abs}$) for Frost in the Temperate Regime @ ($P_{0}$) = 0.1 bar')
#
# this is quadrant 2, [H2O Temp]
axes_01 = plt.subplot(G[1,0]) # Frost Block - Right 
# M_abs
# H2O - Temp 
axes_01.plot(TempWaterIce_Pure[:,0],TempWaterIce_Pure[:,1], label = 'Frost Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsWaterIceT7H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-7.0P0_0.1.cfg.txt'])
axes_01.plot(MabsWaterIceT7H2O, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
MabsWaterIceT6H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-6.0P0_0.1.cfg.txt'])
axes_01.plot(MabsWaterIceT6H2O, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsWaterIceT5H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-5.0P0_0.1.cfg.txt'])
axes_01.plot(MabsWaterIceT5H2O, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsWaterIceT4H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-4.0P0_0.1.cfg.txt'])
axes_01.plot(MabsWaterIceT4H2O, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsWaterIceT3H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-3.0P0_0.1.cfg.txt'])
axes_01.plot(MabsWaterIceT3H2O, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsWaterIceT2H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-2.0P0_0.1.cfg.txt'])
axes_01.plot(MabsWaterIceT2H2O, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsWaterIceT1H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-1.0P0_0.1.cfg.txt'])
axes_01.plot(MabsWaterIceT1H2O, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsWaterIceT01H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-0.01P0_0.1.cfg.txt'])
axes_01.plot(MabsWaterIceT01H2O, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.text(0.25,0.1,'($H_{2}O$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.5)
axes_01.set_yscale("log")
# boop
# this is quadrant 2, formally quadrant 3 [CO2 Cold]
axes_10 = plt.subplot(G[0,1])  
# M_abs
# CO2 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_10.plot(ColdWaterIce_Pure[:,0],ColdWaterIce_Pure[:,1], label = 'Frost Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsWaterIceT7CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-7.0P0_0.1.cfg.txt'])
axes_10.plot(MabsWaterIceT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsWaterIceT6CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-6.0P0_0.1.cfg.txt'])
axes_10.plot(MabsWaterIceT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsWaterIceT5CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-5.0P0_0.1.cfg.txt'])
axes_10.plot(MabsWaterIceT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsWaterIceT4CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-4.0P0_0.1.cfg.txt'])
axes_10.plot(MabsWaterIceT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsWaterIceT3CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-3.0P0_0.1.cfg.txt'])
axes_10.plot(MabsWaterIceT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsWaterIceT2CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-2.0P0_0.1.cfg.txt'])
axes_10.plot(MabsWaterIceT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsWaterIceT1CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-1.0P0_0.1.cfg.txt'])
axes_10.plot(MabsWaterIceT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsWaterIceT01CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-0.01P0_0.1.cfg.txt'])
axes_10.plot(MabsWaterIceT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_10.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_10.text(0.25,0.5,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_10.set_ylim(0,1)
axes_10.set_xlim(0.2,2.5)
axes_10.set_yscale("log")
axes_10.title.set_text('log($M_{abs}$) for Frost in the Cold Regime @ ($P_{0}$) = 0.1 bar')


#
# this is quadrant 4, [CH4 Cold]
axes_11 = plt.subplot(G[1,1]) # Frost Block - Right 
# M_abs
# CH4 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_11.plot(ColdWaterIce_Pure[:,0],ColdWaterIce_Pure[:,1], label = 'Frost Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsWaterIceT7CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-7.0P0_0.1.cfg.txt'])
axes_11.plot(MabsWaterIceT7CH4, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsWaterIceT6CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-6.0P0_0.1.cfg.txt'])
axes_11.plot(MabsWaterIceT6CH4, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsWaterIceT5CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-5.0P0_0.1.cfg.txt'])
axes_11.plot(MabsWaterIceT5CH4, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsWaterIceT4CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-4.0P0_0.1.cfg.txt'])
axes_11.plot(MabsWaterIceT4CH4, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsWaterIceT3CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-3.0P0_0.1.cfg.txt'])
axes_11.plot(MabsWaterIceT3CH4, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsWaterIceT2CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-2.0P0_0.1.cfg.txt'])
axes_11.plot(MabsWaterIceT2CH4, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsWaterIceT1CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-1.0P0_0.1.cfg.txt'])
axes_11.plot(MabsWaterIceT1CH4, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsWaterIceT01CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-0.01P0_0.1.cfg.txt'])
axes_11.plot(MabsWaterIceT01CH4, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_11.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_11.text(0.25,0.1,'($CH_{4}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_11.set_ylim(0,1)
axes_11.set_xlim(0.2,2.5)
axes_11.set_yscale("log")
G.tight_layout(fig)
print('boop')
plt.show()
#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs/ParamsComparisons/Frost"
if not os.path.exists(dst):
	os.makedirs(dst)
    
  

fig.savefig("CruzArceFigs/ParamsComparisons/Frost/Frost_M_abs_P0_01")


# ###### Now 0.0 = 1 bars = Earth's P0

# In[182]:


fig = plt.figure(figsize=(12, 8))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0.0 , 'Wavelength (um)', ha='center', va='bottom', rotation='horizontal')
G = gridspec.GridSpec(2, 2)
# this is quadrant 1, [CO2 Temp]
axes_00 = plt.subplot(G[0,0]) # Frost Block - Right 
# M_abs
# CO2 - Temp 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_00.plot(TempWaterIce_Pure[:,0],TempWaterIce_Pure[:,1], label = 'Frost Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsWaterIceT7CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-7.0P0_1.0.cfg.txt'])
axes_00.plot(MabsWaterIceT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsWaterIceT6CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-6.0P0_1.0.cfg.txt'])
axes_00.plot(MabsWaterIceT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsWaterIceT5CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-5.0P0_1.0.cfg.txt'])
axes_00.plot(MabsWaterIceT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsWaterIceT4CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-4.0P0_1.0.cfg.txt'])
axes_00.plot(MabsWaterIceT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsWaterIceT3CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-3.0P0_1.0.cfg.txt'])
axes_00.plot(MabsWaterIceT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsWaterIceT2CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_00.plot(MabsWaterIceT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsWaterIceT1CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-1.0P0_1.0.cfg.txt'])
axes_00.plot(MabsWaterIceT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsWaterIceT01CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-0.01P0_1.0.cfg.txt'])
axes_00.plot(MabsWaterIceT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.text(0.25,0.5,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.5)
axes_00.set_yscale("log")
axes_00.title.set_text('log($M_{abs}$) for Frost in the Temperate Regime @ ($P_{0}$) = 1 bars')
#
# this is quadrant 2, [H2O Temp]
axes_01 = plt.subplot(G[1,0]) # Frost Block - Right 
# M_abs
# H2O - Temp 
axes_01.plot(TempWaterIce_Pure[:,0],TempWaterIce_Pure[:,1], label = 'Frost Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsWaterIceT7H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-7.0P0_1.0.cfg.txt'])
axes_01.plot(MabsWaterIceT7H2O, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
MabsWaterIceT6H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-6.0P0_1.0.cfg.txt'])
axes_01.plot(MabsWaterIceT6H2O, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsWaterIceT5H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-5.0P0_1.0.cfg.txt'])
axes_01.plot(MabsWaterIceT5H2O, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsWaterIceT4H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-4.0P0_1.0.cfg.txt'])
axes_01.plot(MabsWaterIceT4H2O, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsWaterIceT3H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-3.0P0_1.0.cfg.txt'])
axes_01.plot(MabsWaterIceT3H2O, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsWaterIceT2H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-2.0P0_1.0.cfg.txt'])
axes_01.plot(MabsWaterIceT2H2O, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsWaterIceT1H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-1.0P0_1.0.cfg.txt'])
axes_01.plot(MabsWaterIceT1H2O, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsWaterIceT01H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-0.01P0_1.0.cfg.txt'])
axes_01.plot(MabsWaterIceT01H2O, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.text(0.25,0.1,'($H_{2}O$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.5)
axes_01.set_yscale("log")
# boop
# this is quadrant 2, formally quadrant 3 [CO2 Cold]
axes_10 = plt.subplot(G[0,1])  
# M_abs
# CO2 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_10.plot(ColdWaterIce_Pure[:,0],ColdWaterIce_Pure[:,1], label = 'Frost Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsWaterIceT7CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-7.0P0_1.0.cfg.txt'])
axes_10.plot(MabsWaterIceT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsWaterIceT6CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-6.0P0_1.0.cfg.txt'])
axes_10.plot(MabsWaterIceT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsWaterIceT5CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-5.0P0_1.0.cfg.txt'])
axes_10.plot(MabsWaterIceT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsWaterIceT4CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-4.0P0_1.0.cfg.txt'])
axes_10.plot(MabsWaterIceT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsWaterIceT3CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-3.0P0_1.0.cfg.txt'])
axes_10.plot(MabsWaterIceT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsWaterIceT2CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_10.plot(MabsWaterIceT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsWaterIceT1CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-1.0P0_1.0.cfg.txt'])
axes_10.plot(MabsWaterIceT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsWaterIceT01CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-0.01P0_1.0.cfg.txt'])
axes_10.plot(MabsWaterIceT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_10.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_10.text(0.25,0.5,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_10.set_ylim(0,1)
axes_10.set_xlim(0.2,2.5)
axes_10.set_yscale("log")
axes_10.title.set_text('log($M_{abs}$) for Frost in the Cold Regime @ ($P_{0}$) = 1 bars')


#
# this is quadrant 4, [CH4 Cold]
axes_11 = plt.subplot(G[1,1]) # Frost Block - Right 
# M_abs
# CH4 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_11.plot(ColdWaterIce_Pure[:,0],ColdWaterIce_Pure[:,1], label = 'Frost Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsWaterIceT7CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-7.0P0_1.0.cfg.txt'])
axes_11.plot(MabsWaterIceT7CH4, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsWaterIceT6CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-6.0P0_1.0.cfg.txt'])
axes_11.plot(MabsWaterIceT6CH4, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsWaterIceT5CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-5.0P0_1.0.cfg.txt'])
axes_11.plot(MabsWaterIceT5CH4, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsWaterIceT4CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-4.0P0_1.0.cfg.txt'])
axes_11.plot(MabsWaterIceT4CH4, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsWaterIceT3CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-3.0P0_1.0.cfg.txt'])
axes_11.plot(MabsWaterIceT3CH4, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsWaterIceT2CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-2.0P0_1.0.cfg.txt'])
axes_11.plot(MabsWaterIceT2CH4, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsWaterIceT1CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-1.0P0_1.0.cfg.txt'])
axes_11.plot(MabsWaterIceT1CH4, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsWaterIceT01CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-0.01P0_1.0.cfg.txt'])
axes_11.plot(MabsWaterIceT01CH4, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_11.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_11.text(0.25,0.1,'($CH_{4}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_11.set_ylim(0,1)
axes_11.set_xlim(0.2,2.5)
axes_11.set_yscale("log")
G.tight_layout(fig)
print('boop')
plt.show()
#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs/ParamsComparisons/Frost"
if not os.path.exists(dst):
	os.makedirs(dst)
    
  

fig.savefig("CruzArceFigs/ParamsComparisons/Frost/Frost_M_abs_P0_1")


# ###### Now 1.0 = 10 bars

# In[181]:


fig = plt.figure(figsize=(12, 8))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0.0 , 'Wavelength (um)', ha='center', va='bottom', rotation='horizontal')
G = gridspec.GridSpec(2, 2)
# this is quadrant 1, [CO2 Temp]
axes_00 = plt.subplot(G[0,0]) # Frost Block - Right 
# M_abs
# CO2 - Temp 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_00.plot(TempWaterIce_Pure[:,0],TempWaterIce_Pure[:,1], label = 'Frost Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsWaterIceT7CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-7.0P0_10.0.cfg.txt'])
axes_00.plot(MabsWaterIceT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsWaterIceT6CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-6.0P0_10.0.cfg.txt'])
axes_00.plot(MabsWaterIceT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsWaterIceT5CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-5.0P0_10.0.cfg.txt'])
axes_00.plot(MabsWaterIceT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsWaterIceT4CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-4.0P0_10.0.cfg.txt'])
axes_00.plot(MabsWaterIceT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsWaterIceT3CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-3.0P0_10.0.cfg.txt'])
axes_00.plot(MabsWaterIceT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsWaterIceT2CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-2.0P0_10.0.cfg.txt'])
axes_00.plot(MabsWaterIceT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsWaterIceT1CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-1.0P0_10.0.cfg.txt'])
axes_00.plot(MabsWaterIceT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsWaterIceT01CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-0.01P0_10.0.cfg.txt'])
axes_00.plot(MabsWaterIceT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.text(0.25,0.5,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.5)
axes_00.set_yscale("log")
axes_00.title.set_text('log($M_{abs}$) for Frost in the Temperate Regime @ ($P_{0}$) = 10 bars')
#
# this is quadrant 2, [H2O Temp]
axes_01 = plt.subplot(G[1,0]) # Frost Block - Right 
# M_abs
# H2O - Temp 
axes_01.plot(TempWaterIce_Pure[:,0],TempWaterIce_Pure[:,1], label = 'Frost Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsWaterIceT7H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-7.0P0_10.0.cfg.txt'])
axes_01.plot(MabsWaterIceT7H2O, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
MabsWaterIceT6H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-6.0P0_10.0.cfg.txt'])
axes_01.plot(MabsWaterIceT6H2O, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsWaterIceT5H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-5.0P0_10.0.cfg.txt'])
axes_01.plot(MabsWaterIceT5H2O, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsWaterIceT4H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-4.0P0_10.0.cfg.txt'])
axes_01.plot(MabsWaterIceT4H2O, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsWaterIceT3H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-3.0P0_10.0.cfg.txt'])
axes_01.plot(MabsWaterIceT3H2O, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsWaterIceT2H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-2.0P0_10.0.cfg.txt'])
axes_01.plot(MabsWaterIceT2H2O, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsWaterIceT1H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-1.0P0_10.0.cfg.txt'])
axes_01.plot(MabsWaterIceT1H2O, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsWaterIceT01H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-0.01P0_10.0.cfg.txt'])
axes_01.plot(MabsWaterIceT01H2O, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.text(0.25,0.1,'($H_{2}O$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.5)
axes_01.set_yscale("log")
# boop
# this is quadrant 2, formally quadrant 3 [CO2 Cold]
axes_10 = plt.subplot(G[0,1])  
# M_abs
# CO2 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_10.plot(ColdWaterIce_Pure[:,0],ColdWaterIce_Pure[:,1], label = 'Frost Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsWaterIceT7CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-7.0P0_10.0.cfg.txt'])
axes_10.plot(MabsWaterIceT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsWaterIceT6CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-6.0P0_10.0.cfg.txt'])
axes_10.plot(MabsWaterIceT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsWaterIceT5CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-5.0P0_10.0.cfg.txt'])
axes_10.plot(MabsWaterIceT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsWaterIceT4CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-4.0P0_10.0.cfg.txt'])
axes_10.plot(MabsWaterIceT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsWaterIceT3CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-3.0P0_10.0.cfg.txt'])
axes_10.plot(MabsWaterIceT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsWaterIceT2CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-2.0P0_10.0.cfg.txt'])
axes_10.plot(MabsWaterIceT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsWaterIceT1CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-1.0P0_10.0.cfg.txt'])
axes_10.plot(MabsWaterIceT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsWaterIceT01CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-0.01P0_10.0.cfg.txt'])
axes_10.plot(MabsWaterIceT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_10.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_10.text(0.25,0.5,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_10.set_ylim(0,1)
axes_10.set_xlim(0.2,2.5)
axes_10.set_yscale("log")
axes_10.title.set_text('log($M_{abs}$) for Frost in the Cold Regime @ ($P_{0}$) = 10 bars')


#
# this is quadrant 4, [CH4 Cold]
axes_11 = plt.subplot(G[1,1]) # Frost Block - Right 
# M_abs
# CH4 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_11.plot(ColdWaterIce_Pure[:,0],ColdWaterIce_Pure[:,1], label = 'Frost Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsWaterIceT7CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-7.0P0_10.0.cfg.txt'])
axes_11.plot(MabsWaterIceT7CH4, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsWaterIceT6CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-6.0P0_10.0.cfg.txt'])
axes_11.plot(MabsWaterIceT6CH4, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsWaterIceT5CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-5.0P0_10.0.cfg.txt'])
axes_11.plot(MabsWaterIceT5CH4, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsWaterIceT4CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-4.0P0_10.0.cfg.txt'])
axes_11.plot(MabsWaterIceT4CH4, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsWaterIceT3CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-3.0P0_10.0.cfg.txt'])
axes_11.plot(MabsWaterIceT3CH4, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsWaterIceT2CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-2.0P0_10.0.cfg.txt'])
axes_11.plot(MabsWaterIceT2CH4, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsWaterIceT1CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-1.0P0_10.0.cfg.txt'])
axes_11.plot(MabsWaterIceT1CH4, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsWaterIceT01CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-0.01P0_10.0.cfg.txt'])
axes_11.plot(MabsWaterIceT01CH4, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_11.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_11.text(0.25,0.1,'($CH_{4}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_11.set_ylim(0,1)
axes_11.set_xlim(0.2,2.5)
axes_11.set_yscale("log")
G.tight_layout(fig)
print('boop')
plt.show()
#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs/ParamsComparisons/Frost"
if not os.path.exists(dst):
	os.makedirs(dst)
    
  

fig.savefig("CruzArceFigs/ParamsComparisons/Frost/Frost_M_abs_P0_10")


# ###### Now 2.0 = 100 bars

# In[180]:


fig = plt.figure(figsize=(12, 8))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0.0 , 'Wavelength (um)', ha='center', va='bottom', rotation='horizontal')
G = gridspec.GridSpec(2, 2)
# this is quadrant 1, [CO2 Temp]
axes_00 = plt.subplot(G[0,0]) # Frost Block - Right 
# M_abs
# CO2 - Temp 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_00.plot(TempWaterIce_Pure[:,0],TempWaterIce_Pure[:,1], label = 'Frost Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsWaterIceT7CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-7.0P0_100.0.cfg.txt'])
axes_00.plot(MabsWaterIceT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsWaterIceT6CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-6.0P0_100.0.cfg.txt'])
axes_00.plot(MabsWaterIceT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsWaterIceT5CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-5.0P0_100.0.cfg.txt'])
axes_00.plot(MabsWaterIceT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsWaterIceT4CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-4.0P0_100.0.cfg.txt'])
axes_00.plot(MabsWaterIceT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsWaterIceT3CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-3.0P0_100.0.cfg.txt'])
axes_00.plot(MabsWaterIceT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsWaterIceT2CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-2.0P0_100.0.cfg.txt'])
axes_00.plot(MabsWaterIceT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsWaterIceT1CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-1.0P0_100.0.cfg.txt'])
axes_00.plot(MabsWaterIceT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsWaterIceT01CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-0.01P0_100.0.cfg.txt'])
axes_00.plot(MabsWaterIceT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.text(0.25,0.5,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.5)
axes_00.set_yscale("log")
axes_00.title.set_text('log($M_{abs}$) for Frost in the Temperate Regime @ ($P_{0}$) = 100 bars')
#
# this is quadrant 2, [H2O Temp]
axes_01 = plt.subplot(G[1,0]) # Frost Block - Right 
# M_abs
# H2O - Temp 
axes_01.plot(TempWaterIce_Pure[:,0],TempWaterIce_Pure[:,1], label = 'Frost Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsWaterIceT7H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-7.0P0_100.0.cfg.txt'])
axes_01.plot(MabsWaterIceT7H2O, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
MabsWaterIceT6H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-6.0P0_100.0.cfg.txt'])
axes_01.plot(MabsWaterIceT6H2O, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsWaterIceT5H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-5.0P0_100.0.cfg.txt'])
axes_01.plot(MabsWaterIceT5H2O, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsWaterIceT4H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-4.0P0_100.0.cfg.txt'])
axes_01.plot(MabsWaterIceT4H2O, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsWaterIceT3H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-3.0P0_100.0.cfg.txt'])
axes_01.plot(MabsWaterIceT3H2O, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsWaterIceT2H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-2.0P0_100.0.cfg.txt'])
axes_01.plot(MabsWaterIceT2H2O, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsWaterIceT1H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-1.0P0_100.0.cfg.txt'])
axes_01.plot(MabsWaterIceT1H2O, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsWaterIceT01H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-0.01P0_100.0.cfg.txt'])
axes_01.plot(MabsWaterIceT01H2O, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.text(0.25,0.1,'($H_{2}O$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.5)
axes_01.set_yscale("log")
# boop
# this is quadrant 2, formally quadrant 3 [CO2 Cold]
axes_10 = plt.subplot(G[0,1])  
# M_abs
# CO2 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_10.plot(ColdWaterIce_Pure[:,0],ColdWaterIce_Pure[:,1], label = 'Frost Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsWaterIceT7CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-7.0P0_100.0.cfg.txt'])
axes_10.plot(MabsWaterIceT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsWaterIceT6CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-6.0P0_100.0.cfg.txt'])
axes_10.plot(MabsWaterIceT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsWaterIceT5CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-5.0P0_100.0.cfg.txt'])
axes_10.plot(MabsWaterIceT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsWaterIceT4CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-4.0P0_100.0.cfg.txt'])
axes_10.plot(MabsWaterIceT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsWaterIceT3CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-3.0P0_100.0.cfg.txt'])
axes_10.plot(MabsWaterIceT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsWaterIceT2CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-2.0P0_100.0.cfg.txt'])
axes_10.plot(MabsWaterIceT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsWaterIceT1CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-1.0P0_100.0.cfg.txt'])
axes_10.plot(MabsWaterIceT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsWaterIceT01CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-0.01P0_100.0.cfg.txt'])
axes_10.plot(MabsWaterIceT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_10.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_10.text(0.25,0.5,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_10.set_ylim(0,1)
axes_10.set_xlim(0.2,2.5)
axes_10.set_yscale("log")
axes_10.title.set_text('log($M_{abs}$) for Frost in the Cold Regime @ ($P_{0}$) = 100 bars')


#
# this is quadrant 4, [CH4 Cold]
axes_11 = plt.subplot(G[1,1]) # Frost Block - Right 
# M_abs
# CH4 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_11.plot(ColdWaterIce_Pure[:,0],ColdWaterIce_Pure[:,1], label = 'Frost Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsWaterIceT7CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-7.0P0_100.0.cfg.txt'])
axes_11.plot(MabsWaterIceT7CH4, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsWaterIceT6CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-6.0P0_100.0.cfg.txt'])
axes_11.plot(MabsWaterIceT6CH4, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsWaterIceT5CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-5.0P0_100.0.cfg.txt'])
axes_11.plot(MabsWaterIceT5CH4, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsWaterIceT4CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-4.0P0_100.0.cfg.txt'])
axes_11.plot(MabsWaterIceT4CH4, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsWaterIceT3CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-3.0P0_100.0.cfg.txt'])
axes_11.plot(MabsWaterIceT3CH4, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsWaterIceT2CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-2.0P0_100.0.cfg.txt'])
axes_11.plot(MabsWaterIceT2CH4, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsWaterIceT1CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-1.0P0_100.0.cfg.txt'])
axes_11.plot(MabsWaterIceT1CH4, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsWaterIceT01CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-0.01P0_100.0.cfg.txt'])
axes_11.plot(MabsWaterIceT01CH4, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_11.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_11.text(0.25,0.1,'($CH_{4}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_11.set_ylim(0,1)
axes_11.set_xlim(0.2,2.5)
axes_11.set_yscale("log")
G.tight_layout(fig)
print('boop')
plt.show()
#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs/ParamsComparisons/Frost"
if not os.path.exists(dst):
	os.makedirs(dst)
    
  

fig.savefig("CruzArceFigs/ParamsComparisons/Frost/Frost_M_abs_P0_100")


# ###### Finally 3.0 = 1000 bars

# In[215]:



###### Finally 3.0 = 1000 bars
fig = plt.figure(figsize=(12, 8))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0.0 , 'Wavelength (um)', ha='center', va='bottom', rotation='horizontal')
G = gridspec.GridSpec(2, 2)
# this is quadrant 1, [CO2 Temp]
axes_00 = plt.subplot(G[0,0]) # WaterIce Block - Right 
# M_abs
# CO2 - Temp 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_00.plot(TempWaterIce_Pure[:,0],TempWaterIce_Pure[:,1], label = 'Frost Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsWaterIceT7CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-7.0P0_1000.0.cfg.txt'])
axes_00.plot(MabsWaterIceT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsWaterIceT6CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-6.0P0_1000.0.cfg.txt'])
axes_00.plot(MabsWaterIceT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsWaterIceT5CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-5.0P0_1000.0.cfg.txt'])
axes_00.plot(MabsWaterIceT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsWaterIceT4CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-4.0P0_1000.0.cfg.txt'])
axes_00.plot(MabsWaterIceT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsWaterIceT3CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-3.0P0_1000.0.cfg.txt'])
axes_00.plot(MabsWaterIceT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsWaterIceT2CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-2.0P0_1000.0.cfg.txt'])
axes_00.plot(MabsWaterIceT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsWaterIceT1CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-1.0P0_1000.0.cfg.txt'])
axes_00.plot(MabsWaterIceT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsWaterIceT01CO2 = df(Temp,columns=['','TempWaterIceCO2Mabs_-0.01P0_1000.0.cfg.txt'])
axes_00.plot(MabsWaterIceT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.text(0.25,0.5,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.5)
axes_00.set_yscale("log")
axes_00.title.set_text('log($M_{abs}$) for Frost in the Temperate Regime @ ($P_{0}$) = 1000 bars')
#
# this is quadrant 2, [H2O Temp]
axes_01 = plt.subplot(G[1,0]) # WaterIce Block - Right 
# M_abs
# H2O - Temp 
axes_01.plot(TempWaterIce_Pure[:,0],TempWaterIce_Pure[:,1], label = 'Frost Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsWaterIceT7H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-7.0P0_1000.0.cfg.txt'])
axes_01.plot(MabsWaterIceT7H2O, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
MabsWaterIceT6H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-6.0P0_1000.0.cfg.txt'])
axes_01.plot(MabsWaterIceT6H2O, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsWaterIceT5H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-5.0P0_1000.0.cfg.txt'])
axes_01.plot(MabsWaterIceT5H2O, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsWaterIceT4H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-4.0P0_1000.0.cfg.txt'])
axes_01.plot(MabsWaterIceT4H2O, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsWaterIceT3H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-3.0P0_1000.0.cfg.txt'])
axes_01.plot(MabsWaterIceT3H2O, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsWaterIceT2H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-2.0P0_1000.0.cfg.txt'])
axes_01.plot(MabsWaterIceT2H2O, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsWaterIceT1H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-1.0P0_1000.0.cfg.txt'])
axes_01.plot(MabsWaterIceT1H2O, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsWaterIceT01H2O = df(Temp,columns=['','TempWaterIceH2OMabs_-0.01P0_1000.0.cfg.txt'])
axes_01.plot(MabsWaterIceT01H2O, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.text(0.25,0.1,'($H_{2}O$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.5)
axes_01.set_yscale("log")
# boop
# this is quadrant 2, formally quadrant 3 [CO2 Cold]
axes_10 = plt.subplot(G[0,1])  
# M_abs
# CO2 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_10.plot(ColdWaterIce_Pure[:,0],ColdWaterIce_Pure[:,1], label = 'Frost Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsWaterIceT7CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-7.0P0_1000.0.cfg.txt'])
axes_10.plot(MabsWaterIceT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsWaterIceT6CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-6.0P0_1000.0.cfg.txt'])
axes_10.plot(MabsWaterIceT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsWaterIceT5CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-5.0P0_1000.0.cfg.txt'])
axes_10.plot(MabsWaterIceT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsWaterIceT4CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-4.0P0_1000.0.cfg.txt'])
axes_10.plot(MabsWaterIceT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsWaterIceT3CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-3.0P0_1000.0.cfg.txt'])
axes_10.plot(MabsWaterIceT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsWaterIceT2CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-2.0P0_1000.0.cfg.txt'])
axes_10.plot(MabsWaterIceT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsWaterIceT1CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-1.0P0_1000.0.cfg.txt'])
axes_10.plot(MabsWaterIceT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsWaterIceT01CO2 = df(Cold,columns=['','ColdWaterIceCO2Mabs_-0.01P0_1000.0.cfg.txt'])
axes_10.plot(MabsWaterIceT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_10.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_10.text(0.25,0.5,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_10.set_ylim(0,1)
axes_10.set_xlim(0.2,2.5)
axes_10.set_yscale("log")
axes_10.title.set_text('log($M_{abs}$) for Frost in the Cold Regime @ ($P_{0}$) = 1000 bars')


#
# this is quadrant 4, [CH4 Cold]
axes_11 = plt.subplot(G[1,1]) # WaterIce Block - Right 
# M_abs
# CH4 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_11.plot(ColdWaterIce_Pure[:,0],ColdWaterIce_Pure[:,1], label = 'Frost Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsWaterIceT7CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-7.0P0_1000.0.cfg.txt'])
axes_11.plot(MabsWaterIceT7CH4, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsWaterIceT6CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-6.0P0_1000.0.cfg.txt'])
axes_11.plot(MabsWaterIceT6CH4, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsWaterIceT5CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-5.0P0_1000.0.cfg.txt'])
axes_11.plot(MabsWaterIceT5CH4, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsWaterIceT4CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-4.0P0_1000.0.cfg.txt'])
axes_11.plot(MabsWaterIceT4CH4, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsWaterIceT3CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-3.0P0_1000.0.cfg.txt'])
axes_11.plot(MabsWaterIceT3CH4, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsWaterIceT2CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-2.0P0_1000.0.cfg.txt'])
axes_11.plot(MabsWaterIceT2CH4, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsWaterIceT1CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-1.0P0_1000.0.cfg.txt'])
axes_11.plot(MabsWaterIceT1CH4, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsWaterIceT01CH4 = df(Cold,columns=['','ColdWaterIceCH4Mabs_-0.01P0_1000.0.cfg.txt'])
axes_11.plot(MabsWaterIceT01CH4, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_11.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_11.text(0.25,0.1,'($CH_{4}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_11.set_ylim(0,1)
axes_11.set_xlim(0.2,2.5)
axes_11.set_yscale("log")
G.tight_layout(fig)
print('boop')
plt.show()
# #Perfecto, now we create a folder to save these in.
# dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs/ParamsComparisons/Frost"
# if not os.path.exists(dst):
#     os.makedirs(dst)
    
  

# fig.savefig("CruzArceFigs/ParamsComparisons/Frost/WaterIce_M_abs_P0_1000")


# ##### Dope! Now let's do the same with the other surface end-members.

# ##### Basalt - m_abs for all p0

# ###### Starting from the top. p0 = 3.0 = 1000 bars

# In[218]:


fig = plt.figure(figsize=(15, 8.5))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0.0 , 'Wavelength (um)', ha='center', va='bottom', rotation='horizontal')
G = gridspec.GridSpec(2, 3)
G.update( wspace = 0 ) # set the spacing between axes. wspace=0.025,hspace=0.0
# this is quadrant 1, [CO2 Temp]
axes_00 = plt.subplot(G[0,0]) # Basalt Block - Right 
# M_abs
# CO2 - Temp 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_00.plot(TempBasalt_Pure[:,0],TempBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-7.0P0_1000.0.cfg.txt'])
axes_00.plot(MabsBasaltT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsBasaltT6CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-6.0P0_1000.0.cfg.txt'])
axes_00.plot(MabsBasaltT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsBasaltT5CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-5.0P0_1000.0.cfg.txt'])
axes_00.plot(MabsBasaltT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsBasaltT4CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-4.0P0_1000.0.cfg.txt'])
axes_00.plot(MabsBasaltT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsBasaltT3CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-3.0P0_1000.0.cfg.txt'])
axes_00.plot(MabsBasaltT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsBasaltT2CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-2.0P0_1000.0.cfg.txt'])
axes_00.plot(MabsBasaltT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsBasaltT1CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-1.0P0_1000.0.cfg.txt'])
axes_00.plot(MabsBasaltT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsBasaltT01CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-0.01P0_1000.0.cfg.txt'])
axes_00.plot(MabsBasaltT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.text(0.25,0.5,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.5)
axes_00.set_yscale("log")
axes_00.title.set_text('log($M_{abs}$) Basalt Temperate Regime @ ($P_{0}$) = 1000 bars')
#
# this is quadrant 2, [H2O Temp]
axes_01 = plt.subplot(G[1,0]) # Basalt Block - Right 
# M_abs
# H2O - Temp 
axes_01.plot(TempBasalt_Pure[:,0],TempBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7H2O = df(Temp,columns=['','TempBasaltH2OMabs_-7.0P0_1000.0.cfg.txt'])
axes_01.plot(MabsBasaltT7H2O, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
MabsBasaltT6H2O = df(Temp,columns=['','TempBasaltH2OMabs_-6.0P0_1000.0.cfg.txt'])
axes_01.plot(MabsBasaltT6H2O, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsBasaltT5H2O = df(Temp,columns=['','TempBasaltH2OMabs_-5.0P0_1000.0.cfg.txt'])
axes_01.plot(MabsBasaltT5H2O, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsBasaltT4H2O = df(Temp,columns=['','TempBasaltH2OMabs_-4.0P0_1000.0.cfg.txt'])
axes_01.plot(MabsBasaltT4H2O, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsBasaltT3H2O = df(Temp,columns=['','TempBasaltH2OMabs_-3.0P0_1000.0.cfg.txt'])
axes_01.plot(MabsBasaltT3H2O, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsBasaltT2H2O = df(Temp,columns=['','TempBasaltH2OMabs_-2.0P0_1000.0.cfg.txt'])
axes_01.plot(MabsBasaltT2H2O, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsBasaltT1H2O = df(Temp,columns=['','TempBasaltH2OMabs_-1.0P0_1000.0.cfg.txt'])
axes_01.plot(MabsBasaltT1H2O, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsBasaltT01H2O = df(Temp,columns=['','TempBasaltH2OMabs_-0.01P0_1000.0.cfg.txt'])
axes_01.plot(MabsBasaltT01H2O, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.text(0.25,0.01,'($H_{2}O$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.5)
axes_01.set_yscale("log")
# boop
# this is quadrant 2, formally quadrant 3 [CO2 Cold]
axes_10 = plt.subplot(G[0,1])  
# M_abs
# CO2 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_10.plot(ColdBasalt_Pure[:,0],ColdBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-7.0P0_1000.0.cfg.txt'])
axes_10.plot(MabsBasaltT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsBasaltT6CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-6.0P0_1000.0.cfg.txt'])
axes_10.plot(MabsBasaltT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsBasaltT5CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-5.0P0_1000.0.cfg.txt'])
axes_10.plot(MabsBasaltT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsBasaltT4CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-4.0P0_1000.0.cfg.txt'])
axes_10.plot(MabsBasaltT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsBasaltT3CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-3.0P0_1000.0.cfg.txt'])
axes_10.plot(MabsBasaltT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsBasaltT2CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-2.0P0_1000.0.cfg.txt'])
axes_10.plot(MabsBasaltT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsBasaltT1CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-1.0P0_1000.0.cfg.txt'])
axes_10.plot(MabsBasaltT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsBasaltT01CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-0.01P0_1000.0.cfg.txt'])
axes_10.plot(MabsBasaltT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_10.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_10.text(0.25,0.5,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_10.set_ylim(0,1)
axes_10.set_xlim(0.2,2.5)
axes_10.set_yscale("log")
axes_10.title.set_text('log($M_{abs}$) Basalt Cold Regime @ ($P_{0}$) = 1000 bars')


#
# this is quadrant 4, [CH4 Cold]
axes_11 = plt.subplot(G[1,1]) # Basalt Block - Right 
# M_abs
# CH4 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_11.plot(ColdBasalt_Pure[:,0],ColdBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-7.0P0_1000.0.cfg.txt'])
axes_11.plot(MabsBasaltT7CH4, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsBasaltT6CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-6.0P0_1000.0.cfg.txt'])
axes_11.plot(MabsBasaltT6CH4, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsBasaltT5CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-5.0P0_1000.0.cfg.txt'])
axes_11.plot(MabsBasaltT5CH4, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsBasaltT4CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-4.0P0_1000.0.cfg.txt'])
axes_11.plot(MabsBasaltT4CH4, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsBasaltT3CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-3.0P0_1000.0.cfg.txt'])
axes_11.plot(MabsBasaltT3CH4, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsBasaltT2CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-2.0P0_1000.0.cfg.txt'])
axes_11.plot(MabsBasaltT2CH4, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsBasaltT1CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-1.0P0_1000.0.cfg.txt'])
axes_11.plot(MabsBasaltT1CH4, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsBasaltT01CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-0.01P0_1000.0.cfg.txt'])
axes_11.plot(MabsBasaltT01CH4, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_11.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_11.text(0.25,0.01,'($CH_{4}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_11.set_ylim(0,1)
axes_11.set_xlim(0.2,2.5)
axes_11.set_yscale("log")
#
axes_02 = plt.subplot(G[0,2])
axes_02.plot(HotBasalt_Pure[:,0],HotBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-7.0P0_1000.0.cfg.txt'])
axes_02.plot(MabsBasaltT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsBasaltT6CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-6.0P0_1000.0.cfg.txt'])
axes_02.plot(MabsBasaltT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsBasaltT5CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-5.0P0_1000.0.cfg.txt'])
axes_02.plot(MabsBasaltT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsBasaltT4CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-4.0P0_1000.0.cfg.txt'])
axes_02.plot(MabsBasaltT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsBasaltT3CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-3.0P0_1000.0.cfg.txt'])
axes_02.plot(MabsBasaltT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsBasaltT2CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-2.0P0_1000.0.cfg.txt'])
axes_02.plot(MabsBasaltT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsBasaltT1CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-1.0P0_1000.0.cfg.txt'])
axes_02.plot(MabsBasaltT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsBasaltT01CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-0.01P0_1000.0.cfg.txt'])
axes_02.plot(MabsBasaltT01CH4, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')


handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_02.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_02.text(0.25,0.5,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
axes_02.set_yscale("log")
axes_02.title.set_text('log($M_{abs}$) Basalt Hot Regime @ ($P_{0}$) = 1000 bars')
plt.grid(True)
#
axes_12 = plt.subplot(G[1,2])
axes_12.plot(HotBasalt_Pure[:,0],HotBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-7.0P0_1000.0.cfg.txt'])
axes_12.plot(MabsBasaltT7SO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsBasaltT6SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-6.0P0_1000.0.cfg.txt'])
axes_12.plot(MabsBasaltT6SO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsBasaltT5SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-5.0P0_1000.0.cfg.txt'])
axes_12.plot(MabsBasaltT5SO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsBasaltT4SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-4.0P0_1000.0.cfg.txt'])
axes_12.plot(MabsBasaltT4SO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsBasaltT3SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-3.0P0_1000.0.cfg.txt'])
axes_12.plot(MabsBasaltT3SO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsBasaltT2SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-2.0P0_1000.0.cfg.txt'])
axes_12.plot(MabsBasaltT2SO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsBasaltT1SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-1.0P0_1000.0.cfg.txt'])
axes_12.plot(MabsBasaltT1SO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsBasaltT01SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-0.01P0_1000.0.cfg.txt'])
axes_12.plot(MabsBasaltT01SO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')


handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_12.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_12.text(2,0.0006,'($SO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
axes_12.set_yscale("log")

G.tight_layout(fig)
print('boop')
plt.show()
#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs/ParamsComparisons/Basalt"
if not os.path.exists(dst):
	os.makedirs(dst)
    
  

fig.savefig("CruzArceFigs/ParamsComparisons/Basalt/Basalt_M_abs_P0_1000")


# ###### p0 = 2.0 = 100 bars

# In[219]:


fig = plt.figure(figsize=(15, 8.5))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0.0 , 'Wavelength (um)', ha='center', va='bottom', rotation='horizontal')
G = gridspec.GridSpec(2, 3)
G.update( wspace = 0 ) # set the spacing between axes. wspace=0.025,hspace=0.0
# this is quadrant 1, [CO2 Temp]
axes_00 = plt.subplot(G[0,0]) # Basalt Block - Right 
# M_abs
# CO2 - Temp 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_00.plot(TempBasalt_Pure[:,0],TempBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-7.0P0_100.0.cfg.txt'])
axes_00.plot(MabsBasaltT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsBasaltT6CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-6.0P0_100.0.cfg.txt'])
axes_00.plot(MabsBasaltT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsBasaltT5CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-5.0P0_100.0.cfg.txt'])
axes_00.plot(MabsBasaltT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsBasaltT4CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-4.0P0_100.0.cfg.txt'])
axes_00.plot(MabsBasaltT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsBasaltT3CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-3.0P0_100.0.cfg.txt'])
axes_00.plot(MabsBasaltT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsBasaltT2CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-2.0P0_100.0.cfg.txt'])
axes_00.plot(MabsBasaltT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsBasaltT1CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-1.0P0_100.0.cfg.txt'])
axes_00.plot(MabsBasaltT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsBasaltT01CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-0.01P0_100.0.cfg.txt'])
axes_00.plot(MabsBasaltT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.text(0.25,0.5,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.5)
axes_00.set_yscale("log")
axes_00.title.set_text('log($M_{abs}$) Basalt Temperate Regime @ ($P_{0}$) = 100 bars')
#
# this is quadrant 2, [H2O Temp]
axes_01 = plt.subplot(G[1,0]) # Basalt Block - Right 
# M_abs
# H2O - Temp 
axes_01.plot(TempBasalt_Pure[:,0],TempBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7H2O = df(Temp,columns=['','TempBasaltH2OMabs_-7.0P0_100.0.cfg.txt'])
axes_01.plot(MabsBasaltT7H2O, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
MabsBasaltT6H2O = df(Temp,columns=['','TempBasaltH2OMabs_-6.0P0_100.0.cfg.txt'])
axes_01.plot(MabsBasaltT6H2O, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsBasaltT5H2O = df(Temp,columns=['','TempBasaltH2OMabs_-5.0P0_100.0.cfg.txt'])
axes_01.plot(MabsBasaltT5H2O, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsBasaltT4H2O = df(Temp,columns=['','TempBasaltH2OMabs_-4.0P0_100.0.cfg.txt'])
axes_01.plot(MabsBasaltT4H2O, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsBasaltT3H2O = df(Temp,columns=['','TempBasaltH2OMabs_-3.0P0_100.0.cfg.txt'])
axes_01.plot(MabsBasaltT3H2O, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsBasaltT2H2O = df(Temp,columns=['','TempBasaltH2OMabs_-2.0P0_100.0.cfg.txt'])
axes_01.plot(MabsBasaltT2H2O, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsBasaltT1H2O = df(Temp,columns=['','TempBasaltH2OMabs_-1.0P0_100.0.cfg.txt'])
axes_01.plot(MabsBasaltT1H2O, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsBasaltT01H2O = df(Temp,columns=['','TempBasaltH2OMabs_-0.01P0_100.0.cfg.txt'])
axes_01.plot(MabsBasaltT01H2O, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.text(0.25,0.01,'($H_{2}O$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.5)
axes_01.set_yscale("log")
# boop
# this is quadrant 2, formally quadrant 3 [CO2 Cold]
axes_10 = plt.subplot(G[0,1])  
# M_abs
# CO2 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_10.plot(ColdBasalt_Pure[:,0],ColdBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-7.0P0_100.0.cfg.txt'])
axes_10.plot(MabsBasaltT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsBasaltT6CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-6.0P0_100.0.cfg.txt'])
axes_10.plot(MabsBasaltT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsBasaltT5CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-5.0P0_100.0.cfg.txt'])
axes_10.plot(MabsBasaltT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsBasaltT4CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-4.0P0_100.0.cfg.txt'])
axes_10.plot(MabsBasaltT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsBasaltT3CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-3.0P0_100.0.cfg.txt'])
axes_10.plot(MabsBasaltT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsBasaltT2CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-2.0P0_100.0.cfg.txt'])
axes_10.plot(MabsBasaltT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsBasaltT1CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-1.0P0_100.0.cfg.txt'])
axes_10.plot(MabsBasaltT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsBasaltT01CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-0.01P0_100.0.cfg.txt'])
axes_10.plot(MabsBasaltT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_10.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_10.text(0.25,0.5,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_10.set_ylim(0,1)
axes_10.set_xlim(0.2,2.5)
axes_10.set_yscale("log")
axes_10.title.set_text('log($M_{abs}$) Basalt Cold Regime @ ($P_{0}$) = 100 bars')


#
# this is quadrant 4, [CH4 Cold]
axes_11 = plt.subplot(G[1,1]) # Basalt Block - Right 
# M_abs
# CH4 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_11.plot(ColdBasalt_Pure[:,0],ColdBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-7.0P0_100.0.cfg.txt'])
axes_11.plot(MabsBasaltT7CH4, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsBasaltT6CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-6.0P0_100.0.cfg.txt'])
axes_11.plot(MabsBasaltT6CH4, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsBasaltT5CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-5.0P0_100.0.cfg.txt'])
axes_11.plot(MabsBasaltT5CH4, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsBasaltT4CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-4.0P0_100.0.cfg.txt'])
axes_11.plot(MabsBasaltT4CH4, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsBasaltT3CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-3.0P0_100.0.cfg.txt'])
axes_11.plot(MabsBasaltT3CH4, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsBasaltT2CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-2.0P0_100.0.cfg.txt'])
axes_11.plot(MabsBasaltT2CH4, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsBasaltT1CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-1.0P0_100.0.cfg.txt'])
axes_11.plot(MabsBasaltT1CH4, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsBasaltT01CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-0.01P0_100.0.cfg.txt'])
axes_11.plot(MabsBasaltT01CH4, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_11.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_11.text(0.25,0.01,'($CH_{4}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_11.set_ylim(0,1)
axes_11.set_xlim(0.2,2.5)
axes_11.set_yscale("log")
#
axes_02 = plt.subplot(G[0,2])
axes_02.plot(HotBasalt_Pure[:,0],HotBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-7.0P0_100.0.cfg.txt'])
axes_02.plot(MabsBasaltT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsBasaltT6CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-6.0P0_100.0.cfg.txt'])
axes_02.plot(MabsBasaltT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsBasaltT5CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-5.0P0_100.0.cfg.txt'])
axes_02.plot(MabsBasaltT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsBasaltT4CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-4.0P0_100.0.cfg.txt'])
axes_02.plot(MabsBasaltT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsBasaltT3CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-3.0P0_100.0.cfg.txt'])
axes_02.plot(MabsBasaltT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsBasaltT2CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-2.0P0_100.0.cfg.txt'])
axes_02.plot(MabsBasaltT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsBasaltT1CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-1.0P0_100.0.cfg.txt'])
axes_02.plot(MabsBasaltT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsBasaltT01CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-0.01P0_100.0.cfg.txt'])
axes_02.plot(MabsBasaltT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')


handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_02.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_02.text(0.25,0.5,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
axes_02.set_yscale("log")
axes_02.title.set_text('log($M_{abs}$) Basalt Hot Regime @ ($P_{0}$) = 100 bars')
plt.grid(True)
#
axes_12 = plt.subplot(G[1,2])
axes_12.plot(HotBasalt_Pure[:,0],HotBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-7.0P0_100.0.cfg.txt'])
axes_12.plot(MabsBasaltT7SO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsBasaltT6SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-6.0P0_100.0.cfg.txt'])
axes_12.plot(MabsBasaltT6SO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsBasaltT5SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-5.0P0_100.0.cfg.txt'])
axes_12.plot(MabsBasaltT5SO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsBasaltT4SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-4.0P0_100.0.cfg.txt'])
axes_12.plot(MabsBasaltT4SO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsBasaltT3SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-3.0P0_100.0.cfg.txt'])
axes_12.plot(MabsBasaltT3SO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsBasaltT2SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-2.0P0_100.0.cfg.txt'])
axes_12.plot(MabsBasaltT2SO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsBasaltT1SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-1.0P0_100.0.cfg.txt'])
axes_12.plot(MabsBasaltT1SO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsBasaltT01SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-0.01P0_100.0.cfg.txt'])
axes_12.plot(MabsBasaltT01SO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')


handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_12.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_12.text(2,0.0006,'($SO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
axes_12.set_yscale("log")

G.tight_layout(fig)
print('boop')
plt.show()
#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs/ParamsComparisons/Basalt"
if not os.path.exists(dst):
	os.makedirs(dst)
    
  

fig.savefig("CruzArceFigs/ParamsComparisons/Basalt/Basalt_M_abs_P0_100")


# ###### p0 = 1.0 = 10 bars

# In[222]:


fig = plt.figure(figsize=(15, 8.5))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0.0 , 'Wavelength (um)', ha='center', va='bottom', rotation='horizontal')
G = gridspec.GridSpec(2, 3)
G.update( wspace = 0 ) # set the spacing between axes. wspace=0.025,hspace=0.0
# this is quadrant 1, [CO2 Temp]
axes_00 = plt.subplot(G[0,0]) # Basalt Block - Right 
# M_abs
# CO2 - Temp 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_00.plot(TempBasalt_Pure[:,0],TempBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-7.0P0_10.0.cfg.txt'])
axes_00.plot(MabsBasaltT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsBasaltT6CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-6.0P0_10.0.cfg.txt'])
axes_00.plot(MabsBasaltT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsBasaltT5CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-5.0P0_10.0.cfg.txt'])
axes_00.plot(MabsBasaltT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsBasaltT4CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-4.0P0_10.0.cfg.txt'])
axes_00.plot(MabsBasaltT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsBasaltT3CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-3.0P0_10.0.cfg.txt'])
axes_00.plot(MabsBasaltT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsBasaltT2CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-2.0P0_10.0.cfg.txt'])
axes_00.plot(MabsBasaltT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsBasaltT1CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-1.0P0_10.0.cfg.txt'])
axes_00.plot(MabsBasaltT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsBasaltT01CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-0.01P0_10.0.cfg.txt'])
axes_00.plot(MabsBasaltT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.text(0.25,0.5,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.5)
axes_00.set_yscale("log")
axes_00.title.set_text('log($M_{abs}$) Basalt Temperate Regime @ ($P_{0}$) = 10bars')
#
# this is quadrant 2, [H2O Temp]
axes_01 = plt.subplot(G[1,0]) # Basalt Block - Right 
# M_abs
# H2O - Temp 
axes_01.plot(TempBasalt_Pure[:,0],TempBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7H2O = df(Temp,columns=['','TempBasaltH2OMabs_-7.0P0_10.0.cfg.txt'])
axes_01.plot(MabsBasaltT7H2O, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
MabsBasaltT6H2O = df(Temp,columns=['','TempBasaltH2OMabs_-6.0P0_10.0.cfg.txt'])
axes_01.plot(MabsBasaltT6H2O, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsBasaltT5H2O = df(Temp,columns=['','TempBasaltH2OMabs_-5.0P0_10.0.cfg.txt'])
axes_01.plot(MabsBasaltT5H2O, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsBasaltT4H2O = df(Temp,columns=['','TempBasaltH2OMabs_-4.0P0_10.0.cfg.txt'])
axes_01.plot(MabsBasaltT4H2O, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsBasaltT3H2O = df(Temp,columns=['','TempBasaltH2OMabs_-3.0P0_10.0.cfg.txt'])
axes_01.plot(MabsBasaltT3H2O, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsBasaltT2H2O = df(Temp,columns=['','TempBasaltH2OMabs_-2.0P0_10.0.cfg.txt'])
axes_01.plot(MabsBasaltT2H2O, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsBasaltT1H2O = df(Temp,columns=['','TempBasaltH2OMabs_-1.0P0_10.0.cfg.txt'])
axes_01.plot(MabsBasaltT1H2O, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsBasaltT01H2O = df(Temp,columns=['','TempBasaltH2OMabs_-0.01P0_10.0.cfg.txt'])
axes_01.plot(MabsBasaltT01H2O, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.text(0.25,0.01,'($H_{2}O$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.5)
axes_01.set_yscale("log")
# boop
# this is quadrant 2, formally quadrant 3 [CO2 Cold]
axes_10 = plt.subplot(G[0,1])  
# M_abs
# CO2 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_10.plot(ColdBasalt_Pure[:,0],ColdBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-7.0P0_10.0.cfg.txt'])
axes_10.plot(MabsBasaltT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsBasaltT6CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-6.0P0_10.0.cfg.txt'])
axes_10.plot(MabsBasaltT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsBasaltT5CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-5.0P0_10.0.cfg.txt'])
axes_10.plot(MabsBasaltT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsBasaltT4CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-4.0P0_10.0.cfg.txt'])
axes_10.plot(MabsBasaltT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsBasaltT3CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-3.0P0_10.0.cfg.txt'])
axes_10.plot(MabsBasaltT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsBasaltT2CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-2.0P0_10.0.cfg.txt'])
axes_10.plot(MabsBasaltT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsBasaltT1CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-1.0P0_10.0.cfg.txt'])
axes_10.plot(MabsBasaltT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsBasaltT01CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-0.01P0_10.0.cfg.txt'])
axes_10.plot(MabsBasaltT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_10.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_10.text(0.25,0.5,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_10.set_ylim(0,1)
axes_10.set_xlim(0.2,2.5)
axes_10.set_yscale("log")
axes_10.title.set_text('log($M_{abs}$) Basalt Cold Regime @ ($P_{0}$) = 10bars')


#
# this is quadrant 4, [CH4 Cold]
axes_11 = plt.subplot(G[1,1]) # Basalt Block - Right 
# M_abs
# CH4 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_11.plot(ColdBasalt_Pure[:,0],ColdBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-7.0P0_10.0.cfg.txt'])
axes_11.plot(MabsBasaltT7CH4, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsBasaltT6CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-6.0P0_10.0.cfg.txt'])
axes_11.plot(MabsBasaltT6CH4, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsBasaltT5CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-5.0P0_10.0.cfg.txt'])
axes_11.plot(MabsBasaltT5CH4, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsBasaltT4CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-4.0P0_10.0.cfg.txt'])
axes_11.plot(MabsBasaltT4CH4, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsBasaltT3CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-3.0P0_10.0.cfg.txt'])
axes_11.plot(MabsBasaltT3CH4, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsBasaltT2CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-2.0P0_10.0.cfg.txt'])
axes_11.plot(MabsBasaltT2CH4, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsBasaltT1CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-1.0P0_10.0.cfg.txt'])
axes_11.plot(MabsBasaltT1CH4, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsBasaltT01CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-0.01P0_10.0.cfg.txt'])
axes_11.plot(MabsBasaltT01CH4, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_11.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_11.text(0.25,0.01,'($CH_{4}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_11.set_ylim(0,1)
axes_11.set_xlim(0.2,2.5)
axes_11.set_yscale("log")
#
axes_02 = plt.subplot(G[0,2])
axes_02.plot(HotBasalt_Pure[:,0],HotBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-7.0P0_10.0.cfg.txt'])
axes_02.plot(MabsBasaltT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsBasaltT6CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-6.0P0_10.0.cfg.txt'])
axes_02.plot(MabsBasaltT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsBasaltT5CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-5.0P0_10.0.cfg.txt'])
axes_02.plot(MabsBasaltT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsBasaltT4CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-4.0P0_10.0.cfg.txt'])
axes_02.plot(MabsBasaltT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsBasaltT3CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-3.0P0_10.0.cfg.txt'])
axes_02.plot(MabsBasaltT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsBasaltT2CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-2.0P0_10.0.cfg.txt'])
axes_02.plot(MabsBasaltT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsBasaltT1CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-1.0P0_10.0.cfg.txt'])
axes_02.plot(MabsBasaltT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsBasaltT01CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-0.01P0_10.0.cfg.txt'])
axes_02.plot(MabsBasaltT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')


handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_02.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_02.text(0.25,0.5,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
axes_02.set_yscale("log")
axes_02.title.set_text('log($M_{abs}$) Basalt Hot Regime @ ($P_{0}$) = 10bars')
plt.grid(True)
#
axes_12 = plt.subplot(G[1,2])
axes_12.plot(HotBasalt_Pure[:,0],HotBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-7.0P0_10.0.cfg.txt'])
axes_12.plot(MabsBasaltT7SO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsBasaltT6SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-6.0P0_10.0.cfg.txt'])
axes_12.plot(MabsBasaltT6SO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsBasaltT5SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-5.0P0_10.0.cfg.txt'])
axes_12.plot(MabsBasaltT5SO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsBasaltT4SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-4.0P0_10.0.cfg.txt'])
axes_12.plot(MabsBasaltT4SO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsBasaltT3SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-3.0P0_10.0.cfg.txt'])
axes_12.plot(MabsBasaltT3SO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsBasaltT2SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-2.0P0_10.0.cfg.txt'])
axes_12.plot(MabsBasaltT2SO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsBasaltT1SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-1.0P0_10.0.cfg.txt'])
axes_12.plot(MabsBasaltT1SO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsBasaltT01SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-0.01P0_10.0.cfg.txt'])
axes_12.plot(MabsBasaltT01SO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')


handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_12.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_12.text(2,0.0006,'($SO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
axes_12.set_yscale("log")

G.tight_layout(fig)
print('boop')
plt.show()
#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs/ParamsComparisons/Basalt"
if not os.path.exists(dst):
	os.makedirs(dst)
    
  

fig.savefig("CruzArceFigs/ParamsComparisons/Basalt/Basalt_M_abs_P0_10")


# ###### p0 = 0.0

# In[4]:


fig = plt.figure(figsize=(15, 8.5))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0.0 , 'Wavelength (um)', ha='center', va='bottom', rotation='horizontal')
G = gridspec.GridSpec(2, 3)
G.update( wspace = 0 ) # set the spacing between axes. wspace=0.025,hspace=0.0
# this is quadrant 1, [CO2 Temp]
axes_00 = plt.subplot(G[0,0]) # Basalt Block - Right 
# M_abs
# CO2 - Temp 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_00.plot(TempBasalt_Pure[:,0],TempBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-7.0P0_1.0.cfg.txt'])
axes_00.plot(MabsBasaltT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsBasaltT6CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-6.0P0_1.0.cfg.txt'])
axes_00.plot(MabsBasaltT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

# MabsBasaltT5CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-5.0P0_1.0.cfg.txt'])
# axes_00.plot(MabsBasaltT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

# MabsBasaltT4CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-4.0P0_1.0.cfg.txt'])
# axes_00.plot(MabsBasaltT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

# MabsBasaltT3CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-3.0P0_1.0.cfg.txt'])
# axes_00.plot(MabsBasaltT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

# MabsBasaltT2CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-2.0P0_1.0.cfg.txt'])
# axes_00.plot(MabsBasaltT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

# MabsBasaltT1CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-1.0P0_1.0.cfg.txt'])
# axes_00.plot(MabsBasaltT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


# MabsBasaltT01CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-0.01P0_1.0.cfg.txt'])
# axes_00.plot(MabsBasaltT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.text(0.25,0.5,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.5)
axes_00.set_yscale("log")
axes_00.title.set_text('log($M_{abs}$) Basalt Temperate Regime @ ($P_{0}$) = 1 bar')
#
# this is quadrant 2, [H2O Temp]
axes_01 = plt.subplot(G[1,0]) # Basalt Block - Right 
# M_abs
# H2O - Temp 
axes_01.plot(TempBasalt_Pure[:,0],TempBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7H2O = df(Temp,columns=['','TempBasaltH2OMabs_-7.0P0_1.0.cfg.txt'])
axes_01.plot(MabsBasaltT7H2O, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
MabsBasaltT6H2O = df(Temp,columns=['','TempBasaltH2OMabs_-6.0P0_1.0.cfg.txt'])
axes_01.plot(MabsBasaltT6H2O, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

# MabsBasaltT5H2O = df(Temp,columns=['','TempBasaltH2OMabs_-5.0P0_1.0.cfg.txt'])
# axes_01.plot(MabsBasaltT5H2O, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

# MabsBasaltT4H2O = df(Temp,columns=['','TempBasaltH2OMabs_-4.0P0_1.0.cfg.txt'])
# axes_01.plot(MabsBasaltT4H2O, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

# MabsBasaltT3H2O = df(Temp,columns=['','TempBasaltH2OMabs_-3.0P0_1.0.cfg.txt'])
# axes_01.plot(MabsBasaltT3H2O, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

# MabsBasaltT2H2O = df(Temp,columns=['','TempBasaltH2OMabs_-2.0P0_1.0.cfg.txt'])
# axes_01.plot(MabsBasaltT2H2O, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

# MabsBasaltT1H2O = df(Temp,columns=['','TempBasaltH2OMabs_-1.0P0_1.0.cfg.txt'])
# axes_01.plot(MabsBasaltT1H2O, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


# MabsBasaltT01H2O = df(Temp,columns=['','TempBasaltH2OMabs_-0.01P0_1.0.cfg.txt'])
# axes_01.plot(MabsBasaltT01H2O, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.text(0.25,0.01,'($H_{2}O$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.5)
axes_01.set_yscale("log")
# boop
# this is quadrant 2, formally quadrant 3 [CO2 Cold]
axes_10 = plt.subplot(G[0,1])  
# M_abs
# CO2 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_10.plot(ColdBasalt_Pure[:,0],ColdBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-7.0P0_1.0.cfg.txt'])
axes_10.plot(MabsBasaltT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsBasaltT6CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-6.0P0_1.0.cfg.txt'])
axes_10.plot(MabsBasaltT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

# MabsBasaltT5CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-5.0P0_1.0.cfg.txt'])
# axes_10.plot(MabsBasaltT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

# MabsBasaltT4CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-4.0P0_1.0.cfg.txt'])
# axes_10.plot(MabsBasaltT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

# MabsBasaltT3CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-3.0P0_1.0.cfg.txt'])
# axes_10.plot(MabsBasaltT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

# MabsBasaltT2CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-2.0P0_1.0.cfg.txt'])
# axes_10.plot(MabsBasaltT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

# MabsBasaltT1CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-1.0P0_1.0.cfg.txt'])
# axes_10.plot(MabsBasaltT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


# MabsBasaltT01CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-0.01P0_1.0.cfg.txt'])
# axes_10.plot(MabsBasaltT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_10.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_10.text(0.25,0.5,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_10.set_ylim(0,1)
axes_10.set_xlim(0.2,2.5)
axes_10.set_yscale("log")
axes_10.title.set_text('log($M_{abs}$) Basalt Cold Regime @ ($P_{0}$) = 1 bar')


#
# this is quadrant 4, [CH4 Cold]
axes_11 = plt.subplot(G[1,1]) # Basalt Block - Right 
# M_abs
# CH4 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_11.plot(ColdBasalt_Pure[:,0],ColdBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-7.0P0_1.0.cfg.txt'])
axes_11.plot(MabsBasaltT7CH4, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsBasaltT6CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-6.0P0_1.0.cfg.txt'])
axes_11.plot(MabsBasaltT6CH4, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

# MabsBasaltT5CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-5.0P0_1.0.cfg.txt'])
# axes_11.plot(MabsBasaltT5CH4, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

# MabsBasaltT4CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-4.0P0_1.0.cfg.txt'])
# axes_11.plot(MabsBasaltT4CH4, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

# MabsBasaltT3CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-3.0P0_1.0.cfg.txt'])
# axes_11.plot(MabsBasaltT3CH4, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

# MabsBasaltT2CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-2.0P0_1.0.cfg.txt'])
# axes_11.plot(MabsBasaltT2CH4, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

# MabsBasaltT1CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-1.0P0_1.0.cfg.txt'])
# axes_11.plot(MabsBasaltT1CH4, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


# MabsBasaltT01CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-0.01P0_1.0.cfg.txt'])
# axes_11.plot(MabsBasaltT01CH4, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_11.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_11.text(0.25,0.01,'($CH_{4}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_11.set_ylim(0,1)
axes_11.set_xlim(0.2,2.5)
axes_11.set_yscale("log")
#
axes_02 = plt.subplot(G[0,2])
axes_02.plot(HotBasalt_Pure[:,0],HotBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-7.0P0_1.0.cfg.txt'])
axes_02.plot(MabsBasaltT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsBasaltT6CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-6.0P0_1.0.cfg.txt'])
axes_02.plot(MabsBasaltT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

# MabsBasaltT5CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-5.0P0_1.0.cfg.txt'])
# axes_02.plot(MabsBasaltT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

# MabsBasaltT4CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-4.0P0_1.0.cfg.txt'])
# axes_02.plot(MabsBasaltT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

# MabsBasaltT3CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-3.0P0_1.0.cfg.txt'])
# axes_02.plot(MabsBasaltT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

# MabsBasaltT2CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-2.0P0_1.0.cfg.txt'])
# axes_02.plot(MabsBasaltT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

# MabsBasaltT1CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-1.0P0_1.0.cfg.txt'])
# axes_02.plot(MabsBasaltT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


# MabsBasaltT01CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-0.01P0_1.0.cfg.txt'])
# axes_02.plot(MabsBasaltT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')


handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_02.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_02.text(0.25,0.5,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
axes_02.set_yscale("log")
axes_02.title.set_text('log($M_{abs}$) Basalt Hot Regime @ ($P_{0}$) = 1 bar')
plt.grid(True)
#
axes_12 = plt.subplot(G[1,2])
axes_12.plot(HotBasalt_Pure[:,0],HotBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-7.0P0_1.0.cfg.txt'])
axes_12.plot(MabsBasaltT7SO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsBasaltT6SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-6.0P0_1.0.cfg.txt'])
axes_12.plot(MabsBasaltT6SO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

# MabsBasaltT5SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-5.0P0_1.0.cfg.txt'])
# axes_12.plot(MabsBasaltT5SO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

# MabsBasaltT4SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-4.0P0_1.0.cfg.txt'])
# axes_12.plot(MabsBasaltT4SO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

# MabsBasaltT3SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-3.0P0_1.0.cfg.txt'])
# axes_12.plot(MabsBasaltT3SO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

# MabsBasaltT2SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-2.0P0_1.0.cfg.txt'])
# axes_12.plot(MabsBasaltT2SO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

# MabsBasaltT1SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-1.0P0_1.0.cfg.txt'])
# axes_12.plot(MabsBasaltT1SO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


# MabsBasaltT01SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-0.01P0_1.0.cfg.txt'])
# axes_12.plot(MabsBasaltT01SO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')


handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_12.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_12.text(2,0.0006,'($SO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
axes_12.set_yscale("log")

G.tight_layout(fig)
print('boop')
plt.show()
#Perfecto, now we create a folder to save these in.
# dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs/ParamsComparisons/Basalt"
# if not os.path.exists(dst):
# 	os.makedirs(dst)
    
  

# fig.savefig("CruzArceFigs/ParamsComparisons/Basalt/Basalt_M_abs_P0_1")


# ###### p0 = -1.0 = 0.1 bars

# In[229]:


fig = plt.figure(figsize=(15, 8.5))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0.0 , 'Wavelength (um)', ha='center', va='bottom', rotation='horizontal')
G = gridspec.GridSpec(2, 3)
G.update( wspace = 0 ) # set the spacing between axes. wspace=0.025,hspace=0.0
# this is quadrant 1, [CO2 Temp]
axes_00 = plt.subplot(G[0,0]) # Basalt Block - Right 
# M_abs
# CO2 - Temp 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_00.plot(TempBasalt_Pure[:,0],TempBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-7.0P0_0.1.cfg.txt'])
axes_00.plot(MabsBasaltT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsBasaltT6CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-6.0P0_0.1.cfg.txt'])
axes_00.plot(MabsBasaltT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsBasaltT5CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-5.0P0_0.1.cfg.txt'])
axes_00.plot(MabsBasaltT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsBasaltT4CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-4.0P0_0.1.cfg.txt'])
axes_00.plot(MabsBasaltT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsBasaltT3CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-3.0P0_0.1.cfg.txt'])
axes_00.plot(MabsBasaltT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsBasaltT2CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-2.0P0_0.1.cfg.txt'])
axes_00.plot(MabsBasaltT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsBasaltT1CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-1.0P0_0.1.cfg.txt'])
axes_00.plot(MabsBasaltT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsBasaltT01CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-0.01P0_0.1.cfg.txt'])
axes_00.plot(MabsBasaltT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.text(0.25,0.2,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.5)
axes_00.set_yscale("log")
axes_00.title.set_text('log($M_{abs}$) Basalt Temperate Regime @ ($P_{0}$)= 0.1  bars')
#
# this is quadrant 2, [H2O Temp]
axes_01 = plt.subplot(G[1,0]) # Basalt Block - Right 
# M_abs
# H2O - Temp 
axes_01.plot(TempBasalt_Pure[:,0],TempBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7H2O = df(Temp,columns=['','TempBasaltH2OMabs_-7.0P0_0.1.cfg.txt'])
axes_01.plot(MabsBasaltT7H2O, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
MabsBasaltT6H2O = df(Temp,columns=['','TempBasaltH2OMabs_-6.0P0_0.1.cfg.txt'])
axes_01.plot(MabsBasaltT6H2O, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsBasaltT5H2O = df(Temp,columns=['','TempBasaltH2OMabs_-5.0P0_0.1.cfg.txt'])
axes_01.plot(MabsBasaltT5H2O, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsBasaltT4H2O = df(Temp,columns=['','TempBasaltH2OMabs_-4.0P0_0.1.cfg.txt'])
axes_01.plot(MabsBasaltT4H2O, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsBasaltT3H2O = df(Temp,columns=['','TempBasaltH2OMabs_-3.0P0_0.1.cfg.txt'])
axes_01.plot(MabsBasaltT3H2O, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsBasaltT2H2O = df(Temp,columns=['','TempBasaltH2OMabs_-2.0P0_0.1.cfg.txt'])
axes_01.plot(MabsBasaltT2H2O, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsBasaltT1H2O = df(Temp,columns=['','TempBasaltH2OMabs_-1.0P0_0.1.cfg.txt'])
axes_01.plot(MabsBasaltT1H2O, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsBasaltT01H2O = df(Temp,columns=['','TempBasaltH2OMabs_-0.01P0_0.1.cfg.txt'])
axes_01.plot(MabsBasaltT01H2O, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.text(0.25,0.01,'($H_{2}O$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.5)
axes_01.set_yscale("log")
# boop
# this is quadrant 2, formally quadrant 3 [CO2 Cold]
axes_10 = plt.subplot(G[0,1])  
# M_abs
# CO2 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_10.plot(ColdBasalt_Pure[:,0],ColdBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-7.0P0_0.1.cfg.txt'])
axes_10.plot(MabsBasaltT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsBasaltT6CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-6.0P0_0.1.cfg.txt'])
axes_10.plot(MabsBasaltT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsBasaltT5CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-5.0P0_0.1.cfg.txt'])
axes_10.plot(MabsBasaltT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsBasaltT4CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-4.0P0_0.1.cfg.txt'])
axes_10.plot(MabsBasaltT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsBasaltT3CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-3.0P0_0.1.cfg.txt'])
axes_10.plot(MabsBasaltT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsBasaltT2CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-2.0P0_0.1.cfg.txt'])
axes_10.plot(MabsBasaltT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsBasaltT1CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-1.0P0_0.1.cfg.txt'])
axes_10.plot(MabsBasaltT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsBasaltT01CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-0.01P0_0.1.cfg.txt'])
axes_10.plot(MabsBasaltT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_10.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_10.text(0.25,0.2,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_10.set_ylim(0,1)
axes_10.set_xlim(0.2,2.5)
axes_10.set_yscale("log")
axes_10.title.set_text('log($M_{abs}$) Basalt Cold Regime @ ($P_{0}$)= 0.1  bars')


#
# this is quadrant 4, [CH4 Cold]
axes_11 = plt.subplot(G[1,1]) # Basalt Block - Right 
# M_abs
# CH4 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_11.plot(ColdBasalt_Pure[:,0],ColdBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-7.0P0_0.1.cfg.txt'])
axes_11.plot(MabsBasaltT7CH4, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsBasaltT6CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-6.0P0_0.1.cfg.txt'])
axes_11.plot(MabsBasaltT6CH4, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsBasaltT5CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-5.0P0_0.1.cfg.txt'])
axes_11.plot(MabsBasaltT5CH4, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsBasaltT4CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-4.0P0_0.1.cfg.txt'])
axes_11.plot(MabsBasaltT4CH4, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsBasaltT3CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-3.0P0_0.1.cfg.txt'])
axes_11.plot(MabsBasaltT3CH4, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsBasaltT2CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-2.0P0_0.1.cfg.txt'])
axes_11.plot(MabsBasaltT2CH4, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsBasaltT1CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-1.0P0_0.1.cfg.txt'])
axes_11.plot(MabsBasaltT1CH4, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsBasaltT01CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-0.01P0_0.1.cfg.txt'])
axes_11.plot(MabsBasaltT01CH4, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_11.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_11.text(0.25,0.01,'($CH_{4}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_11.set_ylim(0,1)
axes_11.set_xlim(0.2,2.5)
axes_11.set_yscale("log")
#
axes_02 = plt.subplot(G[0,2])
axes_02.plot(HotBasalt_Pure[:,0],HotBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-7.0P0_0.1.cfg.txt'])
axes_02.plot(MabsBasaltT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsBasaltT6CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-6.0P0_0.1.cfg.txt'])
axes_02.plot(MabsBasaltT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsBasaltT5CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-5.0P0_0.1.cfg.txt'])
axes_02.plot(MabsBasaltT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsBasaltT4CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-4.0P0_0.1.cfg.txt'])
axes_02.plot(MabsBasaltT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsBasaltT3CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-3.0P0_0.1.cfg.txt'])
axes_02.plot(MabsBasaltT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsBasaltT2CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-2.0P0_0.1.cfg.txt'])
axes_02.plot(MabsBasaltT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsBasaltT1CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-1.0P0_0.1.cfg.txt'])
axes_02.plot(MabsBasaltT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsBasaltT01CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-0.01P0_0.1.cfg.txt'])
axes_02.plot(MabsBasaltT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')


handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_02.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_02.text(0.25,0.2,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
axes_02.set_yscale("log")
axes_02.title.set_text('log($M_{abs}$) Basalt Hot Regime @ ($P_{0}$)= 0.1  bars')
plt.grid(True)
#
axes_12 = plt.subplot(G[1,2])
axes_12.plot(HotBasalt_Pure[:,0],HotBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-7.0P0_0.1.cfg.txt'])
axes_12.plot(MabsBasaltT7SO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsBasaltT6SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-6.0P0_0.1.cfg.txt'])
axes_12.plot(MabsBasaltT6SO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsBasaltT5SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-5.0P0_0.1.cfg.txt'])
axes_12.plot(MabsBasaltT5SO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsBasaltT4SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-4.0P0_0.1.cfg.txt'])
axes_12.plot(MabsBasaltT4SO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsBasaltT3SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-3.0P0_0.1.cfg.txt'])
axes_12.plot(MabsBasaltT3SO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsBasaltT2SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-2.0P0_0.1.cfg.txt'])
axes_12.plot(MabsBasaltT2SO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsBasaltT1SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-1.0P0_0.1.cfg.txt'])
axes_12.plot(MabsBasaltT1SO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsBasaltT01SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-0.01P0_0.1.cfg.txt'])
axes_12.plot(MabsBasaltT01SO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')


handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_12.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_12.text(2,0.0006,'($SO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
axes_12.set_yscale("log")

G.tight_layout(fig)
print('boop')
plt.show()
#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs/ParamsComparisons/Basalt"
if not os.path.exists(dst):
	os.makedirs(dst)
    
  

fig.savefig("CruzArceFigs/ParamsComparisons/Basalt/Basalt_M_abs_P0_01")


# ###### p0 = -2.0

# In[231]:


fig = plt.figure(figsize=(15, 8.5))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0.0 , 'Wavelength (um)', ha='center', va='bottom', rotation='horizontal')
G = gridspec.GridSpec(2, 3)
G.update( wspace = 0 ) # set the spacing between axes. wspace=0.025,hspace=0.0
# this is quadrant 1, [CO2 Temp]
axes_00 = plt.subplot(G[0,0]) # Basalt Block - Right 
# M_abs
# CO2 - Temp 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_00.plot(TempBasalt_Pure[:,0],TempBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-7.0P0_0.01.cfg.txt'])
axes_00.plot(MabsBasaltT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsBasaltT6CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-6.0P0_0.01.cfg.txt'])
axes_00.plot(MabsBasaltT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsBasaltT5CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-5.0P0_0.01.cfg.txt'])
axes_00.plot(MabsBasaltT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsBasaltT4CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-4.0P0_0.01.cfg.txt'])
axes_00.plot(MabsBasaltT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsBasaltT3CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-3.0P0_0.01.cfg.txt'])
axes_00.plot(MabsBasaltT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsBasaltT2CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-2.0P0_0.01.cfg.txt'])
axes_00.plot(MabsBasaltT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsBasaltT1CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-1.0P0_0.01.cfg.txt'])
axes_00.plot(MabsBasaltT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsBasaltT01CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-0.01P0_0.01.cfg.txt'])
axes_00.plot(MabsBasaltT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.text(0.25,0.1,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.5)
axes_00.set_yscale("log")
axes_00.title.set_text('log($M_{abs}$) Basalt Temperate Regime @ ($P_{0}$)= 0.01  bars')
#
# this is quadrant 2, [H2O Temp]
axes_01 = plt.subplot(G[1,0]) # Basalt Block - Right 
# M_abs
# H2O - Temp 
axes_01.plot(TempBasalt_Pure[:,0],TempBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7H2O = df(Temp,columns=['','TempBasaltH2OMabs_-7.0P0_0.01.cfg.txt'])
axes_01.plot(MabsBasaltT7H2O, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
MabsBasaltT6H2O = df(Temp,columns=['','TempBasaltH2OMabs_-6.0P0_0.01.cfg.txt'])
axes_01.plot(MabsBasaltT6H2O, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsBasaltT5H2O = df(Temp,columns=['','TempBasaltH2OMabs_-5.0P0_0.01.cfg.txt'])
axes_01.plot(MabsBasaltT5H2O, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsBasaltT4H2O = df(Temp,columns=['','TempBasaltH2OMabs_-4.0P0_0.01.cfg.txt'])
axes_01.plot(MabsBasaltT4H2O, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsBasaltT3H2O = df(Temp,columns=['','TempBasaltH2OMabs_-3.0P0_0.01.cfg.txt'])
axes_01.plot(MabsBasaltT3H2O, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsBasaltT2H2O = df(Temp,columns=['','TempBasaltH2OMabs_-2.0P0_0.01.cfg.txt'])
axes_01.plot(MabsBasaltT2H2O, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsBasaltT1H2O = df(Temp,columns=['','TempBasaltH2OMabs_-1.0P0_0.01.cfg.txt'])
axes_01.plot(MabsBasaltT1H2O, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsBasaltT01H2O = df(Temp,columns=['','TempBasaltH2OMabs_-0.01P0_0.01.cfg.txt'])
axes_01.plot(MabsBasaltT01H2O, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.text(0.25,0.01,'($H_{2}O$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.5)
axes_01.set_yscale("log")
# boop
# this is quadrant 2, formally quadrant 3 [CO2 Cold]
axes_10 = plt.subplot(G[0,1])  
# M_abs
# CO2 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_10.plot(ColdBasalt_Pure[:,0],ColdBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-7.0P0_0.01.cfg.txt'])
axes_10.plot(MabsBasaltT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsBasaltT6CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-6.0P0_0.01.cfg.txt'])
axes_10.plot(MabsBasaltT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsBasaltT5CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-5.0P0_0.01.cfg.txt'])
axes_10.plot(MabsBasaltT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsBasaltT4CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-4.0P0_0.01.cfg.txt'])
axes_10.plot(MabsBasaltT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsBasaltT3CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-3.0P0_0.01.cfg.txt'])
axes_10.plot(MabsBasaltT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsBasaltT2CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-2.0P0_0.01.cfg.txt'])
axes_10.plot(MabsBasaltT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsBasaltT1CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-1.0P0_0.01.cfg.txt'])
axes_10.plot(MabsBasaltT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsBasaltT01CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-0.01P0_0.01.cfg.txt'])
axes_10.plot(MabsBasaltT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_10.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_10.text(0.25,0.1,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_10.set_ylim(0,1)
axes_10.set_xlim(0.2,2.5)
axes_10.set_yscale("log")
axes_10.title.set_text('log($M_{abs}$) Basalt Cold Regime @ ($P_{0}$)= 0.01  bars')


#
# this is quadrant 4, [CH4 Cold]
axes_11 = plt.subplot(G[1,1]) # Basalt Block - Right 
# M_abs
# CH4 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_11.plot(ColdBasalt_Pure[:,0],ColdBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-7.0P0_0.01.cfg.txt'])
axes_11.plot(MabsBasaltT7CH4, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsBasaltT6CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-6.0P0_0.01.cfg.txt'])
axes_11.plot(MabsBasaltT6CH4, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsBasaltT5CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-5.0P0_0.01.cfg.txt'])
axes_11.plot(MabsBasaltT5CH4, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsBasaltT4CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-4.0P0_0.01.cfg.txt'])
axes_11.plot(MabsBasaltT4CH4, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsBasaltT3CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-3.0P0_0.01.cfg.txt'])
axes_11.plot(MabsBasaltT3CH4, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsBasaltT2CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-2.0P0_0.01.cfg.txt'])
axes_11.plot(MabsBasaltT2CH4, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsBasaltT1CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-1.0P0_0.01.cfg.txt'])
axes_11.plot(MabsBasaltT1CH4, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsBasaltT01CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-0.01P0_0.01.cfg.txt'])
axes_11.plot(MabsBasaltT01CH4, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_11.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_11.text(0.25,0.01,'($CH_{4}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_11.set_ylim(0,1)
axes_11.set_xlim(0.2,2.5)
axes_11.set_yscale("log")
#
axes_02 = plt.subplot(G[0,2])
axes_02.plot(HotBasalt_Pure[:,0],HotBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-7.0P0_0.01.cfg.txt'])
axes_02.plot(MabsBasaltT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsBasaltT6CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-6.0P0_0.01.cfg.txt'])
axes_02.plot(MabsBasaltT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsBasaltT5CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-5.0P0_0.01.cfg.txt'])
axes_02.plot(MabsBasaltT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsBasaltT4CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-4.0P0_0.01.cfg.txt'])
axes_02.plot(MabsBasaltT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsBasaltT3CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-3.0P0_0.01.cfg.txt'])
axes_02.plot(MabsBasaltT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsBasaltT2CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-2.0P0_0.01.cfg.txt'])
axes_02.plot(MabsBasaltT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsBasaltT1CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-1.0P0_0.01.cfg.txt'])
axes_02.plot(MabsBasaltT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsBasaltT01CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-0.01P0_0.01.cfg.txt'])
axes_02.plot(MabsBasaltT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')


handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_02.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_02.text(0.25,0.2,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
axes_02.set_yscale("log")
axes_02.title.set_text('log($M_{abs}$) Basalt Hot Regime @ ($P_{0}$)= 0.01  bars')
plt.grid(True)
#
axes_12 = plt.subplot(G[1,2])
axes_12.plot(HotBasalt_Pure[:,0],HotBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-7.0P0_0.01.cfg.txt'])
axes_12.plot(MabsBasaltT7SO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsBasaltT6SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-6.0P0_0.01.cfg.txt'])
axes_12.plot(MabsBasaltT6SO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsBasaltT5SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-5.0P0_0.01.cfg.txt'])
axes_12.plot(MabsBasaltT5SO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsBasaltT4SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-4.0P0_0.01.cfg.txt'])
axes_12.plot(MabsBasaltT4SO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsBasaltT3SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-3.0P0_0.01.cfg.txt'])
axes_12.plot(MabsBasaltT3SO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsBasaltT2SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-2.0P0_0.01.cfg.txt'])
axes_12.plot(MabsBasaltT2SO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsBasaltT1SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-1.0P0_0.01.cfg.txt'])
axes_12.plot(MabsBasaltT1SO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsBasaltT01SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-0.01P0_0.01.cfg.txt'])
axes_12.plot(MabsBasaltT01SO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')


handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_12.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_12.text(2,0.0006,'($SO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
axes_12.set_yscale("log")

G.tight_layout(fig)
print('boop')
plt.show()
#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs/ParamsComparisons/Basalt"
if not os.path.exists(dst):
	os.makedirs(dst)
    
  

fig.savefig("CruzArceFigs/ParamsComparisons/Basalt/Basalt_M_abs_P0_001")


# ###### p0 = -3.0

# In[235]:


fig = plt.figure(figsize=(15, 8.5))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0.0 , 'Wavelength (um)', ha='center', va='bottom', rotation='horizontal')
G = gridspec.GridSpec(2, 3)
G.update( wspace = 0 ) # set the spacing between axes. wspace=0.025,hspace=0.0
# this is quadrant 1, [CO2 Temp]
axes_00 = plt.subplot(G[0,0]) # Basalt Block - Right 
# M_abs
# CO2 - Temp 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_00.plot(TempBasalt_Pure[:,0],TempBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-7.0P0_0.001.cfg.txt'])
axes_00.plot(MabsBasaltT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsBasaltT6CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-6.0P0_0.001.cfg.txt'])
axes_00.plot(MabsBasaltT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsBasaltT5CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-5.0P0_0.001.cfg.txt'])
axes_00.plot(MabsBasaltT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsBasaltT4CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-4.0P0_0.001.cfg.txt'])
axes_00.plot(MabsBasaltT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsBasaltT3CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-3.0P0_0.001.cfg.txt'])
axes_00.plot(MabsBasaltT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsBasaltT2CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-2.0P0_0.001.cfg.txt'])
axes_00.plot(MabsBasaltT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsBasaltT1CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-1.0P0_0.001.cfg.txt'])
axes_00.plot(MabsBasaltT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsBasaltT01CO2 = df(Temp,columns=['','TempBasaltCO2Mabs_-0.01P0_0.001.cfg.txt'])
axes_00.plot(MabsBasaltT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.text(0.25,0.1,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.5)
axes_00.set_yscale("log")
axes_00.title.set_text('log($M_{abs}$) Basalt Temperate Regime @ ($P_{0}$)= 0.001  bars')
#
# this is quadrant 2, [H2O Temp]
axes_01 = plt.subplot(G[1,0]) # Basalt Block - Right 
# M_abs
# H2O - Temp 
axes_01.plot(TempBasalt_Pure[:,0],TempBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7H2O = df(Temp,columns=['','TempBasaltH2OMabs_-7.0P0_0.001.cfg.txt'])
axes_01.plot(MabsBasaltT7H2O, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
MabsBasaltT6H2O = df(Temp,columns=['','TempBasaltH2OMabs_-6.0P0_0.001.cfg.txt'])
axes_01.plot(MabsBasaltT6H2O, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsBasaltT5H2O = df(Temp,columns=['','TempBasaltH2OMabs_-5.0P0_0.001.cfg.txt'])
axes_01.plot(MabsBasaltT5H2O, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsBasaltT4H2O = df(Temp,columns=['','TempBasaltH2OMabs_-4.0P0_0.001.cfg.txt'])
axes_01.plot(MabsBasaltT4H2O, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsBasaltT3H2O = df(Temp,columns=['','TempBasaltH2OMabs_-3.0P0_0.001.cfg.txt'])
axes_01.plot(MabsBasaltT3H2O, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsBasaltT2H2O = df(Temp,columns=['','TempBasaltH2OMabs_-2.0P0_0.001.cfg.txt'])
axes_01.plot(MabsBasaltT2H2O, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsBasaltT1H2O = df(Temp,columns=['','TempBasaltH2OMabs_-1.0P0_0.001.cfg.txt'])
axes_01.plot(MabsBasaltT1H2O, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsBasaltT01H2O = df(Temp,columns=['','TempBasaltH2OMabs_-0.01P0_0.001.cfg.txt'])
axes_01.plot(MabsBasaltT01H2O, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.text(0.25,0.1,'($H_{2}O$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.5)
axes_01.set_yscale("log")
# boop
# this is quadrant 2, formally quadrant 3 [CO2 Cold]
axes_10 = plt.subplot(G[0,1])  
# M_abs
# CO2 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_10.plot(ColdBasalt_Pure[:,0],ColdBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-7.0P0_0.001.cfg.txt'])
axes_10.plot(MabsBasaltT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsBasaltT6CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-6.0P0_0.001.cfg.txt'])
axes_10.plot(MabsBasaltT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsBasaltT5CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-5.0P0_0.001.cfg.txt'])
axes_10.plot(MabsBasaltT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsBasaltT4CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-4.0P0_0.001.cfg.txt'])
axes_10.plot(MabsBasaltT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsBasaltT3CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-3.0P0_0.001.cfg.txt'])
axes_10.plot(MabsBasaltT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsBasaltT2CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-2.0P0_0.001.cfg.txt'])
axes_10.plot(MabsBasaltT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsBasaltT1CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-1.0P0_0.001.cfg.txt'])
axes_10.plot(MabsBasaltT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsBasaltT01CO2 = df(Cold,columns=['','ColdBasaltCO2Mabs_-0.01P0_0.001.cfg.txt'])
axes_10.plot(MabsBasaltT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_10.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_10.text(0.25,0.1,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_10.set_ylim(0,1)
axes_10.set_xlim(0.2,2.5)
axes_10.set_yscale("log")
axes_10.title.set_text('log($M_{abs}$) Basalt Cold Regime @ ($P_{0}$)= 0.001  bars')


#
# this is quadrant 4, [CH4 Cold]
axes_11 = plt.subplot(G[1,1]) # Basalt Block - Right 
# M_abs
# CH4 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_11.plot(ColdBasalt_Pure[:,0],ColdBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-7.0P0_0.001.cfg.txt'])
axes_11.plot(MabsBasaltT7CH4, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsBasaltT6CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-6.0P0_0.001.cfg.txt'])
axes_11.plot(MabsBasaltT6CH4, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsBasaltT5CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-5.0P0_0.001.cfg.txt'])
axes_11.plot(MabsBasaltT5CH4, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsBasaltT4CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-4.0P0_0.001.cfg.txt'])
axes_11.plot(MabsBasaltT4CH4, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsBasaltT3CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-3.0P0_0.001.cfg.txt'])
axes_11.plot(MabsBasaltT3CH4, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsBasaltT2CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-2.0P0_0.001.cfg.txt'])
axes_11.plot(MabsBasaltT2CH4, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsBasaltT1CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-1.0P0_0.001.cfg.txt'])
axes_11.plot(MabsBasaltT1CH4, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsBasaltT01CH4 = df(Cold,columns=['','ColdBasaltCH4Mabs_-0.01P0_0.001.cfg.txt'])
axes_11.plot(MabsBasaltT01CH4, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_11.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_11.text(0.25,0.1,'($CH_{4}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_11.set_ylim(0,1)
axes_11.set_xlim(0.2,2.5)
axes_11.set_yscale("log")
#
axes_02 = plt.subplot(G[0,2])
axes_02.plot(HotBasalt_Pure[:,0],HotBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-7.0P0_0.001.cfg.txt'])
axes_02.plot(MabsBasaltT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsBasaltT6CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-6.0P0_0.001.cfg.txt'])
axes_02.plot(MabsBasaltT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsBasaltT5CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-5.0P0_0.001.cfg.txt'])
axes_02.plot(MabsBasaltT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsBasaltT4CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-4.0P0_0.001.cfg.txt'])
axes_02.plot(MabsBasaltT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsBasaltT3CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-3.0P0_0.001.cfg.txt'])
axes_02.plot(MabsBasaltT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsBasaltT2CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-2.0P0_0.001.cfg.txt'])
axes_02.plot(MabsBasaltT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsBasaltT1CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-1.0P0_0.001.cfg.txt'])
axes_02.plot(MabsBasaltT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsBasaltT01CO2 = df(Hot,columns=['','HotBasaltCO2Mabs_-0.01P0_0.001.cfg.txt'])
axes_02.plot(MabsBasaltT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')


handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_02.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_02.text(0.25,0.2,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
axes_02.set_yscale("log")
axes_02.title.set_text('log($M_{abs}$) Basalt Hot Regime @ ($P_{0}$)= 0.001  bars')
plt.grid(True)
#
axes_12 = plt.subplot(G[1,2])
axes_12.plot(HotBasalt_Pure[:,0],HotBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsBasaltT7SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-7.0P0_0.001.cfg.txt'])
axes_12.plot(MabsBasaltT7SO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsBasaltT6SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-6.0P0_0.001.cfg.txt'])
axes_12.plot(MabsBasaltT6SO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsBasaltT5SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-5.0P0_0.001.cfg.txt'])
axes_12.plot(MabsBasaltT5SO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsBasaltT4SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-4.0P0_0.001.cfg.txt'])
axes_12.plot(MabsBasaltT4SO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsBasaltT3SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-3.0P0_0.001.cfg.txt'])
axes_12.plot(MabsBasaltT3SO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsBasaltT2SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-2.0P0_0.001.cfg.txt'])
axes_12.plot(MabsBasaltT2SO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsBasaltT1SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-1.0P0_0.001.cfg.txt'])
axes_12.plot(MabsBasaltT1SO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsBasaltT01SO2 = df(Hot,columns=['','HotBasaltSO2Mabs_-0.01P0_0.001.cfg.txt'])
axes_12.plot(MabsBasaltT01SO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')


handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_12.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_12.text(2,0.0006,'($SO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
axes_12.set_yscale("log")

G.tight_layout(fig)
print('boop')
plt.show()
#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs/ParamsComparisons/Basalt"
if not os.path.exists(dst):
	os.makedirs(dst)
    
  

fig.savefig("CruzArceFigs/ParamsComparisons/Basalt/Basalt_M_abs_P0_0001")


# #### Dope! now lets do Sand

# ###### Starting from the top. # p0 = 3.0 = 1000 bars

# In[241]:


fig = plt.figure(figsize=(15, 8.5))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0.0 , 'Wavelength (um)', ha='center', va='bottom', rotation='horizontal')
G = gridspec.GridSpec(2, 3)
G.update( wspace = 0 ) # set the spacing between axes. wspace=0.025,hspace=0.0
# this is quadrant 1, [CO2 Temp]
axes_00 = plt.subplot(G[0,0]) # Sand Block - Right 
# M_abs
# CO2 - Temp 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_00.plot(TempSand_Pure[:,0],TempSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7CO2 = df(Temp,columns=['','TempSandCO2Mabs_-7.0P0_1000.0.cfg.txt'])
axes_00.plot(MabsSandT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsSandT6CO2 = df(Temp,columns=['','TempSandCO2Mabs_-6.0P0_1000.0.cfg.txt'])
axes_00.plot(MabsSandT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsSandT5CO2 = df(Temp,columns=['','TempSandCO2Mabs_-5.0P0_1000.0.cfg.txt'])
axes_00.plot(MabsSandT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsSandT4CO2 = df(Temp,columns=['','TempSandCO2Mabs_-4.0P0_1000.0.cfg.txt'])
axes_00.plot(MabsSandT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsSandT3CO2 = df(Temp,columns=['','TempSandCO2Mabs_-3.0P0_1000.0.cfg.txt'])
axes_00.plot(MabsSandT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsSandT2CO2 = df(Temp,columns=['','TempSandCO2Mabs_-2.0P0_1000.0.cfg.txt'])
axes_00.plot(MabsSandT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsSandT1CO2 = df(Temp,columns=['','TempSandCO2Mabs_-1.0P0_1000.0.cfg.txt'])
axes_00.plot(MabsSandT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')

MabsSandT01CO2 = df(Temp,columns=['','TempSandCO2Mabs_-0.01P0_1000.0.cfg.txt'])
axes_00.plot(MabsSandT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.text(0.25,0.01,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.5)
axes_00.set_yscale("log")
axes_00.title.set_text('log($M_{abs}$) Sand Temperate Regime @ ($P_{0}$) = 1000 bars')
#
# this is quadrant 2, [H2O Temp]
axes_01 = plt.subplot(G[1,0]) # Sand Block - Right 
# M_abs
# H2O - Temp 
axes_01.plot(TempSand_Pure[:,0],TempSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7H2O = df(Temp,columns=['','TempSandH2OMabs_-7.0P0_1000.0.cfg.txt'])
axes_01.plot(MabsSandT7H2O, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
MabsSandT6H2O = df(Temp,columns=['','TempSandH2OMabs_-6.0P0_1000.0.cfg.txt'])
axes_01.plot(MabsSandT6H2O, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsSandT5H2O = df(Temp,columns=['','TempSandH2OMabs_-5.0P0_1000.0.cfg.txt'])
axes_01.plot(MabsSandT5H2O, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsSandT4H2O = df(Temp,columns=['','TempSandH2OMabs_-4.0P0_1000.0.cfg.txt'])
axes_01.plot(MabsSandT4H2O, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsSandT3H2O = df(Temp,columns=['','TempSandH2OMabs_-3.0P0_1000.0.cfg.txt'])
axes_01.plot(MabsSandT3H2O, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsSandT2H2O = df(Temp,columns=['','TempSandH2OMabs_-2.0P0_1000.0.cfg.txt'])
axes_01.plot(MabsSandT2H2O, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsSandT1H2O = df(Temp,columns=['','TempSandH2OMabs_-1.0P0_1000.0.cfg.txt'])
axes_01.plot(MabsSandT1H2O, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')


MabsSandT01H2O = df(Temp,columns=['','TempSandH2OMabs_-0.01P0_1000.0.cfg.txt'])
axes_01.plot(MabsSandT01H2O, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.text(0.25,0.01,'($H_{2}O$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.5)
axes_01.set_yscale("log")
# boop
# this is quadrant 2, formally quadrant 3 [CO2 Cold]
axes_10 = plt.subplot(G[0,1])  
# M_abs
# CO2 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_10.plot(ColdSand_Pure[:,0],ColdSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-7.0P0_1000.0.cfg.txt'])
axes_10.plot(MabsSandT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsSandT6CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-6.0P0_1000.0.cfg.txt'])
axes_10.plot(MabsSandT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsSandT5CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-5.0P0_1000.0.cfg.txt'])
axes_10.plot(MabsSandT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsSandT4CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-4.0P0_1000.0.cfg.txt'])
axes_10.plot(MabsSandT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsSandT3CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-3.0P0_1000.0.cfg.txt'])
axes_10.plot(MabsSandT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsSandT2CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-2.0P0_1000.0.cfg.txt'])
axes_10.plot(MabsSandT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsSandT1CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-1.0P0_1000.0.cfg.txt'])
axes_10.plot(MabsSandT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')

MabsSandT01CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-0.01P0_1000.0.cfg.txt'])
axes_10.plot(MabsSandT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_10.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_10.text(0.25,0.05,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_10.set_ylim(0,1)
axes_10.set_xlim(0.2,2.5)
axes_10.set_yscale("log")
axes_10.title.set_text('log($M_{abs}$) Sand Cold Regime @ ($P_{0}$) = 1000 bars')

#
# this is quadrant 4, [CH4 Cold]
axes_11 = plt.subplot(G[1,1]) # Sand Block - Right 
# M_abs
# CH4 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_11.plot(ColdSand_Pure[:,0],ColdSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-7.0P0_1000.0.cfg.txt'])
axes_11.plot(MabsSandT7CH4, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsSandT6CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-6.0P0_1000.0.cfg.txt'])
axes_11.plot(MabsSandT6CH4, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsSandT5CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-5.0P0_1000.0.cfg.txt'])
axes_11.plot(MabsSandT5CH4, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsSandT4CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-4.0P0_1000.0.cfg.txt'])
axes_11.plot(MabsSandT4CH4, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsSandT3CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-3.0P0_1000.0.cfg.txt'])
axes_11.plot(MabsSandT3CH4, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsSandT2CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-2.0P0_1000.0.cfg.txt'])
axes_11.plot(MabsSandT2CH4, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsSandT1CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-1.0P0_1000.0.cfg.txt'])
axes_11.plot(MabsSandT1CH4, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')

MabsSandT01CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-0.01P0_1000.0.cfg.txt'])
axes_11.plot(MabsSandT01CH4, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_11.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_11.text(0.25,0.01,'($CH_{4}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_11.set_ylim(0,1)
axes_11.set_xlim(0.2,2.5)
axes_11.set_yscale("log")
#
axes_02 = plt.subplot(G[0,2])
axes_02.plot(HotSand_Pure[:,0],HotSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7CO2 = df(Hot,columns=['','HotSandCO2Mabs_-7.0P0_1000.0.cfg.txt'])
axes_02.plot(MabsSandT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsSandT6CO2 = df(Hot,columns=['','HotSandCO2Mabs_-6.0P0_1000.0.cfg.txt'])
axes_02.plot(MabsSandT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsSandT5CO2 = df(Hot,columns=['','HotSandCO2Mabs_-5.0P0_1000.0.cfg.txt'])
axes_02.plot(MabsSandT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsSandT4CO2 = df(Hot,columns=['','HotSandCO2Mabs_-4.0P0_1000.0.cfg.txt'])
axes_02.plot(MabsSandT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsSandT3CO2 = df(Hot,columns=['','HotSandCO2Mabs_-3.0P0_1000.0.cfg.txt'])
axes_02.plot(MabsSandT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsSandT2CO2 = df(Hot,columns=['','HotSandCO2Mabs_-2.0P0_1000.0.cfg.txt'])
axes_02.plot(MabsSandT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsSandT1CO2 = df(Hot,columns=['','HotSandCO2Mabs_-1.0P0_1000.0.cfg.txt'])
axes_02.plot(MabsSandT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')

MabsSandT01CO2 = df(Hot,columns=['','HotSandCO2Mabs_-0.01P0_1000.0.cfg.txt'])
axes_02.plot(MabsSandT01CH4, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_02.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_02.text(0.25,0.05,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
axes_02.set_yscale("log")
axes_02.title.set_text('log($M_{abs}$) Sand Hot Regime @ ($P_{0}$) = 1000 bars')
plt.grid(True)
#
axes_12 = plt.subplot(G[1,2])
axes_12.plot(HotSand_Pure[:,0],HotSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7SO2 = df(Hot,columns=['','HotSandSO2Mabs_-7.0P0_1000.0.cfg.txt'])
axes_12.plot(MabsSandT7SO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')

MabsSandT6SO2 = df(Hot,columns=['','HotSandSO2Mabs_-6.0P0_1000.0.cfg.txt'])
axes_12.plot(MabsSandT6SO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')

MabsSandT5SO2 = df(Hot,columns=['','HotSandSO2Mabs_-5.0P0_1000.0.cfg.txt'])
axes_12.plot(MabsSandT5SO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')

MabsSandT4SO2 = df(Hot,columns=['','HotSandSO2Mabs_-4.0P0_1000.0.cfg.txt'])
axes_12.plot(MabsSandT4SO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')

MabsSandT3SO2 = df(Hot,columns=['','HotSandSO2Mabs_-3.0P0_1000.0.cfg.txt'])
axes_12.plot(MabsSandT3SO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')

MabsSandT2SO2 = df(Hot,columns=['','HotSandSO2Mabs_-2.0P0_1000.0.cfg.txt'])
axes_12.plot(MabsSandT2SO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')

MabsSandT1SO2 = df(Hot,columns=['','HotSandSO2Mabs_-1.0P0_1000.0.cfg.txt'])
axes_12.plot(MabsSandT1SO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')

MabsSandT01SO2 = df(Hot,columns=['','HotSandSO2Mabs_-0.01P0_1000.0.cfg.txt'])
axes_12.plot(MabsSandT01SO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')


handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_12.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_12.text(2,0.0006,'($SO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
axes_12.set_yscale("log")

G.tight_layout(fig)
print('boop')
plt.show()
#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs/ParamsComparisons/Sand"
if not os.path.exists(dst):
    os.makedirs(dst)
    
  

fig.savefig("CruzArceFigs/ParamsComparisons/Sand/Sand_M_abs_P0_1000")


# ##### p0 = 2.0 = 100 bars

# In[245]:


fig = plt.figure(figsize=(15, 8.5))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0.0 , 'Wavelength (um)', ha='center', va='bottom', rotation='horizontal')
G = gridspec.GridSpec(2, 3)
G.update( wspace = 0 ) # set the spacing between axes. wspace=0.025,hspace=0.0
# this is quadrant 1, [CO2 Temp]
axes_00 = plt.subplot(G[0,0]) # Sand Block - Right 
# M_abs
# CO2 - Temp 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_00.plot(TempSand_Pure[:,0],TempSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7CO2 = df(Temp,columns=['','TempSandCO2Mabs_-7.0P0_100.0.cfg.txt'])
axes_00.plot(MabsSandT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSandT6CO2 = df(Temp,columns=['','TempSandCO2Mabs_-6.0P0_100.0.cfg.txt'])
axes_00.plot(MabsSandT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSandT5CO2 = df(Temp,columns=['','TempSandCO2Mabs_-5.0P0_100.0.cfg.txt'])
axes_00.plot(MabsSandT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSandT4CO2 = df(Temp,columns=['','TempSandCO2Mabs_-4.0P0_100.0.cfg.txt'])
axes_00.plot(MabsSandT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSandT3CO2 = df(Temp,columns=['','TempSandCO2Mabs_-3.0P0_100.0.cfg.txt'])
axes_00.plot(MabsSandT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSandT2CO2 = df(Temp,columns=['','TempSandCO2Mabs_-2.0P0_100.0.cfg.txt'])
axes_00.plot(MabsSandT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSandT1CO2 = df(Temp,columns=['','TempSandCO2Mabs_-1.0P0_100.0.cfg.txt'])
axes_00.plot(MabsSandT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSandT01CO2 = df(Temp,columns=['','TempSandCO2Mabs_-0.01P0_100.0.cfg.txt'])
axes_00.plot(MabsSandT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.text(0.25,0.01,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.5)
axes_00.set_yscale("log")
axes_00.title.set_text('log($M_{abs}$) Sand Temperate Regime @ ($P_{0}$) = 100 bars')
#
# this is quadrant 2, [H2O Temp]
axes_01 = plt.subplot(G[1,0]) # Sand Block - Right 
# M_abs
# H2O - Temp 
axes_01.plot(TempSand_Pure[:,0],TempSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7H2O = df(Temp,columns=['','TempSandH2OMabs_-7.0P0_100.0.cfg.txt'])
axes_01.plot(MabsSandT7H2O, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
MabsSandT6H2O = df(Temp,columns=['','TempSandH2OMabs_-6.0P0_100.0.cfg.txt'])
axes_01.plot(MabsSandT6H2O, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSandT5H2O = df(Temp,columns=['','TempSandH2OMabs_-5.0P0_100.0.cfg.txt'])
axes_01.plot(MabsSandT5H2O, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSandT4H2O = df(Temp,columns=['','TempSandH2OMabs_-4.0P0_100.0.cfg.txt'])
axes_01.plot(MabsSandT4H2O, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSandT3H2O = df(Temp,columns=['','TempSandH2OMabs_-3.0P0_100.0.cfg.txt'])
axes_01.plot(MabsSandT3H2O, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSandT2H2O = df(Temp,columns=['','TempSandH2OMabs_-2.0P0_100.0.cfg.txt'])
axes_01.plot(MabsSandT2H2O, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSandT1H2O = df(Temp,columns=['','TempSandH2OMabs_-1.0P0_100.0.cfg.txt'])
axes_01.plot(MabsSandT1H2O, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSandT01H2O = df(Temp,columns=['','TempSandH2OMabs_-0.01P0_100.0.cfg.txt'])
axes_01.plot(MabsSandT01H2O, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.text(0.25,0.01,'($H_{2}O$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.5)
axes_01.set_yscale("log")
# all done
# this is quadrant 2, formally quadrant 3 [CO2 Cold]
axes_10 = plt.subplot(G[0,1])  
# M_abs
# CO2 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_10.plot(ColdSand_Pure[:,0],ColdSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-7.0P0_100.0.cfg.txt'])
axes_10.plot(MabsSandT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSandT6CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-6.0P0_100.0.cfg.txt'])
axes_10.plot(MabsSandT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSandT5CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-5.0P0_100.0.cfg.txt'])
axes_10.plot(MabsSandT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSandT4CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-4.0P0_100.0.cfg.txt'])
axes_10.plot(MabsSandT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSandT3CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-3.0P0_100.0.cfg.txt'])
axes_10.plot(MabsSandT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSandT2CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-2.0P0_100.0.cfg.txt'])
axes_10.plot(MabsSandT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSandT1CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-1.0P0_100.0.cfg.txt'])
axes_10.plot(MabsSandT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSandT01CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-0.01P0_100.0.cfg.txt'])
axes_10.plot(MabsSandT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_10.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_10.text(0.25,0.05,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_10.set_ylim(0,1)
axes_10.set_xlim(0.2,2.5)
axes_10.set_yscale("log")
axes_10.title.set_text('log($M_{abs}$) Sand Cold Regime @ ($P_{0}$) = 100 bars')
#
# this is quadrant 4, [CH4 Cold]
axes_11 = plt.subplot(G[1,1]) # Sand Block - Right 
# M_abs
# CH4 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_11.plot(ColdSand_Pure[:,0],ColdSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-7.0P0_100.0.cfg.txt'])
axes_11.plot(MabsSandT7CH4, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSandT6CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-6.0P0_100.0.cfg.txt'])
axes_11.plot(MabsSandT6CH4, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSandT5CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-5.0P0_100.0.cfg.txt'])
axes_11.plot(MabsSandT5CH4, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSandT4CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-4.0P0_100.0.cfg.txt'])
axes_11.plot(MabsSandT4CH4, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSandT3CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-3.0P0_100.0.cfg.txt'])
axes_11.plot(MabsSandT3CH4, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSandT2CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-2.0P0_100.0.cfg.txt'])
axes_11.plot(MabsSandT2CH4, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSandT1CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-1.0P0_100.0.cfg.txt'])
axes_11.plot(MabsSandT1CH4, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSandT01CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-0.01P0_100.0.cfg.txt'])
axes_11.plot(MabsSandT01CH4, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_11.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_11.text(0.25,0.01,'($CH_{4}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_11.set_ylim(0,1)
axes_11.set_xlim(0.2,2.5)
axes_11.set_yscale("log")
#
axes_02 = plt.subplot(G[0,2])
axes_02.plot(HotSand_Pure[:,0],HotSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7CO2 = df(Hot,columns=['','HotSandCO2Mabs_-7.0P0_100.0.cfg.txt'])
axes_02.plot(MabsSandT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSandT6CO2 = df(Hot,columns=['','HotSandCO2Mabs_-6.0P0_100.0.cfg.txt'])
axes_02.plot(MabsSandT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSandT5CO2 = df(Hot,columns=['','HotSandCO2Mabs_-5.0P0_100.0.cfg.txt'])
axes_02.plot(MabsSandT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSandT4CO2 = df(Hot,columns=['','HotSandCO2Mabs_-4.0P0_100.0.cfg.txt'])
axes_02.plot(MabsSandT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSandT3CO2 = df(Hot,columns=['','HotSandCO2Mabs_-3.0P0_100.0.cfg.txt'])
axes_02.plot(MabsSandT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSandT2CO2 = df(Hot,columns=['','HotSandCO2Mabs_-2.0P0_100.0.cfg.txt'])
axes_02.plot(MabsSandT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSandT1CO2 = df(Hot,columns=['','HotSandCO2Mabs_-1.0P0_100.0.cfg.txt'])
axes_02.plot(MabsSandT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSandT01CO2 = df(Hot,columns=['','HotSandCO2Mabs_-0.01P0_100.0.cfg.txt'])
axes_02.plot(MabsSandT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_02.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_02.text(0.25,0.01,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
axes_02.set_yscale("log")
axes_02.title.set_text('log($M_{abs}$) Sand Hot Regime @ ($P_{0}$) = 100 bars')
plt.grid(True)
#
axes_12 = plt.subplot(G[1,2])
axes_12.plot(HotSand_Pure[:,0],HotSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7SO2 = df(Hot,columns=['','HotSandSO2Mabs_-7.0P0_100.0.cfg.txt'])
axes_12.plot(MabsSandT7SO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSandT6SO2 = df(Hot,columns=['','HotSandSO2Mabs_-6.0P0_100.0.cfg.txt'])
axes_12.plot(MabsSandT6SO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSandT5SO2 = df(Hot,columns=['','HotSandSO2Mabs_-5.0P0_100.0.cfg.txt'])
axes_12.plot(MabsSandT5SO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSandT4SO2 = df(Hot,columns=['','HotSandSO2Mabs_-4.0P0_100.0.cfg.txt'])
axes_12.plot(MabsSandT4SO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSandT3SO2 = df(Hot,columns=['','HotSandSO2Mabs_-3.0P0_100.0.cfg.txt'])
axes_12.plot(MabsSandT3SO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSandT2SO2 = df(Hot,columns=['','HotSandSO2Mabs_-2.0P0_100.0.cfg.txt'])
axes_12.plot(MabsSandT2SO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSandT1SO2 = df(Hot,columns=['','HotSandSO2Mabs_-1.0P0_100.0.cfg.txt'])
axes_12.plot(MabsSandT1SO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSandT01SO2 = df(Hot,columns=['','HotSandSO2Mabs_-0.01P0_100.0.cfg.txt'])
axes_12.plot(MabsSandT01SO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_12.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_12.text(2,0.0006,'($SO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
axes_12.set_yscale("log")
G.tight_layout(fig)
print('all done')
plt.show()
#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs/ParamsComparisons/Sand"
if not os.path.exists(dst):
    os.makedirs(dst)

fig.savefig("CruzArceFigs/ParamsComparisons/Sand/Sand_M_abs_P0_100")


# ##### p0= 1.0 = 10 bars

# In[247]:


# # p0 = 1.0 = 10 bars
# [222]:
fig = plt.figure(figsize=(15, 8.5))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0.0 , 'Wavelength (um)', ha='center', va='bottom', rotation='horizontal')
G = gridspec.GridSpec(2, 3)
G.update( wspace = 0 ) # set the spacing between axes. wspace=0.025,hspace=0.0
# this is quadrant 1, [CO2 Temp]
axes_00 = plt.subplot(G[0,0]) # Sand Block - Right 
# M_abs
# CO2 - Temp 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_00.plot(TempSand_Pure[:,0],TempSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7CO2 = df(Temp,columns=['','TempSandCO2Mabs_-7.0P0_10.0.cfg.txt'])
axes_00.plot(MabsSandT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSandT6CO2 = df(Temp,columns=['','TempSandCO2Mabs_-6.0P0_10.0.cfg.txt'])
axes_00.plot(MabsSandT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSandT5CO2 = df(Temp,columns=['','TempSandCO2Mabs_-5.0P0_10.0.cfg.txt'])
axes_00.plot(MabsSandT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSandT4CO2 = df(Temp,columns=['','TempSandCO2Mabs_-4.0P0_10.0.cfg.txt'])
axes_00.plot(MabsSandT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSandT3CO2 = df(Temp,columns=['','TempSandCO2Mabs_-3.0P0_10.0.cfg.txt'])
axes_00.plot(MabsSandT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSandT2CO2 = df(Temp,columns=['','TempSandCO2Mabs_-2.0P0_10.0.cfg.txt'])
axes_00.plot(MabsSandT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSandT1CO2 = df(Temp,columns=['','TempSandCO2Mabs_-1.0P0_10.0.cfg.txt'])
axes_00.plot(MabsSandT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSandT01CO2 = df(Temp,columns=['','TempSandCO2Mabs_-0.01P0_10.0.cfg.txt'])
axes_00.plot(MabsSandT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.text(0.25,0.05,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.5)
axes_00.set_yscale("log")
axes_00.title.set_text('log($M_{abs}$) Sand Temperate Regime @ ($P_{0}$) = 10bars')
#
# this is quadrant 2, [H2O Temp]
axes_01 = plt.subplot(G[1,0]) # Sand Block - Right 
# M_abs
# H2O - Temp 
axes_01.plot(TempSand_Pure[:,0],TempSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7H2O = df(Temp,columns=['','TempSandH2OMabs_-7.0P0_10.0.cfg.txt'])
axes_01.plot(MabsSandT7H2O, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
MabsSandT6H2O = df(Temp,columns=['','TempSandH2OMabs_-6.0P0_10.0.cfg.txt'])
axes_01.plot(MabsSandT6H2O, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSandT5H2O = df(Temp,columns=['','TempSandH2OMabs_-5.0P0_10.0.cfg.txt'])
axes_01.plot(MabsSandT5H2O, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSandT4H2O = df(Temp,columns=['','TempSandH2OMabs_-4.0P0_10.0.cfg.txt'])
axes_01.plot(MabsSandT4H2O, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSandT3H2O = df(Temp,columns=['','TempSandH2OMabs_-3.0P0_10.0.cfg.txt'])
axes_01.plot(MabsSandT3H2O, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSandT2H2O = df(Temp,columns=['','TempSandH2OMabs_-2.0P0_10.0.cfg.txt'])
axes_01.plot(MabsSandT2H2O, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSandT1H2O = df(Temp,columns=['','TempSandH2OMabs_-1.0P0_10.0.cfg.txt'])
axes_01.plot(MabsSandT1H2O, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSandT01H2O = df(Temp,columns=['','TempSandH2OMabs_-0.01P0_10.0.cfg.txt'])
axes_01.plot(MabsSandT01H2O, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.text(0.25,0.01,'($H_{2}O$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.5)
axes_01.set_yscale("log")
# all done
# this is quadrant 2, formally quadrant 3 [CO2 Cold]
axes_10 = plt.subplot(G[0,1])  
# M_abs
# CO2 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_10.plot(ColdSand_Pure[:,0],ColdSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-7.0P0_10.0.cfg.txt'])
axes_10.plot(MabsSandT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSandT6CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-6.0P0_10.0.cfg.txt'])
axes_10.plot(MabsSandT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSandT5CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-5.0P0_10.0.cfg.txt'])
axes_10.plot(MabsSandT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSandT4CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-4.0P0_10.0.cfg.txt'])
axes_10.plot(MabsSandT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSandT3CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-3.0P0_10.0.cfg.txt'])
axes_10.plot(MabsSandT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSandT2CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-2.0P0_10.0.cfg.txt'])
axes_10.plot(MabsSandT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSandT1CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-1.0P0_10.0.cfg.txt'])
axes_10.plot(MabsSandT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSandT01CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-0.01P0_10.0.cfg.txt'])
axes_10.plot(MabsSandT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_10.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_10.text(0.25,0.05,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_10.set_ylim(0,1)
axes_10.set_xlim(0.2,2.5)
axes_10.set_yscale("log")
axes_10.title.set_text('log($M_{abs}$) Sand Cold Regime @ ($P_{0}$) = 10bars')
#
# this is quadrant 4, [CH4 Cold]
axes_11 = plt.subplot(G[1,1]) # Sand Block - Right 
# M_abs
# CH4 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_11.plot(ColdSand_Pure[:,0],ColdSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-7.0P0_10.0.cfg.txt'])
axes_11.plot(MabsSandT7CH4, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSandT6CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-6.0P0_10.0.cfg.txt'])
axes_11.plot(MabsSandT6CH4, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSandT5CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-5.0P0_10.0.cfg.txt'])
axes_11.plot(MabsSandT5CH4, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSandT4CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-4.0P0_10.0.cfg.txt'])
axes_11.plot(MabsSandT4CH4, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSandT3CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-3.0P0_10.0.cfg.txt'])
axes_11.plot(MabsSandT3CH4, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSandT2CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-2.0P0_10.0.cfg.txt'])
axes_11.plot(MabsSandT2CH4, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSandT1CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-1.0P0_10.0.cfg.txt'])
axes_11.plot(MabsSandT1CH4, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSandT01CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-0.01P0_10.0.cfg.txt'])
axes_11.plot(MabsSandT01CH4, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_11.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_11.text(0.25,0.01,'($CH_{4}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_11.set_ylim(0,1)
axes_11.set_xlim(0.2,2.5)
axes_11.set_yscale("log")
#
axes_02 = plt.subplot(G[0,2])
axes_02.plot(HotSand_Pure[:,0],HotSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7CO2 = df(Hot,columns=['','HotSandCO2Mabs_-7.0P0_10.0.cfg.txt'])
axes_02.plot(MabsSandT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSandT6CO2 = df(Hot,columns=['','HotSandCO2Mabs_-6.0P0_10.0.cfg.txt'])
axes_02.plot(MabsSandT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSandT5CO2 = df(Hot,columns=['','HotSandCO2Mabs_-5.0P0_10.0.cfg.txt'])
axes_02.plot(MabsSandT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSandT4CO2 = df(Hot,columns=['','HotSandCO2Mabs_-4.0P0_10.0.cfg.txt'])
axes_02.plot(MabsSandT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSandT3CO2 = df(Hot,columns=['','HotSandCO2Mabs_-3.0P0_10.0.cfg.txt'])
axes_02.plot(MabsSandT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSandT2CO2 = df(Hot,columns=['','HotSandCO2Mabs_-2.0P0_10.0.cfg.txt'])
axes_02.plot(MabsSandT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSandT1CO2 = df(Hot,columns=['','HotSandCO2Mabs_-1.0P0_10.0.cfg.txt'])
axes_02.plot(MabsSandT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSandT01CO2 = df(Hot,columns=['','HotSandCO2Mabs_-0.01P0_10.0.cfg.txt'])
axes_02.plot(MabsSandT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_02.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_02.text(0.25,0.05,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
axes_02.set_yscale("log")
axes_02.title.set_text('log($M_{abs}$) Sand Hot Regime @ ($P_{0}$) = 10bars')
plt.grid(True)
#
axes_12 = plt.subplot(G[1,2])
axes_12.plot(HotSand_Pure[:,0],HotSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7SO2 = df(Hot,columns=['','HotSandSO2Mabs_-7.0P0_10.0.cfg.txt'])
axes_12.plot(MabsSandT7SO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSandT6SO2 = df(Hot,columns=['','HotSandSO2Mabs_-6.0P0_10.0.cfg.txt'])
axes_12.plot(MabsSandT6SO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSandT5SO2 = df(Hot,columns=['','HotSandSO2Mabs_-5.0P0_10.0.cfg.txt'])
axes_12.plot(MabsSandT5SO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSandT4SO2 = df(Hot,columns=['','HotSandSO2Mabs_-4.0P0_10.0.cfg.txt'])
axes_12.plot(MabsSandT4SO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSandT3SO2 = df(Hot,columns=['','HotSandSO2Mabs_-3.0P0_10.0.cfg.txt'])
axes_12.plot(MabsSandT3SO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSandT2SO2 = df(Hot,columns=['','HotSandSO2Mabs_-2.0P0_10.0.cfg.txt'])
axes_12.plot(MabsSandT2SO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSandT1SO2 = df(Hot,columns=['','HotSandSO2Mabs_-1.0P0_10.0.cfg.txt'])
axes_12.plot(MabsSandT1SO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSandT01SO2 = df(Hot,columns=['','HotSandSO2Mabs_-0.01P0_10.0.cfg.txt'])
axes_12.plot(MabsSandT01SO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_12.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_12.text(2,0.0006,'($SO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
axes_12.set_yscale("log")
G.tight_layout(fig)
print('all done')
plt.show()
#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs/ParamsComparisons/Sand"
if not os.path.exists(dst):
    os.makedirs(dst)

fig.savefig("CruzArceFigs/ParamsComparisons/Sand/Sand_M_abs_P0_10")


# ###### p0 = 0.0 = 1 bars

# In[249]:


# p0 = 0.0
#
fig = plt.figure(figsize=(15, 8.5))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0.0 , 'Wavelength (um)', ha='center', va='bottom', rotation='horizontal')
G = gridspec.GridSpec(2, 3)
G.update( wspace = 0 ) # set the spacing between axes. wspace=0.025,hspace=0.0
# this is quadrant 1, [CO2 Temp]
axes_00 = plt.subplot(G[0,0]) # Sand Block - Right 
# M_abs
# CO2 - Temp 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_00.plot(TempSand_Pure[:,0],TempSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7CO2 = df(Temp,columns=['','TempSandCO2Mabs_-7.0P0_1.0.cfg.txt'])
axes_00.plot(MabsSandT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSandT6CO2 = df(Temp,columns=['','TempSandCO2Mabs_-6.0P0_1.0.cfg.txt'])
axes_00.plot(MabsSandT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSandT5CO2 = df(Temp,columns=['','TempSandCO2Mabs_-5.0P0_1.0.cfg.txt'])
axes_00.plot(MabsSandT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSandT4CO2 = df(Temp,columns=['','TempSandCO2Mabs_-4.0P0_1.0.cfg.txt'])
axes_00.plot(MabsSandT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSandT3CO2 = df(Temp,columns=['','TempSandCO2Mabs_-3.0P0_1.0.cfg.txt'])
axes_00.plot(MabsSandT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSandT2CO2 = df(Temp,columns=['','TempSandCO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_00.plot(MabsSandT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSandT1CO2 = df(Temp,columns=['','TempSandCO2Mabs_-1.0P0_1.0.cfg.txt'])
axes_00.plot(MabsSandT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSandT01CO2 = df(Temp,columns=['','TempSandCO2Mabs_-0.01P0_1.0.cfg.txt'])
axes_00.plot(MabsSandT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.text(0.25,0.05,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.5)
axes_00.set_yscale("log")
axes_00.title.set_text('log($M_{abs}$) Sand Temperate Regime @ ($P_{0}$) = 1 bar')
#
# this is quadrant 2, [H2O Temp]
axes_01 = plt.subplot(G[1,0]) # Sand Block - Right 
# M_abs
# H2O - Temp 
axes_01.plot(TempSand_Pure[:,0],TempSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7H2O = df(Temp,columns=['','TempSandH2OMabs_-7.0P0_1.0.cfg.txt'])
axes_01.plot(MabsSandT7H2O, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
MabsSandT6H2O = df(Temp,columns=['','TempSandH2OMabs_-6.0P0_1.0.cfg.txt'])
axes_01.plot(MabsSandT6H2O, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSandT5H2O = df(Temp,columns=['','TempSandH2OMabs_-5.0P0_1.0.cfg.txt'])
axes_01.plot(MabsSandT5H2O, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSandT4H2O = df(Temp,columns=['','TempSandH2OMabs_-4.0P0_1.0.cfg.txt'])
axes_01.plot(MabsSandT4H2O, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSandT3H2O = df(Temp,columns=['','TempSandH2OMabs_-3.0P0_1.0.cfg.txt'])
axes_01.plot(MabsSandT3H2O, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSandT2H2O = df(Temp,columns=['','TempSandH2OMabs_-2.0P0_1.0.cfg.txt'])
axes_01.plot(MabsSandT2H2O, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSandT1H2O = df(Temp,columns=['','TempSandH2OMabs_-1.0P0_1.0.cfg.txt'])
axes_01.plot(MabsSandT1H2O, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSandT01H2O = df(Temp,columns=['','TempSandH2OMabs_-0.01P0_1.0.cfg.txt'])
axes_01.plot(MabsSandT01H2O, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.text(0.25,0.01,'($H_{2}O$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.5)
axes_01.set_yscale("log")
# all done
# this is quadrant 2, formally quadrant 3 [CO2 Cold]
axes_10 = plt.subplot(G[0,1])  
# M_abs
# CO2 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_10.plot(ColdSand_Pure[:,0],ColdSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-7.0P0_1.0.cfg.txt'])
axes_10.plot(MabsSandT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSandT6CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-6.0P0_1.0.cfg.txt'])
axes_10.plot(MabsSandT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSandT5CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-5.0P0_1.0.cfg.txt'])
axes_10.plot(MabsSandT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSandT4CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-4.0P0_1.0.cfg.txt'])
axes_10.plot(MabsSandT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSandT3CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-3.0P0_1.0.cfg.txt'])
axes_10.plot(MabsSandT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSandT2CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_10.plot(MabsSandT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSandT1CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-1.0P0_1.0.cfg.txt'])
axes_10.plot(MabsSandT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSandT01CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-0.01P0_1.0.cfg.txt'])
axes_10.plot(MabsSandT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_10.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_10.text(0.25,0.05,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_10.set_ylim(0,1)
axes_10.set_xlim(0.2,2.5)
axes_10.set_yscale("log")
axes_10.title.set_text('log($M_{abs}$) Sand Cold Regime @ ($P_{0}$) = 1 bar')
#
# this is quadrant 4, [CH4 Cold]
axes_11 = plt.subplot(G[1,1]) # Sand Block - Right 
# M_abs
# CH4 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_11.plot(ColdSand_Pure[:,0],ColdSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-7.0P0_1.0.cfg.txt'])
axes_11.plot(MabsSandT7CH4, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSandT6CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-6.0P0_1.0.cfg.txt'])
axes_11.plot(MabsSandT6CH4, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSandT5CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-5.0P0_1.0.cfg.txt'])
axes_11.plot(MabsSandT5CH4, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSandT4CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-4.0P0_1.0.cfg.txt'])
axes_11.plot(MabsSandT4CH4, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSandT3CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-3.0P0_1.0.cfg.txt'])
axes_11.plot(MabsSandT3CH4, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSandT2CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-2.0P0_1.0.cfg.txt'])
axes_11.plot(MabsSandT2CH4, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSandT1CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-1.0P0_1.0.cfg.txt'])
axes_11.plot(MabsSandT1CH4, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSandT01CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-0.01P0_1.0.cfg.txt'])
axes_11.plot(MabsSandT01CH4, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_11.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_11.text(0.25,0.01,'($CH_{4}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_11.set_ylim(0,1)
axes_11.set_xlim(0.2,2.5)
axes_11.set_yscale("log")
#
axes_02 = plt.subplot(G[0,2])
axes_02.plot(HotSand_Pure[:,0],HotSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7CO2 = df(Hot,columns=['','HotSandCO2Mabs_-7.0P0_1.0.cfg.txt'])
axes_02.plot(MabsSandT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSandT6CO2 = df(Hot,columns=['','HotSandCO2Mabs_-6.0P0_1.0.cfg.txt'])
axes_02.plot(MabsSandT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSandT5CO2 = df(Hot,columns=['','HotSandCO2Mabs_-5.0P0_1.0.cfg.txt'])
axes_02.plot(MabsSandT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSandT4CO2 = df(Hot,columns=['','HotSandCO2Mabs_-4.0P0_1.0.cfg.txt'])
axes_02.plot(MabsSandT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSandT3CO2 = df(Hot,columns=['','HotSandCO2Mabs_-3.0P0_1.0.cfg.txt'])
axes_02.plot(MabsSandT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSandT2CO2 = df(Hot,columns=['','HotSandCO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_02.plot(MabsSandT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSandT1CO2 = df(Hot,columns=['','HotSandCO2Mabs_-1.0P0_1.0.cfg.txt'])
axes_02.plot(MabsSandT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSandT01CO2 = df(Hot,columns=['','HotSandCO2Mabs_-0.01P0_1.0.cfg.txt'])
axes_02.plot(MabsSandT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_02.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_02.text(0.25,0.05,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
axes_02.set_yscale("log")
axes_02.title.set_text('log($M_{abs}$) Sand Hot Regime @ ($P_{0}$) = 1 bar')
plt.grid(True)
#
axes_12 = plt.subplot(G[1,2])
axes_12.plot(HotSand_Pure[:,0],HotSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7SO2 = df(Hot,columns=['','HotSandSO2Mabs_-7.0P0_1.0.cfg.txt'])
axes_12.plot(MabsSandT7SO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSandT6SO2 = df(Hot,columns=['','HotSandSO2Mabs_-6.0P0_1.0.cfg.txt'])
axes_12.plot(MabsSandT6SO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSandT5SO2 = df(Hot,columns=['','HotSandSO2Mabs_-5.0P0_1.0.cfg.txt'])
axes_12.plot(MabsSandT5SO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSandT4SO2 = df(Hot,columns=['','HotSandSO2Mabs_-4.0P0_1.0.cfg.txt'])
axes_12.plot(MabsSandT4SO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSandT3SO2 = df(Hot,columns=['','HotSandSO2Mabs_-3.0P0_1.0.cfg.txt'])
axes_12.plot(MabsSandT3SO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSandT2SO2 = df(Hot,columns=['','HotSandSO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_12.plot(MabsSandT2SO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSandT1SO2 = df(Hot,columns=['','HotSandSO2Mabs_-1.0P0_1.0.cfg.txt'])
axes_12.plot(MabsSandT1SO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSandT01SO2 = df(Hot,columns=['','HotSandSO2Mabs_-0.01P0_1.0.cfg.txt'])
axes_12.plot(MabsSandT01SO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_12.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_12.text(2,0.0006,'($SO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
axes_12.set_yscale("log")
G.tight_layout(fig)
print('all done')
plt.show()
#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs/ParamsComparisons/Sand"
if not os.path.exists(dst):
    os.makedirs(dst)

fig.savefig("CruzArceFigs/ParamsComparisons/Sand/Sand_M_abs_P0_1")


# ###### p0 = -1.0 = 0.1 bars

# In[259]:


# p0 = -1.0 = 0.1 bars
#[229]:
fig = plt.figure(figsize=(15, 8.5))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0.0 , 'Wavelength (um)', ha='center', va='bottom', rotation='horizontal')
G = gridspec.GridSpec(2, 3)
G.update( wspace = 0 ) # set the spacing between axes. wspace=0.025,hspace=0.0
# this is quadrant 1, [CO2 Temp]
axes_00 = plt.subplot(G[0,0]) # Sand Block - Right 
# M_abs
# CO2 - Temp 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_00.plot(TempSand_Pure[:,0],TempSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7CO2 = df(Temp,columns=['','TempSandCO2Mabs_-7.0P0_0.1.cfg.txt'])
axes_00.plot(MabsSandT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSandT6CO2 = df(Temp,columns=['','TempSandCO2Mabs_-6.0P0_0.1.cfg.txt'])
axes_00.plot(MabsSandT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSandT5CO2 = df(Temp,columns=['','TempSandCO2Mabs_-5.0P0_0.1.cfg.txt'])
axes_00.plot(MabsSandT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSandT4CO2 = df(Temp,columns=['','TempSandCO2Mabs_-4.0P0_0.1.cfg.txt'])
axes_00.plot(MabsSandT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSandT3CO2 = df(Temp,columns=['','TempSandCO2Mabs_-3.0P0_0.1.cfg.txt'])
axes_00.plot(MabsSandT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSandT2CO2 = df(Temp,columns=['','TempSandCO2Mabs_-2.0P0_0.1.cfg.txt'])
axes_00.plot(MabsSandT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSandT1CO2 = df(Temp,columns=['','TempSandCO2Mabs_-1.0P0_0.1.cfg.txt'])
axes_00.plot(MabsSandT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSandT01CO2 = df(Temp,columns=['','TempSandCO2Mabs_-0.01P0_0.1.cfg.txt'])
axes_00.plot(MabsSandT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.text(0.25,0.02,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.5)
axes_00.set_yscale("log")
axes_00.title.set_text('log($M_{abs}$) Sand Temperate Regime @ ($P_{0}$)= 0.1  bars')
#
# this is quadrant 2, [H2O Temp]
axes_01 = plt.subplot(G[1,0]) # Sand Block - Right 
# M_abs
# H2O - Temp 
axes_01.plot(TempSand_Pure[:,0],TempSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7H2O = df(Temp,columns=['','TempSandH2OMabs_-7.0P0_0.1.cfg.txt'])
axes_01.plot(MabsSandT7H2O, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
MabsSandT6H2O = df(Temp,columns=['','TempSandH2OMabs_-6.0P0_0.1.cfg.txt'])
axes_01.plot(MabsSandT6H2O, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSandT5H2O = df(Temp,columns=['','TempSandH2OMabs_-5.0P0_0.1.cfg.txt'])
axes_01.plot(MabsSandT5H2O, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSandT4H2O = df(Temp,columns=['','TempSandH2OMabs_-4.0P0_0.1.cfg.txt'])
axes_01.plot(MabsSandT4H2O, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSandT3H2O = df(Temp,columns=['','TempSandH2OMabs_-3.0P0_0.1.cfg.txt'])
axes_01.plot(MabsSandT3H2O, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSandT2H2O = df(Temp,columns=['','TempSandH2OMabs_-2.0P0_0.1.cfg.txt'])
axes_01.plot(MabsSandT2H2O, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSandT1H2O = df(Temp,columns=['','TempSandH2OMabs_-1.0P0_0.1.cfg.txt'])
axes_01.plot(MabsSandT1H2O, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSandT01H2O = df(Temp,columns=['','TempSandH2OMabs_-0.01P0_0.1.cfg.txt'])
axes_01.plot(MabsSandT01H2O, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.text(0.25,0.01,'($H_{2}O$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.5)
axes_01.set_yscale("log")
# all done
# this is quadrant 2, formally quadrant 3 [CO2 Cold]
axes_10 = plt.subplot(G[0,1])  
# M_abs
# CO2 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_10.plot(ColdSand_Pure[:,0],ColdSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-7.0P0_0.1.cfg.txt'])
axes_10.plot(MabsSandT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSandT6CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-6.0P0_0.1.cfg.txt'])
axes_10.plot(MabsSandT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSandT5CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-5.0P0_0.1.cfg.txt'])
axes_10.plot(MabsSandT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSandT4CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-4.0P0_0.1.cfg.txt'])
axes_10.plot(MabsSandT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSandT3CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-3.0P0_0.1.cfg.txt'])
axes_10.plot(MabsSandT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSandT2CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-2.0P0_0.1.cfg.txt'])
axes_10.plot(MabsSandT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSandT1CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-1.0P0_0.1.cfg.txt'])
axes_10.plot(MabsSandT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSandT01CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-0.01P0_0.1.cfg.txt'])
axes_10.plot(MabsSandT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_10.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_10.text(0.25,0.02,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_10.set_ylim(0,1)
axes_10.set_xlim(0.2,2.5)
axes_10.set_yscale("log")
axes_10.title.set_text('log($M_{abs}$) Sand Cold Regime @ ($P_{0}$)= 0.1  bars')
#
# this is quadrant 4, [CH4 Cold]
axes_11 = plt.subplot(G[1,1]) # Sand Block - Right 
# M_abs
# CH4 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_11.plot(ColdSand_Pure[:,0],ColdSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-7.0P0_0.1.cfg.txt'])
axes_11.plot(MabsSandT7CH4, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSandT6CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-6.0P0_0.1.cfg.txt'])
axes_11.plot(MabsSandT6CH4, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSandT5CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-5.0P0_0.1.cfg.txt'])
axes_11.plot(MabsSandT5CH4, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSandT4CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-4.0P0_0.1.cfg.txt'])
axes_11.plot(MabsSandT4CH4, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSandT3CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-3.0P0_0.1.cfg.txt'])
axes_11.plot(MabsSandT3CH4, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSandT2CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-2.0P0_0.1.cfg.txt'])
axes_11.plot(MabsSandT2CH4, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSandT1CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-1.0P0_0.1.cfg.txt'])
axes_11.plot(MabsSandT1CH4, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSandT01CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-0.01P0_0.1.cfg.txt'])
axes_11.plot(MabsSandT01CH4, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_11.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_11.text(0.25,0.01,'($CH_{4}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_11.set_ylim(0,1)
axes_11.set_xlim(0.2,2.5)
axes_11.set_yscale("log")
#
axes_02 = plt.subplot(G[0,2])
axes_02.plot(HotSand_Pure[:,0],HotSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7CO2 = df(Hot,columns=['','HotSandCO2Mabs_-7.0P0_0.1.cfg.txt'])
axes_02.plot(MabsSandT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSandT6CO2 = df(Hot,columns=['','HotSandCO2Mabs_-6.0P0_0.1.cfg.txt'])
axes_02.plot(MabsSandT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSandT5CO2 = df(Hot,columns=['','HotSandCO2Mabs_-5.0P0_0.1.cfg.txt'])
axes_02.plot(MabsSandT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSandT4CO2 = df(Hot,columns=['','HotSandCO2Mabs_-4.0P0_0.1.cfg.txt'])
axes_02.plot(MabsSandT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSandT3CO2 = df(Hot,columns=['','HotSandCO2Mabs_-3.0P0_0.1.cfg.txt'])
axes_02.plot(MabsSandT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSandT2CO2 = df(Hot,columns=['','HotSandCO2Mabs_-2.0P0_0.1.cfg.txt'])
axes_02.plot(MabsSandT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSandT1CO2 = df(Hot,columns=['','HotSandCO2Mabs_-1.0P0_0.1.cfg.txt'])
axes_02.plot(MabsSandT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSandT01CO2 = df(Hot,columns=['','HotSandCO2Mabs_-0.01P0_0.1.cfg.txt'])
axes_02.plot(MabsSandT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_02.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_02.text(0.25,0.62,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
axes_02.set_yscale("log")
axes_02.title.set_text('log($M_{abs}$) Sand Hot Regime @ ($P_{0}$)= 0.1  bars')
plt.grid(True)
#
axes_12 = plt.subplot(G[1,2])
axes_12.plot(HotSand_Pure[:,0],HotSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7SO2 = df(Hot,columns=['','HotSandSO2Mabs_-7.0P0_0.1.cfg.txt'])
axes_12.plot(MabsSandT7SO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSandT6SO2 = df(Hot,columns=['','HotSandSO2Mabs_-6.0P0_0.1.cfg.txt'])
axes_12.plot(MabsSandT6SO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSandT5SO2 = df(Hot,columns=['','HotSandSO2Mabs_-5.0P0_0.1.cfg.txt'])
axes_12.plot(MabsSandT5SO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSandT4SO2 = df(Hot,columns=['','HotSandSO2Mabs_-4.0P0_0.1.cfg.txt'])
axes_12.plot(MabsSandT4SO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSandT3SO2 = df(Hot,columns=['','HotSandSO2Mabs_-3.0P0_0.1.cfg.txt'])
axes_12.plot(MabsSandT3SO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSandT2SO2 = df(Hot,columns=['','HotSandSO2Mabs_-2.0P0_0.1.cfg.txt'])
axes_12.plot(MabsSandT2SO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSandT1SO2 = df(Hot,columns=['','HotSandSO2Mabs_-1.0P0_0.1.cfg.txt'])
axes_12.plot(MabsSandT1SO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSandT01SO2 = df(Hot,columns=['','HotSandSO2Mabs_-0.01P0_0.1.cfg.txt'])
axes_12.plot(MabsSandT01SO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_12.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_12.text(2,0.0006,'($SO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
axes_12.set_yscale("log")
G.tight_layout(fig)
print('all done')
plt.show()
#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs/ParamsComparisons/Sand"
if not os.path.exists(dst):
    os.makedirs(dst)

fig.savefig("CruzArceFigs/ParamsComparisons/Sand/Sand_M_abs_P0_01")


# ##### p0 = -2.0 = 0.01 bars

# In[263]:


# p0 = -2.0¶
#[231]:
fig = plt.figure(figsize=(15, 8.5))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0.0 , 'Wavelength (um)', ha='center', va='bottom', rotation='horizontal')
G = gridspec.GridSpec(2, 3)
G.update( wspace = 0 ) # set the spacing between axes. wspace=0.025,hspace=0.0
# this is quadrant 1, [CO2 Temp]
axes_00 = plt.subplot(G[0,0]) # Sand Block - Right 
# M_abs
# CO2 - Temp 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_00.plot(TempSand_Pure[:,0],TempSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7CO2 = df(Temp,columns=['','TempSandCO2Mabs_-7.0P0_0.01.cfg.txt'])
axes_00.plot(MabsSandT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSandT6CO2 = df(Temp,columns=['','TempSandCO2Mabs_-6.0P0_0.01.cfg.txt'])
axes_00.plot(MabsSandT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSandT5CO2 = df(Temp,columns=['','TempSandCO2Mabs_-5.0P0_0.01.cfg.txt'])
axes_00.plot(MabsSandT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSandT4CO2 = df(Temp,columns=['','TempSandCO2Mabs_-4.0P0_0.01.cfg.txt'])
axes_00.plot(MabsSandT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSandT3CO2 = df(Temp,columns=['','TempSandCO2Mabs_-3.0P0_0.01.cfg.txt'])
axes_00.plot(MabsSandT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSandT2CO2 = df(Temp,columns=['','TempSandCO2Mabs_-2.0P0_0.01.cfg.txt'])
axes_00.plot(MabsSandT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSandT1CO2 = df(Temp,columns=['','TempSandCO2Mabs_-1.0P0_0.01.cfg.txt'])
axes_00.plot(MabsSandT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSandT01CO2 = df(Temp,columns=['','TempSandCO2Mabs_-0.01P0_0.01.cfg.txt'])
axes_00.plot(MabsSandT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.text(0.25,0.58,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.5)
axes_00.set_yscale("log")
axes_00.title.set_text('log($M_{abs}$) Sand Temperate Regime @ ($P_{0}$)= 0.01  bars')
#
# this is quadrant 2, [H2O Temp]
axes_01 = plt.subplot(G[1,0]) # Sand Block - Right 
# M_abs
# H2O - Temp 
axes_01.plot(TempSand_Pure[:,0],TempSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7H2O = df(Temp,columns=['','TempSandH2OMabs_-7.0P0_0.01.cfg.txt'])
axes_01.plot(MabsSandT7H2O, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
MabsSandT6H2O = df(Temp,columns=['','TempSandH2OMabs_-6.0P0_0.01.cfg.txt'])
axes_01.plot(MabsSandT6H2O, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSandT5H2O = df(Temp,columns=['','TempSandH2OMabs_-5.0P0_0.01.cfg.txt'])
axes_01.plot(MabsSandT5H2O, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSandT4H2O = df(Temp,columns=['','TempSandH2OMabs_-4.0P0_0.01.cfg.txt'])
axes_01.plot(MabsSandT4H2O, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSandT3H2O = df(Temp,columns=['','TempSandH2OMabs_-3.0P0_0.01.cfg.txt'])
axes_01.plot(MabsSandT3H2O, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSandT2H2O = df(Temp,columns=['','TempSandH2OMabs_-2.0P0_0.01.cfg.txt'])
axes_01.plot(MabsSandT2H2O, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSandT1H2O = df(Temp,columns=['','TempSandH2OMabs_-1.0P0_0.01.cfg.txt'])
axes_01.plot(MabsSandT1H2O, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSandT01H2O = df(Temp,columns=['','TempSandH2OMabs_-0.01P0_0.01.cfg.txt'])
axes_01.plot(MabsSandT01H2O, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.text(0.25,0.58,'($H_{2}O$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.5)
axes_01.set_yscale("log")
# all done
# this is quadrant 2, formally quadrant 3 [CO2 Cold]
axes_10 = plt.subplot(G[0,1])  
# M_abs
# CO2 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_10.plot(ColdSand_Pure[:,0],ColdSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-7.0P0_0.01.cfg.txt'])
axes_10.plot(MabsSandT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSandT6CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-6.0P0_0.01.cfg.txt'])
axes_10.plot(MabsSandT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSandT5CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-5.0P0_0.01.cfg.txt'])
axes_10.plot(MabsSandT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSandT4CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-4.0P0_0.01.cfg.txt'])
axes_10.plot(MabsSandT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSandT3CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-3.0P0_0.01.cfg.txt'])
axes_10.plot(MabsSandT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSandT2CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-2.0P0_0.01.cfg.txt'])
axes_10.plot(MabsSandT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSandT1CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-1.0P0_0.01.cfg.txt'])
axes_10.plot(MabsSandT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSandT01CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-0.01P0_0.01.cfg.txt'])
axes_10.plot(MabsSandT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_10.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_10.text(0.25,0.58,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_10.set_ylim(0,1)
axes_10.set_xlim(0.2,2.5)
axes_10.set_yscale("log")
axes_10.title.set_text('log($M_{abs}$) Sand Cold Regime @ ($P_{0}$)= 0.01  bars')
#
# this is quadrant 4, [CH4 Cold]
axes_11 = plt.subplot(G[1,1]) # Sand Block - Right 
# M_abs
# CH4 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_11.plot(ColdSand_Pure[:,0],ColdSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-7.0P0_0.01.cfg.txt'])
axes_11.plot(MabsSandT7CH4, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSandT6CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-6.0P0_0.01.cfg.txt'])
axes_11.plot(MabsSandT6CH4, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSandT5CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-5.0P0_0.01.cfg.txt'])
axes_11.plot(MabsSandT5CH4, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSandT4CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-4.0P0_0.01.cfg.txt'])
axes_11.plot(MabsSandT4CH4, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSandT3CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-3.0P0_0.01.cfg.txt'])
axes_11.plot(MabsSandT3CH4, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSandT2CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-2.0P0_0.01.cfg.txt'])
axes_11.plot(MabsSandT2CH4, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSandT1CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-1.0P0_0.01.cfg.txt'])
axes_11.plot(MabsSandT1CH4, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSandT01CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-0.01P0_0.01.cfg.txt'])
axes_11.plot(MabsSandT01CH4, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_11.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_11.text(0.25,0.01,'($CH_{4}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_11.set_ylim(0,1)
axes_11.set_xlim(0.2,2.5)
axes_11.set_yscale("log")
#
axes_02 = plt.subplot(G[0,2])
axes_02.plot(HotSand_Pure[:,0],HotSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7CO2 = df(Hot,columns=['','HotSandCO2Mabs_-7.0P0_0.01.cfg.txt'])
axes_02.plot(MabsSandT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSandT6CO2 = df(Hot,columns=['','HotSandCO2Mabs_-6.0P0_0.01.cfg.txt'])
axes_02.plot(MabsSandT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSandT5CO2 = df(Hot,columns=['','HotSandCO2Mabs_-5.0P0_0.01.cfg.txt'])
axes_02.plot(MabsSandT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSandT4CO2 = df(Hot,columns=['','HotSandCO2Mabs_-4.0P0_0.01.cfg.txt'])
axes_02.plot(MabsSandT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSandT3CO2 = df(Hot,columns=['','HotSandCO2Mabs_-3.0P0_0.01.cfg.txt'])
axes_02.plot(MabsSandT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSandT2CO2 = df(Hot,columns=['','HotSandCO2Mabs_-2.0P0_0.01.cfg.txt'])
axes_02.plot(MabsSandT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSandT1CO2 = df(Hot,columns=['','HotSandCO2Mabs_-1.0P0_0.01.cfg.txt'])
axes_02.plot(MabsSandT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSandT01CO2 = df(Hot,columns=['','HotSandCO2Mabs_-0.01P0_0.01.cfg.txt'])
axes_02.plot(MabsSandT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_02.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_02.text(0.25,0.58,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
axes_02.set_yscale("log")
axes_02.title.set_text('log($M_{abs}$) Sand Hot Regime @ ($P_{0}$)= 0.01  bars')
plt.grid(True)
#
axes_12 = plt.subplot(G[1,2])
axes_12.plot(HotSand_Pure[:,0],HotSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7SO2 = df(Hot,columns=['','HotSandSO2Mabs_-7.0P0_0.01.cfg.txt'])
axes_12.plot(MabsSandT7SO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSandT6SO2 = df(Hot,columns=['','HotSandSO2Mabs_-6.0P0_0.01.cfg.txt'])
axes_12.plot(MabsSandT6SO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSandT5SO2 = df(Hot,columns=['','HotSandSO2Mabs_-5.0P0_0.01.cfg.txt'])
axes_12.plot(MabsSandT5SO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSandT4SO2 = df(Hot,columns=['','HotSandSO2Mabs_-4.0P0_0.01.cfg.txt'])
axes_12.plot(MabsSandT4SO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSandT3SO2 = df(Hot,columns=['','HotSandSO2Mabs_-3.0P0_0.01.cfg.txt'])
axes_12.plot(MabsSandT3SO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSandT2SO2 = df(Hot,columns=['','HotSandSO2Mabs_-2.0P0_0.01.cfg.txt'])
axes_12.plot(MabsSandT2SO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSandT1SO2 = df(Hot,columns=['','HotSandSO2Mabs_-1.0P0_0.01.cfg.txt'])
axes_12.plot(MabsSandT1SO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSandT01SO2 = df(Hot,columns=['','HotSandSO2Mabs_-0.01P0_0.01.cfg.txt'])
axes_12.plot(MabsSandT01SO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_12.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_12.text(2,0.0006,'($SO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
axes_12.set_yscale("log")
G.tight_layout(fig)
print('all done')
plt.show()
#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs/ParamsComparisons/Sand"
if not os.path.exists(dst):
    os.makedirs(dst)

fig.savefig("CruzArceFigs/ParamsComparisons/Sand/Sand_M_abs_P0_001")


# ###### lastly p0 = -3.0 = 0.001 bars

# In[265]:


# p0 = -3.0
#[235]:
fig = plt.figure(figsize=(15, 8.5))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0.0 , 'Wavelength (um)', ha='center', va='bottom', rotation='horizontal')
G = gridspec.GridSpec(2, 3)
G.update( wspace = 0 ) # set the spacing between axes. wspace=0.025,hspace=0.0
# this is quadrant 1, [CO2 Temp]
axes_00 = plt.subplot(G[0,0]) # Sand Block - Right 
# M_abs
# CO2 - Temp 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_00.plot(TempSand_Pure[:,0],TempSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7CO2 = df(Temp,columns=['','TempSandCO2Mabs_-7.0P0_0.001.cfg.txt'])
axes_00.plot(MabsSandT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSandT6CO2 = df(Temp,columns=['','TempSandCO2Mabs_-6.0P0_0.001.cfg.txt'])
axes_00.plot(MabsSandT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSandT5CO2 = df(Temp,columns=['','TempSandCO2Mabs_-5.0P0_0.001.cfg.txt'])
axes_00.plot(MabsSandT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSandT4CO2 = df(Temp,columns=['','TempSandCO2Mabs_-4.0P0_0.001.cfg.txt'])
axes_00.plot(MabsSandT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSandT3CO2 = df(Temp,columns=['','TempSandCO2Mabs_-3.0P0_0.001.cfg.txt'])
axes_00.plot(MabsSandT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSandT2CO2 = df(Temp,columns=['','TempSandCO2Mabs_-2.0P0_0.001.cfg.txt'])
axes_00.plot(MabsSandT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSandT1CO2 = df(Temp,columns=['','TempSandCO2Mabs_-1.0P0_0.001.cfg.txt'])
axes_00.plot(MabsSandT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSandT01CO2 = df(Temp,columns=['','TempSandCO2Mabs_-0.01P0_0.001.cfg.txt'])
axes_00.plot(MabsSandT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.text(0.25,0.58,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.5)
axes_00.set_yscale("log")
axes_00.title.set_text('log($M_{abs}$) Sand Temperate Regime @ ($P_{0}$)= 0.001  bars')
#
# this is quadrant 2, [H2O Temp]
axes_01 = plt.subplot(G[1,0]) # Sand Block - Right 
# M_abs
# H2O - Temp 
axes_01.plot(TempSand_Pure[:,0],TempSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7H2O = df(Temp,columns=['','TempSandH2OMabs_-7.0P0_0.001.cfg.txt'])
axes_01.plot(MabsSandT7H2O, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
MabsSandT6H2O = df(Temp,columns=['','TempSandH2OMabs_-6.0P0_0.001.cfg.txt'])
axes_01.plot(MabsSandT6H2O, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSandT5H2O = df(Temp,columns=['','TempSandH2OMabs_-5.0P0_0.001.cfg.txt'])
axes_01.plot(MabsSandT5H2O, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSandT4H2O = df(Temp,columns=['','TempSandH2OMabs_-4.0P0_0.001.cfg.txt'])
axes_01.plot(MabsSandT4H2O, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSandT3H2O = df(Temp,columns=['','TempSandH2OMabs_-3.0P0_0.001.cfg.txt'])
axes_01.plot(MabsSandT3H2O, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSandT2H2O = df(Temp,columns=['','TempSandH2OMabs_-2.0P0_0.001.cfg.txt'])
axes_01.plot(MabsSandT2H2O, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSandT1H2O = df(Temp,columns=['','TempSandH2OMabs_-1.0P0_0.001.cfg.txt'])
axes_01.plot(MabsSandT1H2O, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSandT01H2O = df(Temp,columns=['','TempSandH2OMabs_-0.01P0_0.001.cfg.txt'])
axes_01.plot(MabsSandT01H2O, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.text(0.25,0.58,'($H_{2}O$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.5)
axes_01.set_yscale("log")
# all done
# this is quadrant 2, formally quadrant 3 [CO2 Cold]
axes_10 = plt.subplot(G[0,1])  
# M_abs
# CO2 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_10.plot(ColdSand_Pure[:,0],ColdSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-7.0P0_0.001.cfg.txt'])
axes_10.plot(MabsSandT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSandT6CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-6.0P0_0.001.cfg.txt'])
axes_10.plot(MabsSandT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSandT5CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-5.0P0_0.001.cfg.txt'])
axes_10.plot(MabsSandT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSandT4CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-4.0P0_0.001.cfg.txt'])
axes_10.plot(MabsSandT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSandT3CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-3.0P0_0.001.cfg.txt'])
axes_10.plot(MabsSandT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSandT2CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-2.0P0_0.001.cfg.txt'])
axes_10.plot(MabsSandT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSandT1CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-1.0P0_0.001.cfg.txt'])
axes_10.plot(MabsSandT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSandT01CO2 = df(Cold,columns=['','ColdSandCO2Mabs_-0.01P0_0.001.cfg.txt'])
axes_10.plot(MabsSandT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_10.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_10.text(0.25,0.58,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_10.set_ylim(0,1)
axes_10.set_xlim(0.2,2.5)
axes_10.set_yscale("log")
axes_10.title.set_text('log($M_{abs}$) Sand Cold Regime @ ($P_{0}$)= 0.001  bars')
#
# this is quadrant 4, [CH4 Cold]
axes_11 = plt.subplot(G[1,1]) # Sand Block - Right 
# M_abs
# CH4 - Cold 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_11.plot(ColdSand_Pure[:,0],ColdSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-7.0P0_0.001.cfg.txt'])
axes_11.plot(MabsSandT7CH4, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSandT6CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-6.0P0_0.001.cfg.txt'])
axes_11.plot(MabsSandT6CH4, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSandT5CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-5.0P0_0.001.cfg.txt'])
axes_11.plot(MabsSandT5CH4, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSandT4CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-4.0P0_0.001.cfg.txt'])
axes_11.plot(MabsSandT4CH4, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSandT3CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-3.0P0_0.001.cfg.txt'])
axes_11.plot(MabsSandT3CH4, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSandT2CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-2.0P0_0.001.cfg.txt'])
axes_11.plot(MabsSandT2CH4, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSandT1CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-1.0P0_0.001.cfg.txt'])
axes_11.plot(MabsSandT1CH4, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSandT01CH4 = df(Cold,columns=['','ColdSandCH4Mabs_-0.01P0_0.001.cfg.txt'])
axes_11.plot(MabsSandT01CH4, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_11.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_11.text(0.25,0.58,'($CH_{4}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_11.set_ylim(0,1)
axes_11.set_xlim(0.2,2.5)
axes_11.set_yscale("log")
#
axes_02 = plt.subplot(G[0,2])
axes_02.plot(HotSand_Pure[:,0],HotSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7CO2 = df(Hot,columns=['','HotSandCO2Mabs_-7.0P0_0.001.cfg.txt'])
axes_02.plot(MabsSandT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSandT6CO2 = df(Hot,columns=['','HotSandCO2Mabs_-6.0P0_0.001.cfg.txt'])
axes_02.plot(MabsSandT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSandT5CO2 = df(Hot,columns=['','HotSandCO2Mabs_-5.0P0_0.001.cfg.txt'])
axes_02.plot(MabsSandT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSandT4CO2 = df(Hot,columns=['','HotSandCO2Mabs_-4.0P0_0.001.cfg.txt'])
axes_02.plot(MabsSandT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSandT3CO2 = df(Hot,columns=['','HotSandCO2Mabs_-3.0P0_0.001.cfg.txt'])
axes_02.plot(MabsSandT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSandT2CO2 = df(Hot,columns=['','HotSandCO2Mabs_-2.0P0_0.001.cfg.txt'])
axes_02.plot(MabsSandT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSandT1CO2 = df(Hot,columns=['','HotSandCO2Mabs_-1.0P0_0.001.cfg.txt'])
axes_02.plot(MabsSandT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSandT01CO2 = df(Hot,columns=['','HotSandCO2Mabs_-0.01P0_0.001.cfg.txt'])
axes_02.plot(MabsSandT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_02.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_02.text(0.25,0.58,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
axes_02.set_yscale("log")
axes_02.title.set_text('log($M_{abs}$) Sand Hot Regime @ ($P_{0}$)= 0.001  bars')
plt.grid(True)
#
axes_12 = plt.subplot(G[1,2])
axes_12.plot(HotSand_Pure[:,0],HotSand_Pure[:,1], label = 'Sand Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSandT7SO2 = df(Hot,columns=['','HotSandSO2Mabs_-7.0P0_0.001.cfg.txt'])
axes_12.plot(MabsSandT7SO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSandT6SO2 = df(Hot,columns=['','HotSandSO2Mabs_-6.0P0_0.001.cfg.txt'])
axes_12.plot(MabsSandT6SO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSandT5SO2 = df(Hot,columns=['','HotSandSO2Mabs_-5.0P0_0.001.cfg.txt'])
axes_12.plot(MabsSandT5SO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSandT4SO2 = df(Hot,columns=['','HotSandSO2Mabs_-4.0P0_0.001.cfg.txt'])
axes_12.plot(MabsSandT4SO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSandT3SO2 = df(Hot,columns=['','HotSandSO2Mabs_-3.0P0_0.001.cfg.txt'])
axes_12.plot(MabsSandT3SO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSandT2SO2 = df(Hot,columns=['','HotSandSO2Mabs_-2.0P0_0.001.cfg.txt'])
axes_12.plot(MabsSandT2SO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSandT1SO2 = df(Hot,columns=['','HotSandSO2Mabs_-1.0P0_0.001.cfg.txt'])
axes_12.plot(MabsSandT1SO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSandT01SO2 = df(Hot,columns=['','HotSandSO2Mabs_-0.01P0_0.001.cfg.txt'])
axes_12.plot(MabsSandT01SO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_12.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_12.text(2,0.0006,'($SO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
axes_12.set_yscale("log")
G.tight_layout(fig)
print('all done')
plt.show()
#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs/ParamsComparisons/Sand"
if not os.path.exists(dst):
    os.makedirs(dst)

fig.savefig("CruzArceFigs/ParamsComparisons/Sand/Sand_M_abs_P0_0001")


# #### Seawater

# In[ ]:


#Seawater - m_abs for all p0
#Starting from the top. # p0 = 3.0 = 1000 bars
#[218]:
fig = plt.figure(figsize=(12, 8))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0.0 , 'Wavelength (um)', ha='center', va='bottom', rotation='horizontal')
G = gridspec.GridSpec(2, 1)
G.update( wspace = 0 ) # set the spacing between axes. wspace=0.025,hspace=0.0
# this is quadrant 1, [CO2 Temp]
axes_00 = plt.subplot(G[0,:]) # Seawater Block - Right 
# M_abs
# CO2 - Temp 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_00.plot(TempSeawater_Pure[:,0],TempSeawater_Pure[:,1], label = 'Seawater Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSeawaterT7CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-7.0P0_1000.0.cfg.txt'])
axes_00.plot(MabsSeawaterT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSeawaterT6CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-6.0P0_1000.0.cfg.txt'])
axes_00.plot(MabsSeawaterT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSeawaterT5CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-5.0P0_1000.0.cfg.txt'])
axes_00.plot(MabsSeawaterT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSeawaterT4CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-4.0P0_1000.0.cfg.txt'])
axes_00.plot(MabsSeawaterT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSeawaterT3CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-3.0P0_1000.0.cfg.txt'])
axes_00.plot(MabsSeawaterT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSeawaterT2CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-2.0P0_1000.0.cfg.txt'])
axes_00.plot(MabsSeawaterT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSeawaterT1CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-1.0P0_1000.0.cfg.txt'])
axes_00.plot(MabsSeawaterT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSeawaterT01CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-0.01P0_1000.0.cfg.txt'])
axes_00.plot(MabsSeawaterT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.text(0.25,0.5,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.5)
axes_00.set_yscale("log")
axes_00.title.set_text('log($M_{abs}$) Seawater Temperate Regime @ ($P_{0}$) = 1000 bars')
#
# this is quadrant 2, [H2O Temp]
axes_01 = plt.subplot(G[1,:]) # Seawater Block - Right 
# M_abs
# H2O - Temp 
axes_01.plot(TempSeawater_Pure[:,0],TempSeawater_Pure[:,1], label = 'Seawater Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSeawaterT7H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-7.0P0_1000.0.cfg.txt'])
axes_01.plot(MabsSeawaterT7H2O, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
MabsSeawaterT6H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-6.0P0_1000.0.cfg.txt'])
axes_01.plot(MabsSeawaterT6H2O, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSeawaterT5H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-5.0P0_1000.0.cfg.txt'])
axes_01.plot(MabsSeawaterT5H2O, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSeawaterT4H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-4.0P0_1000.0.cfg.txt'])
axes_01.plot(MabsSeawaterT4H2O, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSeawaterT3H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-3.0P0_1000.0.cfg.txt'])
axes_01.plot(MabsSeawaterT3H2O, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSeawaterT2H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-2.0P0_1000.0.cfg.txt'])
axes_01.plot(MabsSeawaterT2H2O, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSeawaterT1H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-1.0P0_1000.0.cfg.txt'])
axes_01.plot(MabsSeawaterT1H2O, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSeawaterT01H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-0.01P0_1000.0.cfg.txt'])
axes_01.plot(MabsSeawaterT01H2O, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.text(0.25,0.01,'($H_{2}O$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.5)
axes_01.set_yscale("log")
# all done
G.tight_layout(fig)
print('all done')
plt.show()
#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs/ParamsComparisons/Seawater"
if not os.path.exists(dst):
    os.makedirs(dst)

fig.savefig("CruzArceFigs/ParamsComparisons/Seawater/Seawater_M_abs_P0_1000")

# # p0 = 2.0 = 100 bars
# [219]:
fig = plt.figure(figsize=((12, 8)))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0.0 , 'Wavelength (um)', ha='center', va='bottom', rotation='horizontal')
G = gridspec.GridSpec(2, 1)
G.update( wspace = 0 ) # set the spacing between axes. wspace=0.025,hspace=0.0
# this is quadrant 1, [CO2 Temp]
axes_00 = plt.subplot(G[0, :]) # Seawater Block - Right 
# M_abs
# CO2 - Temp 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_00.plot(TempSeawater_Pure[:,0],TempSeawater_Pure[:,1], label = 'Seawater Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSeawaterT7CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-7.0P0_100.0.cfg.txt'])
axes_00.plot(MabsSeawaterT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSeawaterT6CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-6.0P0_100.0.cfg.txt'])
axes_00.plot(MabsSeawaterT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSeawaterT5CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-5.0P0_100.0.cfg.txt'])
axes_00.plot(MabsSeawaterT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSeawaterT4CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-4.0P0_100.0.cfg.txt'])
axes_00.plot(MabsSeawaterT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSeawaterT3CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-3.0P0_100.0.cfg.txt'])
axes_00.plot(MabsSeawaterT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSeawaterT2CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-2.0P0_100.0.cfg.txt'])
axes_00.plot(MabsSeawaterT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSeawaterT1CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-1.0P0_100.0.cfg.txt'])
axes_00.plot(MabsSeawaterT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSeawaterT01CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-0.01P0_100.0.cfg.txt'])
axes_00.plot(MabsSeawaterT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.text(0.25,0.5,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.5)
axes_00.set_yscale("log")
axes_00.title.set_text('log($M_{abs}$) Seawater Temperate Regime @ ($P_{0}$) = 100 bars')
#
# this is quadrant 2, [H2O Temp]
axes_01 = plt.subplot(G[1, :]) # Seawater Block - Right 
# M_abs
# H2O - Temp 
axes_01.plot(TempSeawater_Pure[:,0],TempSeawater_Pure[:,1], label = 'Seawater Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSeawaterT7H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-7.0P0_100.0.cfg.txt'])
axes_01.plot(MabsSeawaterT7H2O, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
MabsSeawaterT6H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-6.0P0_100.0.cfg.txt'])
axes_01.plot(MabsSeawaterT6H2O, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSeawaterT5H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-5.0P0_100.0.cfg.txt'])
axes_01.plot(MabsSeawaterT5H2O, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSeawaterT4H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-4.0P0_100.0.cfg.txt'])
axes_01.plot(MabsSeawaterT4H2O, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSeawaterT3H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-3.0P0_100.0.cfg.txt'])
axes_01.plot(MabsSeawaterT3H2O, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSeawaterT2H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-2.0P0_100.0.cfg.txt'])
axes_01.plot(MabsSeawaterT2H2O, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSeawaterT1H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-1.0P0_100.0.cfg.txt'])
axes_01.plot(MabsSeawaterT1H2O, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSeawaterT01H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-0.01P0_100.0.cfg.txt'])
axes_01.plot(MabsSeawaterT01H2O, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.text(0.25,0.01,'($H_{2}O$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.5)
axes_01.set_yscale("log")
# all done
G.tight_layout(fig)
print('all done')
plt.show()
#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs/ParamsComparisons/Seawater"
if not os.path.exists(dst):
    os.makedirs(dst)

fig.savefig("CruzArceFigs/ParamsComparisons/Seawater/Seawater_M_abs_P0_100")

# # p0 = 1.0 = 10 bars
# [222]:
fig = plt.figure(figsize=((12, 8)))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0.0 , 'Wavelength (um)', ha='center', va='bottom', rotation='horizontal')
G = gridspec.GridSpec(2, 1)
G.update( wspace = 0 ) # set the spacing between axes. wspace=0.025,hspace=0.0
# this is quadrant 1, [CO2 Temp]
axes_00 = plt.subplot(G[0, :]) # Seawater Block - Right 
# M_abs
# CO2 - Temp 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_00.plot(TempSeawater_Pure[:,0],TempSeawater_Pure[:,1], label = 'Seawater Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSeawaterT7CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-7.0P0_10.0.cfg.txt'])
axes_00.plot(MabsSeawaterT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSeawaterT6CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-6.0P0_10.0.cfg.txt'])
axes_00.plot(MabsSeawaterT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSeawaterT5CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-5.0P0_10.0.cfg.txt'])
axes_00.plot(MabsSeawaterT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSeawaterT4CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-4.0P0_10.0.cfg.txt'])
axes_00.plot(MabsSeawaterT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSeawaterT3CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-3.0P0_10.0.cfg.txt'])
axes_00.plot(MabsSeawaterT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSeawaterT2CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-2.0P0_10.0.cfg.txt'])
axes_00.plot(MabsSeawaterT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSeawaterT1CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-1.0P0_10.0.cfg.txt'])
axes_00.plot(MabsSeawaterT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSeawaterT01CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-0.01P0_10.0.cfg.txt'])
axes_00.plot(MabsSeawaterT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.text(0.25,0.5,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.5)
axes_00.set_yscale("log")
axes_00.title.set_text('log($M_{abs}$) Seawater Temperate Regime @ ($P_{0}$) = 10bars')
#
# this is quadrant 2, [H2O Temp]
axes_01 = plt.subplot(G[1, :]) # Seawater Block - Right 
# M_abs
# H2O - Temp 
axes_01.plot(TempSeawater_Pure[:,0],TempSeawater_Pure[:,1], label = 'Seawater Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSeawaterT7H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-7.0P0_10.0.cfg.txt'])
axes_01.plot(MabsSeawaterT7H2O, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
MabsSeawaterT6H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-6.0P0_10.0.cfg.txt'])
axes_01.plot(MabsSeawaterT6H2O, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSeawaterT5H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-5.0P0_10.0.cfg.txt'])
axes_01.plot(MabsSeawaterT5H2O, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSeawaterT4H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-4.0P0_10.0.cfg.txt'])
axes_01.plot(MabsSeawaterT4H2O, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSeawaterT3H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-3.0P0_10.0.cfg.txt'])
axes_01.plot(MabsSeawaterT3H2O, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSeawaterT2H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-2.0P0_10.0.cfg.txt'])
axes_01.plot(MabsSeawaterT2H2O, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSeawaterT1H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-1.0P0_10.0.cfg.txt'])
axes_01.plot(MabsSeawaterT1H2O, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSeawaterT01H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-0.01P0_10.0.cfg.txt'])
axes_01.plot(MabsSeawaterT01H2O, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.text(0.25,0.01,'($H_{2}O$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.5)
axes_01.set_yscale("log")
# all done
G.tight_layout(fig)
print('all done')
plt.show()
#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs/ParamsComparisons/Seawater"
if not os.path.exists(dst):
    os.makedirs(dst)

fig.savefig("CruzArceFigs/ParamsComparisons/Seawater/Seawater_M_abs_P0_10")

# p0 = 0.0
#
fig = plt.figure(figsize=((12, 8)))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0.0 , 'Wavelength (um)', ha='center', va='bottom', rotation='horizontal')
G = gridspec.GridSpec(2, 1)
G.update( wspace = 0 ) # set the spacing between axes. wspace=0.025,hspace=0.0
# this is quadrant 1, [CO2 Temp]
axes_00 = plt.subplot(G[0, :]) # Seawater Block - Right 
# M_abs
# CO2 - Temp 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_00.plot(TempSeawater_Pure[:,0],TempSeawater_Pure[:,1], label = 'Seawater Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSeawaterT7CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-7.0P0_1.0.cfg.txt'])
axes_00.plot(MabsSeawaterT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSeawaterT6CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-6.0P0_1.0.cfg.txt'])
axes_00.plot(MabsSeawaterT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSeawaterT5CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-5.0P0_1.0.cfg.txt'])
axes_00.plot(MabsSeawaterT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSeawaterT4CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-4.0P0_1.0.cfg.txt'])
axes_00.plot(MabsSeawaterT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSeawaterT3CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-3.0P0_1.0.cfg.txt'])
axes_00.plot(MabsSeawaterT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSeawaterT2CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_00.plot(MabsSeawaterT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSeawaterT1CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-1.0P0_1.0.cfg.txt'])
axes_00.plot(MabsSeawaterT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSeawaterT01CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-0.01P0_1.0.cfg.txt'])
axes_00.plot(MabsSeawaterT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.text(0.25,0.5,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.5)
axes_00.set_yscale("log")
axes_00.title.set_text('log($M_{abs}$) Seawater Temperate Regime @ ($P_{0}$) = 1 bar')
#
# this is quadrant 2, [H2O Temp]
axes_01 = plt.subplot(G[1, :]) # Seawater Block - Right 
# M_abs
# H2O - Temp 
axes_01.plot(TempSeawater_Pure[:,0],TempSeawater_Pure[:,1], label = 'Seawater Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSeawaterT7H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-7.0P0_1.0.cfg.txt'])
axes_01.plot(MabsSeawaterT7H2O, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
MabsSeawaterT6H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-6.0P0_1.0.cfg.txt'])
axes_01.plot(MabsSeawaterT6H2O, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSeawaterT5H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-5.0P0_1.0.cfg.txt'])
axes_01.plot(MabsSeawaterT5H2O, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSeawaterT4H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-4.0P0_1.0.cfg.txt'])
axes_01.plot(MabsSeawaterT4H2O, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSeawaterT3H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-3.0P0_1.0.cfg.txt'])
axes_01.plot(MabsSeawaterT3H2O, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSeawaterT2H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-2.0P0_1.0.cfg.txt'])
axes_01.plot(MabsSeawaterT2H2O, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSeawaterT1H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-1.0P0_1.0.cfg.txt'])
axes_01.plot(MabsSeawaterT1H2O, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSeawaterT01H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-0.01P0_1.0.cfg.txt'])
axes_01.plot(MabsSeawaterT01H2O, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.text(0.25,0.01,'($H_{2}O$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.5)
axes_01.set_yscale("log")
# all done
# this is quadrant 2, formally quadrant 3 [CO2 ]

G.tight_layout(fig)
print('all done')
plt.show()
#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs/ParamsComparisons/Seawater"
if not os.path.exists(dst):
    os.makedirs(dst)

fig.savefig("CruzArceFigs/ParamsComparisons/Seawater/Seawater_M_abs_P0_1")

# p0 = -1.0 = 0.1 bars
#[229]:
fig = plt.figure(figsize=((12, 8)))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0.0 , 'Wavelength (um)', ha='center', va='bottom', rotation='horizontal')
G = gridspec.GridSpec(2, 1)
G.update( wspace = 0 ) # set the spacing between axes. wspace=0.025,hspace=0.0
# this is quadrant 1, [CO2 Temp]
axes_00 = plt.subplot(G[0, :]) # Seawater Block - Right 
# M_abs
# CO2 - Temp 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_00.plot(TempSeawater_Pure[:,0],TempSeawater_Pure[:,1], label = 'Seawater Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSeawaterT7CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-7.0P0_0.1.cfg.txt'])
axes_00.plot(MabsSeawaterT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSeawaterT6CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-6.0P0_0.1.cfg.txt'])
axes_00.plot(MabsSeawaterT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSeawaterT5CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-5.0P0_0.1.cfg.txt'])
axes_00.plot(MabsSeawaterT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSeawaterT4CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-4.0P0_0.1.cfg.txt'])
axes_00.plot(MabsSeawaterT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSeawaterT3CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-3.0P0_0.1.cfg.txt'])
axes_00.plot(MabsSeawaterT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSeawaterT2CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-2.0P0_0.1.cfg.txt'])
axes_00.plot(MabsSeawaterT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSeawaterT1CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-1.0P0_0.1.cfg.txt'])
axes_00.plot(MabsSeawaterT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSeawaterT01CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-0.01P0_0.1.cfg.txt'])
axes_00.plot(MabsSeawaterT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.text(0.25,0.2,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.5)
axes_00.set_yscale("log")
axes_00.title.set_text('log($M_{abs}$) Seawater Temperate Regime @ ($P_{0}$)= 0.1  bars')
#
# this is quadrant 2, [H2O Temp]
axes_01 = plt.subplot(G[1, :]) # Seawater Block - Right 
# M_abs
# H2O - Temp 
axes_01.plot(TempSeawater_Pure[:,0],TempSeawater_Pure[:,1], label = 'Seawater Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSeawaterT7H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-7.0P0_0.1.cfg.txt'])
axes_01.plot(MabsSeawaterT7H2O, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
MabsSeawaterT6H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-6.0P0_0.1.cfg.txt'])
axes_01.plot(MabsSeawaterT6H2O, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSeawaterT5H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-5.0P0_0.1.cfg.txt'])
axes_01.plot(MabsSeawaterT5H2O, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSeawaterT4H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-4.0P0_0.1.cfg.txt'])
axes_01.plot(MabsSeawaterT4H2O, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSeawaterT3H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-3.0P0_0.1.cfg.txt'])
axes_01.plot(MabsSeawaterT3H2O, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSeawaterT2H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-2.0P0_0.1.cfg.txt'])
axes_01.plot(MabsSeawaterT2H2O, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSeawaterT1H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-1.0P0_0.1.cfg.txt'])
axes_01.plot(MabsSeawaterT1H2O, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSeawaterT01H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-0.01P0_0.1.cfg.txt'])
axes_01.plot(MabsSeawaterT01H2O, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.text(0.25,0.01,'($H_{2}O$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.5)
axes_01.set_yscale("log")
# all done
G.tight_layout(fig)
print('all done')
plt.show()
#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs/ParamsComparisons/Seawater"
if not os.path.exists(dst):
    os.makedirs(dst)

fig.savefig("CruzArceFigs/ParamsComparisons/Seawater/Seawater_M_abs_P0_01")

# p0 = -2.0¶
#[231]:
fig = plt.figure(figsize=((12, 8)))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0.0 , 'Wavelength (um)', ha='center', va='bottom', rotation='horizontal')
G = gridspec.GridSpec(2, 1)
G.update( wspace = 0 ) # set the spacing between axes. wspace=0.025,hspace=0.0
# this is quadrant 1, [CO2 Temp]
axes_00 = plt.subplot(G[0, :]) # Seawater Block - Right 
# M_abs
# CO2 - Temp 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_00.plot(TempSeawater_Pure[:,0],TempSeawater_Pure[:,1], label = 'Seawater Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSeawaterT7CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-7.0P0_0.01.cfg.txt'])
axes_00.plot(MabsSeawaterT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSeawaterT6CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-6.0P0_0.01.cfg.txt'])
axes_00.plot(MabsSeawaterT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSeawaterT5CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-5.0P0_0.01.cfg.txt'])
axes_00.plot(MabsSeawaterT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSeawaterT4CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-4.0P0_0.01.cfg.txt'])
axes_00.plot(MabsSeawaterT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSeawaterT3CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-3.0P0_0.01.cfg.txt'])
axes_00.plot(MabsSeawaterT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSeawaterT2CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-2.0P0_0.01.cfg.txt'])
axes_00.plot(MabsSeawaterT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSeawaterT1CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-1.0P0_0.01.cfg.txt'])
axes_00.plot(MabsSeawaterT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSeawaterT01CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-0.01P0_0.01.cfg.txt'])
axes_00.plot(MabsSeawaterT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.text(0.25,0.01,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.5)
axes_00.set_yscale("log")
axes_00.title.set_text('log($M_{abs}$) Seawater Temperate Regime @ ($P_{0}$)= 0.01  bars')
#
# this is quadrant 2, [H2O Temp]
axes_01 = plt.subplot(G[1, :]) # Seawater Block - Right 
# M_abs
# H2O - Temp 
axes_01.plot(TempSeawater_Pure[:,0],TempSeawater_Pure[:,1], label = 'Seawater Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSeawaterT7H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-7.0P0_0.01.cfg.txt'])
axes_01.plot(MabsSeawaterT7H2O, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
MabsSeawaterT6H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-6.0P0_0.01.cfg.txt'])
axes_01.plot(MabsSeawaterT6H2O, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSeawaterT5H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-5.0P0_0.01.cfg.txt'])
axes_01.plot(MabsSeawaterT5H2O, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSeawaterT4H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-4.0P0_0.01.cfg.txt'])
axes_01.plot(MabsSeawaterT4H2O, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSeawaterT3H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-3.0P0_0.01.cfg.txt'])
axes_01.plot(MabsSeawaterT3H2O, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSeawaterT2H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-2.0P0_0.01.cfg.txt'])
axes_01.plot(MabsSeawaterT2H2O, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSeawaterT1H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-1.0P0_0.01.cfg.txt'])
axes_01.plot(MabsSeawaterT1H2O, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSeawaterT01H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-0.01P0_0.01.cfg.txt'])
axes_01.plot(MabsSeawaterT01H2O, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.text(0.25,0.01,'($H_{2}O$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.5)
axes_01.set_yscale("log")
# all done
G.tight_layout(fig)
print('all done')
plt.show()
#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs/ParamsComparisons/Seawater"
if not os.path.exists(dst):
    os.makedirs(dst)

fig.savefig("CruzArceFigs/ParamsComparisons/Seawater/Seawater_M_abs_P0_001")

# p0 = -3.0
#[235]:
fig = plt.figure(figsize=((12, 8)))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0.0 , 'Wavelength (um)', ha='center', va='bottom', rotation='horizontal')
G = gridspec.GridSpec(2, 1)
G.update( wspace = 0 ) # set the spacing between axes. wspace=0.025,hspace=0.0
# this is quadrant 1, [CO2 Temp]
axes_00 = plt.subplot(G[0, :]) # Seawater Block - Right 
# M_abs
# CO2 - Temp 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_00.plot(TempSeawater_Pure[:,0],TempSeawater_Pure[:,1], label = 'Seawater Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSeawaterT7CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-7.0P0_0.001.cfg.txt'])
axes_00.plot(MabsSeawaterT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSeawaterT6CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-6.0P0_0.001.cfg.txt'])
axes_00.plot(MabsSeawaterT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSeawaterT5CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-5.0P0_0.001.cfg.txt'])
axes_00.plot(MabsSeawaterT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSeawaterT4CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-4.0P0_0.001.cfg.txt'])
axes_00.plot(MabsSeawaterT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSeawaterT3CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-3.0P0_0.001.cfg.txt'])
axes_00.plot(MabsSeawaterT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSeawaterT2CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-2.0P0_0.001.cfg.txt'])
axes_00.plot(MabsSeawaterT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSeawaterT1CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-1.0P0_0.001.cfg.txt'])
axes_00.plot(MabsSeawaterT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSeawaterT01CO2 = df(Temp,columns=['','TempSeawaterCO2Mabs_-0.01P0_0.001.cfg.txt'])
axes_00.plot(MabsSeawaterT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.text(0.25,0.01,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.5)
axes_00.set_yscale("log")
axes_00.title.set_text('log($M_{abs}$) Seawater Temperate Regime @ ($P_{0}$)= 0.001  bars')
#
# this is quadrant 2, [H2O Temp]
axes_01 = plt.subplot(G[1, :]) # Seawater Block - Right 
# M_abs
# H2O - Temp 
axes_01.plot(TempSeawater_Pure[:,0],TempSeawater_Pure[:,1], label = 'Seawater Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSeawaterT7H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-7.0P0_0.001.cfg.txt'])
axes_01.plot(MabsSeawaterT7H2O, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
MabsSeawaterT6H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-6.0P0_0.001.cfg.txt'])
axes_01.plot(MabsSeawaterT6H2O, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSeawaterT5H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-5.0P0_0.001.cfg.txt'])
axes_01.plot(MabsSeawaterT5H2O, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSeawaterT4H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-4.0P0_0.001.cfg.txt'])
axes_01.plot(MabsSeawaterT4H2O, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSeawaterT3H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-3.0P0_0.001.cfg.txt'])
axes_01.plot(MabsSeawaterT3H2O, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSeawaterT2H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-2.0P0_0.001.cfg.txt'])
axes_01.plot(MabsSeawaterT2H2O, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSeawaterT1H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-1.0P0_0.001.cfg.txt'])
axes_01.plot(MabsSeawaterT1H2O, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSeawaterT01H2O = df(Temp,columns=['','TempSeawaterH2OMabs_-0.01P0_0.001.cfg.txt'])
axes_01.plot(MabsSeawaterT01H2O, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.text(0.25,0.01,'($H_{2}O$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.5)
axes_01.set_yscale("log")
# all done
G.tight_layout(fig)
print('all done')
plt.show()
#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs/ParamsComparisons/Seawater"
if not os.path.exists(dst):
    os.makedirs(dst)

fig.savefig("CruzArceFigs/ParamsComparisons/Seawater/Seawater_M_abs_P0_0001")


# #### Grass

# In[ ]:


#Grass - m_abs for all p0
#Starting from the top. # p0 = 3.0 = 1000 bars
#[218]:
fig = plt.figure(figsize=(12, 8))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0.0 , 'Wavelength (um)', ha='center', va='bottom', rotation='horizontal')
G = gridspec.GridSpec(2, 1)
G.update( wspace = 0 ) # set the spacing between axes. wspace=0.025,hspace=0.0
# this is quadrant 1, [CO2 Temp]
axes_00 = plt.subplot(G[0,:]) # Grass Block - Right 
# M_abs
# CO2 - Temp 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_00.plot(TempGrass_Pure[:,0],TempGrass_Pure[:,1], label = 'Grass Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsGrassT7CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-7.0P0_1000.0.cfg.txt'])
axes_00.plot(MabsGrassT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsGrassT6CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-6.0P0_1000.0.cfg.txt'])
axes_00.plot(MabsGrassT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsGrassT5CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-5.0P0_1000.0.cfg.txt'])
axes_00.plot(MabsGrassT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsGrassT4CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-4.0P0_1000.0.cfg.txt'])
axes_00.plot(MabsGrassT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsGrassT3CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-3.0P0_1000.0.cfg.txt'])
axes_00.plot(MabsGrassT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsGrassT2CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-2.0P0_1000.0.cfg.txt'])
axes_00.plot(MabsGrassT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsGrassT1CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-1.0P0_1000.0.cfg.txt'])
axes_00.plot(MabsGrassT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsGrassT01CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-0.01P0_1000.0.cfg.txt'])
axes_00.plot(MabsGrassT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.text(0.25,0.5,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.5)
axes_00.set_yscale("log")
axes_00.title.set_text('log($M_{abs}$) Grass Temperate Regime @ ($P_{0}$) = 1000 bars')
#
# this is quadrant 2, [H2O Temp]
axes_01 = plt.subplot(G[1,:]) # Grass Block - Right 
# M_abs
# H2O - Temp 
axes_01.plot(TempGrass_Pure[:,0],TempGrass_Pure[:,1], label = 'Grass Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsGrassT7H2O = df(Temp,columns=['','TempGrassH2OMabs_-7.0P0_1000.0.cfg.txt'])
axes_01.plot(MabsGrassT7H2O, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
MabsGrassT6H2O = df(Temp,columns=['','TempGrassH2OMabs_-6.0P0_1000.0.cfg.txt'])
axes_01.plot(MabsGrassT6H2O, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsGrassT5H2O = df(Temp,columns=['','TempGrassH2OMabs_-5.0P0_1000.0.cfg.txt'])
axes_01.plot(MabsGrassT5H2O, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsGrassT4H2O = df(Temp,columns=['','TempGrassH2OMabs_-4.0P0_1000.0.cfg.txt'])
axes_01.plot(MabsGrassT4H2O, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsGrassT3H2O = df(Temp,columns=['','TempGrassH2OMabs_-3.0P0_1000.0.cfg.txt'])
axes_01.plot(MabsGrassT3H2O, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsGrassT2H2O = df(Temp,columns=['','TempGrassH2OMabs_-2.0P0_1000.0.cfg.txt'])
axes_01.plot(MabsGrassT2H2O, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsGrassT1H2O = df(Temp,columns=['','TempGrassH2OMabs_-1.0P0_1000.0.cfg.txt'])
axes_01.plot(MabsGrassT1H2O, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsGrassT01H2O = df(Temp,columns=['','TempGrassH2OMabs_-0.01P0_1000.0.cfg.txt'])
axes_01.plot(MabsGrassT01H2O, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.text(0.25,0.01,'($H_{2}O$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.5)
axes_01.set_yscale("log")
# all done
G.tight_layout(fig)
print('all done')
plt.show()
#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs/ParamsComparisons/Grass"
if not os.path.exists(dst):
    os.makedirs(dst)

fig.savefig("CruzArceFigs/ParamsComparisons/Grass/Grass_M_abs_P0_1000")

# # p0 = 2.0 = 100 bars
# [219]:
fig = plt.figure(figsize=((12, 8)))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0.0 , 'Wavelength (um)', ha='center', va='bottom', rotation='horizontal')
G = gridspec.GridSpec(2, 1)
G.update( wspace = 0 ) # set the spacing between axes. wspace=0.025,hspace=0.0
# this is quadrant 1, [CO2 Temp]
axes_00 = plt.subplot(G[0, :]) # Grass Block - Right 
# M_abs
# CO2 - Temp 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_00.plot(TempGrass_Pure[:,0],TempGrass_Pure[:,1], label = 'Grass Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsGrassT7CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-7.0P0_100.0.cfg.txt'])
axes_00.plot(MabsGrassT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsGrassT6CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-6.0P0_100.0.cfg.txt'])
axes_00.plot(MabsGrassT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsGrassT5CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-5.0P0_100.0.cfg.txt'])
axes_00.plot(MabsGrassT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsGrassT4CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-4.0P0_100.0.cfg.txt'])
axes_00.plot(MabsGrassT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsGrassT3CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-3.0P0_100.0.cfg.txt'])
axes_00.plot(MabsGrassT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsGrassT2CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-2.0P0_100.0.cfg.txt'])
axes_00.plot(MabsGrassT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsGrassT1CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-1.0P0_100.0.cfg.txt'])
axes_00.plot(MabsGrassT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsGrassT01CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-0.01P0_100.0.cfg.txt'])
axes_00.plot(MabsGrassT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.text(0.25,0.5,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.5)
axes_00.set_yscale("log")
axes_00.title.set_text('log($M_{abs}$) Grass Temperate Regime @ ($P_{0}$) = 100 bars')
#
# this is quadrant 2, [H2O Temp]
axes_01 = plt.subplot(G[1, :]) # Grass Block - Right 
# M_abs
# H2O - Temp 
axes_01.plot(TempGrass_Pure[:,0],TempGrass_Pure[:,1], label = 'Grass Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsGrassT7H2O = df(Temp,columns=['','TempGrassH2OMabs_-7.0P0_100.0.cfg.txt'])
axes_01.plot(MabsGrassT7H2O, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
MabsGrassT6H2O = df(Temp,columns=['','TempGrassH2OMabs_-6.0P0_100.0.cfg.txt'])
axes_01.plot(MabsGrassT6H2O, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsGrassT5H2O = df(Temp,columns=['','TempGrassH2OMabs_-5.0P0_100.0.cfg.txt'])
axes_01.plot(MabsGrassT5H2O, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsGrassT4H2O = df(Temp,columns=['','TempGrassH2OMabs_-4.0P0_100.0.cfg.txt'])
axes_01.plot(MabsGrassT4H2O, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsGrassT3H2O = df(Temp,columns=['','TempGrassH2OMabs_-3.0P0_100.0.cfg.txt'])
axes_01.plot(MabsGrassT3H2O, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsGrassT2H2O = df(Temp,columns=['','TempGrassH2OMabs_-2.0P0_100.0.cfg.txt'])
axes_01.plot(MabsGrassT2H2O, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsGrassT1H2O = df(Temp,columns=['','TempGrassH2OMabs_-1.0P0_100.0.cfg.txt'])
axes_01.plot(MabsGrassT1H2O, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsGrassT01H2O = df(Temp,columns=['','TempGrassH2OMabs_-0.01P0_100.0.cfg.txt'])
axes_01.plot(MabsGrassT01H2O, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.text(0.25,0.01,'($H_{2}O$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.5)
axes_01.set_yscale("log")
# all done
G.tight_layout(fig)
print('all done')
plt.show()
#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs/ParamsComparisons/Grass"
if not os.path.exists(dst):
    os.makedirs(dst)

fig.savefig("CruzArceFigs/ParamsComparisons/Grass/Grass_M_abs_P0_100")

# # p0 = 1.0 = 10 bars
# [222]:
fig = plt.figure(figsize=((12, 8)))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0.0 , 'Wavelength (um)', ha='center', va='bottom', rotation='horizontal')
G = gridspec.GridSpec(2, 1)
G.update( wspace = 0 ) # set the spacing between axes. wspace=0.025,hspace=0.0
# this is quadrant 1, [CO2 Temp]
axes_00 = plt.subplot(G[0, :]) # Grass Block - Right 
# M_abs
# CO2 - Temp 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_00.plot(TempGrass_Pure[:,0],TempGrass_Pure[:,1], label = 'Grass Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsGrassT7CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-7.0P0_10.0.cfg.txt'])
axes_00.plot(MabsGrassT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsGrassT6CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-6.0P0_10.0.cfg.txt'])
axes_00.plot(MabsGrassT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsGrassT5CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-5.0P0_10.0.cfg.txt'])
axes_00.plot(MabsGrassT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsGrassT4CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-4.0P0_10.0.cfg.txt'])
axes_00.plot(MabsGrassT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsGrassT3CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-3.0P0_10.0.cfg.txt'])
axes_00.plot(MabsGrassT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsGrassT2CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-2.0P0_10.0.cfg.txt'])
axes_00.plot(MabsGrassT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsGrassT1CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-1.0P0_10.0.cfg.txt'])
axes_00.plot(MabsGrassT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsGrassT01CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-0.01P0_10.0.cfg.txt'])
axes_00.plot(MabsGrassT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.text(0.25,0.5,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.5)
axes_00.set_yscale("log")
axes_00.title.set_text('log($M_{abs}$) Grass Temperate Regime @ ($P_{0}$) = 10bars')
#
# this is quadrant 2, [H2O Temp]
axes_01 = plt.subplot(G[1, :]) # Grass Block - Right 
# M_abs
# H2O - Temp 
axes_01.plot(TempGrass_Pure[:,0],TempGrass_Pure[:,1], label = 'Grass Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsGrassT7H2O = df(Temp,columns=['','TempGrassH2OMabs_-7.0P0_10.0.cfg.txt'])
axes_01.plot(MabsGrassT7H2O, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
MabsGrassT6H2O = df(Temp,columns=['','TempGrassH2OMabs_-6.0P0_10.0.cfg.txt'])
axes_01.plot(MabsGrassT6H2O, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsGrassT5H2O = df(Temp,columns=['','TempGrassH2OMabs_-5.0P0_10.0.cfg.txt'])
axes_01.plot(MabsGrassT5H2O, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsGrassT4H2O = df(Temp,columns=['','TempGrassH2OMabs_-4.0P0_10.0.cfg.txt'])
axes_01.plot(MabsGrassT4H2O, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsGrassT3H2O = df(Temp,columns=['','TempGrassH2OMabs_-3.0P0_10.0.cfg.txt'])
axes_01.plot(MabsGrassT3H2O, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsGrassT2H2O = df(Temp,columns=['','TempGrassH2OMabs_-2.0P0_10.0.cfg.txt'])
axes_01.plot(MabsGrassT2H2O, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsGrassT1H2O = df(Temp,columns=['','TempGrassH2OMabs_-1.0P0_10.0.cfg.txt'])
axes_01.plot(MabsGrassT1H2O, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsGrassT01H2O = df(Temp,columns=['','TempGrassH2OMabs_-0.01P0_10.0.cfg.txt'])
axes_01.plot(MabsGrassT01H2O, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.text(0.25,0.01,'($H_{2}O$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.5)
axes_01.set_yscale("log")
# all done
G.tight_layout(fig)
print('all done')
plt.show()
#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs/ParamsComparisons/Grass"
if not os.path.exists(dst):
    os.makedirs(dst)

fig.savefig("CruzArceFigs/ParamsComparisons/Grass/Grass_M_abs_P0_10")

# p0 = 0.0
#
fig = plt.figure(figsize=((12, 8)))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0.0 , 'Wavelength (um)', ha='center', va='bottom', rotation='horizontal')
G = gridspec.GridSpec(2, 1)
G.update( wspace = 0 ) # set the spacing between axes. wspace=0.025,hspace=0.0
# this is quadrant 1, [CO2 Temp]
axes_00 = plt.subplot(G[0, :]) # Grass Block - Right 
# M_abs
# CO2 - Temp 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_00.plot(TempGrass_Pure[:,0],TempGrass_Pure[:,1], label = 'Grass Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsGrassT7CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-7.0P0_1.0.cfg.txt'])
axes_00.plot(MabsGrassT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsGrassT6CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-6.0P0_1.0.cfg.txt'])
axes_00.plot(MabsGrassT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsGrassT5CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-5.0P0_1.0.cfg.txt'])
axes_00.plot(MabsGrassT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsGrassT4CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-4.0P0_1.0.cfg.txt'])
axes_00.plot(MabsGrassT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsGrassT3CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-3.0P0_1.0.cfg.txt'])
axes_00.plot(MabsGrassT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsGrassT2CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-2.0P0_1.0.cfg.txt'])
axes_00.plot(MabsGrassT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsGrassT1CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-1.0P0_1.0.cfg.txt'])
axes_00.plot(MabsGrassT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsGrassT01CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-0.01P0_1.0.cfg.txt'])
axes_00.plot(MabsGrassT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.text(0.25,0.5,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.5)
axes_00.set_yscale("log")
axes_00.title.set_text('log($M_{abs}$) Grass Temperate Regime @ ($P_{0}$) = 1 bar')
#
# this is quadrant 2, [H2O Temp]
axes_01 = plt.subplot(G[1, :]) # Grass Block - Right 
# M_abs
# H2O - Temp 
axes_01.plot(TempGrass_Pure[:,0],TempGrass_Pure[:,1], label = 'Grass Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsGrassT7H2O = df(Temp,columns=['','TempGrassH2OMabs_-7.0P0_1.0.cfg.txt'])
axes_01.plot(MabsGrassT7H2O, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
MabsGrassT6H2O = df(Temp,columns=['','TempGrassH2OMabs_-6.0P0_1.0.cfg.txt'])
axes_01.plot(MabsGrassT6H2O, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsGrassT5H2O = df(Temp,columns=['','TempGrassH2OMabs_-5.0P0_1.0.cfg.txt'])
axes_01.plot(MabsGrassT5H2O, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsGrassT4H2O = df(Temp,columns=['','TempGrassH2OMabs_-4.0P0_1.0.cfg.txt'])
axes_01.plot(MabsGrassT4H2O, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsGrassT3H2O = df(Temp,columns=['','TempGrassH2OMabs_-3.0P0_1.0.cfg.txt'])
axes_01.plot(MabsGrassT3H2O, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsGrassT2H2O = df(Temp,columns=['','TempGrassH2OMabs_-2.0P0_1.0.cfg.txt'])
axes_01.plot(MabsGrassT2H2O, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsGrassT1H2O = df(Temp,columns=['','TempGrassH2OMabs_-1.0P0_1.0.cfg.txt'])
axes_01.plot(MabsGrassT1H2O, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsGrassT01H2O = df(Temp,columns=['','TempGrassH2OMabs_-0.01P0_1.0.cfg.txt'])
axes_01.plot(MabsGrassT01H2O, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.text(0.25,0.01,'($H_{2}O$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.5)
axes_01.set_yscale("log")
# all done
# this is quadrant 2, formally quadrant 3 [CO2 ]

G.tight_layout(fig)
print('all done')
plt.show()
#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs/ParamsComparisons/Grass"
if not os.path.exists(dst):
    os.makedirs(dst)

fig.savefig("CruzArceFigs/ParamsComparisons/Grass/Grass_M_abs_P0_1")

# p0 = -1.0 = 0.1 bars
#[229]:
fig = plt.figure(figsize=((12, 8)))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0.0 , 'Wavelength (um)', ha='center', va='bottom', rotation='horizontal')
G = gridspec.GridSpec(2, 1)
G.update( wspace = 0 ) # set the spacing between axes. wspace=0.025,hspace=0.0
# this is quadrant 1, [CO2 Temp]
axes_00 = plt.subplot(G[0, :]) # Grass Block - Right 
# M_abs
# CO2 - Temp 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_00.plot(TempGrass_Pure[:,0],TempGrass_Pure[:,1], label = 'Grass Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsGrassT7CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-7.0P0_0.1.cfg.txt'])
axes_00.plot(MabsGrassT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsGrassT6CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-6.0P0_0.1.cfg.txt'])
axes_00.plot(MabsGrassT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsGrassT5CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-5.0P0_0.1.cfg.txt'])
axes_00.plot(MabsGrassT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsGrassT4CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-4.0P0_0.1.cfg.txt'])
axes_00.plot(MabsGrassT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsGrassT3CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-3.0P0_0.1.cfg.txt'])
axes_00.plot(MabsGrassT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsGrassT2CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-2.0P0_0.1.cfg.txt'])
axes_00.plot(MabsGrassT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsGrassT1CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-1.0P0_0.1.cfg.txt'])
axes_00.plot(MabsGrassT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsGrassT01CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-0.01P0_0.1.cfg.txt'])
axes_00.plot(MabsGrassT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.text(0.25,0.2,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.5)
axes_00.set_yscale("log")
axes_00.title.set_text('log($M_{abs}$) Grass Temperate Regime @ ($P_{0}$)= 0.1  bars')
#
# this is quadrant 2, [H2O Temp]
axes_01 = plt.subplot(G[1, :]) # Grass Block - Right 
# M_abs
# H2O - Temp 
axes_01.plot(TempGrass_Pure[:,0],TempGrass_Pure[:,1], label = 'Grass Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsGrassT7H2O = df(Temp,columns=['','TempGrassH2OMabs_-7.0P0_0.1.cfg.txt'])
axes_01.plot(MabsGrassT7H2O, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
MabsGrassT6H2O = df(Temp,columns=['','TempGrassH2OMabs_-6.0P0_0.1.cfg.txt'])
axes_01.plot(MabsGrassT6H2O, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsGrassT5H2O = df(Temp,columns=['','TempGrassH2OMabs_-5.0P0_0.1.cfg.txt'])
axes_01.plot(MabsGrassT5H2O, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsGrassT4H2O = df(Temp,columns=['','TempGrassH2OMabs_-4.0P0_0.1.cfg.txt'])
axes_01.plot(MabsGrassT4H2O, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsGrassT3H2O = df(Temp,columns=['','TempGrassH2OMabs_-3.0P0_0.1.cfg.txt'])
axes_01.plot(MabsGrassT3H2O, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsGrassT2H2O = df(Temp,columns=['','TempGrassH2OMabs_-2.0P0_0.1.cfg.txt'])
axes_01.plot(MabsGrassT2H2O, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsGrassT1H2O = df(Temp,columns=['','TempGrassH2OMabs_-1.0P0_0.1.cfg.txt'])
axes_01.plot(MabsGrassT1H2O, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsGrassT01H2O = df(Temp,columns=['','TempGrassH2OMabs_-0.01P0_0.1.cfg.txt'])
axes_01.plot(MabsGrassT01H2O, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.text(0.25,0.01,'($H_{2}O$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.5)
axes_01.set_yscale("log")
# all done
G.tight_layout(fig)
print('all done')
plt.show()
#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs/ParamsComparisons/Grass"
if not os.path.exists(dst):
    os.makedirs(dst)

fig.savefig("CruzArceFigs/ParamsComparisons/Grass/Grass_M_abs_P0_01")

# p0 = -2.0¶
#[231]:
fig = plt.figure(figsize=((12, 8)))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0.0 , 'Wavelength (um)', ha='center', va='bottom', rotation='horizontal')
G = gridspec.GridSpec(2, 1)
G.update( wspace = 0 ) # set the spacing between axes. wspace=0.025,hspace=0.0
# this is quadrant 1, [CO2 Temp]
axes_00 = plt.subplot(G[0, :]) # Grass Block - Right 
# M_abs
# CO2 - Temp 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_00.plot(TempGrass_Pure[:,0],TempGrass_Pure[:,1], label = 'Grass Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsGrassT7CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-7.0P0_0.01.cfg.txt'])
axes_00.plot(MabsGrassT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsGrassT6CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-6.0P0_0.01.cfg.txt'])
axes_00.plot(MabsGrassT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsGrassT5CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-5.0P0_0.01.cfg.txt'])
axes_00.plot(MabsGrassT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsGrassT4CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-4.0P0_0.01.cfg.txt'])
axes_00.plot(MabsGrassT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsGrassT3CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-3.0P0_0.01.cfg.txt'])
axes_00.plot(MabsGrassT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsGrassT2CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-2.0P0_0.01.cfg.txt'])
axes_00.plot(MabsGrassT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsGrassT1CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-1.0P0_0.01.cfg.txt'])
axes_00.plot(MabsGrassT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsGrassT01CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-0.01P0_0.01.cfg.txt'])
axes_00.plot(MabsGrassT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.text(0.25,0.01,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.5)
axes_00.set_yscale("log")
axes_00.title.set_text('log($M_{abs}$) Grass Temperate Regime @ ($P_{0}$)= 0.01  bars')
#
# this is quadrant 2, [H2O Temp]
axes_01 = plt.subplot(G[1, :]) # Grass Block - Right 
# M_abs
# H2O - Temp 
axes_01.plot(TempGrass_Pure[:,0],TempGrass_Pure[:,1], label = 'Grass Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsGrassT7H2O = df(Temp,columns=['','TempGrassH2OMabs_-7.0P0_0.01.cfg.txt'])
axes_01.plot(MabsGrassT7H2O, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
MabsGrassT6H2O = df(Temp,columns=['','TempGrassH2OMabs_-6.0P0_0.01.cfg.txt'])
axes_01.plot(MabsGrassT6H2O, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsGrassT5H2O = df(Temp,columns=['','TempGrassH2OMabs_-5.0P0_0.01.cfg.txt'])
axes_01.plot(MabsGrassT5H2O, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsGrassT4H2O = df(Temp,columns=['','TempGrassH2OMabs_-4.0P0_0.01.cfg.txt'])
axes_01.plot(MabsGrassT4H2O, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsGrassT3H2O = df(Temp,columns=['','TempGrassH2OMabs_-3.0P0_0.01.cfg.txt'])
axes_01.plot(MabsGrassT3H2O, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsGrassT2H2O = df(Temp,columns=['','TempGrassH2OMabs_-2.0P0_0.01.cfg.txt'])
axes_01.plot(MabsGrassT2H2O, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsGrassT1H2O = df(Temp,columns=['','TempGrassH2OMabs_-1.0P0_0.01.cfg.txt'])
axes_01.plot(MabsGrassT1H2O, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsGrassT01H2O = df(Temp,columns=['','TempGrassH2OMabs_-0.01P0_0.01.cfg.txt'])
axes_01.plot(MabsGrassT01H2O, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.text(0.25,0.01,'($H_{2}O$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.5)
axes_01.set_yscale("log")
# all done
G.tight_layout(fig)
print('all done')
plt.show()
#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs/ParamsComparisons/Grass"
if not os.path.exists(dst):
    os.makedirs(dst)

fig.savefig("CruzArceFigs/ParamsComparisons/Grass/Grass_M_abs_P0_001")

# p0 = -3.0
#[235]:
fig = plt.figure(figsize=((12, 8)))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0.0 , 'Wavelength (um)', ha='center', va='bottom', rotation='horizontal')
G = gridspec.GridSpec(2, 1)
G.update( wspace = 0 ) # set the spacing between axes. wspace=0.025,hspace=0.0
# this is quadrant 1, [CO2 Temp]
axes_00 = plt.subplot(G[0, :]) # Grass Block - Right 
# M_abs
# CO2 - Temp 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_00.plot(TempGrass_Pure[:,0],TempGrass_Pure[:,1], label = 'Grass Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsGrassT7CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-7.0P0_0.001.cfg.txt'])
axes_00.plot(MabsGrassT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsGrassT6CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-6.0P0_0.001.cfg.txt'])
axes_00.plot(MabsGrassT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsGrassT5CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-5.0P0_0.001.cfg.txt'])
axes_00.plot(MabsGrassT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsGrassT4CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-4.0P0_0.001.cfg.txt'])
axes_00.plot(MabsGrassT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsGrassT3CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-3.0P0_0.001.cfg.txt'])
axes_00.plot(MabsGrassT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsGrassT2CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-2.0P0_0.001.cfg.txt'])
axes_00.plot(MabsGrassT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsGrassT1CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-1.0P0_0.001.cfg.txt'])
axes_00.plot(MabsGrassT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsGrassT01CO2 = df(Temp,columns=['','TempGrassCO2Mabs_-0.01P0_0.001.cfg.txt'])
axes_00.plot(MabsGrassT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.text(0.25,0.01,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.5)
axes_00.set_yscale("log")
axes_00.title.set_text('log($M_{abs}$) Grass Temperate Regime @ ($P_{0}$)= 0.001  bars')
#
# this is quadrant 2, [H2O Temp]
axes_01 = plt.subplot(G[1, :]) # Grass Block - Right 
# M_abs
# H2O - Temp 
axes_01.plot(TempGrass_Pure[:,0],TempGrass_Pure[:,1], label = 'Grass Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsGrassT7H2O = df(Temp,columns=['','TempGrassH2OMabs_-7.0P0_0.001.cfg.txt'])
axes_01.plot(MabsGrassT7H2O, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
MabsGrassT6H2O = df(Temp,columns=['','TempGrassH2OMabs_-6.0P0_0.001.cfg.txt'])
axes_01.plot(MabsGrassT6H2O, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsGrassT5H2O = df(Temp,columns=['','TempGrassH2OMabs_-5.0P0_0.001.cfg.txt'])
axes_01.plot(MabsGrassT5H2O, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsGrassT4H2O = df(Temp,columns=['','TempGrassH2OMabs_-4.0P0_0.001.cfg.txt'])
axes_01.plot(MabsGrassT4H2O, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsGrassT3H2O = df(Temp,columns=['','TempGrassH2OMabs_-3.0P0_0.001.cfg.txt'])
axes_01.plot(MabsGrassT3H2O, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsGrassT2H2O = df(Temp,columns=['','TempGrassH2OMabs_-2.0P0_0.001.cfg.txt'])
axes_01.plot(MabsGrassT2H2O, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsGrassT1H2O = df(Temp,columns=['','TempGrassH2OMabs_-1.0P0_0.001.cfg.txt'])
axes_01.plot(MabsGrassT1H2O, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsGrassT01H2O = df(Temp,columns=['','TempGrassH2OMabs_-0.01P0_0.001.cfg.txt'])
axes_01.plot(MabsGrassT01H2O, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.text(0.25,0.01,'($H_{2}O$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.5)
axes_01.set_yscale("log")
# all done
G.tight_layout(fig)
print('all done')
plt.show()
#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs/ParamsComparisons/Grass"
if not os.path.exists(dst):
    os.makedirs(dst)

fig.savefig("CruzArceFigs/ParamsComparisons/Grass/Grass_M_abs_P0_0001")


# #### Lastly: Glass!

# In[273]:


fig = plt.figure(figsize=((12, 8)))
# Set common labels
fig.text(0.006,.5 , 'Geometric Albedo (I/F)', ha='center', va='center', rotation='vertical')
fig.text(0.5,0.0 , 'Wavelength (um)', ha='center', va='bottom', rotation='horizontal')
G = gridspec.GridSpec(2, 1)
G.update( wspace = 0 ) # set the spacing between axes. wspace=0.025,hspace=0.0
# this is quadrant 1, [CO2 Hot]
axes_00 = plt.subplot(G[0, :]) # Silicates Block - Right 
# M_abs
# CO2 - Hot 
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
axes_00.plot(HotSilicates_Pure[:,0],HotSilicates_Pure[:,1], label = 'Silicates Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSilicatesT7CO2 = df(Hot,columns=['','HotSilicatesCO2Mabs_-7.0P0_0.001.cfg.txt'])
axes_00.plot(MabsSilicatesT7CO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
MabsSilicatesT6CO2 = df(Hot,columns=['','HotSilicatesCO2Mabs_-6.0P0_0.001.cfg.txt'])
axes_00.plot(MabsSilicatesT6CO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSilicatesT5CO2 = df(Hot,columns=['','HotSilicatesCO2Mabs_-5.0P0_0.001.cfg.txt'])
axes_00.plot(MabsSilicatesT5CO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSilicatesT4CO2 = df(Hot,columns=['','HotSilicatesCO2Mabs_-4.0P0_0.001.cfg.txt'])
axes_00.plot(MabsSilicatesT4CO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSilicatesT3CO2 = df(Hot,columns=['','HotSilicatesCO2Mabs_-3.0P0_0.001.cfg.txt'])
axes_00.plot(MabsSilicatesT3CO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSilicatesT2CO2 = df(Hot,columns=['','HotSilicatesCO2Mabs_-2.0P0_0.001.cfg.txt'])
axes_00.plot(MabsSilicatesT2CO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSilicatesT1CO2 = df(Hot,columns=['','HotSilicatesCO2Mabs_-1.0P0_0.001.cfg.txt'])
axes_00.plot(MabsSilicatesT1CO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSilicatesT01CO2 = df(Hot,columns=['','HotSilicatesCO2Mabs_-0.01P0_0.001.cfg.txt'])
axes_00.plot(MabsSilicatesT01CO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_00.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_00.text(0.25,.1,'($CO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_00.set_ylim(0,1)
axes_00.set_xlim(0.2,2.5)
axes_00.set_yscale("log")
axes_00.title.set_text('log($M_{abs}$) Glass Hot Regime @ ($P_{0}$)= 0.001  bars')
#
# this is quadrant 2, [SO2 Hot]
axes_01 = plt.subplot(G[1, :]) # Silicates Block - Right 
# M_abs
# SO2 - Hot 
axes_01.plot(HotSilicates_Pure[:,0],HotSilicates_Pure[:,1], label = 'Silicates Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
MabsSilicatesT7SO2 = df(Hot,columns=['','HotSilicatesSO2Mabs_-7.0P0_0.001.cfg.txt'])
axes_01.plot(MabsSilicatesT7SO2, label = '-7.0', marker = ',', color = 'magenta', linestyle= 'solid')
# magenta, darkblue, skyblue, aquamarine, green, yellow, orange, red
MabsSilicatesT6SO2 = df(Hot,columns=['','HotSilicatesSO2Mabs_-6.0P0_0.001.cfg.txt'])
axes_01.plot(MabsSilicatesT6SO2, label = '-6.0', marker = ',', color = 'darkblue', linestyle= 'solid')
MabsSilicatesT5SO2 = df(Hot,columns=['','HotSilicatesSO2Mabs_-5.0P0_0.001.cfg.txt'])
axes_01.plot(MabsSilicatesT5SO2, label = '-5.0', marker = ',', color = 'skyblue', linestyle= 'solid')
MabsSilicatesT4SO2 = df(Hot,columns=['','HotSilicatesSO2Mabs_-4.0P0_0.001.cfg.txt'])
axes_01.plot(MabsSilicatesT4SO2, label = '-4.0', marker = ',', color = 'aquamarine', linestyle= 'solid')
MabsSilicatesT3SO2 = df(Hot,columns=['','HotSilicatesSO2Mabs_-3.0P0_0.001.cfg.txt'])
axes_01.plot(MabsSilicatesT3SO2, label = '-3.0', marker = ',', color = 'tab:green', linestyle= 'solid')
MabsSilicatesT2SO2 = df(Hot,columns=['','HotSilicatesSO2Mabs_-2.0P0_0.001.cfg.txt'])
axes_01.plot(MabsSilicatesT2SO2, label = '-2.0', marker = ',', color = 'yellow', linestyle= 'solid')
MabsSilicatesT1SO2 = df(Hot,columns=['','HotSilicatesSO2Mabs_-1.0P0_0.001.cfg.txt'])
axes_01.plot(MabsSilicatesT1SO2, label = '-1.0', marker = ',', color = 'tab:orange', linestyle= 'solid')
MabsSilicatesT01SO2 = df(Hot,columns=['','HotSilicatesSO2Mabs_-0.01P0_0.001.cfg.txt'])
axes_01.plot(MabsSilicatesT01SO2, label = '-0.01', marker = ',', color = 'tab:red', linestyle= 'solid')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes_01.legend(by_label.values(), by_label.keys(), handlelength=.5, borderpad=.2, labelspacing=.2, handletextpad =.4, borderaxespad = .2 )#,bbox_to_anchor=(.7, -.09))
axes_01.text(0.25,0.1,'($SO_{2}$)', ha='left', va='top',fontweight = 'normal',size = 12,clip_box='bbox')
plt.grid(True)
#axes_01.set_ylim(0,1)
axes_01.set_xlim(0.2,2.5)
axes_01.set_yscale("log")
# all done
G.tight_layout(fig)
print('all done')
plt.show()
#Perfecto, now we create a folder to save these in.
dst = "/Users/ccruzarc/Desktop/PSG_GPC_framework_master/CruzArceFigs/ParamsComparisons/Silicates"
if not os.path.exists(dst):
    os.makedirs(dst)

fig.savefig("CruzArceFigs/ParamsComparisons/Silicates/Silicates_M_abs_P0_0001")


# In[221]:


# Question to answer: Where 
plt.plot(HotBasalt_Pure[:,0],HotBasalt_Pure[:,1], label = 'Basalt Reflectance',marker = '*', color = 'black', linestyle= 'dotted' )
plt.yscale('log')


# In[ ]:




