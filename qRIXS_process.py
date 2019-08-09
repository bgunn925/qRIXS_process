'''
File: 		qRIXS_process.py
Updated: 	Aug 9, 2019

To run: 	python3 qRIXS_process.py
w/ verbose:	python3 -u qRIXS_process.py | tee logfile.txt

Description: 
	This program is used to process qRIXS data from ALS by: 
	- correcting 2D scans for tilt/curvature
	- integrating 2D scans
	- outputing CSV files with intensity vs energy/pixel 

Instructions:
	1) Set paths to input and output directories
	2) Set up Dark_Scan, CT_Scan, and CCD_Scan objects
		2a) Use MATLAB (imagesc) or another method to find 
			upper/lower bounds for CT_Scan peaks
		2b) Do not initialize CT_Scan objects w/ slope/intercept
	3) Comment out CCD_Scan objects which are not CT Scans
	4) Run once over CT Scans 
	5) Plot CSV files (intensity vs pixel), fit peaks, 
		plot energy vs pixel, fit line to obtain slope/intercept
	6) Add slope/intercept values to CT_Scan objects
	7) Run again over all CCD_Scan objects to create CSV files
		with intensity vs energy
'''

from sys import argv
import numpy as np 
import matplotlib.pyplot as plt 
import scipy
import scipy.misc
import os
import itertools
import pylab
from pylab import *
from scipy import integrate
import itertools
from scipy.optimize import leastsq # Levenberg-Marquadt Algorithm #
from scipy.optimize import least_squares # Least squares with bounds
import sys
import csv
import cv2
import imutils
import time
import datetime


######################################################################################################
#-------------------------------------  USER INPUT 1/2 ----------------------------------------------#
######################################################################################################

verbose					= True		# prints info to terminal
Andor_9634 				= True 		# includes Andor 9634 scans
Andor_20039 			= True 		# includes Andor 20039 scans
show_plots				= False		# pauses and prints every plot -> mostly for debugging
axis_swap 				= False 	# only use if axes swapped for energy vs pixel fit
elastic 				= True		# print elastic contributions
elastic_roi				= [735,25] 	# Elastic ROI [center, halfwidth] in pixels (NOTE: first pixel = 0)
									# --> ROI is from (center-halfwidth) to (center+halfwidth)
									# --> [0,0] to turn off


# set path to raw data directory
input_dir_path 			= "/home/bgunn/Desktop/qrixs/Raw"

# set path to processed data directory
output_dir_path  		= "/home/bgunn/Desktop/qrixs/Processed"

######################################################################################################
#---------------------------------- END OF USER INPUT 1/2 -------------------------------------------#
######################################################################################################

# builds detector list
Detector_Dir 			= []
if Andor_9634:
	Detector_Dir.append("/Andor 9634")
if Andor_20039:
	Detector_Dir.append("/Andor 20039")

if (elastic_roi[0]==0 and elastic_roi[1]==0):
	roi_on = False
else:
	roi_on = True

#------------- Class Definitions ----------------------------------------------------------------------------

class IterScans(type):											# defines a self-iterable metaclass
	def __iter__(cls):											
		return iter(cls._CCD_Scans)


class CCD_Scan(metaclass=IterScans):							# defines CCD Scan class

	_CCD_Scans = []												# list of all CCD Scan objects

	def __init__(	self, 
					date, 										# scan date list --> [year, month, day]
					data_image,  								# number of CCD scan 
					dark_image, 								# dark image object name
					correction_image): 							# correction image object name

		self._CCD_Scans.append(self) 							# adds to CCD_Scans list 
		self.Year			= date[0] 							# returns year of scan
		self.Month			= date[1] 							# returns month of scan
		self.Day 			= date[2] 							# returns day of scan
		self.Scan_Num		= data_image 						# returns scan number
		self.Date_Dir 		= "/" + str(date[0]) \
							+ " " + str(date[1]) \
							+ " " + str(date[2]) 				# returns date directory name
		self.Data_Image_Dir = "/CCD Scan " \
							+ str(data_image)					# returns CCD Scan directory name
		self.Dark_Image 	= dark_image 						# returns dark scan object
		self.CT_Image 		= correction_image 					# returns CT scan object


class Dark_Scan:  												# dark scan object
	def __init__(	self, 
					A9634_Darkpath, 							# path to Andor 9634 dark image
					A20039_Darkpath): 							# path to Andor 20039 dark image
		
		# returns list of dark paths
		self.Dark_Path 			= []
		if Andor_9634:
			self.Dark_Path.append(A9634_Darkpath)
		if Andor_20039:
			self.Dark_Path.append(A20039_Darkpath)

		# returns particular dark scan paths
		self.A9634_Dark_Path 	= A9634_Darkpath 				
		self.A20039_Dark_Path 	= A20039_Darkpath 				

class CT_Scan: 													# CT scan object
	def __init__(	self, 
					A9634_CTpath,  								# path to Andor 9634 CT scan
					A9634_lowerbound, 							# Andor 9634 CT scan signal lower bound pixel number
					A9634_upperbound, 							# Andor 9634 CT scan signal upper bound pixel number
					A20039_CTpath, 								# repeat for Andor 20039
					A20039_lowerbound,
					A20039_upperbound,
					A9634_slope=1,
					A9634_intercept=0,
					A20039_slope=1,
					A20039_intercept=0):

		# Returns lists of paths/bounds/fit parameters
		self.CT_Path 			= []
		self.lowerbound 		= []
		self.upperbound 		= []
		self.slope 				= []
		self.intercept 			= []

		if Andor_9634:
			self.CT_Path.append(A9634_CTpath)
			self.lowerbound.append(A9634_lowerbound)
			self.upperbound.append(A9634_upperbound)
			self.slope.append(A9634_slope)
			self.intercept.append(A9634_intercept)

		if Andor_20039:
			self.CT_Path.append(A20039_CTpath)
			self.lowerbound.append(A20039_lowerbound)
			self.upperbound.append(A20039_upperbound)
			self.slope.append(A20039_slope)
			self.intercept.append(A20039_intercept)

		# Returns particular path/bound/fit parameter
		self.A9634_CT_Path 		= A9634_CTpath
		self.A9634_lowerbound 	= A9634_lowerbound		
		self.A9634_upperbound 	= A9634_upperbound
		self.A9634_slope 		= A9634_slope
		self.A9634_intercept 	= A9634_intercept
		self.A20039_CT_Path 	= A20039_CTpath
		self.A20039_lowerbound 	= A20039_lowerbound
		self.A20039_upperbound 	= A20039_upperbound
		self.A20039_slope 		= A20039_slope
		self.A20039_intercept 	= A20039_intercept


#----------------------- Function Definitions -----------------------------------------------------------

# Here we define the function we want to fit
def lorentzian(x,p):
	numerator 	= (p[0] / 2)
	denominator = (x - (p[1]))**2 + (p[0] / 2)**2
	y 			= p[2] / pi * (numerator / denominator)
	return y

# We define the residual value (data-model)
def residuals(p,y,x):
	err = y - lorentzian(x,p)
	return err		

######################################################################################################
#-------------------------------------- USER INPUT 2/2 ----------------------------------------------#
######################################################################################################

#----------------------- Create all Dark Scan objects ---------------------------------------------------
# Template:
# Dark_Scan_Num = Dark_Scan( Andor_9634_path, Andor_20039_path )

Dark_Scan_5281=Dark_Scan(	input_dir_path + "/2018 12 18/CCD Scan 5281/Andor 9634/NNO_LAO5281-00001.tif",
							input_dir_path + "/2018 12 18/CCD Scan 5281/Andor 20039/NNO_LAO5281-00001.tif")



#----------------------- Create all CT Scan objects ------------------------------------------------------
# If: A) running only CT scans to find pixel->energy conversion
#	  -or-
# 	  B) you don't need the pixels converted to energy	 
#	  use template: 
#
# CT_Scan_Num = CT_Scan( Andor_9634_path, 
# 						 Andor_9634_lowerbound, Andor_9634_upperbound, 
# 						 Andor_20039_path, 
# 						 Andor_20039_lowerbound, Andor_20039_upperbound )
#
# NOTE: load CT Scans in MATLAB and use 'imagesc' command to find the upper/lower peak bounds  
#---------------------------------------------------------------------------------------------------------
# If  you do care about the  pixel-> energy conversion:
# 	After finding slope/intercept for pixel->energy, use template:
#
# CT_Scan_Num = CT_Scan( Andor_9634_path, 
# 						 Andor_9634_lowerbound, Andor_9634_upperbound, 
# 						 Andor_20039_path, 
# 						 Andor_20039_lowerbound, Andor_20039_upperbound,
# 						 Andor_9634_slope, Andor_9634_intercept,
# 						 Andor_20039_slope, Andor_20039_intercept )


'''# with energy conversion info
CT_Scan_5279=CT_Scan(	input_dir_path + "/2018 12 18/CCD Scan 5279/Andor 9634/NNO_LAO5279-00006.tif", 
						676, 711,
						input_dir_path + "/2018 12 18/CCD Scan 5279/Andor 20039/NNO_LAO5279-00006.tif",
						680, 699,
						-0.27277, 1046.9, 
						-0.2678, 1039.5)
'''

# without energy conversion info
CT_Scan_5279=CT_Scan( 	"/home/bgunn/Desktop/qrixs/Raw/2018 12 18/CCD Scan 5279/Andor 9634/NNO_LAO5279-00006.tif", 
						675, 725, 
						"/home/bgunn/Desktop/qrixs/Raw/2018 12 18/CCD Scan 5279/Andor 20039/NNO_LAO5279-00006.tif",
						675, 700)

#----------------------- Create all data scan objects -------------------------------------------------
# Template:
# CCD_Scan_Num = CCD_Scan( [year, month, day] , scan_number, dark_scan_object, CT_scan_object )
#
# NOTE: only need to loop over CT Scans while obtaining pixel->energy parameters
#       (comment out any CCD scans which are not CT scans)

CCD_Scan_5278=CCD_Scan([2018, 12, 18], 5278, Dark_Scan_5281, CT_Scan_5279)

######################################################################################################
#---------------------------------- END OF USER INPUT 2/2 -------------------------------------------#
######################################################################################################

#----------------------- Apply analysis, looping over CCD_Scan class ---------------------------------
file_counter 	= 0
start_time 		= time.time()
now 	 		= datetime.datetime.now()
print("\n\nqRIXS processing started at:\t", now.strftime("%Y-%m-%d %H:%M:%S"), "\n")

for scan in CCD_Scan: 		 													# loops over CCD Scans
	for det_i in range(len(Detector_Dir)): 										# loops over detectors

		# CCD Scan
		local_path = scan.Date_Dir + scan.Data_Image_Dir + Detector_Dir[det_i] 	# local path
		output_path = output_dir_path + local_path 								# output path
		data_path = input_dir_path + local_path 								# input path

		# Dark Scan
		dark_path = scan.Dark_Image.Dark_Path[det_i] 							# sets path to dark image
		dark_data = scipy.misc.imread(dark_path, flatten=True) 					# reads dark image
		if dark_data.shape[0] < dark_data.shape[1]: 							# transposes dark image if wrong orientation
			dark_data = dark_data.T

		# CT Scan
		corr_path = scan.CT_Image.CT_Path[det_i] 								# sets path to CT image
		corr_data = scipy.misc.imread(corr_path, flatten=True) 					# reads CT image
		if corr_data.shape[0] < corr_data.shape[1]: 							# transposes CT image if wrong orientation
			corr_data = corr_data.T

		# Creates Mirror Folder Hierarchy for Output
		if os.path.exists(output_path)==False: 			
			os.makedirs(output_path) 

		#------------Calculate Curvature / Tilt Correction  --------------------------------------------
		width = 0  																# widens both ends of correction region by this amount
		centers=[]
		ypos=[]
		bin_pixel_number=32  
		bin=round(len(corr_data[0,:])/bin_pixel_number)

		for j in range(0,bin_pixel_number):

			#Now we fit a lorentzian to the data, with the x column an array of pixels from (0,range desired) and y is a summed column along the x-pixels
			y=sum(corr_data[:,i] for i in range(j*bin,j*bin+(bin-1)))/bin_pixel_number #This is how to sum a bunch of columns into a "sum" column
			x=np.array(range(0,len(y)))

			# set fit bounds
			low_bound=scan.CT_Image.lowerbound[det_i]-width
			high_bound=scan.CT_Image.upperbound[det_i]+width
			avg_bound=mean([low_bound, high_bound])

			# We define the conditions x meets as the 'background'
			ind_bg_low = (x > min(x)) & (x <low_bound)
			ind_bg_high = (x > high_bound) & (x < max(x))

			#We concantenate the high and low part of the background, both x and y
			x_bg = np.concatenate((x[ind_bg_low],x[ind_bg_high]))
			y_bg = np.concatenate((y[ind_bg_low],y[ind_bg_high]))
			
			# fitting the background to a line 
			m, c = np.polyfit(x_bg, y_bg, 1)
			
			# removing fitted background 
			background = m*x + c
			y_bg_corr = y - background
			
			#Now we do the fit:
			# initial values 
			sat1_p = [6,avg_bound,1]  # [hwhm, peak center, intensity] 
			low = [0, low_bound, 0]
			upp = [10, high_bound, np.amax(corr_data)]
			
			# optimization 
			pbest = least_squares(residuals, sat1_p, bounds=(low,upp), method = 'trf', args=(y_bg_corr,x))
			best_parameters = pbest.x

			# fit to data 
			fit = lorentzian(x,best_parameters)
			ypos.append(np.mean(range(j*bin,j*bin+(bin-1))))
			centers.append(best_parameters[1])

		#############
		## Add another step to check if there are still outliers even after using lower
		## and upper bounds during the fitting procedure    
		#############    
			
		# find length of centers and create a copy that will be used for filtered centers    
		lc = len(centers)    
		filt_centers = centers.copy()

		# create another copy of centers and add a value at the beginning and end that
		# is the median of the starting and ending values, respectively
		dummy_list = centers.copy()
		dummy_list.insert(0, np.median(centers[:6]))
		dummy_list.append(np.median(centers[-6:]))

		# compute the average slope of centers
		dummy1 = 0
		for i in range(0,lc-1):
			dummy1 += abs(centers[i+1]-centers[i])	
		dummy1 = dummy1 / (lc-1)

		# if any value is an outlier and thus has a very different slope than the average
		# slope stored in dummy1 change the value to the median of surrounding values
		for i in range(0,lc):
			dummy2 = (abs(dummy_list[i+1] - dummy_list[i]) + abs(dummy_list[i+2] - dummy_list[i+1]))/2
			if dummy2 > 1.5 * dummy1:
				filt_centers[i] = np.median(filt_centers[max([0, i-4]) : min([lc-1, i+4])])

		#This fits a polinomial to the x vs y position of the peak. In other words, this would eventuually correct for tilts and curvature.
		poly_fit= np.polyfit(ypos, filt_centers, 3)
		f=np.poly1d(poly_fit)
	
		#---------------Apply Correction To Dark Image-----------------------------------------------
		darkmod=np.zeros((dark_data.shape[0],dark_data.shape[1]))

		for i, j in itertools.product(range(dark_data[:,0].size), range(dark_data[0,:].size)):
			new_i = int(round(i-(f(j)-f(0))))
			if new_i >= dark_data.shape[0]:
				new_i = new_i % dark_data.shape[0]
			darkmod[new_i,j]=dark_data[i,j] #This is where we correct for curvature or tilt, by shifting the pixel positions based on the fit from line 86 (no longer line 86...).

		# prints info to terminal
		if verbose:
			print("\n----------------------------------------------------------------------------------")
			print("CCD Scan:     \t", scan.Scan_Num)
			print("Detector:     \t", Detector_Dir[det_i])
			print("CCD Scan Path:\t", data_path)
			print("\nCT Scan:\t", corr_path)
			print("Lower bound:  \t", low_bound)
			print("Upper bound:  \t", high_bound)
			print("\nPolynomial fit coefficients: ")
			print("x^3:          \t", poly_fit[0])
			print("x^2:          \t", poly_fit[1])
			print("x^1:          \t", poly_fit[2])
			print("x^0:          \t", poly_fit[3])			
			print("\nDark Scan:    \t", dark_path)

		#-------------------- Apply Correction, Sum, and Write to CSV for each image -----------------------------
		if elastic:
			elastic_list=[]													# Create list to hold elastic peak information
			el_filename=""

		for root, dirs, files in os.walk(data_path): 						# walks through file hierarchy in CCD Scan directory			
			for file in files:  											# loops over each file in CCD Scan directory
				if file.endswith('.tif'): 									# only loop over TIFF files

					# Load Data File
					data_file = data_path+"/"+file 							# sets data image path
					raw_data=scipy.misc.imread(data_file, flatten=True) 	# loads data image
					if raw_data.shape[0] < raw_data.shape[1]: 				# transposes if wrong orientation
						raw_data = raw_data.T					
					data = raw_data - darkmod 								# subtracts dark image 
					
					# Apply Curve/Tilt Correction
					datamod=np.zeros((data.shape[0],data.shape[1]))
					for i, j in itertools.product(range(data[:,0].size), range(data[0,:].size)):
						new_i = int(round(i-(f(j)-f(0))))
						if new_i >= data.shape[0]:
							new_i = new_i % data.shape[0]
						datamod[new_i,j]=data[i,j] 

					if show_plots:
						fig = plt.figure()
						plt.plot(ypos,centers, marker='s', markersize=8)
						plt.plot(x, f(x), marker='s', markersize=8)	
									
					# Integrate Intensities
					ysum=np.zeros(len(datamod[:,0]))
					for i in range(len(datamod[:,0])):
						for j in range(len(datamod[0,:])):
							ysum[i]=ysum[i] + datamod[i,j]


					# Elastic contribution
					if elastic:
						filenum=int(file.split("-")[-1].split(".")[0])			# parse file number
						if filenum==1:
							el_filename=file.rsplit("-",1)[0] 					# parse file name
						ymax = max(ysum)										# find max value 
						ymaxpixel = np.argmax(ysum) 							# find max value pixel number
						if roi_on:
							yroi=0  											# variable to store roi sum
							roi_low = int(elastic_roi[0]-elastic_roi[1]) 		# roi lower bound
							roi_high = int(elastic_roi[0]+elastic_roi[1]) 		# roi upper bound
							for i in range(roi_high-roi_low+1):
								if roi_low+i>=0:
									yroi += ysum[roi_low+i] 					# sum intensities in roi range
								else:
									print("WARNING: Low end of ROI is less than zero! Subzero pixels not summed!") 
							elastic_list.append([filenum, ymax, ymaxpixel, yroi])
						else:
							elastic_list.append([filenum, ymax, ymaxpixel])
							
					# Calculate energy axis
					energy_slope = scan.CT_Image.slope[det_i]
					energy_intercept = scan.CT_Image.intercept[det_i]

					# Write output to CSV
					filename= os.path.splitext(file)[0]
					csvfilename=filename+".csv"
					csvpath=output_path+"/"+csvfilename
					csvfile=open(csvpath, 'w+')
					csvwriter=csv.writer(csvfile)

					if (energy_slope==1 and energy_intercept==0):
						csvwriter.writerow(['pixel', 'int'])
					else:
						csvwriter.writerow(['energy', 'int'])
						energy_array=np.zeros(len(ysum))

					for i in range(len(ysum)):
						if not axis_swap:
							energy_value = format( energy_slope * x[i] + energy_intercept, '.2f')
						else:
							energy_value = format( (x[i] - energy_intercept)/energy_slope , '.2f')
						csvwriter.writerow([energy_value, ysum[i]])
						if not (energy_slope==1 and energy_intercept==0):
							energy_array[i] = energy_value

					file_counter = file_counter + 1
					if verbose:
						print("\nFile:\t", file_counter)
						print("Image:\t", file)
						print("Path: \t", data_file)
						if elastic:
							print("Max: \t", int(ymax), " at pixel number ", ymaxpixel)

					# ------ Plot ----------------------------------------
					# Corrected Image
					datamod_plot=output_path+"/"+filename+"_corr.png"
					fig = plt.figure()
					plt.imshow(datamod)
					savefig(datamod_plot)
					if show_plots:
						plt.show()
					plt.close(fig)

					# Integrated Plot
					sum_plot=output_path+"/"+filename+"_sum.png"
					if elastic and roi_on:
						fig, ax=plt.subplots()
					else:
						fig=plt.figure()
					if (energy_slope==1 and energy_intercept==0):
						plt.plot(x,ysum)
						plt.xlabel("Pixel")
					else:
						plt.plot(energy_array,ysum)
						plt.xlabel("Energy [eV]")
					if elastic and roi_on:
						ax.axvspan(roi_low, roi_high, alpha=0.5, color='red')
						plt.legend(('Intensity', 'ROI'))
					plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
					plt.ylabel("Intensity")
					plt.title(filename,loc='right')
					savefig(sum_plot)
					if show_plots:
						plt.show()
					plt.close(fig)			

		# Elastic contribution
		if elastic:
			elastic_list.sort()														
			el_scan_num 	= [i[0] for i in elastic_list]				
			el_ymax 		= [i[1] for i in elastic_list]
			el_ymaxpixel 	= [i[2] for i in elastic_list]
			if roi_on:
				el_roi 		= [i[3] for i in elastic_list]

			el_path = output_path+"/Elastic"
			if os.path.exists(el_path)==False:
				os.makedirs(el_path)

			el_roi_plot = el_path+"/"+el_filename+"_elastic_roi_plot.png"
			el_max_plot = el_path+"/"+el_filename+"_elastic_max_plot.png"
			el_csvpath = el_path+"/"+el_filename+"_elastic.csv"
			el_csvfile=open(el_csvpath, 'w+')
			el_csvwriter=csv.writer(el_csvfile)

			if roi_on:
				el_csvwriter.writerow(['scan','max_int', 'max_pixel', 'roi_sum'])
			else:
				el_csvwriter.writerow(['scan','max_int', 'max_pixel'])
			
			for i in range(len(elastic_list)):
				if roi_on:
					el_csvwriter.writerow([el_scan_num[i], el_ymax[i], el_ymaxpixel[i], el_roi[i]]) 
				else:
					el_csvwriter.writerow([el_scan_num[i], el_ymax[i], el_ymaxpixel[i]]) 

			if roi_on:
				fig=plt.figure()
				plt.plot(el_scan_num, el_roi)
				plt.title(el_filename, loc='right')
				plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
				plt.xlabel("Scan Number")
				plt.ylabel("ROI Sum")
				savefig(el_roi_plot)
				if show_plots:
					plt.show()
				plt.close(fig)

			fig=plt.figure()
			plt.plot(el_scan_num, el_ymax)
			plt.title(el_filename,loc='right')
			plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
			plt.ylabel("Max Intensity")
			plt.xlabel("Scan Number")
			savefig(el_max_plot)
			if show_plots:
				plt.show()
			plt.close(fig)

end_time = time.time()
run_time = end_time - start_time
now 	 = datetime.datetime.now()
print("\n\nqRIXS processing finished at:\t", now.strftime("%Y-%m-%d %H:%M:%S"), "\n")
print("\n\n", file_counter, " files processed in ", run_time, " seconds.\n\n")