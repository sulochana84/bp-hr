""" PPG signal calculation"""
from face_detection import *
from facedetection_mtcnn import *
from vidstab import VidStab
from derive_pca import *
from optimalsize_face import *
import cv2
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.fft import fft
from findpeaks import findpeaks
import scipy
from scipy.signal import find_peaks
from scipy import signal
from scipy.signal import butter, convolve, find_peaks,filtfilt
from facedetection_mtcnn import*
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from Bpm_calculation import*
from find_features import*
import pandas as pd
from Process_PPG import *
from bp_regression_model import *
import sys

def video_clip(folder):

    """ clip the input video using this function
    t1 = starting time
    t2 = ending time
    ffmpeg_extract_subclip(input_video_path, t1, t2, targetname=output_video_path)"""

    ffmpeg_extract_subclip(folder, 3, 20, targetname="clipped_video/test.MOV")

def video_stabilisation(folder):
    """ This funtion stabilze the video using
    VidStab function"""
    stabilizer = VidStab()
    stabilizer.stabilize(input_path=folder,output_path='stablevideo.MOV')

def frame_length(folder):
    """ This function return the number of frames in
    the video"""
    lst = os.listdir(folder)
    number_files = len(lst)
    return number_files

def frame_parameters(path):
    """This fuction return rows and colums of the
     face bouding box"""
    for img in glob.glob(path):
        r_image = cv2.imread(img)
        r_image1 = np.asarray(r_image)
        if (type(r_image1) is np.ndarray):
            break
    num_rows = r_image1.shape[0]
    num_columns = r_image1.shape[1]
    total_pixels = num_rows * num_columns
    return num_rows, num_columns,total_pixels

def deleteprevious(path):
    """ function deletes the images in the folder from previous run"""
    removing_files = glob.glob(path)
    for i in removing_files:
        os.remove(i)

def limit_xaxis(vid_length,n_files):
    """ this function to specify the x -axis limit"""
    a_range = vid_length / n_files
    xaxis1 = []
    for i in np.arange(0, vid_length, a_range):
        xaxis1.append(i)
    xaxis2 = np.array(xaxis1)
    xaxis = np.around(xaxis2, 2)
    return xaxis

def main(folder):
    lst = []
    hr_peak = []
    hr_fft = []
    hr_semi = []
    hr_fft_interp=[]
    count =0
    video_name =[]
    actual_BP = pd.read_excel("Book1_Actualdata.xlsx")
    actual_BP_np = np.array(actual_BP)

    #load the folder of videos
    for file in os.listdir(folder):
        if file.endswith(".MOV"):
            path = os.path.join(folder, file)
            #cropped_video = cv2.VideoCapture(path)
            #delete the previous in the folder
            deleteprevious("clipped_video/Test.MOV")
            # clip the videos
            video_clip(path)
            cropped_video = cv2.VideoCapture("clipped_video/Test.MOV")
            # Information about the video
            #frames per second
            fps = cropped_video .get(cv2.CAP_PROP_FPS)
            fps = round(fps)
            #Number of frames
            frame_count = int( cropped_video.get(cv2.CAP_PROP_FRAME_COUNT))
            #video length
            vid_length = round(frame_count / fps)
            #Time between frames
            time_bw_frame = 1 / fps
            #print(" video length" , vid_length)
            #print("Time between frames",time_bw_frame)
            #print("frames per second", fps)
            # path to save frames inside th current directory
            path = "outframe/*.png"
            folder1 = "outframe"
            #clean the folder for each interation
            deleteprevious(path)
            # extract the bounding box of the face region
            x,y,x2,y2 = face_boundingbox_size( cropped_video)
            # crop the face region in each frame
            face_bbox(  cropped_video ,x,y,x2,y2)
            #the number of the frames in the vidoe
            n_files = frame_length(folder1)
            # rows and columns of the video frames
            rows, cols, t_pixels = frame_parameters(path)
            #print("The number of rows in the frame:", rows)
            #print("The number of columns in the frame:", cols)
            #print("The number of frames in the video:", n_files)
            #print("The number of pixels in each of the frame:", t_pixels)
            ## extract the red channel from the video frames
            red_channel_frames = redchannel_extraction("outframe/*.png",t_pixels, n_files)
            ## calculated the PCA and recontruct the PCA
            recontruct_pca_r = pca_redchannel(red_channel_frames)
            #PPG Signal
            final_ppg = ppg_signal(red_channel_frames, recontruct_pca_r,rows, cols,n_files)
            final_ppg = np.round(final_ppg, 2)
            # x-axis limit
            xaxis= limit_xaxis(vid_length, n_files)
            # plotting the points
            plt.plot(xaxis,final_ppg)
            # naming the x axis
            #plt.xlabel('time in seconds')
            # naming the y axis
            #plt.ylabel('amplitude')
            # giving a title to my graph
            #plt.title('ppg signal!')
            # function to show the plot
            #fig = plt.figure(1)
            #plt.show()
            #fig.clf()
            #band pass filtering of PPG signal
            high_pass_filtered = filter_s(final_ppg,fps)
            #y = filter_all(final_ppg, fps, order=5,cutoff_high= 3,cutoff_low=0.3)
            #y = butter_bandpass_filter(final_ppg, lowcut=0.5, highcut=7, fs=fps, order=4)
            #fig = plt.figure(2)
            xaxis = range(0,n_files)
            #plt.plot(xaxis, high_pass_filtered)
            #fig = plt.figure(2)
            #plt.xlabel('number of frames')
            # naming the y axis
            #plt.ylabel(' Amplitude')
            # giving a title to my graph
            #plt.title(' filtered  signal!')
            #plt.show()
            #fig.clf()
            # smooth the denoised signal
            filtered_window = simple_moving_average(high_pass_filtered, window=3)
            #plt.plot(filtered_window)
            #plt.title('Averaged Signal!')
            #plt.show()
            # measure the heart rate
            #heart rate measurement using peak infmroantion
            heart_rate_peak = give_bpm(filtered_window,time_bw_frame)
            domi_psd, peak_fre = bpm_fft(filtered_window , fps)
            max_freq = bpm_semilogy(filtered_window, fps)
            heartrate_fft_interp = bpm_fft_interp(filtered_window,fps)
            # print the heart_rate
            hr_peak.append(heart_rate_peak )
            hr_fft.append(peak_fre*60)
            hr_semi.append(max_freq * 60)
            hr_fft_interp.append(heartrate_fft_interp)
            # extract time domain features
            values = timedomain_features(filtered_window, vid_length, n_files)
            #feture vector
            lst.append(values)
            video_name.append(file)
            count = count+1
    #convert filename list to dataFrame
    #convert the list into pandas dataframe
    arr = np.array(lst)
    num_of_patients = []
    for i in range(0, count ):
        num_of_patients.append(i)
    index = []
    #print patient name
    for patient_num in num_of_patients:
        new_name = 'patient' + str(int(patient_num) + 1)
        index.append(new_name)
    columns = ["PP", "FF", "PF", "FP", "pp_ff", "pp_fp", "fp_ff", "fp_pf", "ppg_height", "af_height", "ad_height",
               "systolic_area", "diastolic_area", "ratio_area", "total_area", "delta_time", "reflection_index"]
    df = pd.DataFrame(arr, video_name, columns)
    df.to_csv('Patient_data.csv', index=True)
    #print("Heart rate using peak analysis", hr_peak)
    #print("Heart rate using FFT analysis", hr_fft)
    #print("Heart rate using Semilogy analysis", hr_semi)
    #print("Heart rate using FFT_interp", hr_fft_interp)
    #load the actual data
    dfs = pd.read_excel("Book1_Actualdata.xlsx")
    sys_pressure = dfs["SYSTOLIC_PRESSURE"]
    dias_pressure = dfs["DIASTOILC_PRESSURE"]
    pickled_model  = bp_regression(sys_pressure, dias_pressure)
    return pickled_model
if __name__ == '__main__':
    pickled_model = main(sys.argv[1])





