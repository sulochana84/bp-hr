""" Prgram to measure heart rate, time and frequency domain features"""
#import all the required libraries
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
from scipy.fftpack import fft, fftfreq
from sys import argv
import sys
import pandas as pd
import pickle

def video_clip(folder):

    ffmpeg_extract_subclip(folder, 2, 18, targetname="clipped_video/test.MOV")

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
def redchannel_extraction(path, total_pixels, number_files):
    """this function coverts each frame into single columns and
    a form matrix"""
    red_channel = np.zeros(shape=(total_pixels, number_files))
    count = 0
    for img in glob.glob(path):
        r_img = cv2.imread(img)
        b1, g1, r1 = cv2.split(r_img)
        n_image = np.asarray(r1)
        red_frame = n_image.ravel()
        red_channel[:, count] = red_frame
        count = count + 1
    return red_channel
def deleteprevious(path):
    """ function deletes the images in the folder from previous run"""
    removing_files = glob.glob(path)
    for i in removing_files:
        os.remove(i)
def ppg_signal(X, X_reconstructed, num_rows, num_columns, number_files ):
    """ PPG signal construction"""
    PPG_signal = X - X_reconstructed
    error_signal = []
    for i in range(number_files):
        row_ppg = PPG_signal[:, i]
        error_frame = np.reshape(row_ppg, (num_rows, num_columns))
        signal_content = error_frame.sum() / (num_rows * num_columns)
        error_signal.append(signal_content)
    ppg_s = np.array(error_signal)
    return ppg_s
def limit_xaxis(vid_length,n_files):
    """ this function to specify the x -axis limit"""
    a_range = vid_length / n_files
    xaxis1 = []
    for i in np.arange(0, vid_length, a_range):
        xaxis1.append(i)
    xaxis2 = np.array(xaxis1)
    xaxis = np.around(xaxis2, 2)
    return xaxis
def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], 'band')
    y = scipy.signal.lfilter(b, a, data)
    return y

def butter_highpass(cutoff, fs, order):
    nyq = 0.5*fs   #Nyquist
    normal_cutoff = cutoff/nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False, output='ba')
    return b, a
#calculating low pass filter co-efficient
def butter_lowpass(cutoff, fs, order):
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False, output='ba')
    return b, a

def filter_all(data, fs, order,cutoff_high, cutoff_low):
    b, a = butter_highpass(cutoff_high, fs, order=order)
    highpassed_signal = filtfilt(b, a, data)
    d, c = butter_lowpass(cutoff_low, fs, order = order)
    bandpassed_signal = filtfilt(d, c, highpassed_signal)
    return bandpassed_signal

def filter_s(signal,Fs):
   """This function retured the denoised PPG singnal"""
   pass_band = (40 / 60.0, 220 / 60.0)
   b, a = scipy.signal.butter(3, pass_band, btype='bandpass', fs=Fs)
   return scipy.signal.filtfilt(b, a, signal)

def simple_moving_average(signal, window):
    """ This function smooth the singal using moving average """
    denoised_signal = np.convolve(signal, np.ones(window)/window, mode='same')
    return denoised_signal

def give_bpm(r_averaged,fps):
    """ This function returns heart rate using peak
    information"""
    # returns peak height and index at which peak occured
    r_peaks = find_peaks(r_averaged>0)
    time_bw_frame = 1 / fps
    # calculating average frames between peaks
    diff_sum = 0
    total_peaks = len(r_peaks[0])
    i = 0
    while i < total_peaks - 1:
        diff_sum = diff_sum + r_peaks[0][i + 1] - r_peaks[0][i]
        i = i + 1

    avg_diff = diff_sum / (total_peaks - 1)
    # final calculation of Heart rate and error percentage
    avg_time_bw_peaks = avg_diff * time_bw_frame

    bpm = 60 / avg_time_bw_peaks

    #print("Calculated Heart rate = ", bpm)
    return bpm
def new_peaks(filtered, idx_p):
    """ This function is to eliminate unwanted positive
    and negative peaks from the denoised PPG signal"""
    val_p = filtered[idx_p]
    size_peaks = idx_p.size
    idx_p1 = []
    for i in range(0, size_peaks):
        if val_p[i] > 0:
            idx_p1.append(idx_p[i])

    val_p1 = val_p[val_p > 0]
    return idx_p1,val_p1
def main(video_path):
    """ main function to process the video and return heart_rate ,
    time and frequency domain parameters"""
    #specify the video path
    video_cap = video_path
    #clipt the video
    video_clip( video_cap )
    #load the cropped vieod from te folder
    cropped_video = cv2.VideoCapture(video_cap )
    # Information about the video
    #calculate the number of frames per second
    fps = cropped_video .get(cv2.CAP_PROP_FPS)
    # calculate the total number of frames in the video
    frame_count = int( cropped_video.get(cv2.CAP_PROP_FRAME_COUNT))
    #length of the video
    vid_length = frame_count / fps
    #time between frames
    time_bw_frame = 1 / fps
    #print(" video length" , vid_length)
    #print("Time between frames",time_bw_frame)
    #print("frames per second", fps)
    #this folder store the video frames
    path = "outframe/*.png"
    folder1 = "outframe"
    #clean the folder
    deleteprevious(path)
    # read the stable video
    x,y,x2,y2 = face_boundingbox_size( cropped_video)
    face_bbox(  cropped_video ,x,y,x2,y2)
    n_files = frame_length(folder1)
    rows, cols, t_pixels = frame_parameters(path)
    #print("The number of rows in the frame:", rows)
    #print("The number of columns in the frame:", cols)
    #print("The number of frames in the video:", n_files)
    #print("The number of pixels in each of the frame:", t_pixels)
    #extract the red channel of each frame in the video
    red_channel_frames = redchannel_extraction("outframe/*.png",t_pixels, n_files)
    #reconstruct the PCA of the signal
    recontruct_pca_redchannel = pca_redchannel(red_channel_frames)
    # Calcultae the PPG Signal
    final_ppg = ppg_signal(red_channel_frames, recontruct_pca_redchannel,rows, cols,n_files)
    # round the ppg signal
    final_ppg = np.round(final_ppg, 2)
    xaxis= limit_xaxis(vid_length, n_files)
    # plotting the points
    #plt.plot(xaxis,final_ppg)
    # naming the x axis
    #plt.xlabel('time in seconds')
    # naming the y axis
    #plt.ylabel('amplitude')
    # giving a title to my graph
    #plt.title('ppg signal!')
    # function to show the plot
    #fig = plt.figure(1)
    ##plt.show()
    #fig.clf()
    #band pass filtering of PPG signal
    y = filter_s(final_ppg,fps)
    #y = filter_all(final_ppg, fps, order=5,cutoff_high= 3,cutoff_low=0.3)
    #y = butter_bandpass_filter(final_ppg, lowcut=0.5, highcut=5, fs=fps, order=4)
    #fig = plt.figure(2)
    xaxis = range(0,n_files)
   # plt.plot(xaxis, y)
    fig = plt.figure(2)
    #plt.xlabel('time in seconds')
    # naming the y axis
    ##plt.ylabel(' Amplitude')
    # giving a title to my graph
    #plt.title(' filtered  signal!')
    #plt.show()
    #fig.clf()
    #smooth the signal
    filtered_signal = simple_moving_average(y, window=5)
    #plt.plot(filtered_signal)
    ##plt.title('Averaged Signal!')
    #plt.show()
    # measure the heart rate using peak finformation
    heart_rate = give_bpm(filtered_signal,fps)
    #positive peaks of the signal
    p_peaks, p_a = find_peaks(filtered_signal, distance=10)  # returns peak height and index at which peak occured
    #eliminate unwanted peaks
    idx_p , val_p = new_peaks(filtered_signal, p_peaks)
    #negative peaks of the signal
    n_peaks, n_a = find_peaks(-filtered_signal,distance=10)
    # eliminate unwanted peaks
    idx_n, val_n = new_peaks(-filtered_signal, n_peaks)
    ##fig = plt.figure(5)
    #plt.plot(filtered_signal)
    #plt.plot( idx_p, filtered_signal[ idx_p], "x")
    #plt.plot(np.zeros_like(filtered_signal), "--", color="gray")
    #plt.plot(idx_n, filtered_signal[idx_n], "x")
    #plt.plot(np.zeros_like(filtered_signal), "--", color="green")
    #plt.ylabel(' Amplitude')
    #plt.title(' Positive and negative peaks of the final signal!')
    #plt.xlabel('time in seconds')
    #plt.show()
    #fig.clf()
    #Derivative of the PPG signal
    #first derivative of the signal
    first_deri_signal = np.diff(filtered_signal, n=1)
    #second derivative of the signal
    second_deri_signal= np.diff(filtered_signal, n=2)
    #plt.plot(first_deri_signal)
    # plt.plot(y_spl_2d1)
    #plt.plot(filtered_signal)
    ##plt.title(' Derivative of the PPG signal!')
    #plt.show()
    # find peaks in the derivative of the signal
    m_peaks, m_a = find_peaks(first_deri_signal>0,distance=10)
    # eliminate unwanted peaks from first derivative signal
    val_l = first_deri_signal [m_peaks]
    #size of the peaks of first derivative signal
    val_length = val_l.size
    # plt.plot(filtered)
    #plt.plot(first_deri_signal)
    #plt.plot(m_peaks, first_deri_signal[m_peaks], "x")
    #plt.plot(np.zeros_like(first_deri_signal ), "--", color="gray")
    #plt.title(' Peaks in the derivative of the PPG signal!')
    #plt.show()
    #Dicrotic notch detection
    a_range = vid_length / n_files
    time_info = []
    #initialise empty list to store valley points of the derivative signal
    v_points = []
    for i in range(0, val_length - 1):
        if (val_l[i] - val_l[i + 1]) > 0:
            v_points.append(m_peaks[i + 1])
    v_points1 = np.asarray(v_points)
    val_m = filtered_signal[v_points1]
    plt.plot(filtered_signal)
    #plt.plot(y_spl_2d)
    #plt.plot(v_points1, filtered_signal[v_points1], "x")
    #plt.plot(np.zeros_like(first_deri_signal), "--", color="gray")
    #plt.title("Dicrotic notch of the signal")
    #plt.show()
    # Peak Processing and parameter extraction
    #calculate the time using video length
    a_range = vid_length / n_files
    for i in np.arange(0, vid_length, a_range):
        time_info.append(i)
    time_info1 = np.array(time_info)
    time_info2 = np.around(time_info1, 2)
    #find the peak to calculate peak parameters
    #peak selection from positive and negative peaks
    p_peak_length = len(p_peaks)
    select_signal = round(p_peak_length/2)
    # peak selection from Dicrotic notches
    select_dicrotic_signal = round(val_m.size/2)
    p_peakinfo = p_peaks[select_signal]
    p_peakinfo1 = p_peaks[select_signal-1]
    #feature 1  time difference between two systolic peaks
    PP = time_info2[p_peakinfo] - time_info2[ p_peakinfo1]
    n_peakinfo = n_peaks[select_signal]
    n_peakinfo1 = n_peaks[select_signal-1]
    #feature 2 time difference between two diastolic peaks
    FF = time_info2[n_peakinfo] - time_info2[n_peakinfo1]
    #feature 3 time difference between diastolic and systolic peaks
    PF = time_info2[ p_peakinfo1] - time_info2[n_peakinfo1]
    # feature 4 time difference between systolic and diastolic peaks
    FP = time_info2[ p_peakinfo1]-time_info2[n_peakinfo]
    # feature 5
    pp_ff = PP/FF
    #feature 6
    pp_fp = PP/FP
    # feature 7
    fp_ff = FP/FF
    # feature 7
    fp_pf = FP
    #feature 8
    ppg_height = val_p[select_signal]
    # feature 9
    af_height = val_n[select_signal]
    # feature 10
    ad_height = val_m[select_dicrotic_signal]
    #feature 11
    #area under the curve
    # Systolic Area
    x1=  n_peaks[select_signal]
    x2= p_peaks[select_signal]
    y1=  val_p[select_signal+1]
    y2=  val_n[select_signal]
    average_y = (y1 + y2) / 2
    delta_x = (x2 - x1)
    systolic_area = average_y * delta_x
    # feature 12 Diastolic area
    x3 = n_peaks[select_dicrotic_signal]
    x4 = m_peaks[select_dicrotic_signal]
    y3 = val_n[select_dicrotic_signal]
    y4 = val_m[select_dicrotic_signal]
    average_y1 = (y3 + y4) / 2
    delta_x1 = (x4 - x3)
    diastolic_area = average_y1 * delta_x1
    #feature 13
    ratio_area =  systolic_area /diastolic_area
    #feature 14
    total_area = systolic_area + diastolic_area
    #feature 15
    delta_time = delta_x = (x2 - x1)
    #feature 16
    augmentation_index = ad_height-af_height/ppg_height-af_height
    #feature 17
    reflection_index = 1-augmentation_index
    #Frequency domain analysis
    #first method
    # Perform FFT with SciPy
    from scipy.fftpack import fft, fftfreq
    signalFFT = abs(fft(filtered_signal))
    signalPSD = np.abs(signalFFT) ** 2
    spacing = 1/fps
    fftfreqency = fftfreq(len(signalPSD), spacing)
    i = fftfreqency>0
    plt.figurefigsize = (8, 4)
    #plt.plot(fftfreq[i], 10*np.log10(signalPSD[i]))
    #plt.plot(fftfreqency[i], signalPSD[i])
    #plt.xlabel('Frequency [Hz]')
    #plt.ylabel('PSD [dB]')
    #plt.show()
    # Find the peak frequency: we can focus on only the positive frequencies
    pos_mask = np.where(fftfreqency > 0)
    freqs = fftfreqency[pos_mask]
    peak_freq = freqs[signalPSD[pos_mask].argmax()]
    #print("peak frequency",peak_freq)
    heart_rate_fft = peak_freq*60
    # second method
    f, S = signal.periodogram(filtered_signal, fps)
    #(S, f) = plt.psd(filtered, Fs=fps)
    #plt.semilogy(f, S)
    #plt.xlabel('frequency [Hz]')
    #plt.ylabel('PSD [V**2/Hz]')
    #plt.show()
    max_y = max(S)  # Find the maximum y value
    max_x = f[S.argmax()]  # Find the x value corresponding to the maximum y value
    heartrate_semilogy = max_x*60
    # Third method
    #length of the signal
    L = len(filtered_signal)
    #limit of the x-axis
    even_times = np.linspace(0, L, L)
    #interpolate the signal
    interpolated = np.interp(even_times, even_times, filtered_signal)
    # apply hamming window to the interpolated signal
    interpolated = np.hamming(L) * interpolated
    # normalised signal
    norm_signal = interpolated / np.linalg.norm(interpolated)
    raw_signal = np.fft.rfft(norm_signal * fps)
    #frequency calculatiom
    freq = np.fft.rfftfreq(L, 1 / fps) * 60
    #power spectral density of the signal
    fft = np.abs(raw_signal) ** 2
    #plt.plot(freq, fft, color="blue")
    #plt.show()
    peak_freq_interpol = freq[fft.argmax()]
    #print("heart_rate_FFT_first method", heart_rate_fft)
    #print("heart_rate_semilogy_second method", heartrate_semilogy)
    #print("heart_rate_no.of peaks", heart_rate)
    #print("heart_rate_FFT_interp", peak_freq_interpol )
    values = [PP, FF, PF, FP, pp_ff, pp_fp, fp_ff, fp_pf, ppg_height, af_height, ad_height, systolic_area,
              diastolic_area,ratio_area, total_area, delta_time, reflection_index]

    return peak_freq_interpol, heart_rate, values

if __name__== "__main__":
     peak_freq_interpol, heart_rate,values = main(sys.argv[1])
     print("heart_rate_no.of peaks", heart_rate)
     print("heart_rate_FFT_interp", peak_freq_interpol)

     arr = np.array(values)
     num_of_patients = []
     columns = ["PP", "FF", "PF", "FP", "pp_ff", "pp_fp", "fp_ff", "fp_pf", "ppg_height", "af_height", "ad_height",
                "systolic_area", "diastolic_area", "ratio_area", "total_area", "delta_time", "reflection_index"]
     #create a dataFrame
     df = pd.DataFrame(arr, columns)
     #transpose of the dataFrame
     df = df.T
     # load the pretrained model
     pickled_model = pickle.load(open('lr_model.pkl', 'rb'))
     #prediction
     ypred = pickled_model.predict(df)
     print("Systolic pressure & Diastolic pressure",ypred)
