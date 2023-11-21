#Peak Processing and parameter extraction
from scipy.signal import find_peaks
from scipy import signal
import numpy as np
def timedomain_features(filtered_window,vid_length,n_files):

    p_peaks, p_a = find_peaks(filtered_window, distance=10)
    n_peaks, n_a = find_peaks(-filtered_window, distance=10)
    y_spl_2d = np.diff(filtered_window, n=1)
    m_peaks, m_a = find_peaks(y_spl_2d)
    idx_p, props = find_peaks(filtered_window, distance = 10)
    val_p = filtered_window[idx_p]
    #print(val_p)
    #print(props)
    idx_n, props = find_peaks(-filtered_window, distance=10)
    val_n = filtered_window[idx_n]
    # derivative of the signal
    # first derivative of the signal
    y_spl_2d = np.diff(filtered_window, n=1)
    # Second derivative of the signal
    y_spl_2d1 = np.diff(filtered_window, n=2)
    # plt.plot(y_spl_2d1)
    # find peaks of the derivative signal
    m_peaks, m_a = find_peaks(y_spl_2d)
    val_l = y_spl_2d[m_peaks]
    val_length = val_l.size
    # define empty matrix to calculate dicrotic notch points
    dicrotic_points = []
    for i in range(0, val_length - 1):
        if (val_l[i] - val_l[i + 1]) > 0:
            dicrotic_points.append(m_peaks[i + 1])
            array_dicrotic_points = np.asarray(dicrotic_points)
    val_m = filtered_window[array_dicrotic_points]
    #Time domain parameters
    a_range = vid_length / n_files
    time_info = []
    for i in np.arange(0, vid_length, a_range):
        time_info.append(i)
    time_info1 = np.array(time_info)
    time_info2 = np.around(time_info1, 2)
    # peak selection from positive and negative peaks
    p_peak_length = len(p_peaks)
    select_signal = round(p_peak_length / 2)
    # peak selection from Dicrotic notches
    select_dicrotic_signal = round(val_m.size / 2)
    # time difference between two systolic peaks(T2n+1-T2n)
    p_peakinfo = p_peaks[select_signal]
    p_peakinfo1 = p_peaks[select_signal - 1]
    #feature1
    PP = time_info2[p_peakinfo] - time_info2[ p_peakinfo1]
    # time difference between two diastolic peaks(T1n+1-T1n)
    n_peakinfo = n_peaks[select_signal]
    n_peakinfo1 = n_peaks[select_signal - 1]
    #feature2
    FF = time_info2[n_peakinfo] - time_info2[n_peakinfo1]
    #feature3
    PF = time_info2[ p_peakinfo1] - time_info2[n_peakinfo1]
    #feature4
    FP = time_info2[ p_peakinfo1]-time_info2[n_peakinfo]
    #feature5
    pp_ff = PP/FF
    #feature6
    pp_fp = PP/FP
    #feature7
    fp_ff = FP/FF
    #feature8
    fp_pf = FP/PF
    #feature9(ppg signal height)
    ppg_height = val_p[select_signal]
    #feaure10
    af_height = val_n[select_signal]
    #feature11
    ad_height = val_m[select_dicrotic_signal]
    #area under the curve
    # Systolic Area
    x1 = n_peaks[select_signal]
    x2 = p_peaks[select_signal]
    y1 = val_p[select_signal + 1]
    y2 = val_n[select_signal]
    average_y = (y1 + y2) / 2
    delta_x = (x2 - x1)
    # feature11
    systolic_area = average_y * delta_x
    # Diastolic area
    x3 = n_peaks[ select_dicrotic_signal]
    x4 = m_peaks[ select_dicrotic_signal]
    y3 = val_n[ select_dicrotic_signal]
    y4 = val_m[ select_dicrotic_signal]
    average_y1 = (y3 + y4) / 2
    delta_x1 = (x3 - x4)
    #feature12
    diastolic_area = average_y1 * delta_x1
    # feature13
    ratio_area = systolic_area /diastolic_area
    # feature14
    total_area = systolic_area + diastolic_area
    # feature15
    delta_time = delta_x = (x2 - x1)
    # feature16
    augmentation_index = ad_height-af_height/ppg_height-af_height
    # feature17
    reflection_index = 1-augmentation_index
    values = [PP, FF, PF, FP, pp_ff, pp_fp,  fp_ff,fp_pf,  ppg_height,af_height,ad_height,systolic_area,diastolic_area,
               ratio_area,total_area,delta_time,reflection_index]
    return values

