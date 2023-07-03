# ------------------------------------------------------------------------
# Data operation utilities
# ------------------------------------------------------------------------
# Author: Marius Bock
# Email: marius.bock(at)uni-siegen.de
# ------------------------------------------------------------------------

import numpy as np
import pandas as pd


def sliding_window_samples(data, win_len, overlap_ratio=None):
    """
    Return a sliding window measured in seconds over a data array.

    :param data: dataframe
        Input array, can be numpy or pandas dataframe
    :param length_in_seconds: int, default: 1
        Window length as seconds
    :param sampling_rate: int, default: 50
        Sampling rate in hertz as integer value
    :param overlap_ratio: int, default: None
        Overlap is meant as percentage and should be an integer value
    :return: tuple of windows and indices
    """
    windows = []
    indices = []
    curr = 0
    overlapping_elements = 0
    win_len = int(np.ceil(win_len))
    if overlap_ratio is not None:
        if not ((overlap_ratio / 100) * win_len).is_integer():
            float_prec = True
        else:
            float_prec = False
        overlapping_elements = int((overlap_ratio / 100) * win_len)
        if overlapping_elements >= win_len:
            print('Number of overlapping elements exceeds window size.')
            return
    changing_bool = True
    while curr < len(data) - win_len:
        windows.append(data[curr:curr + win_len])
        indices.append([curr, curr + win_len])
        if (float_prec == True) and (changing_bool == True):
            curr = curr + win_len - overlapping_elements - 1
            changing_bool = False
        else:
            curr = curr + win_len - overlapping_elements
            changing_bool = True

    return np.array(windows), np.array(indices)

def apply_sliding_window(data, window_size, window_overlap):
    output_x = None
    output_y = None
    output_sbj = []
    for i, subject in enumerate(np.unique(data[:, 0])):
        subject_data = data[data[:, 0] == subject]
        subject_x, subject_y = subject_data[:, :-1], subject_data[:, -1]
        tmp_x, _ = sliding_window_samples(subject_x, window_size, window_overlap)
        tmp_y, _ = sliding_window_samples(subject_y, window_size, window_overlap)

        if output_x is None:
            output_x = tmp_x
            output_y = tmp_y
            output_sbj = np.full(len(tmp_y), subject)
        else:
            output_x = np.concatenate((output_x, tmp_x), axis=0)
            output_y = np.concatenate((output_y, tmp_y), axis=0)
            output_sbj = np.concatenate((output_sbj, np.full(len(tmp_y), subject)), axis=0)

    output_y = [[i[-1]] for i in output_y]
    return output_sbj, output_x, np.array(output_y).flatten()

def unwindow_inertial_data(orig, ids, preds, win_size, win_overlap):
    unseg_preds = []

    if not ((win_overlap / 100) * win_size).is_integer():
        float_prec = True
    else:
        float_prec = False

    for sbj in np.unique(orig[:, 0]):
        sbj_data = orig[orig[:, 0] == sbj]
        sbj_preds = preds[ids==sbj]
        sbj_unseg_preds = []
        changing_bool = True
        for i, pred in enumerate(sbj_preds):
            if (float_prec == True) and (changing_bool == True):
                sbj_unseg_preds = np.concatenate((sbj_unseg_preds, [pred] * (int(win_size * (1 - win_overlap * 0.01)) + 1)))
                if i + 1 == len(preds):
                    sbj_unseg_preds = np.concatenate((sbj_unseg_preds, [pred] * (int(win_size * (win_overlap * 0.01)) + 1)))
                changing_bool = False
            else:
                sbj_unseg_preds = np.concatenate((sbj_unseg_preds, [pred] * (int(win_size * (1 - win_overlap * 0.01)))))
                if i + 1 == len(preds):
                    sbj_unseg_preds = np.concatenate((sbj_unseg_preds, [pred] * int(win_size * (win_overlap * 0.01))))
                changing_bool = True
        sbj_unseg_preds = np.concatenate((sbj_unseg_preds, np.full(len(sbj_data) - len(sbj_unseg_preds), sbj_preds[-1])))
        unseg_preds = np.concatenate((unseg_preds, sbj_unseg_preds))
    assert len(unseg_preds) == len(orig)    
    return unseg_preds, orig[:, -1]
