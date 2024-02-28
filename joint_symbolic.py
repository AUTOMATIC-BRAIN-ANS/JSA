import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from scipy.signal import find_peaks

plt.style.use('seaborn-v0_8-ticks')


def get_peaks(signal):
    peaks, _ = find_peaks(signal, distance=115)
    difference = np.diff(peaks)
    return peaks, signal[peaks], difference


def make_empty_matrix():
    my_permutations = [''.join(map(str, triplet)) for triplet in list((itertools.product([1, 0], repeat=3)))]
    my_matrix = pd.DataFrame()
    for triplet_1 in my_permutations:
        for triplet_2 in my_permutations:
            my_matrix.at[
                "'" + triplet_1, "'" + triplet_2] = 0  # dash needed so that R does not remove the initial zeros
    return my_matrix


def get_diff_df(dataframe, is_fv, is_left, is_ekg):
    sap = get_peaks(dataframe['abp_arm'])[1]
    if is_fv:
        sap = get_peaks(dataframe['fv_l'])[1]
        if not is_left:
            sap = get_peaks(dataframe['fv_r'])[1]

    rr = get_peaks(dataframe['abp_arm'])[2] * 0.005  # in those without an ECG signal, we use ABP
    if is_ekg:
        rr = get_peaks(dataframe['ekg'])[2] * 0.005

    sap_diff = np.diff(sap)
    rr_diff = np.diff(rr)
    diff_dataframe = pd.DataFrame(list(zip(sap_diff, rr_diff)), columns=['sap_diff', 'rr_diff'])
    return diff_dataframe


def make_and_save_heatmap(relative_array, breath_frequency, is_fv, heatmap_path):
    my_custom_palette = sns.color_palette("ch:-.24", as_cmap=True)
    my_heatmap = sns.heatmap(relative_array, cmap=my_custom_palette, annot=True)
    my_heatmap.xaxis.tick_top()
    my_heatmap.xaxis.set_ticks_position('none')
    fontsize = 10
    plt.xlabel('SAP', fontsize=fontsize)
    my_heatmap.xaxis.set_label_position('top')
    plt.ylabel('RR', fontsize=fontsize)
    if is_fv:
        plt.savefig(heatmap_path + f"\\{breath_frequency}_heatmap_fv.pdf", bbox_inches='tight', dpi=600)
        relative_array.to_csv(heatmap_path + f"\\{breath_frequency}_matrix_fv.csv")
    else:
        plt.savefig(heatmap_path + f"\\{breath_frequency}_heatmap_abp.pdf", bbox_inches='tight', dpi=600)
        relative_array.to_csv(heatmap_path + f"\\{breath_frequency}_matrix_abp.csv")
    plt.show()
    plt.close()


def save_jsd(is_fv, jsd_df, jsd_path):
    if is_fv:
        jsd_df.T.to_csv(jsd_path + "\\jsd_fv.csv")
    else:
        jsd_df.T.to_csv(jsd_path + "\\jsd_abp.csv")


def add_mean_params(jsd_df, dataframe, is_left, patient_number, freq):
    window_size = 2000
    for col_name in ['abp_arm', 'etco2']:
        if col_name in dataframe.columns:
            nan_percent = dataframe[col_name].isnull().sum() * 100 / len(dataframe)
            if nan_percent != 100.0:
                jsd_df.at["mean_" + col_name + "_" + freq, patient_number] = dataframe[col_name].rolling(
                    window=window_size).mean().dropna().mean()
    if is_left:
        jsd_df.at["mean_fvr/l_" + freq, patient_number] = dataframe['fv_l'].rolling(
            window=window_size).mean().dropna().mean()
    else:
        jsd_df.at["mean_fvr/l_" + freq, patient_number] = dataframe['fv_r'].rolling(
            window=window_size).mean().dropna().mean()
    return jsd_df


def joint_symbolic(data_dict, is_fv, metadata_path, heatmap_path, jsd_path, is_ekg):
    jsd_df = pd.DataFrame()
    metadata = pd.read_csv(metadata_path, sep=";")
    for freq in data_dict:
        freq_matrix = make_empty_matrix()
        for patient_number, dataframe in data_dict[freq].items():
            is_left = bool(metadata.at[metadata.loc[metadata['L.poj.'] == int(patient_number)].index[0], 'FV_LEWA_MCA'])
            patient_freq_matrix = make_empty_matrix()
            diff_df = get_diff_df(dataframe, is_fv, is_left, is_ekg)

            for col_name in ['sap_diff', 'rr_diff']:
                diff_df[f'{col_name}_binary'] = diff_df.apply(lambda df_row: 1 if df_row[col_name] > 0 else 0, axis=1)

            for index in range(0, len(diff_df)):
                sap_triplet = ''.join(diff_df['sap_diff_binary'].iloc[index:index + 3].astype(str))
                rr_triplet = ''.join(diff_df['rr_diff_binary'].iloc[index:index + 3].astype(str))
                if len(rr_triplet) > 2 and len(sap_triplet) > 2:
                    freq_matrix.at[
                        "'" + rr_triplet, "'" + sap_triplet] += 1  # dash needed so that R does not remove the
                    # initial zeros
                    patient_freq_matrix.at["'" + rr_triplet, "'" + sap_triplet] += 1

            relative_patient_matrix = patient_freq_matrix.div(np.sum(patient_freq_matrix.values))
            signal_map = {True: 'FV_', False: 'ABP_'}
            jsd_df.at[signal_map[is_fv] + "jsd_sym_" + freq, patient_number] = np.sum(np.diag(relative_patient_matrix))
            jsd_df.at[signal_map[is_fv] + "jsd_diam_" + freq, patient_number] = np.sum(
                np.diag(np.rot90(relative_patient_matrix)))
            jsd_df = add_mean_params(jsd_df, dataframe, is_left, patient_number, freq)

        relative_freq_matrix = freq_matrix.div(np.sum(freq_matrix.values))
        make_and_save_heatmap(relative_freq_matrix, freq, is_fv, heatmap_path)
    save_jsd(is_fv, jsd_df, jsd_path)
