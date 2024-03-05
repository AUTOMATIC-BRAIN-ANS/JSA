import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import ndarray
from scipy.signal import find_peaks

plt.style.use('seaborn-v0_8-ticks')


def get_peaks(signal: pd.Series, distance: int = 115) -> tuple[ndarray, pd.Series, ndarray]:
    """
    Find peaks inside a signal based on minimal distance between peaks
    Params:
        signal (pd.Series):
        distance (int): minimal number of samples between two consecutive peaks
    Returns:
        peaks (ndarray): array of sample numbers corresponding to peaks found
        signal[peaks] (pd.Series): signal values corresponding to subsequent sample numbers
        difference (ndarray[Any, dtype[int]]): array containing distances between two consecutive peaks
    """
    peaks, _ = find_peaks(signal, distance=distance)
    difference = np.diff(peaks)
    return peaks, signal[peaks], difference


def make_empty_matrix() -> pd.DataFrame:
    """
    Prepare empty word distribution density matrix
    Returns:
         my_matrix (pd.DataFrame): empty word distribution density matrix
    """
    symbolic_sequences = ['111', '110', '101', '100', '011', '010', '001', '000']
    empty_matrix = pd.DataFrame(index=symbolic_sequences, columns=symbolic_sequences)
    empty_matrix = empty_matrix.fillna(0)
    return empty_matrix


def get_diff_df(dataframe: pd.DataFrame, is_fv: bool) -> pd.DataFrame:
    """
     Find RR intervals and Systolic Arterial Pressure (SAP) values, and construct a dataframe with two columns
     containing differences between two consecutive values, one for RR intervals and the other for SAP values
    Params:
        dataframe (pd.DataFrame): dataframe containing volunteer's signals as different columns
        is_fv (bool): controls whether ABP or FV is used to find peak values
    Returns:
        diff_dataframe (pd.DataFrame): dataframe containing two columns representing differences between consecutive
        values, one for differences between RR intervals and the other for differences between SAP values
    """
    if not is_fv:
        sap = get_peaks(dataframe['abp_arm'])[1]
    else:
        sap = get_peaks(dataframe['fv[cm/s]'])[1]

    rr = get_peaks(dataframe['abp_arm'])[2]

    sap_diff = np.diff(sap)
    rr_diff = np.diff(rr)
    min_length = min(len(sap_diff), len(rr_diff))  # sap and rr columns do not have equal length

    diff_dataframe = pd.DataFrame({
        "sap_diff": sap_diff[:min_length],
        "rr_diff": rr_diff[:min_length]
    })
    return diff_dataframe


def make_and_save_heatmap(relative_array: pd.DataFrame, breath_frequency: str, is_fv: bool, heatmap_path: str,
                          fontsize: int = 10, fig_extension: str = ".pdf", dpi: int = 600) -> None:
    """
    Save word density distribution matrix as a PDF and a CSV file to selected path
    Params:
        relative_array (pd.DataFrame): word distribution density matrix
        breath_frequency (str): respiratory rate during controlled breathing
        is_fv (bool): controls which signal (ABP or FV) is used to find peak values (SAP)
        heatmap_path (str): the location where the files will be saved
        fontsize (int): chart label font size, by default set to 10 pt
        fig_extension (str): extension of the saved figure, by default set to .pdf
        dpi (int): dpi of the saved figure, by default set to 600
    """
    color_palette = sns.color_palette("ch:-.24", as_cmap=True)
    my_heatmap = sns.heatmap(relative_array, cmap=color_palette, annot=True)
    my_heatmap.xaxis.tick_top()
    my_heatmap.xaxis.set_ticks_position('none')
    plt.xlabel('SAP', fontsize=fontsize)
    my_heatmap.xaxis.set_label_position('top')
    plt.ylabel('RR', fontsize=fontsize)
    if not is_fv:
        plt.savefig(heatmap_path + f"\\{breath_frequency}_heatmap_abp" + fig_extension, bbox_inches='tight', dpi=dpi)
        relative_array.to_csv(heatmap_path + f"\\{breath_frequency}_matrix_abp.csv")
    else:
        plt.savefig(heatmap_path + f"\\{breath_frequency}_heatmap_fv" + fig_extension, bbox_inches='tight', dpi=dpi)
        relative_array.to_csv(heatmap_path + f"\\{breath_frequency}_matrix_fv.csv")
    plt.show()
    plt.close()


def save_jsd(is_fv: bool, jsd_df: pd.DataFrame, jsd_path: str) -> None:
    """
    Save JSD table as a CSV file to selected path
    Params:
        is_fv (bool): controls which signal (ABP or FV) is used to find peak values (SAP)
        jsd_df (pd.DataFrame): dataframe containing volunteer's JSDsym, JSDdiam, mean ABP, mean EtCO2,
        and mean FV values
        jsd_path (str): the location where the table will be saved
    """
    if not is_fv:
        jsd_df.T.to_csv(jsd_path + "\\jsd_abp.csv")
    else:
        jsd_df.T.to_csv(jsd_path + "\\jsd_fv.csv")


def add_mean_params(jsd_df: pd.DataFrame, dataframe: pd.DataFrame, patient_number: str, freq: str,
                    window_size: int = 2000) -> pd.DataFrame:
    """
    Add volunteer's mean ABP, mean EtCO2, and mean FV values as columns to dataframe containing volunteer's JSDsym
    and JSDdiam values
    Params:
        jsd_df (pd.DataFrame): dataframe containing volunteer's JSDsym and JSDdiam values as columns
        dataframe (pd.DataFrame): dataframe containing volunteer's signals as columns
        patient_number (str): volunteer's id number
        freq (str): respiratory rate during controlled breathing
        window_size (int): moving average window size
    Returns:
        jsd_df (pd.DataFrame): same dataframe with added volunteer's mean ABP, mean EtCO2, and mean FV values
    """
    for col_name in ['abp_arm', 'etco2', 'fv[cm/s]']:
        if col_name in dataframe.columns:
            nan_percent = dataframe[col_name].isnull().sum() * 100 / len(dataframe)
            if nan_percent != 100.0:
                jsd_df.at["mean_" + col_name + "_" + freq, patient_number] = dataframe[col_name].rolling(
                    window=window_size).mean().dropna().mean()
    return jsd_df


def joint_symbolic(data_dict: dict, is_fv: bool, heatmap_path: str, jsd_path: str) -> None:
    """
    Perform Joint Symbolic Analysis on a dictionary containing signals acquired during controlled breathing sessions

    Params:
        data_dict (dict): nested dictionary, outer dictionary contains keys corresponding to breathing
        frequencies per minute (e.g., 6_breaths, 10_breaths, 15_breaths) and each value associated with a breathing
        frequency is an inner dictionary. For inner dictionary, the key is volunteer's ID number and the value
        is a dataframe that contains various signals as different columns
        is_fv (bool): controls which signal (ABP or FV) is used to find peak values (SAP)
        heatmap_path (str): the location where the word density distribution matrix will be saved
        jsd_path (str): the location where dataframe containing trace of word density distribution matrix (JSDsym),
        anti-trace of word distribution density matrix (JSDdiam), mean end-tidal carbon dioxide (EtCO2), mean
        arterial blood pressure (ABP), and mean flow velocity (FV) values will be saved
    """
    jsd_df = pd.DataFrame()
    for freq in data_dict:
        freq_matrix = make_empty_matrix()
        for patient_number, dataframe in data_dict[freq].items():
            patient_freq_matrix = make_empty_matrix()
            diff_df = get_diff_df(dataframe, is_fv)

            for col_name in ['sap_diff', 'rr_diff']:
                diff_df[f'{col_name}_binary'] = (diff_df[col_name] > 0).astype(int)

            for index in range(0, len(diff_df)):
                sap_triplet = ''.join(diff_df['sap_diff_binary'].iloc[index:index + 3].astype(str))
                rr_triplet = ''.join(diff_df['rr_diff_binary'].iloc[index:index + 3].astype(str))
                if len(rr_triplet) > 2 and len(sap_triplet) > 2:
                    freq_matrix.at[rr_triplet, sap_triplet] += 1
                    patient_freq_matrix.at[rr_triplet, sap_triplet] += 1

            relative_patient_matrix = patient_freq_matrix.div(np.sum(patient_freq_matrix.values))
            signal_map = {True: 'FV_', False: 'ABP_'}
            jsd_df.at[signal_map[is_fv] + "jsd_sym_" + freq, patient_number] = np.sum(np.diag(relative_patient_matrix))
            jsd_df.at[signal_map[is_fv] + "jsd_diam_" + freq, patient_number] = np.sum(
                np.diag(np.rot90(relative_patient_matrix)))
            jsd_df = add_mean_params(jsd_df, dataframe, patient_number, freq)

        relative_freq_matrix = freq_matrix.div(np.sum(freq_matrix.values))
        make_and_save_heatmap(relative_freq_matrix, freq, is_fv, heatmap_path)
    save_jsd(is_fv, jsd_df, jsd_path)



