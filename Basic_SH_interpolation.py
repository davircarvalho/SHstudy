'''
Spherical Harmonics interpolation experiment
'''
# %% Libs
from scipy.special import sph_harm
import numpy as np
import SOFASonix as sofa
import matplotlib.pyplot as plt


# %% load SOFA HRTF
path = r'D:\Documentos\1 - Work\Individualized_HRTF_Synthesis\Datasets\HUTUBS\pp1_HRIRs_simulated.sofa'
Obj = sofa.SOFAFile.load(path)
H = np.fft.fft(Obj.Data_IR, axis=-1)
N_pos = Obj.SourcePosition.shape[0]

# %% Desired output positions
resolution = 2  # degrees
des_azi = np.arange(0, 360, resolution)
des_ele = np.arange(-90, 90, resolution)
des_pos = np.zeros((len(des_azi) * len(des_ele), 2))
k = 0
for el in des_ele:
    for az in des_azi:
        des_pos[k, :] = [az, el]
        k += 1


# %% Interpolation
def get_SH(Lmax, pos):
    # Lmax: max SH order
    # pos: array [N_pos x 2] : 2= azimuth, elevation
    theta = np.deg2rad(pos[:, 0])  # must be in [0, 2*pi]
    phi = np.deg2rad(pos[:, 1] + 90)  # must be in [0, pi].
    N_SH = int((Lmax + 1)**2)  # number of SH coefficients
    Y = np.zeros((pos.shape[0], N_SH), dtype=complex)
    count = 0
    for ni in np.arange(0, Lmax + 1):
        for mi in np.arange(-ni, ni + 1):
            Y[:, count] = sph_harm(mi, ni, theta, phi)
            count += 1
    return Y


# Calculate coefficients for input positions
Lmax = int(np.floor(np.sqrt(N_pos) - 1))  # max order
SH = get_SH(Lmax, Obj.SourcePosition[:, :2])
Sinv = np.linalg.pinv(SH)
# Direct SFT
H_sh_L = Sinv @ H[:, 0, :]
H_sh_R = Sinv @ H[:, 1, :]

# Calculate desired positions SH coefficients
SH_des_pos = get_SH(Lmax, des_pos)
H_interp = np.zeros((des_pos.shape[0], 2, H.shape[-1]), dtype=complex)
# Inverse SFT
H_interp[:, 0, :] = SH_des_pos @ H_sh_L
H_interp[:, 1, :] = SH_des_pos @ H_sh_R

HRIRs = np.fft.ifft(H_interp, axis=-1)


# %% PLOTS ------------------------------------------------------------------------------
def plot_mag_horizontal(HRIRs, pos, fs, title):
    # find horizontal plane
    ear = 0
    idx_pos = np.where(np.logical_and(pos[:, 1] > -1, pos[:, 1] < 1))
    pos = np.squeeze(pos[idx_pos, :2])
    hM = np.squeeze(HRIRs[:, ear, :])  # left ear
    M = np.squeeze(20 * np.log10(np.abs(np.fft.rfft(hM[idx_pos, :], axis=-1))))

    azi = np.sort(pos[:, 0], axis=0)
    i = np.argsort(pos[:, 0], axis=0)
    M = M[i, :]

    freq = np.fft.rfftfreq(HRIRs.shape[-1], 1 / fs)
    fig, ax = plt.subplots()
    ax.pcolormesh(freq, azi, M, shading='auto')
    plt.ylim([0, 360])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Azimuth (deg)')
    plt.title(title)
    plt.savefig(f'Images/{title}.jpeg', dpi=800)
    plt.show()


def plot_mag_vertical(HRIRs, pos, fs, title):
    # find horizontal plane
    ear = 1
    azi = 0
    idx_pos = np.where(np.logical_and(pos[:, 0] > -1, pos[:, 0] < 1))
    pos = np.squeeze(pos[idx_pos, :2])
    hM = np.squeeze(HRIRs[:, ear, :])  # left ear
    M = np.squeeze(20 * np.log10(np.abs(np.fft.rfft(hM[idx_pos, :], axis=-1))))

    ele = np.sort(pos[:, 1], axis=0)
    i = np.argsort(pos[:, 1], axis=0)
    M = M[i, :]
    freq = np.fft.rfftfreq(HRIRs.shape[-1], 1 / fs)
    fig, ax = plt.subplots()
    ax.pcolormesh(freq, ele, M, shading='auto')
    plt.ylim([-90, 90])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Elevation (deg)')
    plt.title(title)
    plt.savefig(f'Images/{title}.jpeg', dpi=800)
    plt.show()


fs = Obj.Data_SamplingRate
plot_mag_vertical(HRIRs, des_pos, fs, title='Vertical (interpolated)')
plot_mag_vertical(Obj.Data_IR, Obj.SourcePosition, fs, title='Vertical (reference)')

plot_mag_horizontal(HRIRs, des_pos, fs, title='Horizontal (interpolated)')
plot_mag_horizontal(Obj.Data_IR, Obj.SourcePosition, fs, title='Horizontal (reference)')
