'''
Spherical Harmonics interpolation,
ref.: Pollow 012
'''
# %% Libs
import numpy as np
import SOFASonix as sofa
import matplotlib.pyplot as plt
from scipy.special import sph_harm, hankel2
from scipy.spatial import SphericalVoronoi
from scipy.sparse import csr_matrix
import copy


'''
TODO - check distance variation interpolation

'''


def sph2cart(positions):
    azimuth = positions[:, 0]
    elevation = positions[:, 1]
    r = positions[:, 2]
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return np.array([x, y, z]).T


def interpSH(HRIR, input_pos, target_pos, fs, epsilon=1e-8):
    '''
    Interpolate a set of HRIRs at the known positions to the target positions using
    sherical harmonics
    ------
    HRIR: [Npos x ears x samples]
    input_pos: [Npos x 3]
    target_pos: [Npos x 3]
    epsilon: regularization value
    fs: sampling frequency
    ------
    all the variables above are expected to follow the SimpleFreeFieldHRIR SOFA conventions
    '''
    def SH_coeffs(Lmax, pos):
        # Lmax: max SH order
        # pos: array [N_pos x 2] : 2= azimuth, elevation
        N_pos = pos.shape[0]
        theta = np.deg2rad(pos[:, 0])  # must be in [0, 2*pi]
        phi = np.deg2rad(pos[:, 1] + 90)  # must be in [0, pi].
        N_SH = int((Lmax + 1)**2)  # number of SH coefficients
        Y = np.zeros((N_pos, N_SH), dtype=complex)
        count = 0
        for ni in np.arange(0, Lmax + 1):
            for mi in np.arange(-ni, ni + 1):
                Y[:, count] = sph_harm(mi, ni, theta, phi)
                count += 1
        return Y

    def sph_linear2degreeorder(vals):
        return np.ceil(np.sqrt(vals)) - 1

    H = np.fft.fft(HRIR, axis=-1)  # Convert HRIR -> HRTF
    r0 = np.array([input_pos[0, 2]])
    r1 = np.array([target_pos[0, 2]])
    if input_pos[0, 2] != target_pos[0, 2]:  # input and target radius are different
        radius_extrapolation = True
    else:
        radius_extrapolation = False

    # Calculate coeffs for known positions
    N_pos = input_pos.shape[0]
    Lmax = int(np.ceil(np.sqrt(N_pos) - 1))  # max order
    if Lmax > 50:
        Lmax = 50
    print(f'SH order {Lmax}')

    # calculate weighting coefficients (Voronoi surfaces <-> measurement points)
    input_pos_cart = sph2cart(input_pos)  # convert spherical to cartesian
    vor = SphericalVoronoi(input_pos_cart, r0)
    w = vor.calculate_areas()
    W = csr_matrix(np.diag(w))  # diagonal sparse matrix containing weights
    N_SH = (Lmax + 1)**2
    I_mtx = np.eye(N_SH)
    n = sph_linear2degreeorder(np.arange(1, N_SH + 1))
    D = I_mtx * np.diag(1 + n * (n + 1))  # decomposition order-dependent Tikhonov regularization
    Y = SH_coeffs(Lmax, input_pos)  # calculate real-valued SHs using the measurement grid
    Yest = SH_coeffs(Lmax, target_pos)

    if radius_extrapolation:
        # spherical hankel functions of second kind (for r0 and r1) (used for range extrapolation)
        nsamples = HRIR.shape[-1]
        f = np.linspace(0, fs - fs / nsamples, nsamples)
        c0 = 343
        k = 2 * np.pi * f / c0
        kr0 = k * r0  # measurement radius [m]
        kr1 = k * r1  # extrapolation radius [m]

        nn = sph_linear2degreeorder(np.arange(1, Lmax + 1))
        hankel_r0 = hankel2(nn, kr0)
        hankel_r1 = hankel2(nn, kr1)
        hankel_div = hankel_r1 / hankel_r0
        hankel_rep = np.expand_dims(np.nan_to_num(hankel_div[:, 0]), axis=-1)

        print(f'k {k.shape},  r0:{r0.shape}, kr1:{kr1.shape}, hankel_rep:{hankel_rep.shape}')

    # Calculate HRTFs for field points
    H_interp = np.zeros((target_pos.shape[0], 2, H.shape[-1]), dtype=complex)

    for ear in range(2):
        a0 = np.linalg.solve(Y.T @ W @ Y + epsilon * D, Y.T @ W @ H[:, ear, :])

        # range extrapolation
        if radius_extrapolation:
            # calculate range-extrapolated HRTFs
            print(f'a0 {a0.shape},  hankel_rep.T:{hankel_rep.T.shape}')
            a1 = np.multiply(a0, hankel_rep.T)
            print(f' a1:{a1.shape}, Yest:{Yest.shape}')
            # reconstruction to spatial data
            H_interp[:, ear, :] = (Yest @ a1[0:N_SH, :])  # interpolated + range-extrapolated HRTFs
        else:
            H_interp[:, ear, :] = (Yest @ a0[0:N_SH, :])  # interpolated HRTFs

    return np.fft.ifft(H_interp, axis=-1)


# %% Test interpolation
# load HRTF
path = r'C:\Users\rdavi\Desktop\SOFA\pp22_HRIRs_simulated.sofa'
Obj = sofa.SOFAFile.load(path)
fs = Obj.Data_SamplingRate

resolution = 5  # degrees
des_azi = np.arange(0, 360, resolution)
des_ele = np.arange(-90, 90, resolution)
r = Obj.SourcePosition[0][2]
r = 4
target_pos = np.zeros((len(des_azi) * len(des_ele), 3))
k = 0
for el in des_ele:
    for az in des_azi:
        target_pos[k, :] = [az, el, r]
        k += 1


# Interpolate
HRIRs = interpSH(Obj.Data_IR, Obj.SourcePosition, target_pos, fs, epsilon=1e-8)


# %% PLOTS ------------------------------------------------------------------------------
def plot_mag_horizontal(HRIRs, pos, fs, title):
    # find horizontal plane
    ear = 1
    idx_pos = np.where(np.logical_and(pos[:, 1] > -4, pos[:, 1] < 4))
    pos = np.squeeze(pos[idx_pos, :2])
    hM = np.squeeze(HRIRs[:, ear, :])  # left ear
    M = np.squeeze(20 * np.log10(np.abs(np.fft.rfft(hM[idx_pos, :], axis=-1))))

    azi = np.sort(pos[:, 0], axis=0)
    i = np.argsort(pos[:, 0], axis=0)
    M = M[i, :]

    freq = np.fft.rfftfreq(HRIRs.shape[-1], 1 / fs)
    fig, ax = plt.subplots()
    noisefloor = -50
    ax.pcolormesh(freq, azi, M, shading='nearest', vmin=noisefloor)
    plt.ylim([0, 360])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Azimuth (deg)')
    plt.title(title)
    plt.show()


def plot_mag_vertical(HRIRs, pos, fs, title):
    # find horizontal plane
    ear = 1
    idx_pos = np.where(np.logical_and(pos[:, 0] > -4, pos[:, 0] < 4))
    pos = np.squeeze(pos[idx_pos, :2])
    hM = np.squeeze(HRIRs[:, ear, :])  # left ear
    M = np.squeeze(20 * np.log10(np.abs(np.fft.rfft(hM[idx_pos, :], axis=-1))))

    ele = np.sort(pos[:, 1], axis=0)
    i = np.argsort(pos[:, 1], axis=0)
    M = M[i, :]
    freq = np.fft.rfftfreq(HRIRs.shape[-1], 1 / fs)
    fig, ax = plt.subplots()
    noisefloor = -50
    ax.pcolormesh(freq, ele, M, shading='nearest', vmin=noisefloor)
    plt.ylim([-90, 90])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Elevation (deg)')
    plt.title(title)
    plt.show()


fs = Obj.Data_SamplingRate
plot_mag_vertical(Obj.Data_IR, Obj.SourcePosition, fs, title='Vertical (reference)')
plot_mag_vertical(HRIRs, target_pos, fs, title='Vertical (interpolated)')

plot_mag_horizontal(Obj.Data_IR, Obj.SourcePosition, fs, title='Horizontal (reference)')
plot_mag_horizontal(HRIRs, target_pos, fs, title='Horizontal (interpolated)')


# %%
def plot_mag(HRIR1, HRIR2, fs, pos1, pos2, azi=90, elev=0):
    # find horizontal plane
    ear = 1
    idx_pos1 = np.sqrt((pos1[:, 0] - azi)**2 + (pos1[:, 1] - elev)**2).argmin()
    idx_pos2 = np.sqrt((pos2[:, 0] - azi)**2 + (pos2[:, 1] - elev)**2).argmin()
    hM1 = np.squeeze(HRIR1[:, ear, :])  # left ear
    hM2 = np.squeeze(HRIR2[:, ear, :])  # left ear

    M1 = np.squeeze(20 * np.log10(np.abs(np.fft.rfft(hM1[idx_pos1, :], axis=-1))))
    M2 = np.squeeze(20 * np.log10(np.abs(np.fft.rfft(hM2[idx_pos2, :], axis=-1))))

    freq = np.fft.rfftfreq(HRIR1.shape[-1], 1 / fs)
    fig, ax = plt.subplots()
    ax.semilogx(freq, M1, label='original')
    ax.semilogx(freq, M2, label='interpolated')
    plt.ylim([-60, 40])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Mag (dB)')
    plt.title(f'azi:{azi}, elev:{elev}')
    plt.legend()
    plt.show()


plot_mag(Obj.Data_IR, HRIRs, fs, Obj.SourcePosition, target_pos)

# %% Export
Obj_out = copy.deepcopy(Obj)
Obj_out.Data_IR = HRIRs
Obj_out.SourcePosition = target_pos
Obj_out.export('fabian_extrap')