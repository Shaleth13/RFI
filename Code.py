import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
from tqdm import tqdm
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import tensorflow as tf
from sklearn.model_selection import train_test_split

def generate_gaussian_grid(NChan, NTime, mean=0, std=1):
    """
    Generate a grid of complex Gaussian random variables.

    Parameters:
        NChan (int): Number of frequency channels.
        NTime (int): Number of time samples.
        mean (float): Mean of the Gaussian distribution.
        std (float): Standard deviation of the Gaussian distribution.

    Returns:
        np.ndarray: A grid of shape (NChan, NTime) with complex Gaussian random variables.
    """
    # Generate real and imaginary parts as independent Gaussian random variables
    real_part = np.random.normal(mean, std, (NTime, NChan))
    imag_part = np.random.normal(mean, std, (NTime, NChan))

    # Combine into complex numbers
    complex_grid = real_part + 1j * imag_part
    mask = np.asarray(real_part)*0.

    return complex_grid, mask

def insert_RFI1(data, mask, chanInsert, timeInsert, mean, std):
    """
    Add complex Gaussian random numbers with a different mean to specified grid points.

    Parameters:
        grid (np.ndarray): Original grid of complex Gaussian random variables.
      #  start_idx (tuple): Start index (row, column) of the sub-grid to modify.
      #  end_idx (tuple): End index (row, column) of the sub-grid to modify.
        mean (float): Mean of the new Gaussian distribution.
        std (float): Standard deviation of the new Gaussian distribution.

    Returns:
        np.ndarray: Modified grid with new values added.
    """
    # Extract sub-grid indices
    row_start, row_end = timeInsert
    col_start, col_end = chanInsert
    #print(row_start, row_end, col_start, col_end)
    # Generate new Gaussian random values for the specified sub-grid
    new_real = np.random.normal(mean, std, (row_end - row_start, col_end - col_start))
    new_imag = np.random.normal(mean, std, (row_end - row_start, col_end - col_start))
    new_values = new_real + 1j * new_imag

    # Add the new values to the original grid
    #print(data.shape, new_values.shape, row_start,row_end, col_start,col_end)
    data[row_start:row_end, col_start:col_end] += new_values
    mask[row_start:row_end, col_start:col_end] = 1.
    return data, mask

def plot_gaussian_grid(data, mask):
    """
    Plot the magnitude of a complex Gaussian grid.

    Parameters:
        grid (np.ndarray): A grid of complex Gaussian random variables.
    """
    magnitude = np.abs(grid)  # Compute the magnitude of the complex numbers

    plt.figure(figsize=(10, 4.5))
    plt.subplot(121)
    plt.title('real')
    plt.imshow(data.real, aspect='auto', cmap='viridis')
    plt.colorbar(label='real')
    plt.xlabel('Time Samples')
    plt.ylabel('Frequency Channels')
    plt.subplot(122)
    plt.title('mask')
    plt.imshow(mask, aspect='auto', cmap='binary')
    plt.colorbar(label='mask')
    plt.xlabel('Time Samples')
    plt.ylabel('Frequency Channels')
    plt.show()

def save_grid_to_fits(datarfi, dataorg, dmask, filename):
    """
    Save the grid data and mask as a FITS file with three axes: real, imaginary, and mask.

    Parameters:
        data (np.ndarray): Complex data grid.
        mask (np.ndarray): Mask grid.
        filename (str): Name of the FITS file to save.
    """

    # Create a 3D array with axes: [real/imag/mask, NChan, NTime]
    fits_data = np.stack([datarfi.real, datarfi.imag, dataorg.real, dataorg.imag, dmask], axis=0)

    # Create a FITS HDU and save to file
    hdu = fits.PrimaryHDU(fits_data)
    hdu.header['CTYPE1'] = 'Frequency Channels'
    hdu.header['CTYPE2'] = 'Time Samples'
    hdu.header['CTYPE3'] = 'Components'
    hdu.header['CRVAL3'] = 1
    hdu.header['CDELT3'] = 1
    hdu.header['CRPIX3'] = 1
    hdu.header['COMP1'] = 'data Real Part'
    hdu.header['COMP2'] = 'data Imaginary Part'
    hdu.header['COMP3'] = 'Real Part'
    hdu.header['COMP4'] = 'Imaginary Part'
    hdu.header['COMP5'] = 'Mask'
    hdu.writeto(filename, overwrite=True)

def create_data_with_rfi(NChan, NTime, dpar, RFIparam_list, filename):
    """
    Generate data with Gaussian random values and inject RFI, then save to a FITS file.

    Parameters:
        NChan (int): Number of frequency channels.
        NTime (int): Number of time samples.
        dpar (tuple): Parameters for Gaussian data (mean, std).
        RFIparam_list (list): List of RFI parameter dictionaries.
        filename (str): Name of the output FITS file.

    Returns:
        tuple: Data grid and mask after RFI injection.
    """
    mean, std = dpar
    grid, mask = generate_gaussian_grid(NChan, NTime, mean, std)
    data = np.copy(grid)
    for RFIparam in RFIparam_list:
        NRFI = RFIparam['NRFI']
        maxRFIchan = RFIparam['maxRFIchan']
        maxRFItime = RFIparam['maxRFItime']

        chanstart = np.random.randint(0, NChan, NRFI)
        timestart = np.random.randint(0, NTime, NRFI)
        RFImean = np.random.uniform(RFIparam['RFImean_min'], RFIparam['RFImean_max'], NRFI)
        RFIstd = np.random.uniform(RFIparam['RFIstd_min'], RFIparam['RFIstd_max'], NRFI)

        chanend = np.random.randint(1, maxRFIchan, NRFI) + chanstart
        timeend = np.random.randint(1, maxRFItime, NRFI) + timestart
        chanend = np.minimum(chanend, NChan-1)
        timeend = np.minimum(timeend, NTime-1)

        for tt in range(NRFI):
            grid, mask = insert_RFI1(grid, mask, (chanstart[tt], chanend[tt]), (timestart[tt], timeend[tt]), RFImean[tt], RFIstd[tt])

    bandshape = np.outer(NTime, generate_bandshape(NChan=NChan))
    grid = grid*bandshape
    data = data*bandshape

    save_grid_to_fits(grid, data, mask, filename)
    return grid, mask

def create_multiple_realizations(NChan, NTime, dpar, RFIparam_list, NReal, savedir):
    """
    Generate multiple realizations of data with RFI and save them to FITS files.

    Parameters:
        NChan (int): Number of frequency channels.
        NTime (int): Number of time samples.
        dpar (tuple): Parameters for Gaussian data (mean, std).
        RFIparam_list (list): RFI Parameters list for RFI injection
            RFIparam (dict): Parameters for RFI injection:
              - NRFI: Number of RFI regions.
              - RFImean_min: Minimum mean for RFI.
              - RFImean_max: Maximum mean for RFI.
              - RFIstd_min: Minimum std deviation for RFI.
              - RFIstd_max: Maximum std deviation for RFI.
              - maxRFIchan: Maximum width of RFI in frequency channels.
              - maxRFItime: Maximum width of RFI in time samples.
        NReal (int): Number of realizations to generate.
        savedir (str): Directory to save the FITS files.
    """
    # Ensure the save directory exists
    os.makedirs(savedir, exist_ok=True)

    percentages = []
    for i in tqdm(range(NReal), desc="Generating realizations"):
        filename = f"{savedir}/realization_{i+1}.fits"
        _, mask = create_data_with_rfi(NChan, NTime, dpar, RFIparam_list, filename)
        # Calculate the percentage of unmasked regions
        unmasked_percentage = 100 * np.sum(mask == 0) / mask.size
        percentages.append(unmasked_percentage)

    mean_percentage = np.mean(percentages)
    std_percentage = np.std(percentages)

    print(f"Mean percentage of unmasked regions: {mean_percentage:.2f}%")
    print(f"Standard deviation of unmasked percentages: {std_percentage:.2f}%")

    # Plot a histogram of the percentages
    plt.figure(figsize=(8, 6))
    plt.hist(percentages, bins=20, color='skyblue', edgecolor='black')
    plt.title("Histogram of Unmasked Percentages")
    plt.xlabel("Percentage of Unmasked Regions")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

def generate_bandpass(NChan, bandshape_params):
    """
    Generate a bandpass response B(c) defined as:
    B(c) = 1 / [1 + (c / ccR)^Nord]

    Parameters:
        NChan (int): Number of frequency channels.
        cR (float): Rise parameter as a fraction of NChan.
        Nord (int): Order of the response.

    Returns:
        np.ndarray: Bandpass response for each channel.
    """
    chan_param, amp_param, Nord = bandshape_params
    cR, cF = chan_param
    AR, AF = amp_param
    ccR = cR * NChan
    ccF = cF * NChan
    #print(ccR, ccF, AR, AF, Nord)
    channels = np.arange(1, NChan + 1)  # Channel indices starting from 1
    rise = 1 - 1 / (1 + (channels / ccR) ** Nord)
    interpolation = interp1d(rise, channels, kind='linear', fill_value="extrapolate")
    rise_chan = int(np.ceil(interpolation(AR)))
    fall = 1 - 1 / (1 + (channels / (NChan-ccF)) ** Nord)
    fall = fall[::-1]
    interpolation = interp1d(fall, channels, kind='linear', fill_value="extrapolate")
    fall_chan = int(np.ceil(interpolation(AF)))
    #print(rise_chan, fall_chan)
    # Combine rise and fall
    bandp = np.minimum(rise, fall)[rise_chan:fall_chan]

    intchan = np.linspace(1, NChan+1, len(bandp))  # Channel indices starting from 1
    interpolation = interp1d(intchan, bandp, kind='linear', fill_value="extrapolate")
    bandpass = interpolation(channels)

    return bandpass


def generate_bandshape(NChan=1024, bandshape_params=((0.1,0.9), (0.3, 0.3), 4),  ripple_params=[(10, 0.0025), (8, 0.001), (4, 0.005)]):
    """
    Generate a bandshape that rises and sets using a Butterworth filter and includes ripples.

    Parameters:
        NChan (int): Number of frequency channels.
        base_value (float): The baseline value for the bandshape.
        ripple_params (list of tuples): Each tuple contains (ripple_frequency, ripple_amplitude).
        cutoff (float): Cutoff frequency for the Butterworth filter (normalized to Nyquist frequency).
        order (int): Order of the Butterworth filter.

    Returns:
        np.ndarray: Bandshape values for each channel.
    """
    smoothed_rise_set = generate_bandpass(NChan, bandshape_params)
    # Add ripples
    chans = np.linspace(0, 1, NChan)
    ripples = sum([amplitude * np.sin(2 * np.pi * frequency * chans) for frequency, amplitude in ripple_params])

    return smoothed_rise_set* (1+ ripples)



def plot_bandshape(bandshape=None):
    """
    Plot the bandshape with channel indices.

    Parameters:
        bandshape (np.ndarray): Bandshape values.
        NChan (int): Number of frequency channels.
    """
    if bandshape==None:
        bandshape = generate_bandshape()
    channels = np.arange(len(bandshape))
    plt.figure(figsize=(8, 5))
    plt.plot(channels, bandshape, label="Bandshape")
    plt.title("Bandshape with Ripples")
    plt.xlabel("Channel Index")
    plt.ylabel("Amplitude")
    plt.ylim(0,1.1)
    plt.grid(True)
    plt.legend()
    plt.show()

plot_bandshape()
#from RFIfuncs import *

# Parameters for Gaussian data
dpar = (0, 1.)  # Mean = 0, Standard Deviation = 1

# Parameters for RFI injection
RFIparam_list = []
RFIparam = {
    'NRFI':        32,         # Number of RFI regions
    'RFImean_min':  0,         # Minimum mean of RFI
    'RFImean_max': 2,         # Maximum mean of RFI
    'RFIstd_min':   0,         # Minimum standard deviation of RFI
    'RFIstd_max':  2,         # Maximum standard deviation of RFI
    'maxRFIchan': 200,         # Maximum width of RFI in frequency channels
    'maxRFItime': 20          # Maximum width of RFI in time samples
}
RFIparam_list.append(RFIparam)
RFIparam = {
    'NRFI':        16,         # Number of RFI regions
    'RFImean_min':  0,         # Minimum mean of RFI
    'RFImean_max': 1,         # Maximum mean of RFI
    'RFIstd_min':   0,         # Minimum standard deviation of RFI
    'RFIstd_max':  2,         # Maximum standard deviation of RFI
    'maxRFIchan': 20,          # Maximum width of RFI in frequency channels
    'maxRFItime': 200          # Maximum width of RFI in time samples
}
RFIparam_list.append(RFIparam)


NChan, NTime = 256, 256
NReal = 10
create_multiple_realizations(NChan, NTime, dpar, RFIparam_list, NReal, savedir='../outdir1')

import os

fits_dir = "../outdir1/"
fits_files = [f for f in os.listdir(fits_dir) if f.endswith(".fits")]

print("Generated FITS files:", fits_files)

from astropy.io import fits

fitsfile = "../outdir1/realization_5.fits"
file = fits.open(fitsfile, READONLY=True)
data = file[0].data
header = file[0].header
print(data.shape)
print(header["NAXIS"])

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

def convert_fits_cnn(fits_dir, fits_names):
    images = []
    masks = []
    for fits_name in fits_names:
        with fits.open(fits_dir + fits_name) as hdul:
            data = hdul[0].data  # Shape (5, NChan, NTime)

            datarfi_real = data[0]
            datarfi_imag = data[1]
            mask = data[4]  # This is your binary mask

            rfi_magnitude = np.sqrt(datarfi_real*2 + datarfi_imag*2)

            # Make binary mask
            mask_binary = np.where(mask > 0, 1, 0)

            # Masked input image for CNN (RFI-magnitude only where mask is 1)
            image_masked = np.where(mask_binary == 1, rfi_magnitude, 0)

            images.append(image_masked)
            masks.append(mask_binary)
    return np.array(images), np.array(masks)

import os
fits_dir = "../outdir1/"
fits_names = [f for f in os.listdir(fits_dir) if f.endswith(".fits")]

images, masks = convert_fits_cnn(fits_dir, fits_names)  # All 10 files
  # Load 1 file

X = np.array(images)[..., np.newaxis]  # (N, 256, 256, 1)
Y = np.array(masks)[..., np.newaxis]   # (N, 256, 256, 1)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(images[0], aspect='auto', cmap='viridis')
plt.title('Masked RFI Image')

plt.subplot(1,2,2)
plt.imshow(masks[0], aspect='auto', cmap='gray')
plt.title('Mask')
plt.show()

# Train/Val Split
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

import tensorflow as tf
from tensorflow.keras import layers, models

def build_rfi_cnn(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),

        layers.Conv2D(1, (1, 1), activation='sigmoid')  # Output mask
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

input_shape = X_train.shape[1:]  # (256, 256, 1)
model = build_rfi_cnn(input_shape)
model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=2  # Increase if memory allows
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=2  # Increase if memory allows
)

# Evaluate
loss, acc = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {acc:.4f}")

# Predict a sample
pred_mask = model.predict(X_val[0:1])[0, :, :, 0]

# Plot prediction
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(X_val[0, :, :, 0], cmap='viridis')
plt.title("Input")

plt.subplot(1,3,2)
plt.imshow(y_val[0, :, :, 0], cmap='gray')
plt.title("True Mask")

plt.subplot(1,3,3)
plt.imshow(pred_mask > 0.5, cmap='gray')  # Thresholded
plt.title("Predicted Mask")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Load FITS

fits_file = "../outdir1/realization_1.fits" 
with fits.open(fits_file) as hdul:
    data = hdul[0].data 

# Extract RFI data
datarfi_real = data[0]
datarfi_imag = data[1]
mask = data[4]

# Compute RFI magnitude and mask
rfi_magnitude = np.sqrt(datarfi_real**2 + datarfi_imag**2)
mask_binary = np.where(mask > 0, 1, 0)

# Shape parameters
NChan, NTime = mask_binary.shape
t = np.arange(NTime)

# FITS-based E-field (masked)
E_field_2D = rfi_magnitude * mask_binary

# Doppler Electric Field Function
def E_moving_inverseR(ax, ay, az, t, obs_point, E_stat, f0=1e9, c=3e8):
    t = np.atleast_1d(t).astype(float)
    Xs = sum(a * (t ** i) for i, a in enumerate(ax))
    Ys = sum(a * (t ** i) for i, a in enumerate(ay))
    Zs = sum(a * (t ** i) for i, a in enumerate(az))

    # Relative position vector
    rx, ry, rz = obs_point[0]-Xs, obs_point[1]-Ys, obs_point[2]-Zs
    r_vec = np.array([rx, ry, rz], dtype=float)
    Rmag = np.linalg.norm(r_vec, axis=0)

    if np.any(Rmag == 0):
        raise ValueError("Source coincides with observer.")

    # Velocity vector (linear motion)
    vx, vy, vz = ax[1], ay[1], az[1]
    v_vec = np.array([vx, vy, vz])

    # Radial velocity and Doppler frequency
    r_hat = r_vec / Rmag
    v_r = np.dot(v_vec, r_hat)
    f_obs = f0 * (1 + v_r / c)

    # Phase and amplitude
    phase = 2 * np.pi * np.cumsum(f_obs) * (t[1] - t[0])
    A_t = E_stat * (1 / Rmag)

    # Complex E-field
    E_t = A_t * np.exp(1j * phase)
    return E_t


# Fixed antenna location
obs_point = np.array([0.0, 0.0, 0.0])

# Source motion coefficients
ax = [1e4, 10]  # x(t) = 1.0 + 0.01*t
ay = [1e4, 20]  # y(t) = 1.0 + 0.02*t
az = [1e4, 30]  # z(t) = 1.0 + 0.03*t
E_stat = 1.0    # Ref amp

# time-dependent Doppler E-field (single source)
E_t = E_moving_inverseR(ax, ay, az, t, obs_point, E_stat)

E_mag_time = np.abs(E_t)

#E_field_2D_full = rfi_magnitude * mask_binary 
freqs = np.linspace(0.9e9, 1.1e9, NChan)  # observation frequency range
E_field_2D_full = np.zeros((NChan, NTime), dtype=float)
E_field_2D_full = rfi_magnitude * mask_binary

for i, f in enumerate(freqs):
    # Frequency-dependent phase & amplitude modulation
    phase_shift = 2 * np.pi * (f * t / 3e8 + np.sin(2*np.pi * f * t / 1e9))
    #amp_mod = 1 + 0.2 * np.sin(2 * np.pi * (i / NChan) * 5)
    E_field_2D_full[i, :] = np.abs(E_t * np.exp(1j * phase_shift))

# Apply mask from FITS
E_field_2D_full = E_field_2D_full * mask_binary


# plots
plt.figure(figsize=(14, 4))
plt.subplot(1, 3, 1)
plt.imshow(E_field_2D_full, origin='lower', aspect='auto', cmap='magma')
plt.colorbar(label='|E| amplitude')
plt.xlabel('Time Index')
plt.ylabel('Frequency Channel')
plt.title('Doppler E-field (Freq-dependent)')

plt.subplot(1, 3, 2)
plt.imshow(E_field_2D, origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='|E| amplitude')
plt.xlabel('Time Index')
plt.ylabel('Frequency Channel')
plt.title('RFI Magnitude from FITS')

plt.subplot(1, 3, 3)
plt.imshow(mask_binary, origin='lower', aspect='auto', cmap='binary')
plt.colorbar(label='Mask / RFI')
plt.xlabel('Time Index')
plt.ylabel('Frequency Channel')
plt.title('RFI Mask from FITS')

plt.tight_layout()
plt.show()




