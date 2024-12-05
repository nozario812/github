import numpy as np
from scipy.signal import lfilter

def median_filter(arr, sigma):
    """
    Noise filter using Median and Median Absolute Deviation for 1-dimentional array
    """

    if sigma is None:
        return np.ones(arr.size, dtype='b1')

    med = np.median(arr)
    mad = np.median(np.abs(arr - med))

    # Tiny cheeting for mad = 0 case
    if mad == 0:
        absl = np.abs(arr - med)
        if len(absl[absl > 0]) > 0:
            mad = (absl[absl > 0])[0]
        else:
            mad = np.std(arr) / 1.4826

    return (arr >= med - mad*1.4826*sigma) & (arr <= med + mad*1.4826*sigma)

def reduction(data, sigma=3, **kwargs):
    """
    Do data reduction with sum, max and min for pulse/noise using median filter (or manual min/max)

    Parameters (and their default values):
        data:   array of pulse/noise data (NxM or N array-like)
        sigma:  sigmas allowed for median filter

    Valid keywords:
        min:    tuple of (min, max) for min
        max:    tuple of (min, max) for max
        sum:    tuple of (min, max) for sum

    Return (mask)
        mask:   boolean array for indexing filtered data
    """

    data = np.asarray(data)

    if "min" in kwargs:
        min_mask = (data.min(axis=1) > kwargs["min"][0]) & (data.min(axis=1) < kwargs["min"][1])
    else:
        min_mask = median_filter(data.min(axis=1), sigma)

    if "max" in kwargs:
        max_mask = (data.max(axis=1) > kwargs["max"][0]) & (data.max(axis=1) < kwargs["max"][1])
    else:
        max_mask = median_filter(data.max(axis=1), sigma)

    if "sum" in kwargs:
        sum_mask = (data.sum(axis=1) > kwargs["sum"][0]) & (data.sum(axis=1) < kwargs["sum"][1])
    else:
        sum_mask = median_filter(data.sum(axis=1), sigma)

    return min_mask & max_mask & sum_mask

def ntrigger(pulse, noise, threshold=20, sigma=3, smooth=10, avg_pulse=None, **kwargs):
    """
    Number of trigger

    Parameters (and their default values):
        pulse:      array of pulse data (NxM)
        noise:      array of noise data (NxM or N array-like)
        threshold:  sigmas of noise to use as threshold (Default: 20)
        sigma:      sigmas allowed for median filter, or None to disable noise filtering (Default: 3)
        smooth:     number of boxcar taps to smooth pulse (Default: 10)
        avg_pulse:  averaged pulse to use for offset subtraction (Default: None)

    Valid keywords:
        min:    tuple of (min, max) for min
        max:    tuple of (min, max) for max
        sum:    tuple of (min, max) for sum

    Return (mask)
        mask:   array of boolean
    """

    # if given data is not numpy array, convert them
    pulse = np.asarray(pulse).copy()
    noise = np.asarray(noise).copy()

    # Subtract offset from pulse
    p = pulse - offset(pulse, max_shift=0, avg_pulse=avg_pulse)[:, np.newaxis]

    # Smooth pulse
    p = lfilter([smooth**-1]*smooth, 1, p)

    # Data reduction for noise
    if sigma is not None or 'min' in kwargs or 'max' in kwargs or 'sum' in kwargs:
        noise = noise[reduction(noise, sigma, **kwargs)]

    thre = np.std(noise) * threshold

    # Trigger
    ntrigger = np.sum((p >= thre) & np.roll(p < thre, 1), axis=-1)
    ntrigger += np.sum((p <= -thre) & np.roll(p > -thre, 1), axis=-1)

    return ntrigger

def average_pulse(pulse, sigma=3, r=0.2, rr=0.1, max_shift=None, return_shift=False, **kwargs):
    """
    Generate an averaged pulse

    Parameters (and their default values):
        pulse:          array of pulse data (NxM or N array-like, obviously the latter makes no sense though)
        sigma:          sigmas allowed for median filter, or None to disable filtering (Default: 3)
        r:              amount in ratio of removal in total for data reduction (Default: 0.2)
        rr:             amount in ratio of removal for each step for data reduction (Default: 0.1)
        max_shift:      maximum allowed shifts to calculate maximum cross correlation (Default: None = length / 2)
        return_shift:   return array of shifts if True (Default: False)

    Valid keywords:
        min:    tuple of (min, max) for min
        max:    tuple of (min, max) for max
        sum:    tuple of (min, max) for sum

    Return (averaged_pulse)
        averaged_pulse:     averaged pulse
        shift:              shifted values (only if return_shift is True)
    """

    # if given data is not numpy array, convert them
    pulse = np.asarray(pulse).copy()

    s = []

    # Calculate averaged pulse
    if pulse.ndim == 2:
        # Data reduction
        if sigma is not None or 'min' in kwargs or 'max' in kwargs or 'sum' in kwargs:
            pulse = pulse[reduction(pulse, sigma, **kwargs)]

        plen = len(pulse)

        while (len(pulse) > (plen*(1.0-r))):
            avg = np.average(pulse, axis=0)
            pulse = pulse[((pulse - avg)**2).sum(axis=-1).argsort() < (len(pulse) - plen*rr - 1)]

        # Align pulses to the first pulse
        if max_shift is None or max_shift > 0:
            max_shift = pulse.shape[-1]/2 if max_shift is None else max_shift
            if len(pulse) > 1:
                s.append(0)
                for i in range(1, len(pulse)):
                    _s = cross_correlate(pulse[0], pulse[i], max_shift=max_shift)[1]
                    pulse[i] = np.roll(pulse[i], _s)
                    s.append(_s)

        avg_pulse = np.average(pulse, axis=0)

    elif pulse.ndim == 1:
        # Only one pulse data. No need to average
        avg_pulse = pulse

    else:
        raise ValueError("object too deep for desired array")

    if return_shift:
        return avg_pulse, s
    else:
        return avg_pulse

def power(data):
    """
    Calculate power spectrum

    Parameter:
        data:   pulse/noise data (NxM or N array-like)

    Return (power)
        power:  calculated power spectrum
    """

    data = np.asarray(data)

    # Real DFT
    ps = np.abs(np.fft.rfft(data) / data.shape[-1])**2

    if data.shape[-1] % 2:
        # Odd
        ps[...,1:] *= 2
    else:
        # Even
        ps[...,1:-1] *= 2

    return ps

def average_noise(noise, sigma=3, r=0.2, rr=0.1, **kwargs):
    """
    Calculate averaged noise power

    Parameters (and their default values):
        noise:      array of pulse data (NxM or N array-like, obviously the latter makes no sense though)
        sigma:      sigmas allowed for median filter, or None to disable filtering (Default: 3)
        r:          amount in ratio of removal in total for data reduction (Default: 0.2)
        rr:         amount in ratio of removal for each step for data reduction (Default: 0.1)

    Valid keywords:
        min:    tuple of (min, max) for min
        max:    tuple of (min, max) for max
        sum:    tuple of (min, max) for sum

    Return (averaged_pulse)
        power_noise:    calculated averaged noise power in V^2
    """

    # Convert to numpy array
    noise = np.asarray(noise)

    if sigma is not None or 'min' in kwargs or 'max' in kwargs or 'sum' in kwargs:
        noise = noise[reduction(noise, sigma, **kwargs)]

        nlen = len(noise)

        while (len(noise) > (nlen*(1.0-r))):
            avg = np.average(power(noise), axis=0)
            noise = noise[((power(noise) - avg)**2).sum(axis=-1).argsort() < (len(noise) - nlen*rr - 1)]

    return np.average(power(noise), axis=0)

def generate_template(pulse, noise, cutoff=None, lpfc=None, hpfc=None, nulldc=False, **kwargs):
    """
    Generate a template of optimal filter

    Parameters (and their default values):
        pulse:  array of pulse data, will be averaged if dimension is 2
        noise:  array of noise data, will be averaged if dimension is 2
        cutoff: low-pass cut-off bin number for pulse spectrum (Default: None)
                (**note** This option is for backward compatibility only. Will be removed.)
        lpfc:   low-pass cut-off bin number for pulse spectrum (Default: None)
        hpfc:   high-pass cut-off bin number for pulse spectrum (Default: None)
        nulldc: nullify dc bin of template (Default: False)

    Valid keywords:
        min:    tuple of (min, max) for min
        max:    tuple of (min, max) for max
        sum:    tuple of (min, max) for sum

    Return (template)
        template:   generated template
        sn:         calculated signal-to-noise ratio
    """

    # Average pulse
    if pulse.ndim == 2:
        avg_pulse = average_pulse(pulse, **kwargs)
    else:
        avg_pulse = pulse

    # Real-DFT
    fourier = np.fft.rfft(avg_pulse)

    # Apply low-pass/high-pass filter
    m = len(avg_pulse)
    n = len(fourier)

    if lpfc is None and cutoff is not None:
        lpfc = cutoff

    if lpfc is not None and 0 < lpfc < n:
        h = np.blackman(m)*np.sinc(np.float(lpfc)/n*(np.arange(m)-(m-1)*0.5))
        h /= h.sum()
        fourier *= np.abs(np.fft.rfft(h))

    # Apply high-pass filter
    if hpfc is not None and 0 < hpfc < n:
        h = np.blackman(m)*np.sinc(np.float(hpfc)/n*(np.arange(m)-(m-1)*0.5))
        h /= h.sum()
        fourier *= (1 - np.abs(np.fft.rfft(h)))

    # Null DC bin?
    if nulldc:
        fourier[0] = 0

    # Calculate averaged noise power
    if noise.ndim == 2:
        pow_noise = average_noise(noise, **kwargs)
    else:
        pow_noise = noise

    # Calculate S/N ratio
    sn = np.sqrt(power(np.fft.irfft(fourier, len(avg_pulse)))/pow_noise)

    # Generate template (inverse Real-DFT)
    template = np.fft.irfft(fourier / pow_noise, len(avg_pulse))

    # Normalize template
    norm = (avg_pulse.max() - avg_pulse.min()) / ((template * avg_pulse).sum() / len(avg_pulse))

    return template * norm, sn

def cross_correlate(data1, data2, max_shift=None, method='interp'):
    """
    Calculate a cross correlation for a given set of data.

    Parameters (and their default values):
        data1:      pulse/noise data (array-like)
        data2:      pulse/noise data (array-like)
        max_shift:  maximum allowed shifts to calculate maximum cross correlation
                    (Default: None = length / 2)
        method:     interp - perform interpolation for obtained pha and find a maximum
                             (only works if max_shift is given)
                    integ  - integrate for obtained pha
                    none   - take the maximum from obtained pha
                    (Default: interp)

    Return (max_cor, shift)
        max_cor:    calculated max cross correlation
        shift:      required shift to maximize cross correlation
        phase:      calculated phase
    """

    # Sanity check
    if len(data1) != len(data2):
        raise ValueError("data length does not match")

    # if given data set is not numpy array, convert them
    data1 = np.asarray(data1).astype(dtype='float64')
    data2 = np.asarray(data2).astype(dtype='float64')

    # Calculate cross correlation
    if max_shift == 0:
        return np.correlate(data1, data2, 'valid')[0] / len(data1), 0, 0

    # Needs shift
    if max_shift is None:
        max_shift = len(data1) / 2
    else:
        # max_shift should be less than half data length
        max_shift = min(max_shift, len(data1) / 2)

    # Calculate cross correlation
    cor = np.correlate(data1, np.concatenate((data2[-max_shift:], data2, data2[:max_shift])), 'valid')
    ind = cor.argmax()

    if method == 'interp' and 0 < ind < len(cor) - 1:
        return (cor[ind] - (cor[ind-1] - cor[ind+1])**2 / (8 * (cor[ind-1] - 2 * cor[ind] + cor[ind+1]))) / len(data1), ind - max_shift, (cor[ind-1] - cor[ind+1]) / (2 * (cor[ind-1] - 2 * cor[ind] + cor[ind+1]))
    elif method == 'integ':
        return sum(cor), 0, 0
    elif method in ('none', 'interp'):
        # Unable to interpolate, and just return the maximum
        return cor[ind] / len(data1), ind - max_shift, 0
    else:
        raise ValueError("Unsupported method")

def optimal_filter(pulse, template, max_shift=None, method='interp'):
    """
    Perform an optimal filtering for pulse using template

    Parameters (and their default values):
        pulse:      pulses (NxM array-like)
        template:   template (array-like)
        max_shift:  maximum allowed shifts to calculate maximum cross correlation
                    (Default: None = length / 2)
        method:     interp - perform interpolation for obtained pha and find a maximum
                             (only works if max_shift is given)
                    integ  - integrate for obtained pha
                    none   - take the maximum from obtained pha
                    (Default: interp)

    Return (pha, lagphase)
        pha:        pha array
        phase:      phase array
    """

    return np.apply_along_axis(lambda p: cross_correlate(template, p, max_shift=max_shift, method=method), 1, pulse)[...,(0,2)].T

def offset(pulse, bins=None, sigma=3, max_shift=None, avg_pulse=None):
    """
    Calculate an offset (DC level) of pulses

    Parameters (and their default values):
        pulse:      pulses (N or NxM array-like)
        bins:       tuple of (start, end) for bins used for calculating an offset
                    (Default: None = automatic determination)
        sigma:      sigmas allowed for median filter, or None to disable filtering (Default: 3)
        max_shift:  maximum allowed shifts to calculate maximum cross correlation
                    (Default: None = length / 2)
        avg_pulse:  if given, use this for averaged pulse (Default: None)

    Return (offset)
        offset: calculated offset level
    """

    pulse = np.asarray(pulse)

    if bins is None:
        if avg_pulse is None:
            avg_pulse = average_pulse(pulse, sigma=sigma, max_shift=max_shift)
        i = np.correlate(avg_pulse, [1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1]).argmax() - 16
        if i < 1:
            raise ValueError("Pre-trigger is too short")
        return pulse[..., :i].mean(axis=-1)
    else:
        return pulse[..., bins[0]:bins[1]].mean(axis=-1)
    
#import
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit


# Data Load
f = h5py.File('/Users/nozakirio/Desktop/CH1-63uA_CH3-73uA/bnon/cpost02pn_ch1_1.hdf5')
hres = f["waveform/hres"][()]
vres = f["waveform/vres"][()]
pulse = f["waveform/pulse"][()] * vres
noise = f["waveform/noise"][()] * vres
time = np.arange(pulse.shape[-1]) * hres

# Offset to Zero
pulse_med = np.median(pulse[:, 0:100], axis=1)
pulse = pulse - pulse_med[:,np.newaxis]

# Pulse Height
PH = pulse_med - np.min(pulse, axis=1)
plt.figure()
plt.hist(PH, bins=1024, histtype="step")
plt.xlabel(r"$PH/\rm{V}$", fontsize=15)
plt.ylabel(r"$Counts$", fontsize=15)
plt.title("Pulse Height")
plt.show()

# Pulse Height
PH = pulse_med - np.min(pulse, axis=1)
plt.figure()
plt.hist(PH, bins=1024, histtype="step")
plt.xlabel(r"$PH/\rm{V}$", fontsize=15)
plt.ylabel(r"$Counts$", fontsize=15)
plt.title("Pulse Height")
plt.xlim(0.5, 0.6)
plt.show() 

# Optimal Filtering
PH_mask_min = 0.5
PH_mask_max = 0.6
PH_mask = (PH_mask_min < PH) & (PH < PH_mask_max)
tmpl, sn = generate_template(pulse[PH_mask], noise, max_shift=5)
pha, phase = optimal_filter(pulse, tmpl, max_shift=5)

plt.figure()
plt.hist(pha, bins=1024, histtype="step")
plt.xlabel(r"$PHA$", fontsize=15)
plt.ylabel(r"$Counts$", fontsize=15)
plt.title("PHA")
plt.show()

MnKa_mask_min = 0.45
MnKa_mask_max = 0.47
MnKb_mask_min = 0.49
MnKb_mask_max = 0.51
MnKa_mask = (MnKa_mask_min < pha) & (pha < MnKa_mask_max)
MnKb_mask = (MnKb_mask_min < pha) & (pha < MnKb_mask_max)

MnKa = np.median(pha[MnKa_mask])
MnKb = np.median(pha[MnKb_mask])

# Energy Calibration
_x = np.array([0, MnKa, MnKb])
_y = np.array([0, 5.899, 6.490])

model = LinearRegression()
model.fit(_x.reshape(-1, 1), _y)

x_plot = np.linspace(0, 1, 1000)
y_plot = model.predict(x_plot.reshape(-1, 1))

r2 = r2_score(_y, model.predict(_x.reshape(-1, 1)))
print("R2 score: ", r2)

# Energy = a * PHA
a = model.coef_[0]
print("a: ", a)
plt.figure()
plt.scatter(_x, _y, label="data")
plt.plot(x_plot, y_plot, color="r",label="fit")
plt.xlabel(r"$PHA$", fontsize=15)
plt.ylabel(r"$Energy/\rm{keV}$", fontsize=15)
plt.legend()
plt.grid(ls="--", which="both")
plt.show()

Energy = a * pha
plt.figure()
plt.hist(Energy, bins=int(Energy.max()*1e3/0.5), histtype="step")
plt.xlabel(r"$Energy/\rm{keV}$", fontsize=15)
plt.ylabel(r"$Counts$", fontsize=15)
plt.xlim(5.5,7)
plt.show()
 

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


X_train, X_test, y_train, y_test = train_test_split(pulse, Energy, test_size=0.2, shuffle=False)

model = Sequential()
model.add(LSTM(64, input_shape=(pulse.shape[1], 1)))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

# Prediction
new_pulse = np.array([X_test[0]])
predicted_energy = model.predict(new_pulse)
print("Predicted energy/keV:", predicted_energy[0][0])
print("Measured energy/keV:", Energy[0])

plt.figure()
plt.hist(y_pred, bins=100, histtype="step")
plt.show()

plt.figure()
plt.plot(time, X_test.T)
plt.show() 

#以下AE#
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import h5py

# Data Load
f = h5py.File("/content/drive/MyDrive/run008_b64.hdf5")
wave = f["waveform/wave"][()]
hres = f["waveform/hres"][()]
vres = f["waveform/vres"][()]
pulse = wave[:, int(wave.shape[1]/2):] * vres
noise = wave[:, :int(wave.shape[1]/2)] * vres
time = np.arange(0, pulse.shape[1] * hres, hres)

# Offset to Zero
pulse_med = np.median(pulse[:, 0:100], axis=1)
pulse = pulse - pulse_med[:,np.newaxis]

# Normalize
min_val = np.min(pulse, axis=1, keepdims=True)
max_val = np.max(pulse, axis=1, keepdims=True)
pulse = (pulse - min_val) / (max_val - min_val)

# Creat Model
encoding_dim = 2
input_dim = pulse.shape[1]
input_layer = Input(shape=(input_dim,))
x1 = Dense(64, activation='tanh')(input_layer)
x2 = Dense(64, activation='tanh')(x1)
x3 = Dense(64, activation='tanh')(x2)
encoded = Dense(encoding_dim, activation='tanh')(x3)
x3 = Dense(64, activation='tanh')(encoded)
x4 = Dense(64, activation='tanh')(x3)
x5 = Dense(64, activation='tanh')(x4)
decoded = Dense(input_dim, activation='sigmoid')(x5)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mape')
autoencoder.summary()

# Plot Model
tf.keras.utils.plot_model(autoencoder, show_shapes=True, expand_nested=True, show_dtype=True, to_file="model.png")

# Learning
batch_size = 32
epochs = 50
history = autoencoder.fit(pulse[0:500], pulse[0:500],
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(pulse[501:], pulse[501:]))

decoded_data = autoencoder.predict(pulse[501:])

plt.figure(figsize=(10, 2))
plt.plot(time, pulse.T)
plt.show()

plt.figure(figsize=(10, 2))
plt.plot(time, decoded_data.T)
plt.show()

# Learning Curve
plt.plot(range(epochs), history.history['loss'], label="loss")
plt.title("loss")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

plt.plot(range(epochs), history.history['val_loss'], label="val loss")
plt.title("val loss")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


#以下VAE#
import tensorflow as tf
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import h5py

# Data Load
#f = h5py.File('/Users/nozakirio/Desktop/CH1-63uA_CH3-73uA/bnon/cpost02pn_ch1_1.hdf5')
wave = f['waveform/wave'][()]
hres = f['waveform/hres'][()]
vres = f['waveform/vres'][()]
pulse = wave[:, int(wave.shape[1]/2):] * vres
noise = wave[:, :int(wave.shape[1]/2)] * vres
time = np.arange(0, pulse.shape[1] * hres, hres)

# Offset to Zero
pulse_med = np.median(pulse[:, 0:100], axis=1)
pulse = pulse - pulse_med[:,np.newaxis]

# Normalizatoion
# mmscaler = MinMaxScaler()
# pulse = mmscaler.fit_transform(pulse)
min_val = np.min(pulse, axis=1, keepdims=True)
max_val = np.max(pulse, axis=1, keepdims=True)

# データを最小-最大スケールに変換
pulse = (pulse - min_val) / (max_val - min_val)

# Reparametrization Trick
def sampling(args):
    z_mean, z_logvar = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim), seed = 5)
    return z_mean + K.exp(0.5 * z_logvar) * epsilon

# Create VAE Model
input_dim = pulse.shape[1]
input_shape = (input_dim, )
latent_dim = 2

inputs = Input(shape=input_shape)
x1 = Dense(64, activation='relu')(inputs)
x2 = Dense(64, activation='relu')(x1)
x3 = Dense(64, activation='relu')(x2)
z_mean = Dense(latent_dim)(x3)
z_logvar = Dense(latent_dim)(x3)
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_logvar])
encoder = Model(inputs, [z_mean, z_logvar, z], name='encoder')
encoder.summary()

latent_inputs = Input(shape=(latent_dim,))
x4 = Dense(64, activation='relu')(latent_inputs)
x5 = Dense(64, activation='relu')(x4)
x6 = Dense(64, activation='relu')(x5)
outputs = Dense(input_dim, activation='sigmoid')(x6)
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

z_output = encoder(inputs)[2]
outputs = decoder(z_output)

vae = Model(inputs, outputs, name='variational_autoencoder')

# plot model
tf.keras.utils.plot_model(vae, show_shapes=True, expand_nested=True, show_dtype=True, to_file="model.png")

# Loss Function
# Kullback-Leibler Loss
kl_loss = 1 + z_logvar - K.square(z_mean) - K.exp(z_logvar)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
# Reconstruction Loss
reconstruction_loss = mse(inputs, outputs)
reconstruction_loss *= input_dim

vae_loss = K.mean(kl_loss + reconstruction_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

batch_size = 64
epochs = 50

pulse_train, pulse_test = train_test_split(pulse, test_size=0.2, shuffle=False)
history = vae.fit(pulse_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(pulse_test, None))

# Convert
# decoded_data = vae.predict(pulse_test)
decoded_data = vae.predict(pulse)

# Plot
plt.figure(figsize=(10, 2))
plt.plot(time, pulse.T)
plt.title('inputs')
plt.show()

plt.figure(figsize=(10, 2))
plt.plot(time, decoded_data.T)
plt.title('outputs')
plt.show()

# Learning Curve
plt.figure(figsize=(5, 5))
plt.plot(range(epochs), history.history['loss'], label='loss')
plt.title('loss')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

plt.figure(figsize=(5, 5))
plt.plot(range(epochs), history.history['val_loss'], label='val loss')
plt.title('val loss')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

plt.figure()
plt.hist(mse(pulse, decoded_data), bins=1024, histtype="step")
plt.xlabel("MSE")
plt.ylabel("Counts")
plt.xscale("log")
plt.yscale("log")
plt.grid(ls="--")
plt.show()

selected_indices = np.where(mse(pulse, decoded_data) <= 1e-4)[0]
plt.figure()
plt.plot(time, pulse[selected_indices].T)
plt.show()

import matplotlib.cm as cm
def plot_results(encoder,
                 decoder,
                 x_test,
                 batch_size=64):
    z_mean, _, _ = encoder.predict(x_test, batch_size=64)
    plt.figure()
    plt.scatter(z_mean[:, 0], z_mean[:, 1])
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.xlim(-0.5,-0.48)
    plt.ylim(-0.05,-0.046)
    plt.show()

plot_results(encoder,decoder,pulse_test,batch_size=64) 