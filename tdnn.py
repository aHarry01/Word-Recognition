# tdnn.py
# contains functions / classes to implement the pre-processing of audio data,
# time-delay neural network, and training of the neural network

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import scipy
import scipy.io.wavfile as wav
import scipy.signal as signal
import math
import random
import time

filterbank = None # cached mel filterbank so we don't have to recalculate

# Misc Useful links:
#   https://www.tutorialspoint.com/read-and-write-wav-files-using-python-wave
#   https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html


# ========= PRE-PROCESSING FUNCTIONS =======================

def plot_fft(audio_filename):
    # Adapted from: https://stackoverflow.com/questions/23377665/python-scipy-fft-wav-files
    import matplotlib.pyplot as plt
    from scipy.fftpack import fft
    from scipy.io import wavfile # get the api
    fs, data = wavfile.read(audio_filename) # load the data
    #a = data.T[0] # this is a two channel soundtrack, I get the first track
    b=[(ele/2**8.)*2-1 for ele in data] # this is 8-bit track, b is now normalized on [-1,1)
    c = fft(b) # calculate fourier transform (complex numbers list)
    d = int(len(c)/2)  # you only need half of the fft list (real signal symmetry)
    plt.plot(abs(c[0:(d-1)]), 'r')
    plt.show()

def plot_spectrogram(audio_filename):
    # Adapted from: https://stackoverflow.com/questions/43109708/how-to-plot-spectrogram-using-stft-in-python 
    sample_rate, samples = wav.read(audio_filename)
    #print(sample_rate)
    #print(len(samples))
    #plt.specgram(samples, cmap='inferno', Fs=sample_rate) # TODO: configure this!!!
    f, t, Zxx = signal.stft(samples, fs=sample_rate, nperseg=400) #256-point by default, default overlap between segments is 128
   # print(f) # frequency buckets
   # print(t) # time buckets
    #print(Zxx) # fft at each time slot (2D array)
   # print(len(Zxx[1]))
   # print(f"f length {len(f)}, t length {len(t)}, STFT length {len(Zxx)}, FT length {len(Zxx[0])}")
    plt.pcolormesh(t, f, np.abs(Zxx), cmap='inferno', shading="auto")
    plt.show()

'''
Divides audio data into overlapping frames and computes the periodogram estimate of power spectrum of each frame
Heavily based on steps 1 & 2 from:
    http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/

Currently assumes the audio input is 1 second in length (which Google keyword dataset is) otherwise zero-pads it to 1 min 
frame_duration = time duration of each frame (in seconds)
Returns:
    frame_length = samples per frame
    frame_step = samples between frame beginnings
    f = array of center of frequency buckets
    periodogram = list of power spectrum estimates for each frame
'''

def short_time_periodogram(audio_filename, frame_duration=0.025):
    sample_rate, samples = wav.read(audio_filename)
    if sample_rate != 16000 or len(samples) > 16000:
        print(f"ERR: {audio_filename}, {sample_rate}")
    # pre-emphasis filter - https://speechprocessingbook.aalto.fi/Preprocessing/Pre-emphasis.html
    samples = signal.filtfilt([1, -0.68],[1], samples) # not sure if this is really having much of an effect

    #samples = np.sign(samples)*( (np.log(1 + 255*np.abs(samples)))/(np.log(1 + 255)) ) #u-law compression
        
    samples = np.pad(samples,(0,16000 - len(samples)), 'constant', constant_values=(-1, -1)) # zero-pad because some are shorter than 1 second, except we can't zero-pad because it messes things up later (since we take the log), so just do -1
    samples[samples==0] = -1 # can't have any zeros - it makes the log later on infinite. TODO: there might be a better way to deal with this though
    frame_length = int(sample_rate*frame_duration) # number of samples to meet frame_duration
    f, t, spectrogram = signal.stft(samples, fs=sample_rate, nperseg=400, window="hamming") #default noverlap=nperseg//2, window="hamming"
    #plt.xlabel("Time")
    #plt.ylabel("Frequency")
    #plt.title("Spectrogram (FFT applied to every 25ms frame)")
    #plt.pcolormesh(t, f, np.abs(spectrogram), cmap='inferno', shading="auto")
    #plt.show()
    spectrogram = np.transpose(spectrogram) # transpose 'spectrogram' so that each array is the fft for that time frame
    periodograms = (np.abs(spectrogram)**2)/400 # elementwise square each FFT and divide by N=400 to get periodograms (power spectrum estimates)
    return sample_rate, frame_length, t, f, periodograms


'''
Compute mel filterbanks
'''
def get_mel_filterbanks(frame_length, num_mel_filters, sample_rate=16000):
    # compute Mel filterbanks - TODO: probably don't need to this for each MFCC calculation tho...just save results
    upper_mel_freq = 1125*math.log(1+ (sample_rate/2)/700 ) # convert highest freq to Mel
    lower_mel_freq = 401.25 # use 100 Hz as the lowest for now. 100Hz=150.22 mels. 300Hz = 401.25 mels
    filterbank_freqs = []
    mel_filterbank = []
    f = [x*(8000/201.0) for x in range(201)] # only used for plotting mel filterbank

    for x in range(num_mel_filters+2): #need 28 points to compute 26 filterbanks
        mel_freq = lower_mel_freq + x*(upper_mel_freq-lower_mel_freq)/(num_mel_filters+1) # equally spaced in Mel frequency scale
        freq = 700*(math.exp(mel_freq/1125) - 1)# convert back to Hz - will no longer be equally spaced
        # round to the nearest frequency bucket that we actually have
        # this gets the INDEX of the closest frequency in f (the list of frequency bins)
        freq_bucket_index = math.floor( (frame_length+1)*freq/sample_rate + 0.5)
        
        #print(f"{mel_freq} {freq} {freq_bucket_index} {f[freq_bucket_index]}")
        filterbank_freqs.append(freq_bucket_index) #TODO: remove this list and just use start_f_i and end_f_i

        #compute filter peaking starting 2 freqs ago, peaking 1 freq ago and ending on this freq 
        if x >= 2:
            filt = np.array([0 for q in range( int(frame_length/2) + 1)], dtype=np.float32)
            start_f_i = filterbank_freqs[x-2]
            mid_f_i = filterbank_freqs[x-1]
            end_f_i = freq_bucket_index
            # ascending part of the triangular filter
            for (i, f_i) in enumerate(range(start_f_i, mid_f_i)):
                filt[f_i] = i * 1.0/(mid_f_i - start_f_i)

            # descending part of the triangular filter
            for (i, f_i) in enumerate(range(mid_f_i, end_f_i+1)):
                filt[f_i] = -1*i * 1.0/(end_f_i - mid_f_i) + 1

            #plt.plot(f, filt, label = f"line {x}")
            mel_filterbank.append(filt)

    #plt.xlabel('Frequency(Hz)')
    #plt.title(f'Mel Frequency Filterbank ({num_mel_filters} mel filters)')
    #plt.show()
    return mel_filterbank


'''
Get Mel Cepstral coefficients from 1 sec .wav audio file
Based on:
    http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/

num_mel_filters = Number of tringular filters to use in the Mel filterbank
'''
def get_mel_cepstral_coeffs(audio_filename, num_mel_filters=18): # ORIGINAL: 26
    
    # split into frames & get power spectrum estimate for each
    sample_rate, frame_length, t, f, periodograms = short_time_periodogram(audio_filename)
    
    global filterbank
    if not filterbank:
        filterbank = get_mel_filterbanks(frame_length, num_mel_filters)

    # Apply Mel filters (get filterbank energy for each filter by multiplying with power spectrum then summing
    mel_weighted_spectrum = [] # contains the (log of the) filter bank energies at each time frame
    for pwr_spectrum in periodograms:
        filterbank_energies = []
        for filt in filterbank:
            #filterbank_energies.append( sum(np.array(filt)*np.array(pwr_spectrum)) ) # elementwise multiplication then sum
            #print(np.sum(filt*pwr_spectrum))
            filterbank_energies.append(np.log(np.sum(filt*pwr_spectrum)) ) # elementwise mulitplication then sum for filterbank energy, then take log
        mel_weighted_spectrum.append(filterbank_energies)    

    mels = [x for x in range(num_mel_filters)]
    #plt.title("Log of filterbank energies")
    #plt.ylabel("Filterbank number")
    #plt.xlabel("Time")
    #plt.pcolormesh(t, mels, np.transpose(mel_weighted_spectrum), cmap='inferno', shading="auto")
    #plt.show()

    # mean normalization - subtract the mean of each frequency from each frame
    #mel_weighted_spectrum -= (np.mean(mel_weighted_spectrum) + 1e-8)
    #mel_weighted_spectrum = mel_weighted_spectrum.tolist()

    # Decorrelate filter bank energies with DCT to get MFCCs
    # only keep lower 12-14 coefficients?
    for (i,x) in enumerate(mel_weighted_spectrum):
        #mel_weighted_spectrum[i] = scipy.fftpack.dct(x, norm="ortho")
        mel_weighted_spectrum[i] = scipy.fftpack.dct(x, norm="ortho")[1:15] # only MFCCs 2-15

    mels = [x for x in range(len(mel_weighted_spectrum[0]))]
    #plt.title("MFCCs")
    #plt.xlabel('Time')
    #plt.ylabel('Coefficients')
    #plt.pcolormesh(t, mels, np.transpose(mel_weighted_spectrum), cmap='inferno_r', shading="auto")
    #plt.show()

    # mean normalization - subtract the mean of each frequency from each frame
    #mel_weighted_spectrum -= (np.mean(mel_weighted_spectrum) + 1e-8)
    #mel_weighted_spectrum = mel_weighted_spectrum.tolist()

    #TODO: plot mel envelope & original signal??
        # https://speechprocessingbook.aalto.fi/Representations/Melcepstrum.html

    #print(len(mel_weighted_spectrum))
    #print(len(mel_weighted_spectrum[0]))

    return mel_weighted_spectrum

    
# AUDIO PRE-PROCESSING:
# https://wiki.aalto.fi/display/ITSP/Cepstrum+and+MFCC
# http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
# https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
#   stft (divide into sections) - for each frame we have an array of complex #s for the fft for the amplitude at each freq
#   get periodogram power spectrum estimate - for each frame we have an array of real numbers for the power at each freq
#   apply each triangular filter to the periodogram in a frame and for each filter sum up the resulting coefficients and take the log
#       assuming 26 filters, we'll be left w 26 numbers in each frame
#   take DCT of the 26 numbers for each frame and keep only the first 13. now we have the Mel Frequency Cepstral Coefficients for each frame - use as input to TDNN

# Things to possibly adjust
#   ditch DCT decorrelating and possibly log of filterbank energies in pre-processing? (idk, test TDNN with both)
#   number of filters in the filterbank
#   window function on frames?  
#   n-point fft (currently doing 400 to match frame_length)
#

# =====================================================================
# ========= TIME-DELAY NEURAL NETWORK FUNCTIONS =======================
'''
Compute the output of a hidden layer given
I = input array
W = list of trainable weight arrays
B = trainable biases
Uses sigmoid activation function but should try others when testing

Returns output of hidden layer

'''
def compute_hidden_layer_output(I, W, B):
    output = [] # row r of the output is conv(input, weight matrix r) + bias[r]

    for (r,w) in enumerate(W): # apply each weight matrix
        row = np.clip(signal.convolve2d(I, w, mode="valid")[0] + B[r], -600, 600) # if it's too big, np.exp will overflow
        #Apply activation function TODO: try other activation functions
        #row = [1 / (1 + math.exp(-r)) for r in row]
        #try:
        row = 1.0 / (1.0 + np.exp(-row))# apply sigmoid activation function to each element in row
        #except:
        #    print("SIGMOID ERROR: ", end="")
        #    print(row)
        output.append(row)    

    return np.array(output)

'''
Compute the output of the output layer, which
just sums each row in the input and applies an activation function
'''
def compute_output_layer_output(I):
    output = []
    for r in I:
        #output.append( sum(r) )
        #output.append( 1/ (1 + math.exp(-sum(r)) ) )
        # sigmoid time-scaled by 5/len(r) so that if all 'r' inputs in a row are 1, the output will be 0.995
        #output.append( 1/ (1 + np.exp(-sum(r)*(4/len(r))) ) )
        #output.append(1.0 / (1.0 + np.exp(-sum(r))))
        output.append(np.average(r)) # average of all these in a row
    
    return output

'''
Container for all the variables defining a hidden layer
    W = list of weight matrices 
    B = bias matrix
    ri = number of input rows
    ci = numer input columns
    r = number of output rows
    c = number of output cols
'''
class HiddenLayer():
    def __init__(self, W, B, ri, ci): #must be initialized with values
        self.W = W
        self.B = B
        self.rw = len(W[0])
        self.cw = len(W[0][0])
        self.ri = ri
        self.ci = ci
        self.rout = len(W) # output rows = number of weight matrices
        self.cout = ci - len(W[0][0]) + 1 # output cols = input cols - weight matrix cols + 1
        self.output = None

    def compute_output(self, I):
        self.input = I
        self.output = compute_hidden_layer_output(I, self.W, self.B)
        return self.output

'''
Run the given input i through all the HiddenLayers in 'layers'
'''
def run_tdnn(I, layers):
    for layer in layers:
        #print(I)
        I = layer.compute_output(I) # the output of this current layer becomes the input to the next
    return compute_output_layer_output(I) # run the last 'output layer' to get final output

'''
Update weights/matrices of the layers based on last input
Must call run_tdnn first.
output = result of running the current TDNN on an input (run_tdnn returns this)
desired_output = desired result of running TDNN on an input
'''
def train_tdnn(layers, output, desired_output, learning_parameter=0.1):
    
    #compute initial error gradient dE/dO
    output_gradient = np.array([ [O-desired_output[i] for c in range(layers[-1].cout)] for (i,O) in enumerate(output)])

    for (q,layer) in enumerate(reversed(layers)):

        output_gradient_activation = np.array(output_gradient)*layer.output*(1-layer.output) #dE/dO * A'(O), output gradient times derivative of activation func
        
        for w in range(len(layer.W)): # loop thru each weight matrix
            weight_gradients = np.flip( signal.convolve2d(layer.input, [output_gradient_activation[w]], mode="valid"))
            layer.W[w] -= learning_parameter*weight_gradients
        # adjust biases
        layer.B = layer.B - learning_parameter*output_gradient_activation # all operations are elementwise ... bias[i][j] = output_gradient[i][j]*A'(Oij)

        # don't need to do this if it's the last layer...
        if (q != len(layers)-1):            
            # compute output_gradient for the next layer (derivative of error wrt input to this layer

            new_output_gradient = []
            padded_output_gradient = np.pad(output_gradient_activation,((0, 0), (layer.ci-layer.cout, layer.ci-layer.cout))) # pad output gradient so result of conv will have same # rows as input 
            for r in range(0, layer.ri):
                w = [ np.flip(weight_matrix[layer.rw-r-1]) for weight_matrix in reversed(layer.W)]
                new_output_gradient.append( signal.convolve2d(w, padded_output_gradient, mode="valid")[0] )
            output_gradient = new_output_gradient            
        #end = time.time()
        #print(f"OUTPUT GRADIENT {end-st}")
    #print("=============================")

'''
# very simple test case - works
w1 = np.array([[1,0],[0,1]])
w2 = np.array([[2,1],[2,1]])
I= np.array([[1,0,2],[3,4,5]])
b = np.array([[0,3],[1,3]])
W = [w1,w2]
print(compute_hidden_layer_output(I,W,b))
'''

'''
# sample simple test case but use HiddenLayer class
b = np.array([[0,3],[1,3]])
W = [np.array([[1,0],[0,1]]), np.array([[2,1],[2,1]])]
hidden_layer = HiddenLayer(W, b, 2, 3)
I= np.array([[1,0,2],[3,4,5]])
print(compute_hidden_layer_output(I, hidden_layer.W, hidden_layer.B))
print(compute_hidden_layer_output(I,W,b))
'''

# small 2-layer test case
#I = np.array([[1,0,2,1,2],[1,4,8,7,3],[1,7,3,2,1]])
# define hidden layer 1
#W1 = np.array([
#    [[1,0],[0,1],[1,0]], # weight matrix 1
#    [[1,1],[3,3],[3,0]], # weight matrix 2
#    [[5,5],[0,0],[0,1]]  # weight matrix 3
#    ])
#B1 = np.array([[1,3,1,1],[2,2,6,0],[3,1,0,1]]) # biases
#layer1 = HiddenLayer(W1, B1, 3, 5)
#define hidden layer 2
#W2 = np.array([
#        [[1,2,0],[2,0,2],[0,1,0]],
#        [[0,2,1],[1,2,1],[1,3,2]]
#    ])
#B2 = np.array([[1,2],[1,2]])
#layer2 = HiddenLayer(W2, B2, 3, 4)
#print(run_tdnn( I, [layer1, layer2]))
#O1 = compute_hidden_layer_output(I, layer1.W, layer1.B)
#O2 = compute_hidden_layer_output(O1, layer2.W, layer2.B)
#print(O2)
#print(compute_output_layer_output(O2))

'''
# attempt backpropagation on simple 1-layer example
b = np.array([[0,0.5],[0.25,0.5]])
W = [np.array([[-1,0.5],[0,1.4]]), np.array([[-.5,0.25],[0.25,-0.6]])]
hidden_layer = HiddenLayer(W, b, 2, 3)
I= np.array([[1,0,2],[3,4,5]])
#print(hidden_layer.compute_output(I))
#print(compute_hidden_layer_output(I, hidden_layer.W, hidden_layer.B))
for x in range(100):
    output = run_tdnn( I, [hidden_layer])
    print(output)
    y = [0,1] # desired results
    train_tdnn( [hidden_layer], output, y, 0.5)
'''


if __name__ == "__main__":

    custom_mfcc_feat = get_mel_cepstral_coeffs( 'data\\train\\audio\\three\\0a9f9af7_nohash_0.wav' )
    print( len(custom_mfcc_feat))
    from python_speech_features import mfcc
    (rate,sig) = wav.read("data\\train\\audio\\three\\0a9f9af7_nohash_0.wav")
    mfcc_feat = mfcc(sig,rate,winstep=0.0125,numcep=14,nfilt=18,nfft=400,lowfreq=100,preemph=0.68,ceplifter=0)

    mfcc_feat = mfcc(sig,rate,numcep=14)
    print(len(mfcc_feat))
    print(len(mfcc_feat[0]))

    #t = [(200*x)/16000 for x in range(int(16000/200) - 1)]
    #mels = [x for x in range(len(mfcc_feat[0]))]
    #plt.pcolormesh(t, mels, np.transpose(mfcc_feat), cmap='inferno', shading="auto")
    #plt.show()
    '''
    # attempt backpropagation on simple 2-layer example
    I = np.array([[1,0,2,1,2],[1,4,8,7,3],[1,7,3,2,1]])
    # define hidden layer 1
    W1 = np.array([
        [[-0.5,0],[0,0.5],[-0.2,0]], # weight matrix 1
        [[-0.12,0.1],[1,-1],[0.8,0.4]], # weight matrix 2
        [[0.4,0.2],[0,0],[0,11]]  # weight matrix 3
        ])
    B1 = np.array([[0,0,-1,-1],[0.5,-.2,.6,0],[-1,1,0,1]]) # biases
    layer1 = HiddenLayer(W1, B1, 3, 5)
    #define hidden layer 2
    W2 = np.array([
            [[0.1,-0.3,0],[0.2,0,0.2],[0,-0.1,0]],
            [[0,0.25,0.1],[-0.5,1,0.5],[0.51,-0.93,0.2]]
        ])
    B2 = np.array([[1,-1],[1,-1]])
    layer2 = HiddenLayer(W2, B2, 3, 4)

    for t in range(10):
        output = run_tdnn( I, [layer1, layer2])
        print(output)
        y = [1,0]
        train_tdnn( [layer1, layer2], output, y, 1.0 )
    '''


'''
# on real data
# define layer 1
b = np.random.randn(8,77) #same as [[random.random() for c in range(77)] for r in range(8)] except uses a Gaussian with mean=0, stddev=1 
w = np.random.randn(8,14,5) #[ [ [random.random() for c in range(5)] for r in range(14) ] for x in range(8)] # 8 weight matrices, each has 14 rows and 5 cols
layer1 = HiddenLayer(w, b, 14, 81)

# define layer 2
b = np.random.randn(3, 67) #[ [random.random() for c in range(67)] for y in range(3)]
w = np.random.randn(3, 8, 11) #[ [ [random.random() for c in range(11)] for r in range(8) ] for x in range(3)] # 3 weight matrices, each has 8 rows and 11 cols
layer2 = HiddenLayer(w, b, 8, 77)

tdnn_input = np.transpose( get_mel_cepstral_coeffs( 'data\\train\\audio\\three\\0a9f9af7_nohash_0.wav') )
#print(run_tdnn( np.transpose(tdnn_input), [layer1, layer2]))

# attempt backpropagation on real data just by running the same input over and over
y = [0, 0, 1] # output is three
for t in range(100):
    output = run_tdnn( tdnn_input, [layer1, layer2])
    print(output)
    train_tdnn( [layer1, layer2], output, y, 1.0 )
'''


# =====================================================================

#plot_spectrogram('data\\train\\audio\\three\\0a9f9af7_nohash_0.wav')
#tdnn_input = get_mel_cepstral_coeffs( 'data\\train\\audio\\three\\0a9f9af7_nohash_0.wav' )
#print(len(tdnn_input))
#print(len(tdnn_input[0]))

#plot_fft('data\\train\\audio\\yes\\0f3f64d5_nohash_1.wav')
'''
plot_fft('data\\train\\audio\\no\\0f3f64d5_nohash_1.wav')

plot_fft('data\\train\\audio\\one\\1eddce1d_nohash_1.wav')
plot_fft('data\\train\\audio\\three\\0a9f9af7_nohash_0.wav')
'''

#plot_spectrogram('data\\train\\audio\\three\\0a9f9af7_nohash_0.wav')

# TODO: implement hidden convolutional layers of TDNN
#   possibly downsample input if it's too slow?
#   run with test data to be sure they do what you expect

# TODO: backpropagation ... hardest part probably, loo kup many a tutorial/textbook and hope for the best!
#   might have better luck finding things on CNNs instead of TDNNs
# TODO: train on "yes" vs "no"
