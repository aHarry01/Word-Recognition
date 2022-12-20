# main.py
# mostly made this to keep implementation separate from testing calls 

from tdnn import *
import os
import time
import pickle
import matplotlib.pyplot as plt

from python_speech_features import mfcc

# TODO:
# make sure backprop does what we intend (actually finds correct gradient)
# can memorize SMALL (size 1-3) datasets but not larger datasets  - ISSUE WITH PRE-PROCESSING??
# regularization / validation set
# adjust pre-processing (noise reduction? change stft frame length? change # MFCCs?
# entropy cost function
#     Improvements (always check and fiddle with these until they actually improve accuracy...)
#        ** use better / different cost function (entropy cost!)
#        ** regularization
#        ** changing # Mel filters seems to have quite an effect ... maybe I shouldn't throw away higher MFCCs
#        ** try adjusting hidden layer sizes (try simpler 1-layer network first???)
#        * experiment with better / different learning rates (0.01 seems to work OK? narrow it down further?)
#        * use better / different activation function (??)
#        * experiment with initialization of weights/biases
# TODO: mini-batch gradient descent to speed up training (?)
#     Organize notes on implementation, take notes on how changing different shit changes the outcome (change pre-processing??)


'''
Function to train TDNN on given training files

layers = list of hidden layers
training_data = list of list of filenames for each class (ex. training_data[0] contains filenames that fall under class 0)
epochs = how many iterations to make on the training_data 
'''
def train(layers, training_data, epochs, learning_rate=0.01):
    expected_outputs = np.identity(len(training_data))
    #training_order = [(i, j) for j in range(len(training_data[0])) for i in range(len(training_data))]
    training_order = [(i, j) for i in range(len(training_data)) for j in range(len(training_data[0]))]

    num_correct = 0
    n = [0 for i in range(len(training_data))]

    cost = [] # total cost per epoch
    
    st = time.time()
    for i in range(epochs):
        st = time.time()
        cost.append(0)
        #random.shuffle(training_order)
        for (x,y) in training_order:
            filename = training_data[x][y]
            ideal_output = expected_outputs[x]
            #st = time.time()

            tdnn_input = np.transpose( get_mel_cepstral_coeffs( filename ) ) # data pre-processing
            #(rate,sig) = wav.read(filename)
            #sig = np.pad(sig,(0,16000 - len(sig)), 'constant', constant_values=(-1, -1)) # zero-pad to 1 min
            #mfccs = np.pad(mfcc(sig,rate,winstep=0.0125,numcep=14,nfilt=18,nfft=400,lowfreq=100,preemph=0.68,ceplifter=0), ((1,1),(0,0)))
            #tdnn_input = np.transpose( mfccs )

            #end = time.time()
            #print(f"Preprocessing: {end-st}")
            #st = time.time()
            output = run_tdnn( tdnn_input, layers) #forward propagation - what does the network currently output in response to this input?
        
            cost[-1] += sum( [pow(output[c]-ideal_output[c], 2) for c in range(len(ideal_output))]) #quadratic cost
            #end = time.time()
            #print(f"Forward prop: {end-st}")
            #st = time.time()
            train_tdnn( layers, output, ideal_output, learning_rate ) # update weights/biases based on output
            #end = time.time()
            #print(f"Train (back prop): {end-st}")
            #print(ideal_output)
            #print(output)
            end = time.time()
            if i == epochs-1: # final epoch, gather data on how accurate it is on training data
                result = np.argmax(output) # returns index with largest value
                if ideal_output[result] == 1:
                    num_correct += 1
                    n[result] += 1
            #    print(filename)
                #print(ideal_output)
                #print(output)
        
        end = time.time()
        #print(f"Epoch {i} completed in {end-st} sec with avg cost {cost[-1]/len(training_order)}")
    print(f"Final epoch avg cost: {cost[-1]/len(training_order)}")
    print(f"Final epoch training accuracy: {num_correct/(len(training_data)*len(training_data[0]))}")
    #plt.plot(cost)
    #plt.show()
    
'''
Assess accuracy after training

testing_data = list of list of filenames for each class to test on (ex. testing_data[0] contains filenames that fall under class 0)
'''
def test(layers, testing_data):
    total_attempted = 0
    total_correct = 0
    total_correct_by_category = [0 for x in range(len(testing_data))]
    
    for (c,category) in enumerate(testing_data):
        for filename in category:
            tdnn_input = np.transpose( get_mel_cepstral_coeffs( filename ))
            #(rate,sig) = wav.read(filename)
            #sig = np.pad(sig,(0,16000 - len(sig)), 'constant', constant_values=(-1, -1))
            #mfccs = np.pad(mfcc(sig,rate,winstep=0.0125,numcep=14,nfilt=18,nfft=400,lowfreq=100,preemph=0.68,ceplifter=0), ((1,1),(0,0)))
            #tdnn_input = np.transpose( mfccs )
            output = run_tdnn(tdnn_input, layers)
            #print(output)
            result = np.argmax(output) # returns index with largest value
            #print(f"{filename}\nCategory {c}, Result {result}") 
            if result == c: # output was correct
                total_correct += 1
                total_correct_by_category[c] += 1
            total_attempted += 1

    accuracy_overall = total_correct / total_attempted
    accuracy_by_category = [total_correct_by_category[i]/len(testing_data[i]) for i in range(len(testing_data))]

    return accuracy_overall, accuracy_by_category
    

'''
Get lists of training and testing files

directories = list of folders containing examples. Each folder should contain files belonging to one category
num_training = number of files to use in training list for each category
num_testing = number of files to use in testing list for each category

'''
def generate_training_filelist(directories, num_training, num_testing):
    training_data = []
    testing_data = []
    for d in directories:
        files = os.listdir(d)
        #random.shuffle(files)
        files= files[0:num_training+num_testing]
        training_data.append([os.path.join(d, f) for f in files[0:num_training] ])
        testing_data.append([os.path.join(d, f) for f in files[num_training:num_testing+num_training] ])
    return training_data, testing_data


'''
Test performance in various scenarios to evaluate this setup
'''
def bulk_test():
    #train_num = [1,3,5,10,20,50] # number of each class to train on
    train_num = [500]
    test_num = 50 # number of each class to test on
    epochs = 150 # number of times to run through training examples
    learning_rate = 0.0003 # adjust how fast the network learns
    directories = ["data\\train\\audio\\one",
                "data\\train\\audio\\two",
                "data\\train\\audio\\three",
                "data\\train\\audio\\four",
                "data\\train\\audio\\five",
                "data\\train\\audio\\six",
                "data\\train\\audio\\seven",
                "data\\train\\audio\\eight",
                "data\\train\\audio\\nine",
                "data\\train\\audio\\zero"]

    for z in train_num:

        #load from saved object
        with open('random_start_weights_biases_REAL10', 'rb') as f:
            layers = pickle.load(f)

        print(f"N = {z}")
        
        training_data, testing_data = generate_training_filelist(directories, z, test_num)
        train(layers, training_data, epochs, learning_rate)
        acc, acc_by_cat = test(layers, testing_data)
        print(f"Test accuracy: {acc}")
        print(acc_by_cat)


        with open('tdnn_results_22', 'wb') as f:
            pickle.dump(layers, f)
    

# ======== INIT TDNN ===========

'''
# define layer 1
b =  np.random.normal(0, 0.12, (8,77)) #biases 8 rows and 77 cols, uses a Gaussian with mean=0, stddev=1/sqrt(cw*rw) 
w = np.random.normal(0, 0.12, (8,14,5)) #np.random.randn(8,14,5) #8 weight matrices, each has 14 rows and 5 cols
layer1 = HiddenLayer(w, b, 14, 81)

# define layer 2
b =  np.random.normal(0, 0.11, (3,67)) #np.random.randn(3, 67) #[ [random.random() for c in range(67)] for y in range(3)]
w = np.random.normal(0, 0.11, (3,8,11)) #np.random.randn(3, 8, 11) #[ [ [random.random() for c in range(11)] for r in range(8) ] for x in range(3)] # 3 weight matrices, each has 8 rows and 11 cols
layer2 = HiddenLayer(w, b, 8, 77)

layers = [layer1, layer2]

with open('random_start_weights_biases', 'wb') as f:
    pickle.dump(layers, f)
'''

'''
# define layer 1
b =  np.random.normal(0, 0.12, (8,47)) #biases 8 rows and 77 cols, uses a Gaussian with mean=0, stddev=1/sqrt(cw*rw) 
w = np.random.normal(0, 0.12, (8,14,35)) #np.random.randn(8,14,5) #8 weight matrices, each has 14 rows and 35 cols
layer1 = HiddenLayer(w, b, 14, 81)

# define layer 2
b =  np.random.normal(0, 0.11, (4,37)) #np.random.randn(3, 67)
w = np.random.normal(0, 0.11, (4,8,11)) #np.random.randn(3, 8, 11) #3 weight matrices, each has 8 rows and 11 cols
layer2 = HiddenLayer(w, b, 8, 47)

layers = [layer1, layer2]

with open('random_start_weights_biases_REAL4', 'wb') as f:
    pickle.dump(layers, f)
'''

'''
# define layer 1
b =  np.random.normal(0, 0.12, (10,47)) #biases 8 rows and 77 cols, uses a Gaussian with mean=0, stddev=1/sqrt(cw*rw) 
w = np.random.normal(0, 0.12, (10,14,35)) #np.random.randn(8,14,5) #8 weight matrices, each has 14 rows and 35 cols
layer1 = HiddenLayer(w, b, 14, 81)


# define layer 3
b =  np.random.normal(0, 0.11, (3,57)) #np.random.randn(3, 67)
w = np.random.normal(0, 0.11, (3,10,11)) #np.random.randn(3, 8, 11) #3 weight matrices, each has 8 rows and 11 cols
layer2 = HiddenLayer(w, b, 10, 47)

layers = [layer1, layer2]

with open('random_start_weights_biases_14', 'wb') as f:
    pickle.dump(layers, f)
'''

# define layer 1
b =  np.random.normal(0, 0.12, (8,47)) #biases 8 rows and 77 cols, uses a Gaussian with mean=0, stddev=1/sqrt(cw*rw) 
w = np.random.normal(0, 0.12, (8,14,35)) #np.random.randn(8,14,5) #8 weight matrices, each has 14 rows and 35 cols
layer1 = HiddenLayer(w, b, 14, 81)

# define layer 2
b =  np.random.normal(0, 0.11, (10,37)) #np.random.randn(3, 67)
w = np.random.normal(0, 0.11, (10,8,11)) #np.random.randn(3, 8, 11) #3 weight matrices, each has 8 rows and 11 cols
layer2 = HiddenLayer(w, b, 8, 47)

layers = [layer1, layer2]

with open('random_start_weights_biases_REAL10', 'wb') as f:
    pickle.dump(layers, f)


#load from saved object
#with open('random_start_weights_biases', 'rb') as f:
#    layers = pickle.load(f)

'''
# simple 1-layer test example
b = np.random.normal(0, 0.12, (3,57)) # cout = cin - cw + 1 = 81-25+1 = 57
w =  np.random.normal(0, 0.12, (3,14,25)) #3 weight matrices, each has 14 rows and 15 cols
layer = HiddenLayer(w,b,41,81)

layers = [layer]
'''

# ======== TRAIN ===============
# train 'one' vs 'two' vs 'three'

bulk_test()

'''
train_num = 5 # number of each class to train on
test_num = 5 # number of each class to test on
epochs = 200 # number of times to run through training examples
learning_rate = 0.001 #0.00001 # adjust how fast the network learns


directories = ["data\\train\\audio\\one",
               "data\\train\\audio\\two",
               "data\\train\\audio\\three"]
training_data, testing_data = generate_training_filelist(directories, train_num, test_num)
print(training_data)
print(testing_data)
train(layers, training_data, epochs, learning_rate)
acc, acc_by_cat = test(layers, testing_data)
print(f"Overall accuracy: {acc}")
print(acc_by_cat)

with open('tdnn_results', 'wb') as f:
        pickle.dump(layers, f)
'''

'''
for r in range(100):
    print("=========================================")


    tdnn_input = np.transpose( get_mel_cepstral_coeffs( "data\\train\\audio\\three\\00b01445_nohash_0.wav" ) )
    output = run_tdnn( tdnn_input, layers)
    train_tdnn( layers, output, [0,0,1], 0.01 )
    print(output)
    
    tdnn_input = np.transpose( get_mel_cepstral_coeffs( "data\\train\\audio\\two\\0b56bcfe_nohash_0.wav" ) )
    output = run_tdnn( tdnn_input, layers)
    train_tdnn( layers, output, [0,1,0], 0.01 )
    print(output)

    tdnn_input = np.transpose( get_mel_cepstral_coeffs( "data\\train\\audio\\one\\00176480_nohash_0.wav" ) )
    output = run_tdnn( tdnn_input, layers)
    train_tdnn( layers, output, [1,0,0], 0.01 )
    print(output)

'''
