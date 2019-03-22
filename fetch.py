from sys import byteorder
from array import array
from struct import pack

import pyaudio
import wave

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import os
import numpy as np
import random
from skimage.measure import block_reduce

#To find the duration of wave file in seconds
import wave
import contextlib

#Keras imports
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.models import model_from_json

import time
import datetime

os.environ['KMP_DUPLICATE_LIB_OK']  = 'True'
imwidth                             = 50
imheight                            = 34
total_examples                      = 2000
speakers                            = 4
examples_per_speaker                = 50
tt_split                            = 0.1
num_classes                         = 10
test_rec_folder                     = "./testrecs"
log_image_folder                     = "./logims"
recording_directory                 = "../SoundCNN/recordings/"
num_test_files                      = 1

THRESHOLD                           = 1000
CHUNK_SIZE                          = 512
FORMAT                              = pyaudio.paInt16
RATE                                = 8000#44100
WINDOW_SIZE                         = 50
CHECK_THRESH                        = 3
SLEEP_TIME                          = 0.5 #(seconds)
IS_PLOT                             = 1
LOG_MODE                            = 0 # 1 for time, 2 for frequency

#Check for silence
def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

"""
Record a word or words from the microphone and 
return the data as an array of signed shorts.
"""
def record():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > 20:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    return sample_width, r

#Extract relevant signal from the captured audio
def get_bounds(ds):
    np.array(ds)
    lds = len(ds)
    count = 0
    ll=-1
    ul=-1

    #Lower Limit
    for i in range(0,lds,WINDOW_SIZE):
        sum = 0
        for k in range(i,(i+WINDOW_SIZE)%lds):
            sum = sum + np.absolute(ds[k])
        if(sum>THRESHOLD):
            count +=1
        if(count>CHECK_THRESH):
            ll = i - WINDOW_SIZE * CHECK_THRESH
            break
        
    #Upper Limit
    count = 0
    for j in range(i,lds,WINDOW_SIZE):
        sum = 0
        for k in range(j,(j+WINDOW_SIZE)%lds):
            sum = sum + np.absolute(ds[k])
        if(sum<THRESHOLD):
            count +=1
        if(count>CHECK_THRESH):
            ul = j - WINDOW_SIZE * CHECK_THRESH


        if(ul>0 and ll >0):
            break
    return ll, ul 


# Records from the microphone and outputs the resulting data to 'path'
def record_to_file(path):
    
    sample_width, data = record()
    ll, ul = get_bounds(data)
    print(ll,ul)
    if(ul-ll<100):
        return 0
    #nonz  = np.nonzero(data)
    ds = data[ll:ul]
    if(IS_PLOT):
        plt.plot(data)
        plt.axvline(x=ll)
        #plt.axvline(x=ll+5000)
        plt.axvline(x=ul)
        plt.show()

    #data = pack('<' + ('h'*len(data)), *data)
    fname = "0.wav"
    if not os.path.exists(path):
        os.makedirs(path)
    wf = wave.open(os.path.join(path,fname), 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(ds)
    wf.close()
    return 1

# Function to find the duration of the wave file in seconds
def findDuration(fname):
    with contextlib.closing(wave.open(fname,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        sw   = f.getsampwidth()
        chan = f.getnchannels()
        duration = frames / float(rate)
        #print("File:", fname, "--->",frames, rate, sw, chan)
        return duration

#Plot Spectrogram
def graph_spectrogram(wav_file, nfft=512, noverlap=511):
    findDuration(wav_file)
    rate, data = wavfile.read(wav_file)
    #print("")
    fig,ax = plt.subplots(1)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.axis('off')
    pxx, freqs, bins, im = ax.specgram(x=data, Fs=rate, noverlap=noverlap, NFFT=nfft)
    ax.axis('off')
    plt.rcParams['figure.figsize'] = [0.75,0.5]
    #fig.savefig('sp_xyz.png', dpi=300, frameon='false')
    fig.canvas.draw()
    size_inches  = fig.get_size_inches()
    dpi          = fig.get_dpi()
    width, height = fig.get_size_inches() * fig.get_dpi()

    #print(size_inches, dpi, width, height)
    mplimage = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    #print("MPLImage Shape: ", np.shape(mplimage))
    imarray = np.reshape(mplimage, (int(height), int(width), 3))
    plt.close(fig)
    return imarray

#Convert color image to grayscale
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

#Normalize Gray colored image
def normalize_gray(array):
    return (array - array.min())/(array.max() - array.min())

#Split the dataset into test and train sets randomly
def create_train_test(audio_dir):
    file_names = [f for f in os.listdir(audio_dir) if '.wav' in f]
    file_names.sort()
    test_list = []
    train_list = []
    
    for i in range(int(total_examples/examples_per_speaker)):
        test_list.extend(random.sample(file_names[(i*examples_per_speaker+1):(i+1)*examples_per_speaker], int(examples_per_speaker*tt_split)))

    train_list = [x for x in file_names if x not in test_list]

    y_test = np.zeros(len(test_list))
    y_train = np.zeros(len(train_list))
    x_train = np.zeros((len(train_list), imheight, imwidth))
    x_test = np.zeros((len(test_list), imheight, imwidth))

    tuni1   = np.zeros(len(test_list))
    tuni2   = np.zeros(len(test_list))

    for i, f in enumerate(test_list):
        y_test[i]     = int(f[0])
        spectrogram   = graph_spectrogram( audio_dir + f )
        graygram      = rgb2gray(spectrogram)
        normgram      = normalize_gray(graygram)
        norm_shape    = normgram.shape
        if(norm_shape[0]>150):
            continue
        redgram       = block_reduce(normgram, block_size = (3,3), func = np.mean)
        x_test[i,:,:] = redgram
        print("Progress Test Data: {:2.1%}".format(float(i) / len(test_list)), end="\r")

    for i, f in enumerate(train_list):
        y_train[i] = int(f[0])
        spectrogram   = graph_spectrogram( audio_dir + f )
        graygram      = rgb2gray(spectrogram)
        normgram      = normalize_gray(graygram)
        norm_shape    = normgram.shape
        if(norm_shape[0]>150):
            continue
        redgram       = block_reduce(normgram, block_size = (3,3), func = np.mean)
        x_train[i,:,:] = redgram
        print("Progress Training Data: {:2.1%}".format(float(i) / len(train_list)), end="\r")
        
    return x_train, y_train, x_test, y_test

#Create Keras Model
def create_model(path):
    x_train, y_train, x_test, y_test = create_train_test(path)

    print("Size of Training Data:", np.shape(x_train))
    print("Size of Training Labels:", np.shape(y_train))
    print("Size of Test Data:", np.shape(x_test))
    print("Size of Test Labels:", np.shape(y_test))

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train = x_train.reshape(x_train.shape[0], imheight, imwidth, 1)
    x_test = x_test.reshape(x_test.shape[0], imheight, imwidth, 1)
    input_shape = (imheight, imwidth, 1)
    batch_size = 4
    epochs = 1

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam(), metrics=['accuracy'])
    print(model.summary())

    model.fit(x_train, y_train, batch_size=4, epochs=10, verbose=1, validation_data=(x_test, y_test))
    return model

#Extract wave data from recorded audio
def get_wav_data(path):
    input_wav           = path
    spectrogram         = graph_spectrogram( input_wav )
    graygram            = rgb2gray(spectrogram)
    normgram            = normalize_gray(graygram)
    norm_shape          = normgram.shape
    #print("Spec Shape->", norm_shape)
    if(norm_shape[0]>100):
        redgram             = block_reduce(normgram, block_size = (26,26), func = np.mean)
    else:
        redgram             = block_reduce(normgram, block_size = (3,3), func = np.mean)
    redgram             = redgram[0:imheight,0:imwidth]
    red_data            = redgram.reshape(imheight,imwidth, 1)
    empty_data          = np.empty((1,imheight,imwidth,1))
    empty_data[0,:,:,:] = red_data
    new_data            = empty_data
    return new_data

#Save created model
def save_model_to_disk(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

#Load saved model
def load_model_from_disk():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model

#In Loggin mode capture one example of each class. Display it in time and frequency domain.
def generate_log(in_dir, num_samps_per_cat):
    file_names = [f for f in os.listdir(in_dir) if '.wav' in f]
    checklist  = np.zeros(num_samps_per_cat * 10)
    final_list = []
    iternum = 0
    
    #Get a random sample for each category
    while(1):
        print("Iteration Number:", iternum)
        sample_names = random.sample(file_names,10)
        for name in sample_names:
            categ = int(name[0])
            if(checklist[categ]<num_samps_per_cat):
                checklist[categ]+=1
                final_list.append(name)
        if(int(checklist.sum())==(num_samps_per_cat * 10)):
            break 
        iternum+=1
    print(final_list)

    #Generate Images for each sample
    lif = os.path.join(log_image_folder,time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime()))
    if not os.path.exists(lif):
        os.makedirs(lif)
    for name in final_list:      
        #Time Domain Signal
        rate, data = wavfile.read(os.path.join(in_dir,name))
        if(LOG_MODE==1):   
            fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
            ax.set_title('Sound of ' +name[0] + ' - Sampled audio signal in time')
            ax.set_xlabel('Sample number')
            ax.set_ylabel('Amplitude')
            ax.plot(data)
            fig.savefig(os.path.join(lif, name[0:5]+'.png'))   # save the figure to file
            plt.close(fig)
    
        #Frequency Domain Signals
        if(LOG_MODE==2):
            fig,ax = plt.subplots(1)
            #fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
            #ax.axis('off')
            pxx, freqs, bins, im = ax.specgram(x=data, Fs=rate, noverlap=511, NFFT=512)
            #ax.axis('off')
            #plt.rcParams['figure.figsize'] = [0.75,0.5]
            cbar = fig.colorbar(im)
            cbar.set_label('Intensity dB')
            #ax.axis("tight")

            # Prettify
            ax.set_title('Spectrogram of spoken ' +name[0] )
            ax.set_xlabel('time')
            ax.set_ylabel('frequency Hz')
            fig.savefig(os.path.join(lif, name[0]+'_spec.png'), dpi=300, frameon='false')
            plt.close(fig)




if __name__ == '__main__':
    if(not LOG_MODE):
        while(1):
            time.sleep(SLEEP_TIME)
            if(os.path.isfile('model.json')):
                print("please speak a word into the microphone")
                success = record_to_file(test_rec_folder)
                if(not success):
                    print(" Speak Again Clearly")
                    continue
            else:
                print("********************\n\nTraining The Model\n")
            if(os.path.isfile('model.json')):
                model = load_model_from_disk()
            else:
                model = create_model(recording_directory)
                save_model_to_disk(model)
            #fname = 'r4.wav'
            #new_data = get_wav_data(fname)
            for i in range(num_test_files):
            #for i in range(1):
                fname = str(i)+".wav"
                new_data    = get_wav_data(os.path.join(test_rec_folder,fname))    
                predictions = np.array(model.predict(new_data))
                maxpred = predictions.argmax()
                normpred = normalize_gray(predictions)*100
                predarr = np.array(predictions[0])
                sumx = predarr.sum()
                print("TestFile Name: ", fname, " The Model Predicts:", maxpred)
                for nc in range(num_classes):
                    confidence = np.round(100*(predarr[nc]/sumx))
                    print("Class ", nc, " Confidence: ", confidence)
                #print("TestFile Name: ",fname, " Values:", predictions)
                print("_____________________________\n")
    else:
        generate_log(recording_directory,6)












