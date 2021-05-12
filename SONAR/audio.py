import pyaudio
import wave
import numpy as np
import struct
import matplotlib.pyplot as plt
import time
from math import log

SAMPLE_RATE = 44100  # default audio sample rate
# dimensions of the threshold array to feed into visual ML
WIDTH = 300
HEIGHT = 300

class SONAR:
    ''' detect hand positions through SONAR '''
    def __init__(self, samp = SAMPLE_RATE):
        # audio parameters setup
        self.fs = samp  # audio sample rate
        self.chunk = 1024
        self.p = pyaudio.PyAudio()
        self.num_channels = 1  # use mono output for now
        self.format = pyaudio.paFloat32

        # stream for signal output
        # 'output = True' indicates that the sound will be played rather than recorded
        # I have absolutely no idea what device_index is for but it prevents segfaults
        self.output_stream = self.p.open(format = self.format,
                                frames_per_buffer = self.chunk,
                                channels = self.num_channels,
                                rate = self.fs,
                                output = True,
                                output_device_index = None)
        # stream for receiving signals
        self.input_stream = self.p.open(format = self.format,
                                channels = self.num_channels,
                                rate = self.fs,
                                frames_per_buffer = self.chunk,
                                input = True,
                                input_device_index = None)

        # allow other threads to abort this one
        self.terminate = False

        # fft frequency window
        self.f_vec = self.fs * np.arange(self.chunk/2)/self.chunk 

        self.amp = 0.8  # amplitude for signal sending

    # allow camera thread to terminate audio threads
    def abort(self):
        self.terminate = True
                                

    # play a tone at frequency freq for a given duration
    def play_freq(self, freq, duration = 1):
        cur_frame = 0
        # signal: sin (2 pi * freq * time)
        while cur_frame < duration * self.fs and not self.terminate:
            # number of frames to produce on this iteration
            num_frames = self.output_stream.get_write_available()
            times = np.arange(cur_frame, cur_frame + num_frames) / self.fs
            times = times * 2 * np.pi * freq
            # account for amplitude adjustments
            signal = self.amp * np.sin(times)
            signal = signal.astype(np.float32)
            self.output_stream.write(signal.tobytes())
            cur_frame += num_frames


    def play(self, filename):
        # Open the sound file 
        wf = wave.open(filename, 'rb')

        if wf.getnchannels() != self.num_channels:
            raise Exception("Unsupported number of audio channels")

        # Read data in chunks
        data = wf.readframes(self.chunk)

        # Play the sound by writing the audio data to the stream
        # check for abort condition
        while data != b'' and not self.terminate:
            self.output_stream.write(data)
            data = wf.readframes(self.chunk)

        wf.close()

    # receive and process audio input
    def receive(self):
        frames = []
        num = 50
        total_data = np.zeros(num - 1)
        while not self.terminate:  # continuously read until termination
            num_frames = self.input_stream.get_read_available()
            input_signal = np.fromstring(self.input_stream.read(num_frames), dtype=np.float32)
            frames = np.concatenate((frames, input_signal))
            if len(frames) >= self.chunk:  # wait until we have a full chunk before processing
                # fft_data[f] is now the amplitude? of the fth frequency
                fft_data = np.abs(np.fft.rfft(frames[:self.chunk]))
                #max_val = max(fft_data[2:])
                #max_ind = np.where(fft_data == max_val)[0][0] 
                #freq = self.f_vec[max_ind]
                total_data += fft_data[1:num]
                frames = frames[self.chunk:]
        plt.plot(self.f_vec[1:num],total_data)

    # record audio input and write to filename
    def record(self, filename):
        seconds = 1
        print('Recording')

        frames = []  # Initialize array to store frames

        # Store data in chunks for 3 seconds
        for i in range(0, int(self.fs / self.chunk * seconds)):
            if self.terminate: break
            data = self.input_stream.read(self.chunk)

            # Visualize: 
            # https://makersportal.com/blog/2018/9/17/audio-processing-in-python-part-ii-exploring-windowing-sound-pressure-levels-and-a-weighting-using-an-iphone-x
            data_int = np.array(struct.unpack(str(self.chunk*2) + 'B', data), dtype='b')[::2]
            fft_data = (np.abs(np.fft.fft(data_int))[0:int(np.floor(self.chunk/2))])/self.chunk
            fft_data[1:] = 2*fft_data[1:]
            plt.plot(self.f_vec,fft_data)
            
            frames.append(data)

        # Stop the stream
        self.input_stream.stop_stream()

        print('Finished recording')
        plt.show()

        # Save the recorded data as a WAV file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.num_channels)
        wf.setsampwidth(self.p.get_sample_size(self.format))
        wf.setframerate(self.fs)
        wf.writeframes(b''.join(frames))
        wf.close()


    # Records two windows and subtracts them from each other
    def subtract_window(self):
        data = [self.input_stream.read(self.chunk) for _ in range(2)]
        data_int = [np.array(struct.unpack(str(self.chunk*2) + 'B', data[i]), dtype='b')[::2] for i in range(2)]
        fft_data = [(np.abs(np.fft.fft(data))[0:int(np.floor(self.chunk/2))])/self.chunk for data in data_int]
        plt.plot(self.f_vec, fft_data[0])
        plt.plot(self.f_vec, fft_data[1])
        fft_subtract = np.subtract(fft_data[1], fft_data[0])
        fft_subtract[1:] = 2*fft_subtract[1:]
        plt.plot(self.f_vec, fft_subtract)
        plt.show()

    def find_hand(self):
        ''' return a WIDTH x HEIGHT binary determination of 0s and 255s
        representing where the hand is, with (0,0) representing the top 
        left of the screen'''
        return np.zeros((WIDTH, HEIGHT), dtype=np.uint8)
        

    # close all streams and terminate PortAudio interface
    def destruct(self):
        self.output_stream.close()
        self.input_stream.close()
        self.p.terminate()

if __name__ == "__main__":
    s = SONAR()
    s.play_freq(440, 1)
    s.receive()
    #s.subtract_window()
    s.destruct()
    
