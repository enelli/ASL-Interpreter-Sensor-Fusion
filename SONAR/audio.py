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

        # FMCW params
        self.last_freq = 0  # last frequency broadcasted
        self.slope = 0  # FMCW slope
        self.bandwidth = 0  # sweep range
        self.low_ind = 0  # filter out frequencies below f_vec[low_ind]
        self.high_ind = 0  # filter out frequencies above f_vec[high_ind]

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

    def chirp(self, low_freq, high_freq, duration = 2):
        ''' broadcast an FMCW chirp ranging from low_freq
        to high_freq, spanning duration seconds 
        freq at time t is given by (low_freq * (duration - t) + high_freq * t) / duration
        signal is given by sin (2 pi * freq * t)'''
        cur_frame = 0
        while cur_frame < duration * self.fs and not self.terminate:
            # number of frames to produce on this iteration
            num_frames = self.output_stream.get_write_available()
            if num_frames:
                times = np.arange(cur_frame, cur_frame + num_frames) / self.fs
                freq = (low_freq * (duration - times) + high_freq * times) / duration
                arg = np.pi * 2 * np.multiply(freq, times)
                signal = self.amp * np.sin(arg)
                # necessary data type conversions (output is static otherwise)
                signal = signal.astype(np.float32)
                self.output_stream.write(signal.tobytes())
                cur_frame += num_frames
                self.last_freq = freq[-1]  # store most recent frequency as the current freq
    
    def init_fmcw(self, low_freq, high_freq, duration):
        ''' initialize slope and last_freq for fmcw'''
        self.bandwidth = high_freq - low_freq
        self.slope = self.bandwidth / duration
        self.last_freq = low_freq
        # filters for FMCW frequencies
        self.low_ind = int(low_freq * self.chunk / self.fs)
        self.high_ind = int(high_freq * self.chunk / self.fs)

    def transmit(self, low_freq, high_freq, duration = 2):
        ''' continuously broadcast FMCW chirps from low_freq
        to high_freq spanning duration seconds'''
        while not self.terminate:
            self.chirp(low_freq, high_freq, duration)

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
        def time_diff(freq):
            # given a frequency, determine time offset relative to
            # current broadcast frequency
            if (freq > self.last_freq): freq -= self.bandwidth
            # ensure self.last_freq > freq always
            #print(self.last_freq, freq)
            return (self.last_freq - freq) / self.slope
        frames = []
        prev_window = np.zeros(self.high_ind - self.low_ind)
        total_data = np.zeros(self.high_ind - self.low_ind)
        while not self.terminate:  # continuously read until termination
            num_frames = self.input_stream.get_read_available()
            input_signal = np.fromstring(self.input_stream.read(num_frames), dtype=np.float32)
            frames = np.concatenate((frames, input_signal))
            if len(frames) >= self.chunk:  # wait until we have a full chunk before processing
                # fft_data[f] is now the amplitude? of the fth frequency
                # pass just the FMCW frequencies
                fft_data = np.abs(np.fft.rfft(frames[:self.chunk]))[self.low_ind:self.high_ind]
                # extract 4 largest changed frequencies
                max_inds = (fft_data - prev_window).argsort()[-4:]
                freq = [self.f_vec[i] for i in max_inds]
                delays = [time_diff(f) for f in freq]
                #print(delays)
                plt.plot(self.f_vec[self.low_ind:self.high_ind],fft_data)
                total_data += fft_data - prev_window  # show only the total change
                prev_window = fft_data
                frames = frames[self.chunk:]
        plt.plot(self.f_vec[self.low_ind:self.high_ind],total_data)

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

    def match(self, signal):
        ''' given an input signal of frequencies, match them
        to self.fmcw_sweep and return the index of the most similar
        segment '''
        pass

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
    s.chirp(220, 880, 5)
    #s.play_freq(440, 1)
    #s.receive()
    #s.subtract_window()
    s.destruct()
    
