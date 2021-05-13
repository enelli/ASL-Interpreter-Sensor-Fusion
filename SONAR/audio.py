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
BUFFER_SIZE = 2048
SOUND_SPEED = 343
THRESH = 1  # FFT threshold to filter out noise
ENABLE_DRAW = False  # whether to plot data

class SONAR:
    ''' detect hand positions through SONAR '''
    def __init__(self, samp = SAMPLE_RATE):
        # audio parameters setup
        self.fs = samp  # audio sample rate
        self.chunk = BUFFER_SIZE
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

        # fft frequency window, will be clipped to readable frequencies
        self.f_vec = self.fs * np.arange(self.chunk)/self.chunk 
        # set indices on frequency range
        self.low_ind = 0
        self.high_ind = 0

        self.amp = 0.8  # amplitude for signal sending

        self.movement_detected = False  # whether there is current motion

    # allow camera thread to terminate audio threads
    def abort(self):
        self.terminate = True
                                
    def set_freq_range(self, low_freq, high_freq):
        self.low_ind = int(low_freq * self.chunk / self.fs)
        self.high_ind = int(high_freq * self.chunk / self.fs)
        self.f_vec = self.f_vec[self.low_ind:self.high_ind]

    # continuously play a tone at frequency freq
    def play_freq(self, freq):
        cur_frame = 0
        # signal: sin (2 pi * freq * time)
        while not self.terminate:
            # number of frames to produce on this iteration
            num_frames = self.output_stream.get_write_available()
            times = np.arange(cur_frame, cur_frame + num_frames) / self.fs
            arg = times * 2 * np.pi * freq
            # account for amplitude adjustments
            signal = self.amp * np.sin(arg)
            signal = signal.astype(np.float32)
            # log start of signal transmit as accurately as possible
            self.output_stream.write(signal.tobytes())
            cur_frame += num_frames

    # periodically transmit a constant frequency signal every interval seconds
    def transmit(self, freq, interval):
        #signal_length = 0.01  # still detectable
        signal_length = 2
        while not self.terminate:
            self.play_freq(freq, signal_length)
            time.sleep(interval - signal_length)

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

    # detect time it takes for short signal to reach mic
    def receive_burst(self):
        frames = []
        prev_window = np.zeros(self.high_ind - self.low_ind)
        num_moves = 0  # number of consecutive windows with movement
        while not self.terminate:
            num_frames = self.input_stream.get_read_available()
            input_signal = np.frombuffer(self.input_stream.read(num_frames, exception_on_overflow=False), dtype=np.float32)
            if len(input_signal) > 0:
                frames = np.concatenate((frames, input_signal))
            # wait until we have a full chunk before processing; is this a good idea?
            if len(frames) >= self.chunk:  # wait until we have a full chunk before processing
                # fft_data[f] is now the amplitude? of the fth frequency (first two values are garbage)
                fft_data = np.abs(np.fft.rfft(frames[:self.chunk]))[self.low_ind:self.high_ind]
                # filter out low amplitudes
                fft_data = np.where(fft_data < THRESH, 0, fft_data)
                diff = np.abs(fft_data - prev_window)
                diff = np.where(diff < 2 * THRESH, 0, diff)
                # filter out single frequency peaks (these tend to be noise)
                if np.count_nonzero(diff) > 1:
                    num_moves += 1
                    if num_moves > 1: self.movement_detected = True
                elif num_moves > 0:
                    if num_moves > 1:
                        self.movement_detected = False
                        print("Movement ended", num_moves)
                    num_moves = 0
                # assuming near-ultrasound, the extracted frequency should be approximately the transmitted one
                #amp = max(fft_data)
                if ENABLE_DRAW and (len(frames) < 1.5 * self.chunk):  # do not draw every time
                    plt.plot(self.f_vec, diff)
                    plt.draw()
                    plt.pause(1e-6)
                    plt.clf()
                frames = frames[self.chunk:]  # clear frames already read
                prev_window = fft_data

    def is_moving(self):
        return self.movement_detected
            
    # record audio input and write to filename
    def record(self, filename):
        seconds = 10
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

    # close all streams and terminate PortAudio interface
    def destruct(self):
        self.output_stream.close()
        self.input_stream.close()
        self.p.terminate()

if __name__ == "__main__":
    s = SONAR()
    s.set_freq_range(220, 1760)
    #s.transmit(440, 2)
    s.receive_burst()
    s.destruct()
