import pyaudio
import wave
import scipy.io.wavfile as wavfile
import numpy as np
import matplotlib.pyplot as plt
import struct

SAMPLE_RATE = 44100  # default audio sample rate

class SONAR:
    ''' detect hand positions through SONAR '''
    def __init__(self, samp = SAMPLE_RATE):
        # audio parameters setup
        self.fs = samp  # audio sample rate
        self.chunk = 1024
        self.p = pyaudio.PyAudio()
        self.num_channels = 1  # use mono output for now
        self.format = pyaudio.paInt16

        # stream for signal output
        # 'output = True' indicates that the sound will be played rather than recorded
        self.output_stream = self.p.open(format = self.format,
                                frames_per_buffer = self.chunk,
                                channels = self.num_channels,
                                rate = self.fs,
                                output = True)
        # stream for receiving signals
        self.input_stream = self.p.open(format = self.format,
                                channels = self.num_channels,
                                rate = self.fs,
                                frames_per_buffer = self.chunk,
                                input = True)
                                

    # play a tone at frequency freq for a given duration
    def play_freq(self, freq, duration = 1):
        pass


    def play(self, filename):
        # Open the sound file 
        wf = wave.open(filename, 'rb')

        if wf.getnchannels() != self.num_channels:
            raise Exception("Unsupported number of audio channels")

        # Read data in chunks
        data = wf.readframes(self.chunk)

        # Play the sound by writing the audio data to the stream
        while data != b'':
            self.output_stream.write(data)
            data = wf.readframes(self.chunk)

    # record audio input and write to filename
    def record(self, filename):
        seconds = 1
        print('Recording')

        frames = []  # Initialize array to store frames

        # Store data in chunks for 3 seconds
        for i in range(0, int(self.fs / self.chunk * seconds)):
            data = self.input_stream.read(self.chunk)
            data_int = np.array(struct.unpack(str(self.chunk*2) + 'B', data), dtype='b')[::2]
            f_vec = self.fs*np.arange(self.chunk/2)/self.chunk 
            fft_data = (np.abs(np.fft.fft(data_int))[0:int(np.floor(self.chunk/2))])/self.chunk
            fft_data[1:] = 2*fft_data[1:]
            plt.plot(f_vec,fft_data)
            frames.append(data)

        plt.show()
        # Stop the stream
        self.input_stream.stop_stream()

        print('Finished recording')

        # Save the recorded data as a WAV file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.num_channels)
        wf.setsampwidth(self.p.get_sample_size(self.format))
        wf.setframerate(self.fs)
        wf.writeframes(b''.join(frames))
        wf.close()

    # close all streams and terminate PortAudio interface
    def destruct(self):
        self.output_stream.close()
        self.input_stream.close()
        self.p.terminate()

if __name__ == "__main__":
    s = SONAR()
    s.record("record.wav")
    s.destruct()

