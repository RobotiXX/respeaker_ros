from .util import ignore_stderr
import pyaudio
import numpy as np
import math
import time


def capture_stamp_ns(time_info, host_ns):
    """Estimate first-sample UTC by mapping PortAudio's ADC/current clock pair.

    This is a host-clock estimate, not hardware PTP synchronization. Fall back
    to callback receive time when the audio backend supplies no valid times.
    """
    adc = time_info.get('input_buffer_adc_time', 0)
    current = time_info.get('current_time', 0)
    if adc > 0 and current > 0 and math.isfinite(adc) and math.isfinite(current):
        return host_ns + round((adc - current) * 1e9)
    return host_ns


class RespeakerAudio(object):
    def __init__(self, on_audio, channels=None, suppress_error=True, timestamped_callback=False):
        self.on_audio = on_audio
        self.timestamped_callback = timestamped_callback
        with ignore_stderr(enable=suppress_error):
            self.pyaudio = pyaudio.PyAudio()
        self.available_channels = None
        self.channels = channels
        self.device_index = None
        self.rate = 16000
        self.bitwidth = 2
        self.bitdepth = 16

        # find device
        count = self.pyaudio.get_device_count()
        print("%d audio devices found" % count)
        for i in range(count):
            info = self.pyaudio.get_device_info_by_index(i)
            name = info["name"]
            chan = info["maxInputChannels"]
            print(" - %d: %s" % (i, name))
            if name.lower().find("respeaker") >= 0:
                self.available_channels = chan
                self.device_index = i
                print("Found %d: %s (channels: %d)" % (i, name, chan))
                break
        if self.device_index is None:
            print("Failed to find respeaker device by name. Using default input")
            info = self.pyaudio.get_default_input_device_info()
            self.available_channels = info["maxInputChannels"]
            self.device_index = info["index"]

        if self.available_channels != 6:
            print("%d channel is found for respeaker" % self.available_channels)
            print("You may have to update firmware.")
        if self.channels is None:
            self.channels = range(self.available_channels)
        else:
            self.channels = filter(lambda c: 0 <= c < self.available_channels, self.channels)
        if not self.channels:
            raise RuntimeError('Invalid channels %s. (Available channels are %s)' % (
                self.channels, self.available_channels))
        print('Using channels %s' % self.channels)

        self.stream = self.pyaudio.open(
            input=True, start=False,
            format=pyaudio.paInt16,
            channels=self.available_channels,
            rate=self.rate,
            frames_per_buffer=1024,
            stream_callback=self.stream_callback,
            input_device_index=self.device_index,
        )

    def __del__(self):
        self.stop()
        try:
            self.stream.close()
        except:
            pass
        finally:
            self.stream = None
        try:
            self.pyaudio.terminate()
        except:
            pass

    def stream_callback(self, in_data, frame_count, time_info, status):
        stamp_ns = capture_stamp_ns(time_info, time.time_ns())
        # split channel
        data = np.fromstring(in_data, dtype=np.int16)
        chunk_per_channel = len(data) // self.available_channels
        data = np.reshape(data, (chunk_per_channel, self.available_channels))
        for chan in self.channels:
            chan_data = data[:, chan]
            # invoke callback
            if self.timestamped_callback:
                self.on_audio(chan_data, chan, stamp_ns)
            else:
                self.on_audio(chan_data, chan)
        return None, pyaudio.paContinue

    def start(self):
        if self.stream.is_stopped():
            self.stream.start_stream()

    def stop(self):
        if self.stream.is_active():
            self.stream.stop_stream()
