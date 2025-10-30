import socket
import threading
import numpy as np
import time


class UdpMicrophoneDevice(object):
    # Receives raw PCM16 mono audio packets over UDP and invokes on_audio callback.

    def __init__(self, on_audio, logger, rate=48000, port=21885):
        self.on_audio = on_audio
        self.logger = logger
        self.rate = rate
        self.port = port
        self.ip = "0.0.0.0"          # listen on all interfaces
        self.bitdepth = 16
        self.available_channels = 1  # mono stream from your C++ sender
        self.channels = range(self.available_channels)

        self.sock = None
        self.thread = None
        self.running = False

    def create_socket(self):
        # Initialize UDP socket for listening
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))
        self.sock.settimeout(1.0)
        self.logger.info(f"Listening for UDP audio on {self.ip}:{self.port}")

    def destroy_socket(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def start_stream(self):
        if self.sock is None:
            self.create_socket()
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.thread.start()
            self.logger.info("UDP audio receiver started.")

    def stop_stream(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None
        self.destroy_socket()
        self.logger.info("UDP audio receiver stopped.")

    def __del__(self):
        self.stop_stream()

    def _receive_loop(self):
        # Continuously receive UDP packets and call on_audio() with NumPy arrays.
        # Each packet is raw int16 PCM samples.
        
        bufsize = 4096  # maximum UDP payload
        while self.running:
            try:
                data, _ = self.sock.recvfrom(bufsize)
            except socket.timeout:
                continue
            except Exception as e:
                self.logger.error(f"UDP recv error: {e}")
                break

            if not data:
                continue

            # Convert to numpy int16 array
            samples = np.frombuffer(data, dtype=np.int16)

            # Call the same callback signature as original
            try:
                for chan in self.channels:
                    self.on_audio(samples, chan)
            except Exception as e:
                self.logger.error(f"on_audio callback error: {e}")

        self.logger.info("Receiver loop exited.")
