import socket
import threading
import queue
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
        self.net_thread = None
        self.process_thread = None
        self.running = False
        
        # --- OPTIMIZATION 1: Thread Decoupling ---
        # Queue acts as a shock absorber between the fast network and the slower callback
        # maxsize=200 equates to ~2 seconds of audio buffer to prevent runaway memory leaks
        self.audio_queue = queue.Queue(maxsize=200) 

    def create_socket(self):
        # Initialize UDP socket for listening
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # --- OPTIMIZATION 2: The Socket Buffer Trap ---
        # Request a 4MB receive buffer from the Linux kernel
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4194304)
        except Exception as e:
            self.logger.warning(f"Failed to increase OS socket buffer: {e}")

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
            
            # Start the Network Vacuum Thread
            self.net_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.net_thread.start()
            
            # Start the Audio Processing Thread
            self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
            self.process_thread.start()
            
            self.logger.info("UDP audio receiver & processing threads started.")

    def stop_stream(self):
        self.running = False
        if self.net_thread is not None:
            self.net_thread.join(timeout=2.0)
            self.net_thread = None
            
        if self.process_thread is not None:
            self.process_thread.join(timeout=2.0)
            self.process_thread = None
            
        self.destroy_socket()
        
        # Flush the queue to prevent memory lingering
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
                
        self.logger.info("UDP audio receiver stopped.")

    def __del__(self):
        self.stop_stream()

    def _receive_loop(self):
        # Network Thread: ONLY reads from socket and pushes to queue. No processing!
        bufsize = 4096 
        
        while self.running:
            try:
                data, _ = self.sock.recvfrom(bufsize)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    self.logger.error(f"UDP recv error: {e}")
                break

            if not data:
                continue

            # Push raw bytes directly to the queue
            try:
                self.audio_queue.put_nowait(data)
            except queue.Full:
                self.logger.warning("Audio queue full! OS dropping audio packets.")

        self.logger.info("Network receiver loop exited.")

    def _process_loop(self):
        # Worker Thread: Pulls from queue, decodes NumPy, and triggers the callback
        while self.running:
            try:
                # 0.5s timeout allows the thread to check self.running frequently
                data = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Convert to numpy int16 array
            samples = np.frombuffer(data, dtype=np.int16)

            # Fire the callback safely outside the network thread
            try:
                for chan in self.channels:
                    self.on_audio(samples, chan)
            except Exception as e:
                self.logger.error(f"on_audio callback error: {e}")

        self.logger.info("Audio processing loop exited.")