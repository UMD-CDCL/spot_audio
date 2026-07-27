import socket
import wave
import sys

# --- Configuration matching the C++ ALSA Sender ---
UDP_IP = "0.0.0.0"
UDP_PORT = 21885
SAMPLE_RATE = 48000
CHANNELS = 1
SAMPWIDTH = 2  # 16-bit audio (S16_LE) is 2 bytes per sample

OUTPUT_FILENAME = "spot_mic_test.wav"

def main():
    # --- Setup UDP Socket ---
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    print(f"Listening for UDP audio on port {UDP_PORT}...")
    print(f"Audio format: {SAMPLE_RATE} Hz, Mono, 16-bit")
    print(f"Saving recording to: {OUTPUT_FILENAME}")
    print("Press Ctrl+C to stop recording and save the file.\n")

    # --- Setup WAV File ---
    try:
        # Using 'with' ensures the file header is finalized cleanly when we exit
        with wave.open(OUTPUT_FILENAME, 'wb') as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(SAMPWIDTH)
            wav_file.setframerate(SAMPLE_RATE)

            packets_received = 0
            bytes_received = 0

            while True:
                # 2048 buffer is plenty for the 1024-byte payloads
                data, addr = sock.recvfrom(2048)
                
                # Write raw PCM bytes directly to the file
                wav_file.writeframesraw(data)

                packets_received += 1
                bytes_received += len(data)

                # Print status every ~93 packets (roughly 1 second of audio at 512 frames/packet)
                if packets_received % 93 == 0:
                    seconds = bytes_received / (SAMPLE_RATE * SAMPWIDTH)
                    print(f"Received {packets_received} packets... ({seconds:.1f} seconds of audio recorded)", end='\r')

    except KeyboardInterrupt:
        print("\n\nStopped recording.")
        print(f"Total packets received: {packets_received}")
        print(f"Total file size: {bytes_received / 1024.0:.1f} KB")
        print(f"File saved successfully to {OUTPUT_FILENAME}.")
        sys.exit(0)

if __name__ == "__main__":
    main()
