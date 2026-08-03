#!/usr/bin/env python3
"""
Standalone microphone stream tester -- no ROS required.

Listens for 'AUD2' packets from the arm Pi, writes the PCM to a WAV file, and
reports packet loss. This is the first thing to reach for when the microphone
misbehaves: it tells you whether audio is arriving at all, whether the sample rate
matches, and whether the link is losing packets, without any ROS in the way.

Usage:
    python3 mic_udp_receive_test.py [seconds]

Run with no argument to record until Ctrl+C.
"""

import signal
import socket
import struct
import sys
import time
import wave

UDP_IP = "0.0.0.0"
UDP_PORT = 21885
OUTPUT_FILENAME = "spot_mic_test.wav"

# --- wire protocol -----------------------------------------------------------
# MUST match spot-arm-pi/common/spot_arm_wire.h
AUD2_MAGIC = 0x41554432          # 'AUD2'
AUD2_HEADER_FMT = '>IIQIHHI'     # magic, seq, capture_ns, rate, ch, sample_bytes, frames
AUD2_HEADER_LEN = struct.calcsize(AUD2_HEADER_FMT)
assert AUD2_HEADER_LEN == 28, "AudioHeader must stay in sync with spot_arm_wire.h"

_stop = False


def _handle_sigint(_signum, _frame):
    global _stop
    _stop = True


def main():
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else None

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(1.0)

    limit = f" for {duration:.0f}s" if duration else ""
    print(f"Listening for AUD2 UDP audio on port {UDP_PORT}{limit}...")
    print("Press Ctrl+C to stop and finalise the file.\n")

    packets = 0
    lost = 0
    non_aud2 = 0
    pcm_bytes = 0
    expected_seq = None
    stream_rate = None
    wav_file = None
    started = time.monotonic()
    first_capture_ns = None
    last_capture_ns = None

    try:
        while not _stop:
            if duration is not None and (time.monotonic() - started) >= duration:
                break

            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                if packets == 0:
                    print("  ...no packets yet. Is mic-sensor.service running on the Pi?")
                continue

            if len(data) < AUD2_HEADER_LEN:
                non_aud2 += 1
                continue

            magic, seq, capture_ns, rate, channels, sample_bytes, frames = \
                struct.unpack_from(AUD2_HEADER_FMT, data, 0)

            if magic != AUD2_MAGIC:
                non_aud2 += 1
                if non_aud2 == 1:
                    print(f"  WARNING: packet from {addr[0]} is not AUD2 "
                          f"(magic=0x{magic:08x}). Is the Pi running a build of "
                          f"spot-arm-pi that matches this node?")
                continue

            expected_payload = frames * channels * sample_bytes
            if len(data) < AUD2_HEADER_LEN + expected_payload:
                print(f"  WARNING: truncated packet ({len(data)} bytes for "
                      f"{expected_payload}-byte payload)")
                continue

            if wav_file is None:
                stream_rate = rate
                print(f"Stream: {rate} Hz, {channels} ch, {sample_bytes * 8}-bit, "
                      f"{frames} frames/packet  (from {addr[0]})")
                print(f"Saving to: {OUTPUT_FILENAME}\n")
                wav_file = wave.open(OUTPUT_FILENAME, 'wb')
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(sample_bytes)
                wav_file.setframerate(rate)
                first_capture_ns = capture_ns
            elif rate != stream_rate:
                print(f"  WARNING: sample rate changed mid-stream: "
                      f"{stream_rate} -> {rate} Hz")

            if expected_seq is not None and seq != expected_seq:
                gap = (seq - expected_seq) & 0xFFFFFFFF
                if gap < 0x80000000:
                    lost += gap
                    print(f"  lost {gap} packet(s) at seq {expected_seq}")
                else:
                    print("  sequence reset (sender restarted)")
            expected_seq = (seq + 1) & 0xFFFFFFFF

            pcm = data[AUD2_HEADER_LEN:AUD2_HEADER_LEN + expected_payload]
            wav_file.writeframesraw(pcm)

            packets += 1
            pcm_bytes += len(pcm)
            last_capture_ns = capture_ns

            if packets % 93 == 0:  # ~1 s at 512 frames/packet, 48 kHz
                seconds = pcm_bytes / float(stream_rate * sample_bytes * channels)
                loss_pct = 100.0 * lost / (packets + lost) if (packets + lost) else 0.0
                print(f"  {packets} packets, {seconds:6.1f}s audio, "
                      f"loss {loss_pct:5.2f}%", end='\r')

    finally:
        if wav_file is not None:
            wav_file.close()

        print("\n\n--- summary ---")
        print(f"packets received : {packets}")
        print(f"packets lost     : {lost}")
        if packets + lost:
            print(f"loss rate        : {100.0 * lost / (packets + lost):.2f}%")
        if non_aud2:
            print(f"non-AUD2 packets : {non_aud2}")
        if stream_rate and packets:
            audio_s = pcm_bytes / float(stream_rate * 2)
            print(f"audio recorded   : {audio_s:.1f}s -> {OUTPUT_FILENAME}")
            # Capture-time span vs recorded length is the honest end-to-end check:
            # if these disagree, packets went missing or the clock is wrong.
            if first_capture_ns and last_capture_ns and last_capture_ns > first_capture_ns:
                span_s = (last_capture_ns - first_capture_ns) / 1e9
                print(f"capture span     : {span_s:.1f}s "
                      f"(gap vs audio: {span_s - audio_s:+.2f}s)")
        elif packets == 0:
            print("\nNo audio received. Check, in order:")
            print("  1. ssh into the Pi:  systemctl status mic-sensor.service")
            print("  2.                   journalctl -u mic-sensor.service -n 50")
            print("  3.                   arecord -l        (is the NTG enumerated?)")
            print("  4. On this host:     ss -lun | grep 21885")

        try:
            sock.close()
        except OSError:
            pass


if __name__ == "__main__":
    main()
