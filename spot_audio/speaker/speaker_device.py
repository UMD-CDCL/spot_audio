# needed to define the play_sound function in the interface "SpeakerDevice"
from abc import abstractmethod

# needed to make calls to aplay
import subprocess

# needed for giving the user feedback
import warnings

# needed for finding / checking the file system
import os

# needed for error handling when doing i/o operations
from typing import Optional, List

# needed for processing the output of calls to programs called by subprocess
import re


class SpeakerDevice:
    @abstractmethod
    def play_sound(self, path_to_file: str) -> None:
        pass


class USBSpeakerDevice(SpeakerDevice):
    def __init__(self):
        """
        constructor for USB speaker devices
        """
        self.device_name = self._get_device_name("USB Audio [USB Audio]")

    def _find_next_words(self, s: str, keyword: str, n: int, exclusion: Optional[str] = None) -> List[str]:
        """
        finds the first n words after the first instance of a given keyword
        :param s: input string
        :param word: given keyword
        :param n: number of words to find following the given keyword
        :param exclusion: list of characters to optionally exclude
        :return: an empty string if there was no keyword or else the word directly after the first instance of the keyword

        Examples:
        >>> find_next_words('\tchannel 6 (2437 MHz), width: 20 MHz, center1: 2437 MHz\n', 'channel', 1)
            ['6']
        >>> find_next_words('\tchannel 6 (2437 MHz), width: 20 MHz, center1: 2437 MHz\n', 'width:', 2, '.,')
            ['20 MHz']
        """
        # Add on as many words following the keyword as requested by user
        if n == 1:
            trigger = '\S+'
        else:
            trigger = '\S+\s+'*(n-1) + '\S+'
        if exclusion is not None:
            p = re.compile(r'{0}\s+({1}[^{2}\s+])'.format(keyword, trigger, exclusion))
        else:
            p = re.compile(r'{0}\s+({1})'.format(keyword, trigger))
        return re.findall(p, s)

    def _get_device_name(self, name: str) -> str:
        """
        determines the name of the device using "aplay -l"
        :return:
        """
        # list all the audio devices
        list_all_devices = subprocess.run(["aplay", "-l"], capture_output=True)

        # get each indivudal line of the response
        lines = list_all_devices.stdout.splitlines()
        line_with_name = None
        for line in lines:
            if name in line.decode('utf-8') and ('ReSpeaker' not in line.decode('utf-8')):
                line_with_name = line.decode('utf-8')
                break
        card = self._find_next_words(line_with_name, "card", 1)[0][0]
        device = self._find_next_words(line_with_name, "device", 1, "]")[0][0]
        return "plughw:" + card + "," + device

    def play_sound(self, path_to_file: str) -> None:
        """
        plays a .wav file sound by making a system call to `paplay`
        :param path_to_file: path to the .wav file
        :return: null
        """
        # if we have a valid device and the file name is valid, then try to play the sound
        if self.device_name and os.path.exists(path_to_file) and os.path.isfile(path_to_file):
            subprocess.call(['aplay', '-D', self.device_name, path_to_file])


class JackSpeakerDevice(SpeakerDevice):
    def __init__(self, keyword: str):
        """
        finds a Jack Speaker device
        :param keyword:
        """
        self.keyword = keyword
        self.device_name = self._get_device_name()
        if not self.device_name:
            warnings.warn("Could not find device!")

    def _get_device_name(self) -> Optional[str]:
        """
        determines the speakers name by using `pactl list`
        :return: either None, if the speaker wasn't found or else the name of the device
        """
        # pactl list sinks short | awk -F '\t' '{print $1,$2,$5}'
        list_devices = subprocess.run(['pactl', 'list', 'sinks', 'short'], capture_output=True)
        lines = list_devices.stdout.splitlines()
        for line in lines:
            decoded_line = line.decode('utf-8')
            if self.keyword in decoded_line:
                return decoded_line.split('\t')[1]
        return None

    def set_device(self, device_name: str) -> None:
        """
        sets the name of the device directly
        :param device_name:  the name of the device as determined by pactl list sinks short | awk -F '\t' '{print $1,$2,$5}'
        :return: nothing
        """
        self.device_name = device_name

    def play_sound(self, path_to_file: str) -> None:
        """
        plays a .wav file sound by making a system call to `paplay`
        :param path_to_file: path to the .wav file
        :return: null
        """
        # if we have a valid device and the file name is valid, then try to play the sound
        if self.device_name and os.path.exists(path_to_file) and os.path.isfile(path_to_file):
            # paplay --d="alsa_output.pci-0000_00_03.0.hdmi-stereo" filename.ogg
            os.system('paplay --d="' + self.device_name +  '" ' + path_to_file)
