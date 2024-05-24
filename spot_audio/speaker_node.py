#!/usr/bin/env python3


"""
aplay -L

plughw:CARD=ArrayUAC10,DEV=0
    ReSpeaker 4 Mic Array (UAC1.0), USB Audio
    Hardware device with all software conversions

# somtimes doesn't work?
aplay -D plughw <file> 
aplay -D plughw:0,1
"""


# ros2 service call /play_sound_external cdcl_umd_msgs/srv/PlaySound "{file_name: '/home/cdcl/cdcl_ws/src/cdcl_autonomy_stack/spot_audio/data/introduction.wav'}"
# ros2 service call /play_sound_external cdcl_umd_msgs/srv/PlaySound "{file_name: '/home/cdcl/cdcl_ws/src/cdcl_autonomy_stack/spot_audio/data/assistance_prompt.wav'}"
# ros2 service call /play_sound_external cdcl_umd_msgs/srv/PlaySound "{file_name: '/home/cdcl/cdcl_ws/src/cdcl_autonomy_stack/spot_audio/data/stay_still.wav'}"
# ros2 service call /play_sound_external cdcl_umd_msgs/srv/PlaySound "{file_name: '/home/cdcl/cdcl_ws/src/cdcl_autonomy_stack/spot_audio/data/exit.wav'}"

import rclpy
from rclpy.node import Node
from cdcl_umd_msgs.srv import PlaySound
import subprocess
from os.path import exists
from typing import Optional, List
import re

def find_next_words(s: str, keyword: str, n: int, exclusion: Optional[str] = None) -> List[str]:
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
        trigger = r'\S+\s+'*(n-1) + '\S+'
    if exclusion is not None:
        p = re.compile(r'{0}\s+({1}[^{2}\s+])'.format(keyword, trigger, exclusion))
    else:
        p = re.compile(r'{0}\s+({1})'.format(keyword, trigger))
    return re.findall(p, s)

class SpeakerNode(Node):
    def __init__(self):
        super().__init__('speaker_node')
        self.play_sound_srv_ = self.create_service(PlaySound, 'play_sound_external', self.play_sound_callback)
        self.find_device("USB Audio [USB Audio]")
    
    def find_device(self, name: str):
        """ finds the device named """
        "card 3: Device [USB2.0 Device], device 0: USB Audio [USB Audio]"
        list_all_devices = subprocess.run(["aplay", "-l"], capture_output=True)
        lines = list_all_devices.stdout.splitlines()
        line_with_name = None
        for line in lines:
            if name in line.decode('utf-8') and ('ReSpeaker' not in line.decode('utf-8')):
                line_with_name = line.decode('utf-8')
                break
        card = find_next_words(line_with_name, "card", 1)[0][0]
        device = find_next_words(line_with_name, "device", 1, "]")[0][0]
        self.device_str = "plughw:" + card + "," + device

    def play_sound_callback(self, request, response):
        file_name = request.file_name

        # check that file exists
        if not exists(file_name):
            response.success = False
            response.msg = "File doesn't exist."
            return response
        
        # play the sound
        self.play_sound(request.file_name)

        # return success to user
        response.success = True
        response.msg = "played sound"
        return response
    
    def play_sound(self, file):
        subprocess.call(["aplay", "-D", self.device_str, file])


def main(args=None):
    rclpy.init(args=args)

    publisher = SpeakerNode()

    try:
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        pass

    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
