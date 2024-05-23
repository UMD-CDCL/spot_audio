#!/usr/bin/env python3


"""
aplay -L

plughw:CARD=ArrayUAC10,DEV=0
    ReSpeaker 4 Mic Array (UAC1.0), USB Audio
    Hardware device with all software conversions

aplay -D plughw <file>
"""


# ros2 service call /play_sound cdcl_umd_msgs/srv/PlaySound "{file_name: '/home/cdcl/cdcl_ws/src/cdcl_autonomy_stack/spot_audio/data/introduction.wav'}"
# ros2 service call /play_sound cdcl_umd_msgs/srv/PlaySound "{file_name: '/home/cdcl/cdcl_ws/src/cdcl_autonomy_stack/spot_audio/data/assistance_prompt.wav'}"
# ros2 service call /play_sound cdcl_umd_msgs/srv/PlaySound "{file_name: '/home/cdcl/cdcl_ws/src/cdcl_autonomy_stack/spot_audio/data/stay_still.wav'}"
# ros2 service call /play_sound cdcl_umd_msgs/srv/PlaySound "{file_name: '/home/cdcl/cdcl_ws/src/cdcl_autonomy_stack/spot_audio/data/exit.wav'}"

import rclpy
from rclpy.node import Node
from cdcl_umd_msgs.srv import PlaySound
import subprocess
from os.path import exists

class SpeakerNode(Node):
    def __init__(self):
        super().__init__('speaker_node')
        self.play_sound_srv_ = self.create_service(PlaySound, 'play_sound', self.play_sound_callback)

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
        subprocess.call(["aplay", "-D", "plughw", file])


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
