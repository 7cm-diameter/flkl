from typing import Optional

import numpy as np
from amas.agent import Agent, NotWorkingError
from numpy.random import uniform
from utex.agent import AgentAddress
from utex.audio import Speaker, WhiteNoise
from utex.scheduler import (SessionMarker, TrialIterator, blockwise_shuffle2,
                            mix, mixn, repeat)

from flkl.share import (Flkl, as_millis, fixed_time_with_error,
                        flush_message_for, show_progress)

expvars = dict()

async def conditional_discrimination(agent: Agent, ino: Flkl, expvars: dict):
    from itertools import product

    led_pin = expvars.get("led-pin", 3)
    sound_pin = expvars.get("sound-pin", 2)
    speaker_id = expvars.get("speaker-id", 1)
    reward_pin = expvars.get("reward-pin", [4, 5])
    response_pin = expvars.get("response-pin", [6, 7])

    noise = WhiteNoise()
    speaker = Speaker(speaker_id)

    reward_duration = expvars.get("reward-duration", 0.02)
    reward_duration_millis = as_millis(reward_duration)

    stimulus_duration = expvars.get("stimulus-duration", 2.)
    stimulus_duration_millis = as_millis(stimulus_duration)

    decision_duration = expvars.get("decision-duration", 0.5)
    decision_duration_millis = as_millis(decision_duration)

    left_ratio = expvars.get("left-ratio", 1)
    right_ratio = expvars.get("right-ratio", 1)

    led_flick_hz = expvars.get("led-flick-hz", [3, 4, 5, 6, 7, 8])
    sound_flick_hz = expvars.get("sound-flick-hz", [2, 4.5, 6.5, 20])
    boundary = expvars.get("lr-boundary", 5.5)
    left_signals = list(filter(lambda hz: hz < boundary, led_flick_hz))
    right_signals = list(filter(lambda hz: hz > boundary, led_flick_hz))

    audvis_flickr = list(product(led_flick_hz, sound_flick_hz))
    audio_flickr = list(product([0], sound_flick_hz))
    visual_flickr = list(product(led_flick_hz, [0]))

    flickrs = mixn([visual_flickr, audio_flickr, audvis_flickr],
                   [
                       expvars.get("visual-ratio", 1),
                       expvars.get("audio-ratio", 1),
                       expvars.get("audvis-ratio", 1),
                   ])

    trialtypes = mix([0], [1],
                     len(flickrs) * expvars.get("train-ratio", 1),
                     len(flickrs) * expvars.get("test-ratio", 1))

    flickrs *= (expvars.get("train-ratio", 1) + expvars.get("test-ratio", 1))

    maximum_trial = 600
    number_of_reward = expvars.get("number-of-reward", 200)
    iti_mean = expvars.get("ITI", 5.)
    iti_range = expvars.get("ITI-range", 2.)

    flickr_per_trial, type_per_trial = blockwise_shuffle2(
        repeat(flickrs, maximum_trial // len(flickrs) + 1),
        repeat(trialtypes, maximum_trial // len(flickrs) + 1),
        len(flickrs)
    )

    trials = TrialIterator(flickr_per_trial, type_per_trial)

    try:
        speaker.play(noise, blocking=False, loop=True)
        while agent.working() and number_of_reward > 0:
            for i, (vhz, ahz), trialtype in trials:
                iti = uniform(iti_mean - iti_range, iti_mean + iti_range)
                print(f"Trial: {i} Visual: {vhz} Audio: {ahz} Trialtype: {trialtype}")
                await flush_message_for(agent, iti)
                if vhz == 0:
                    correct_idx = 0 if uniform() >= 0.5 else 1
                    ino.flick_for(sound_pin, ahz, stimulus_duration_millis)
                elif ahz == 0:
                    correct_idx = 0 if vhz > boundary else 1
                    ino.flick_for(led_pin, vhz, stimulus_duration_millis)
                else:
                    correct_idx = 0 if vhz > boundary else 1
                    ino.flick_for2(led_pin, sound_pin, vhz, ahz, stimulus_duration_millis)
                correct = await fixed_time_with_error(agent, response_pin[correct_idx], stimulus_duration, decision_duration)
                if trialtype or correct:
                     ino.high_for(reward_pin[correct_idx], reward_duration_millis)
                     number_of_reward -= 1
                await agent.sleep(reward_duration)
                if number_of_reward <= 0:
                    break
            speaker.stop()
            agent.send_to(AgentAddress.OBSERVER.value, SessionMarker.NEND)
            agent.finish()
    except NotWorkingError:
        speaker.stop()
        pass


if __name__ == "__main__":
    from os import mkdir
    from os.path import exists, join

    from amas.agent import Agent
    from amas.connection import Register
    from amas.env import Environment
    from pyno.com import check_connected_board_info
    from pyno.ino import (ArduinoConnecter, ArduinoLineReader, ArduinoSetting,
                          Mode, PinMode)
    from utex.agent import Observer, Recorder, self_terminate
    from utex.clap import PinoClap
    from utex.fs import get_current_file_abspath, namefile
    from utex.scheduler import SessionMarker

    from flkl.share import read

    config = PinoClap().config()
    com_input_config: Optional[dict] = config.comport.get("input")
    com_output_config: Optional[dict] = config.comport.get("output")
    if com_input_config is None or com_output_config is None:
        raise ValueError("`com_input_config` and `com_output_config` are not defined.")
    com_input_config.update({"mode": Mode.readeruno})
    com_output_config.update({"mode": Mode.user})

    if com_input_config is None and com_output_config is None:
        raise Exception()

    available_boards = check_connected_board_info()
    for board in available_boards:
        setting = ArduinoSetting.derive_from_portinfo(board)
        if board.serial_number == com_input_config.get("serial-number"):
            setting.apply_setting(com_input_config)
            serial_number = com_input_config.get("serial-number")
            print(f"Uploading sketch to reader arduino {serial_number}")
        elif board.serial_number == com_output_config.get("serial-number"):
            setting.apply_setting(com_output_config)
            serial_number = com_output_config.get("serial-number")
            print(f"Uploading sketch to controller arduino {serial_number}")
        ArduinoConnecter(setting).write_sketch()

    available_boards = check_connected_board_info()
    reader_ino, flkl = None, None
    for board in available_boards:
        setting = ArduinoSetting.derive_from_portinfo(board)
        if board.serial_number == com_input_config.get("serial-number"):
            setting.apply_setting(com_input_config)
            reader_ino = ArduinoLineReader(ArduinoConnecter(setting).connect())
        elif board.serial_number == com_output_config.get("serial-number"):
            setting.apply_setting(com_output_config)
            flkl = Flkl(ArduinoConnecter(setting).connect())
            [flkl.pin_mode(i, PinMode.OUTPUT) for i in range(2, 6)]
            [flkl.pin_mode(i, PinMode.INPUT) for i in range(6, 8)]

    if reader_ino is None:
        raise ValueError(
            f"Input arduino (serial number: {com_input_config.get('serial-number')}) is not found."
        )

    if flkl is None:
        raise ValueError(
            f"Output arduino (serial number: {com_output_config.get('serial-number')}) is not found."
        )

    data_dir = join(get_current_file_abspath(__file__), "data")
    if not exists(data_dir):
        mkdir(data_dir)
    config.metadata.update({"condition": "lr-test"})
    filename = join(data_dir, namefile(config.metadata))

    controller = (
        Agent("CONTROLLER")
        .assign_task(conditional_discrimination, ino=flkl, expvars=config.experimental)
        .assign_task(self_terminate)
    )

    reader = (
        Agent(AgentAddress.READER.value)
        .assign_task(read, ino=reader_ino, expvars=config.experimental)
        .assign_task(self_terminate)
    )
    observer = Observer()
    recorder = Recorder(filename=filename, timing=True)

    agents = [controller, recorder, reader, observer]
    register = Register(agents)
    env = Environment(agents)

    try:
        env.run()
    except KeyboardInterrupt:
        observer.send_all(SessionMarker.ABEND)
        observer.finish()
