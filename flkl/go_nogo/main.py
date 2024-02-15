from itertools import product
from typing import Optional

import numpy as np
from amas.agent import Agent, NotWorkingError
from numpy.random import uniform
from utex.agent import AgentAddress
from utex.audio import Speaker, WhiteNoise
from utex.scheduler import (SessionMarker, TrialIterator, blockwise_shuffle,
                            blockwise_shuffle2, blockwise_shufflen, mix,
                            repeat)

from flkl.share import (Flkl, as_millis, detect_lick, flush_message_for,
                        go_with_limit, nogo_with_postpone, show_progress)


async def conditional_discrimination(agent: Agent, ino: Flkl, expvars: dict):
    led_pin = expvars.get("led-pin", 3)
    sound_pin = expvars.get("sound-pin", 2)
    speaker_id = expvars.get("speaker-id", 1)
    reward_pin = expvars.get("reward-pin", 4)
    response_pin = expvars.get("response-pin", 6)

    noise = WhiteNoise()
    speaker = Speaker(speaker_id)

    reward_duration = expvars.get("reward-duration", 0.02)
    reward_duration_millis = as_millis(reward_duration)
    timeout_duration = expvars.get("timeout-duration", 5.0)
    FLICK_DURATION_MILLIS = 60000
    flick_duration = expvars.get("flick-duration", 2.0)
    flick_duration_millis = as_millis(flick_duration)
    decision_duration = expvars.get("decision-duration", 1.0)
    decision_duration_millis = as_millis(decision_duration)
    go_max_duration = expvars.get("go-maximum-duration", 4.0)
    nogo_max_duration = expvars.get("nogo-maximum-duration", 20.0)
    flush_duration = flick_duration - decision_duration

    led_flick_hz = expvars.get("led-flick-hz", [2, 3, 4, 5, 6, 7, 8, 9])
    boundary = expvars.get("go-nogo-boundary", 6.5)
    go_signals = list(filter(lambda hz: hz > boundary, led_flick_hz))
    nogo_signals = list(filter(lambda hz: hz < boundary, led_flick_hz))
    go_ratio = expvars.get("go-ratio", 1)
    nogo_ratio = expvars.get("nogo-ratio", 1)
    go_nogo_signals = mix(go_signals, nogo_signals, go_ratio, nogo_ratio)

    sound_flick_hz = expvars.get("sound-flick-hz", [2, 4, 6, 8, 20])
    practice_ratio = expvars.get("practice-ratio", 1)
    test_ratio = expvars.get("test-ratio", 1)
    sound_flickr_with_no_sound = mix([0], sound_flick_hz, test_ratio, practice_ratio)

    number_of_trial = expvars.get("number-of-trial", 200)

    visual_audio_flickr_combination = list(product(go_nogo_signals, sound_flickr_with_no_sound))
    flickr_per_trial = blockwise_shuffle(repeat(visual_audio_flickr_combination,
                                                number_of_trial // len(visual_audio_flickr_combination) + 1),
                                         len(visual_audio_flickr_combination))

    intertrial_interval = expvars.get("ITI", 3.0)
    iti_range = expvars.get("ITI-range", 1.0)
    number_of_trial = expvars.get("number-of-trial", 200)
    itis = np.random.uniform(
        intertrial_interval - iti_range,
        intertrial_interval + iti_range,
        number_of_trial,
    )

    trials = TrialIterator(flickr_per_trial[:number_of_trial], itis)

    try:
        while agent.working():
            speaker.play(noise, blocking=False, loop=True)
            for i, (visual_flickr, audio_flickr), iti in trials:
                print(f"visual={visual_flickr}, audio={audio_flickr}")
                await flush_message_for(agent, iti)
                if audio_flickr > 0:
                    ino.flick_for2(led_pin, sound_pin, visual_flickr, audio_flickr, flick_duration_millis)
                    await flush_message_for(agent, flush_duration)
                    is_licked = await detect_lick(agent, decision_duration, 6)
                    if visual_flickr > boundary and is_licked:
                        ino.high_for(reward_pin, reward_duration_millis)
                        await flush_message_for(agent, reward_duration)
                    elif visual_flickr > boundary and not is_licked:
                        await flush_message_for(agent, reward_duration)
                        await flush_message_for(agent, timeout_duration)
                    elif visual_flickr < boundary and is_licked:
                        await flush_message_for(agent, reward_duration)
                        await flush_message_for(agent, timeout_duration)
                    else:
                        await flush_message_for(agent, reward_duration)
                else:
                    ino.flick_on(led_pin, visual_flickr, FLICK_DURATION_MILLIS)
                    if visual_flickr > boundary:
                        await go_with_limit(agent, response_pin[0], flick_duration_millis, go_max_duration)
                        ino.flick_off()
                        ino.high_for(reward_pin, reward_duration_millis)
                        await flush_message_for(agent, reward_duration)
                    else:
                        await nogo_with_postpone(
                            agent, response_pin[0], flick_duration_millis, nogo_max_duration
                        )
                        ino.flick_off()
                        await flush_message_for(agent, reward_duration)

        speaker.stop()
        agent.send_to(AgentAddress.OBSERVER.value, SessionMarker.NEND)
        agent.finish()
    except NotWorkingError:
        speaker.stop()


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
    config.metadata.update({"condition": "go-nogo-main-task"})
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
