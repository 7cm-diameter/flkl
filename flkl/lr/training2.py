from typing import Optional

import numpy as np
from amas.agent import Agent, NotWorkingError
from numpy.random import uniform
from utex.agent import AgentAddress
from utex.audio import Speaker, WhiteNoise
from utex.scheduler import (SessionMarker, TrialIterator, blockwise_shuffle2,
                            mix, repeat)

from flkl.share import (Flkl, as_millis, fixed_interval_with_postpone,
                        flush_message_for, show_progress)


async def conditional_discrimination(agent: Agent, ino: Flkl, expvars: dict):
    led_pin = expvars.get("led-pin", 3)
    sound_pin = expvars.get("sound-pin", 2)
    speaker_id = expvars.get("speaker-id", 1)
    reward_pin = expvars.get("reward-pin", [4, 5])
    response_pin = expvars.get("response-pin", [6, 7])

    # noise = WhiteNoise()
    # speaker = Speaker(speaker_id)

    reward_duration = expvars.get("reward-duration", 0.02)
    reward_duration_millis = as_millis(reward_duration)
    decision_duration = expvars.get("decision-duration", 1.0)
    decision_duration_millis = as_millis(decision_duration)
    postpone = expvars.get("postpone", 2.0)
    led_flick_hz = expvars.get("led-flick-hz", [2, 10])
    sound_flick_hz = expvars.get("sound-flick-hz", [2, 4, 5, 6, 7, 8, 9, 20])
    boundary = expvars.get("lr-boundary", 6.5)
    right_ratio = expvars.get("right-ratio", 1)
    max_duration = expvars.get("maximum-duration", 10.0)
    MAX_DURATION_MILLIS = 60000
    min_duration = expvars.get("minimum-duration", 2.0)
    left_ratio = expvars.get("left-ratio", 1)
    right_signals = list(filter(lambda hz: hz > boundary, led_flick_hz))
    left_signals = list(filter(lambda hz: hz < boundary, led_flick_hz))
    lr_signals = mix(right_signals, left_signals, right_ratio, left_ratio)
    visual_trial = repeat(1, len(lr_signals))
    sound_trial = repeat(0, len(sound_flick_hz))
    visual_ratio = expvars.get("visual-ratio", 1)
    sound_ratio = expvars.get("sound-ratio", 1)
    flicker_hz_combination = mix(
        lr_signals, sound_flick_hz, visual_ratio, sound_ratio
    )
    modality_combination = mix(visual_trial, sound_trial, visual_ratio, sound_ratio)

    intertrial_interval = expvars.get("ITI", 5.0)
    iti_range = expvars.get("ITI-range", 0.0)
    number_of_trial = expvars.get("number-of-trial", 200)
    itis = np.random.uniform(
        intertrial_interval - iti_range,
        intertrial_interval + iti_range,
        number_of_trial,
    )
    flicker_hz_per_trial, modality_per_trial = blockwise_shuffle2(
        repeat(
            flicker_hz_combination,
            number_of_trial // len(flicker_hz_combination) + 1,
        ),
        repeat(
            modality_combination,
            number_of_trial // len(modality_combination) + 1,
        ),
        len(flicker_hz_combination),
    )
    trials = TrialIterator(
        modality_per_trial[:number_of_trial],
        flicker_hz_per_trial[:number_of_trial],
        itis,
    )

    try:
        while agent.working():
            # speaker.play(noise, blocking=False, loop=True)
            for i, is_visual, flick, iti in trials:
                show_progress(i, iti, flick, led_pin)
                await flush_message_for(agent, iti)
                ino.flick_on2(led_pin, sound_pin, flick, flick, MAX_DURATION_MILLIS)
                if flick > boundary:
                    await fixed_interval_with_postpone(
                        agent, response_pin[0], decision_duration, min_duration, max_duration, postpone
                    )
                    ino.flick_off()
                    ino.high_for(reward_pin[0], reward_duration_millis)
                    await flush_message_for(agent, reward_duration)
                else:
                    await fixed_interval_with_postpone(
                        agent, response_pin[1], decision_duration, min_duration, max_duration, postpone
                    )
                    ino.flick_off()
                    ino.high_for(reward_pin[1], reward_duration_millis)
                await agent.sleep(reward_duration)
            # speaker.stop()
            agent.send_to(AgentAddress.OBSERVER.value, SessionMarker.NEND)
            agent.finish()
    except NotWorkingError:
        # speaker.stop()
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
    config.metadata.update({"condition": "lr-training"})
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
