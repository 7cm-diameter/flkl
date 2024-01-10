from amas.agent import Agent, NotWorkingError
import numpy as np
from numpy.random import uniform
from utex.agent import AgentAddress
from utex.scheduler import SessionMarker, TrialIterator, blockwise_shuffle2, mix, repeat

from flkl.share import Flkl


async def conditional_discrimination(agent: Agent, ino: Flkl, expvars: dict):
    led_pin = expvars.get("led-pin", 5)
    sound_pin = expvars.get("sound-pin", 2)
    reward_pin = expvars.get("reward-pin", 8)

    reward_duration = expvars.get("reward-duration", 20)
    flick_duration = expvars.get("flick-duration", 1000)
    led_flick_hz = expvars.get("led-flick-hz", [2, 10])
    sound_flick_hz = expvars.get("sound-flick-hz", [2, 4, 5, 6, 7, 8, 9, 20])
    boundary = expvars.get("boundary", 6.5)
    go_ratio = expvars.get("go-ratio", 1)
    nogo_ratio = expvars.get("nogo-ratio", 1)
    go_signals = list(filter(lambda hz: hz > boundary, led_flick_hz))
    nogo_signals = list(filter(lambda hz: hz < boundary, led_flick_hz))
    go_nogo_signals = mix(go_signals, nogo_signals, go_ratio, nogo_ratio)
    visual_trial = repeat(1, len(go_nogo_signals))
    sound_trial = repeat(0, len(sound_flick_hz))
    visual_ratio = expvars.get("visual-ratio", 1)
    sound_ratio = expvars.get("audio-ratio", 1)
    go_nogo_signals_with_sound = mix(
        go_nogo_signals, sound_flick_hz, visual_ratio, sound_ratio
    )
    av_trials = mix(visual_trial, sound_trial, visual_ratio, sound_ratio)

    intertrial_interval = expvars.get("ITI", 3.0)
    iti_range = expvars.get("ITI-range", 1.0)
    number_of_trial = expvars.get("number-of-trial", 200)
    itis = np.random.uniform(
        intertrial_interval - iti_range,
        intertrial_interval + iti_range,
        number_of_trial,
    )
    flick_each_trial, av_each_trial = blockwise_shuffle2(
        repeat(
            go_nogo_signals_with_sound,
            number_of_trial // len(go_nogo_signals_with_sound) + 1,
        ),
        repeat(
            av_trials,
            number_of_trial // len(av_trials) + 1,
        ),
        len(go_nogo_signals_with_sound),
    )
    trials = TrialIterator(
        av_each_trial[:number_of_trial], flick_each_trial[:number_of_trial], itis
    )

    try:
        while agent.working():
            for i, is_visual, flick, iti in trials:
                print(f"{i} trial: flick with {flick} hz after {iti} sec")
                await agent.sleep(iti)
                if is_visual:
                    ino.flick_for(led_pin, flick, flick_duration)
                    await agent.sleep(flick_duration / 1000)
                    if flick > boundary:
                        ino.high_for(reward_pin, reward_duration)
                        await agent.sleep(reward_duration / 1000)
                else:
                    ino.flick_for(sound_pin, flick, flick_duration)
                    await agent.sleep(flick_duration / 1000)
                    if uniform() <= 0.5:
                        ino.high_for(reward_pin, reward_duration)
                        await agent.sleep(reward_duration / 1000)
            agent.send_to(AgentAddress.OBSERVER.value, SessionMarker.NEND)
    except NotWorkingError:
        pass


if __name__ == "__main__":
    from pyno.com import check_connected_board_info
    from pyno.ino import (
        ArduinoSetting,
        Mode,
        ArduinoConnecter,
        PinMode,
        ArduinoLineReader,
    )
    from utex.clap import PinoClap
    from utex.agent import self_terminate, Reader, Observer, Recorder
    from utex.scheduler import SessionMarker
    from amas.connection import Register
    from amas.env import Environment

    config = PinoClap().config()
    com_input_config = config.comport.get("input")
    com_input_config.update({"mode": Mode.readeruno})
    com_output_config = config.comport.get("output")
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
    for board in available_boards:
        setting = ArduinoSetting.derive_from_portinfo(board)
        if board.serial_number == com_input_config.get("serial-number"):
            setting.apply_setting(com_input_config)
            reader_ino = ArduinoLineReader(ArduinoConnecter(setting).connect())
        elif board.serial_number == com_output_config.get("serial-number"):
            setting.apply_setting(com_output_config)
            flkl = Flkl(ArduinoConnecter(setting).connect())
            [flkl.pin_mode(i, PinMode.OUTPUT) for i in range(2, 5)]
            [flkl.pin_mode(i, PinMode.INPUT) for i in range(6, 7)]

    controller = (
        Agent("CONTROLLER")
        .assign_task(conditional_discrimination, ino=flkl, expvars=config.experimental)
        .assign_task(self_terminate)
    )

    reader = Reader(reader_ino)
    observer = Observer()
    recorder = Recorder(filename="./hoge.csv")

    agents = [controller, recorder, reader, observer]
    register = Register(agents)
    env = Environment(agents)

    try:
        env.run()
    except KeyboardInterrupt:
        observer.send_all(SessionMarker.ABEND)
        observer.finish()
