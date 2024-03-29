from amas.agent import Agent

from flkl.share import Flkl


def show_progress(trial: int, iti: float, is_visual: int, freq: float):
    modality = "Visual" if is_visual else "Audio"
    print(f"Trial: {trial}    ITI: {float}    Modality: {modality}    Frequency: {freq}")


async def flickr_discrimination(agent: Agent, ino: Flkl, expvars: dict):
    from amas.agent import NotWorkingError
    from numpy import arange
    from numpy.random import uniform
    from utex.agent import AgentAddress
    from utex.audio import Speaker, WhiteNoise
    from utex.scheduler import (SessionMarker, TrialIterator,
                                blockwise_shufflen, mix, repeat)

    from flkl.share import as_millis, flush_message_for

    reward_pin = expvars.get("reward-pin", 4)
    response_pin = expvars.get("response-pin", [6])
    audio_pin = expvars.get("audio-pin", 2)
    visual_pin = expvars.get("visual-pin", 3)

    noise = WhiteNoise()
    speaker = Speaker(expvars.get("speaker-id", 1))

    reward_duration = expvars.get("reward-duration", 0.02)
    flickr_duration = expvars.get("flickr-duration", 2.)

    reward_duration_millis = as_millis(reward_duration)
    flickr_duration_millis = as_millis(flickr_duration)

    reward_probability_for_audio = expvars.get("reward-probability-for-audio", .1)

    visual_target_frequency = expvars.get("target-frequency", 6.)
    visual_distracter_frequency = expvars.get("visual-distracter", [4., 5., 7., 8.])
    visual_traget_flickrs = repeat(visual_target_frequency, len(visual_distracter_frequency))
    visual_flickrs = mix(
        [visual_target_frequency],
        visual_distracter_frequency,
        expvars.get("target-ratio", 4),
        expvars.get("distracter-ratio", 1)
    )

    audio_distracter_frequency = expvars.get("audio-distracter", [2., 5., 6., 7., 18.])

    modality_block = mix(
        repeat([1], len(visual_flickrs)),
        repeat([0], len(audio_distracter_frequency)),
        expvars.get("visual-ratio", 1),
        expvars.get("audio-ratio", 1)
    )
    flickr_block = mix(
        visual_flickrs,
        audio_distracter_frequency,
        expvars.get("visual-ratio", 1),
        expvars.get("audio-ratio", 1)
    )

    number_of_reward = expvars.get("number-of-reward", 200)
    number_of_block = number_of_reward // expvars.get("target-ratio", 5)
    blocksize = len(modality_block)
    flickr_per_trial, modality_per_trial = blockwise_shufflen(
        blocksize,
        repeat(flickr_block, number_of_block),
        repeat(modality_block, number_of_block),
    )

    number_of_trial = len(flickr_per_trial)

    mean_interval = expvars.get("ITI", 10.)
    range_interval = expvars.get("ITI-range", 5.)
    itis = uniform(
        mean_interval - range_interval,
        mean_interval + range_interval,
        number_of_trial
    )

    trials = TrialIterator(
        flickr_per_trial,
        modality_per_trial,
        itis
    )

    try:
        while agent.working():
            speaker.play(noise, False, True)
            for i, freq, is_visual, iti in trials:
                show_progress(i, iti, is_visual, freq)
                if is_visual:
                    rewarded = freq == visual_target_frequency
                    ino.flick_for(visual_pin, freq, flickr_duration_millis)
                    await flush_message_for(agent, flickr_duration)
                    if rewarded:
                        ino.high_for(reward_pin, reward_duration_millis)
                        await flush_message_for(agent, reward_duration)
                    else:
                        await flush_message_for(agent, reward_duration)
                else:
                    rewarded = uniform() <= reward_probability_for_audio
                    ino.flick_for(audio_pin, freq, flickr_duration_millis)
                    await flush_message_for(agent, flickr_duration)
                    if rewarded:
                        ino.high_for(reward_pin, reward_duration_millis)
                        await flush_message_for(agent, reward_duration)
                    else:
                        await flush_message_for(agent, reward_duration)
            speaker.stop()
            agent.send_to(AgentAddress.OBSERVER.value, SessionMarker.NEND)
            agent.finish()
    except NotWorkingError:
        pass


if __name__ == "__main__":
    from os import mkdir
    from os.path import exists, join
    from typing import Optional

    from amas.agent import Agent
    from amas.connection import Register
    from amas.env import Environment
    from pyno.com import check_connected_board_info
    from pyno.ino import (ArduinoConnecter, ArduinoLineReader, ArduinoSetting,
                          Mode, PinMode)
    from utex.agent import AgentAddress, Observer, Recorder, self_terminate
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
    config.metadata.update({"condition": "gnrl-training"})
    filename = join(data_dir, namefile(config.metadata))

    controller = (
        Agent("CONTROLLER")
        .assign_task(flickr_discrimination, ino=flkl, expvars=config.experimental)
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
