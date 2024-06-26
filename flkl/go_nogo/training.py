from amas.agent import Agent

from flkl.share import Flkl


async def flickr_discrimination(agent: Agent, ino: Flkl, expvars: dict):
    from amas.agent import NotWorkingError
    from numpy.random import uniform
    from utex.agent import AgentAddress
    from utex.audio import Speaker, WhiteNoise
    from utex.scheduler import (SessionMarker, TrialIterator,
                                blockwise_shufflen, mix, repeat)

    from flkl.share import (as_millis, count_lick, flush_message_for,
                            go_with_limit, nogo_with_postpone)

    reward_pin = expvars.get("reward-pin", 4)
    response_pin = expvars.get("response-pin", [6])
    audio_pin = expvars.get("audio-pin", 2)
    visual_pin = expvars.get("visual-pin", 3)

    noise = WhiteNoise()
    speaker = Speaker(expvars.get("speaker-id", 1))

    # Used in python program
    reward_duration = expvars.get("reward-duration", 0.02)
    timeout_duration = expvars.get("timeout-duration", 5.)
    maximum_flickr_duration = 60.0
    minimum_flickr_duration = expvars.get("flickr-duration", 2.)
    maximum_go_duration = expvars.get("maximum-go-duration", 4.)
    maximum_nogo_duration = expvars.get("maximum-go-duration", 10.)
    decision_duration = expvars.get("decision-duration", 1.)
    flush_duration = minimum_flickr_duration - decision_duration
    reward_probability_for_audio = expvars.get("reward-probability-for-audio", .5)

    # Used as an argument for arduino command
    reward_duration_millis = as_millis(reward_duration)
    maximum_flickr_duration_millis = as_millis(maximum_flickr_duration)
    minimum_flickr_duration_millis = as_millis(minimum_flickr_duration)

    visual_flickr_frequency = expvars.get("visual-flickr-frequency", [3, 4, 5, 6, 7, 8])
    audio_flickr_frequency = expvars.get("audio-flickr-frequency", [2, 4, 5, 6, 7, 20])
    boundary = expvars.get("boundary", 5.5)

    go_signals = list(filter(lambda freq: freq > boundary, visual_flickr_frequency))
    nogo_signals = list(filter(lambda freq: freq < boundary, visual_flickr_frequency))
    go_ratio = expvars.get("go-ratio", 1)
    nogo_ratio = expvars.get("nogo-ratio", 1)
    go_nogo_mixed = mix(go_signals, nogo_signals, go_ratio, nogo_ratio)

    modality_block = mix(
        repeat(1, len(go_nogo_mixed)),
        repeat(0, len(audio_flickr_frequency)),
        expvars.get("visual-ratio", 1),
        expvars.get("audio-ratio", 1)
    )
    flickr_frequency_block = mix(
        go_nogo_mixed,
        audio_flickr_frequency,
        expvars.get("visual-ratio", 1),
        expvars.get("audio-ratio", 1)
    )

    practice_ratio = expvars.get("practice-ratio", 1)
    test_ratio = expvars.get("test-ratio", 1)
    practice_test_block = mix(
        repeat(0, len(flickr_frequency_block)),
        repeat(1, len(flickr_frequency_block)),
        practice_ratio, test_ratio
    )
    modality_block *= practice_ratio + test_ratio
    flickr_frequency_block *= practice_ratio + test_ratio

    number_of_trial = expvars.get("number-of-trial", 500)
    blocksize = len(flickr_frequency_block)
    flickr_per_trial, modality_per_trial, trialtype_per_trial = blockwise_shufflen(
        len(flickr_frequency_block),
        repeat(
            flickr_frequency_block,
            number_of_trial // blocksize + 1
        ),
        repeat(
            modality_block,
            number_of_trial // blocksize + 1
        ),
        repeat(
            practice_test_block,
            number_of_trial // blocksize + 1
        )
    )

    mean_interval = expvars.get("ITI", 5.)
    range_interval = expvars.get("ITI-range", 0.)
    itis = uniform(
        mean_interval - range_interval,
        mean_interval + range_interval,
        number_of_trial
    )

    trials = TrialIterator(
        flickr_per_trial,
        modality_per_trial,
        trialtype_per_trial,
        itis
    )

    try:
        while agent.working():
            speaker.play(noise, False, True)
            for i, freq, is_visual, trialtype, iti in trials:
                print(f"Trial {i}")
                if trialtype: # 1 => test
                    if is_visual:
                        is_go = freq > boundary
                        ino.flick_for(visual_pin, freq, minimum_flickr_duration_millis)
                        await flush_message_for(agent, flush_duration)
                        licked = await count_lick(agent, decision_duration, response_pin[0])
                        if is_go and licked:
                            ino.high_for(reward_pin, reward_duration_millis)
                            await flush_message_for(agent, reward_duration)
                        elif is_go and not licked:
                            await flush_message_for(agent, reward_duration + timeout_duration)
                        elif not is_go and not licked:
                            await flush_message_for(agent, reward_duration + timeout_duration)
                        else:
                            await flush_message_for(agent, reward_duration)
                    else:
                        is_go = uniform() >= reward_probability_for_audio
                        ino.flick_for(audio_pin, freq, minimum_flickr_duration_millis)
                        if is_go:
                            pass
                        else:
                            pass
                else: # 0 => practice
                    if is_visual:
                        is_go = freq > boundary
                        ino.flick_on(visual_pin, freq, maximum_flickr_duration_millis)
                        if is_go:
                            await go_with_limit(agent, response_pin[0], minimum_flickr_duration, maximum_go_duration)
                            ino.flick_off()
                            ino.high_for(reward_pin, reward_duration_millis)
                            await flush_message_for(agent, reward_duration)
                        else:
                            await nogo_with_postpone(agent, response_pin[0], minimum_flickr_duration, maximum_nogo_duration)
                            ino.flick_off()
                            await flush_message_for(agent, reward_duration)
                    else:
                        is_go = uniform() >= reward_probability_for_audio
                        ino.flick_on(audio_pin, freq, maximum_flickr_duration_millis)
                        if is_go:
                            await go_with_limit(agent, response_pin[0], minimum_flickr_duration, maximum_go_duration)
                            ino.flick_off()
                            ino.high_for(reward_pin, reward_duration_millis)
                            await flush_message_for(agent, reward_duration)
                        else:
                            await nogo_with_postpone(agent, response_pin[0], minimum_flickr_duration, maximum_nogo_duration)
                            ino.flick_off()
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
            [flkl.pin_mode(i, PinMode.OUTPUT) for i in range(2, 5)]
            [flkl.pin_mode(i, PinMode.INPUT) for i in range(6, 7)]

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
    config.metadata.update({"condition": "go-nogo-with-postpone"})
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
