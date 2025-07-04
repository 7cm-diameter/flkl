from amas.agent import Agent

from flkl.share import Flkl


def show_progress(trial: int, iti: float, modality: int, vhz: float, ahz: float):
    mod = ["Sync", "Async", "Visual", "Audio"][modality]
    print(f"Trial: {trial}    ITI: {iti}    Modality: {mod}    Vhz: {vhz}    Ahz: {ahz}")


async def flickr_discrimination(agent: Agent, ino: Flkl, expvars: dict):
    from itertools import product

    from amas.agent import NotWorkingError
    from numpy import arange
    from numpy.random import uniform
    from utex.agent import AgentAddress
    from utex.scheduler import (SessionMarker, TrialIterator,
                                blockwise_shuffle2, mix, mixn, repeat)

    from flkl.share import as_millis, count_lick, flush_message_for

    reward_pin = expvars.get("reward-pin", 4)
    response_pin = expvars.get("response-pin", [6])
    audio_pin = expvars.get("audio-pin", 2)
    visual_pin = expvars.get("visual-pin", 3)

    reward_duration = expvars.get("reward-duration", 0.01)
    flickr_duration = expvars.get("flickr-duration", 2.)
    decision_duration = expvars.get("decision-duration", 1.)
    required_lick = expvars.get("required-lick", 1)

    reward_duration_millis = as_millis(reward_duration)
    flickr_duration_millis = as_millis(flickr_duration)

    audio_reward_probability = expvars.get("reward-probability-for-audio", 0.2)

    dummy_flickr = [0]
    flickr_async_test = expvars.get("test-frequency", 9)
    flickr_sync_rwd = expvars.get("rewarded-frequency", [10, 12, 14])
    flickr_sync_ext = expvars.get("extinction-frequency", [4, 6, 8])

    flickr_sync = mix(
        flickr_sync_rwd,
        flickr_sync_ext,
        expvars.get("reward-ratio", 1),
        expvars.get("extinction-ratio", 1)
    )
    flickr_visual = flickr_sync
    flickr_audio = expvars.get("audio-frequency", [5, 7, 11, 13])

    flickr_async = [tuple(flickr) for flickr in expvars.get("async-flickrs", [[9, 8], [9, 10]])]
    flickr_sync = list(zip(flickr_sync, flickr_sync))
    flickr_visual = list(product(flickr_visual, dummy_flickr))
    flickr_audio = list(product(dummy_flickr, flickr_audio))

    flickrs = mixn(
        [flickr_sync, flickr_async, flickr_visual, flickr_audio],
        [
            expvars.get("sync-ratio", 1),
            expvars.get("async-ratio", 1),
            expvars.get("visual-ratio", 1),
            expvars.get("audio-ratio", 1),
        ]
    )

    modalities = mixn([[0], [1], [2], [3]],
                      [
                          len(flickr_sync) * expvars.get("sync-ratio", 1),
                          len(flickr_async) * expvars.get("async-ratio", 1),
                          len(flickr_visual) * expvars.get("visual-ratio", 1),
                          len(flickr_audio) * expvars.get("audio-ratio", 1)
                      ]
    )

    iti_mean = expvars.get("ITI", 15.0)
    iti_range = expvars.get("ITI-range", 5.0)
    trials_per_stim = expvars.get("trials-per-stimulus", 20)

    flickr_per_trial, modality_per_trial = blockwise_shuffle2(
        repeat(flickrs, trials_per_stim),
        repeat(modalities, trials_per_stim),
        len(flickrs)
    )

    trials = TrialIterator(modality_per_trial, flickr_per_trial)

    try:
        while agent.working():
            for i, modality, flickr in trials:
                vhz, ahz = flickr
                iti = uniform(iti_mean - iti_range, iti_mean + iti_range)
                show_progress(i, iti, modality, vhz, ahz)
                await flush_message_for(agent, iti_mean)
                if modality == 0 or modality == 1:
                    ino.flick_for2(visual_pin, audio_pin, vhz, ahz, flickr_duration_millis)
                    await flush_message_for(agent, flickr_duration - decision_duration)
                    nlick = await count_lick(agent, decision_duration, response_pin[0])
                    if vhz in flickr_sync_rwd and nlick >= required_lick:
                        ino.high_for(reward_pin, reward_duration_millis)
                elif modality == 2:
                    ino.flick_for(visual_pin, vhz, flickr_duration_millis)
                    await flush_message_for(agent, flickr_duration - decision_duration)
                    nlick = await count_lick(agent, decision_duration, response_pin[0])
                    if vhz in flickr_sync_rwd and nlick >= required_lick:
                        ino.high_for(reward_pin, reward_duration_millis)
                else:
                    ino.flick_for(audio_pin, ahz, flickr_duration_millis)
                    await agent.sleep(flickr_duration)
                    if uniform() < audio_reward_probability:
                        ino.high_for(reward_pin, reward_duration_millis)
                await agent.sleep(reward_duration)
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
            [reader_ino.pin_mode(i, PinMode.INPUT) for i in range(0, 14)]
        elif board.serial_number == com_output_config.get("serial-number"):
            setting.apply_setting(com_output_config)
            flkl = Flkl(ArduinoConnecter(setting).connect())
            [flkl.pin_mode(i, PinMode.OUTPUT) for i in range(0, 14)]
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
    config.metadata.update({"condition": "gng-test"})
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
