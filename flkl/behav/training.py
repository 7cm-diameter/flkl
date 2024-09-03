from amas.agent import Agent

from flkl.behav.share import Flkl


def show_progress(trial: int, iti: float, modality: int, freq: float):
    mod = ["Visual-Audio", "Visual", "Audio"][modality]
    print(f"Trial: {trial}    ITI: {iti}    Modality: {mod}    Frequency: {freq}")


async def flickr_discrimination(agent: Agent, ino: Flkl, expvars: dict):
    from amas.agent import NotWorkingError
    from numpy import arange
    from numpy.random import uniform
    from utex.agent import AgentAddress
    from utex.scheduler import (SessionMarker, TrialIterator,
                                blockwise_shuffle2, mix, mixn, repeat)

    from flkl.behav.share import as_millis, count_lick, flush_message_for

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

    flickr_sync_rwd = expvars.get("rewarded-frequency", [12, 14, 16])
    flickr_sync_ext = expvars.get("extinction-frequency", [6, 8, 10])
    flickr_sync = mix(
        flickr_sync_rwd,
        flickr_sync_ext,
        expvars.get("reward-ratio", 1),
        expvars.get("extinction-ratio", 1)
    )
    flickr_visual = flickr_sync
    flickr_audio = expvars.get("audio-frequency", [2, 9, 11, 13, 20])
    flickrs = mixn(
        [flickr_sync, flickr_visual, flickr_audio],
        [
            expvars.get("sync-ratio", 1),
            expvars.get("visual-ratio", 1),
            expvars.get("audio-ratio", 1),
        ]
    )

    modalities = mixn([[0], [1], [2]],
                      [
                          len(flickr_sync) * expvars.get("sync-ratio", 1),
                          len(flickr_visual) * expvars.get("visual-ratio", 1),
                          len(flickr_audio) * expvars.get("audio-ratio", 1)
                      ]
    )

    iti_mean = expvars.get("ITI", 15.0)
    iti_range = expvars.get("ITI-range", 5.0)
    number_of_reward = expvars.get("number-of-reward", 200)
    maximum_trial = 500

    flickr_per_trial, modality_per_trial = blockwise_shuffle2(
        repeat(flickrs, maximum_trial // len(flickrs) + 1),
        repeat(modalities, maximum_trial // len(flickrs) + 1),
        len(flickrs)
    )

    trials = TrialIterator(modality_per_trial[:maximum_trial], flickr_per_trial[:maximum_trial])

    try:
        while agent.working() and number_of_reward > 0:
            for i, modality, flickr in trials:
                iti = uniform(iti_mean - iti_range, iti_mean + iti_range)
                show_progress(i, iti, modality, flickr)
                await flush_message_for(agent, iti_mean)
                if modality == 0:
                    ino.flick_for2(visual_pin, audio_pin, flickr, flickr, flickr_duration_millis)
                    await flush_message_for(agent, flickr_duration - decision_duration)
                    nlick = await count_lick(agent, decision_duration, response_pin[0])
                    if flickr in flickr_sync_rwd and nlick >= required_lick:
                        ino.high_for(reward_pin, reward_duration_millis)
                        number_of_reward -= 1
                elif modality == 1:
                    ino.flick_for(visual_pin, flickr, flickr_duration_millis)
                    await flush_message_for(agent, flickr_duration - decision_duration)
                    nlick = await count_lick(agent, decision_duration, response_pin[0])
                    if flickr in flickr_sync_rwd and nlick >= required_lick:
                        ino.high_for(reward_pin, reward_duration_millis)
                        number_of_reward -= 1
                else:
                    ino.flick_for(audio_pin, flickr, flickr_duration_millis)
                    await agent.sleep(flickr_duration)
                    if uniform() < audio_reward_probability:
                        ino.high_for(reward_pin, reward_duration_millis)
                        number_of_reward -= 1
                await agent.sleep(reward_duration)
                if number_of_reward <= 0:
                    break
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

    from flkl.behav.share import read

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
    config.metadata.update({"condition": "gng-training"})
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
