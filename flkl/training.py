from amas.agent import Agent

from flkl.share import Flkl


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

    from flkl.share import as_millis, count_lick, flush_message_for

    reward_pin = expvars.get("reward-pin", 4)
    audio_pin = expvars.get("speaker-pin", 2)
    visual_pin = expvars.get("led-pin", 3)

    reward_duration = expvars.get("reward-duration", 0.01)
    flickr_duration = expvars.get("flickr-duration", 2.)

    reward_duration_millis = as_millis(reward_duration)
    flickr_duration_millis = as_millis(flickr_duration)

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
                iti = uniform(iti_mean - iti_range, iti_mean + iti_range)
                show_progress(i, iti, modality, flickr)
                await flush_message_for(agent, iti_mean)
                if modality == 0:
                    ino.flick_for2(visual_pin, audio_pin, flickr, flickr, flickr_duration_millis)
                    await agent.sleep(flickr_duration)
                    if flickr in flickr_sync_rwd:
                        ino.high_for(reward_pin, reward_duration_millis)
                elif modality == 1:
                    ino.flick_for(visual_pin, flickr, flickr_duration_millis)
                    await agent.sleep(flickr_duration)
                    if flickr in flickr_sync_rwd:
                        ino.high_for(reward_pin, reward_duration_millis)
                else:
                    ino.flick_for(audio_pin, flickr, flickr_duration_millis)
                    await agent.sleep(flickr_duration + reward_duration)
            agent.send_to(AgentAddress.OBSERVER.value, SessionMarker.NEND)
            agent.finish()
    except NotWorkingError:
        pass

if __name__ == "__main__":
    import argparse
    from datetime import datetime
    from os import mkdir
    from os.path import exists, join
    from typing import Optional

    import yaml
    from amas.agent import Agent
    from amas.connection import Register
    from amas.env import Environment
    from pyno.com import check_connected_board_info
    from pyno.ino import (ArduinoConnecter, ArduinoLineReader, ArduinoSetting,
                          Mode, PinMode)
    from utex.agent import AgentAddress, Observer, Recorder, self_terminate
    from utex.clap import Config
    from utex.fs import get_current_file_abspath, namefile
    from utex.scheduler import SessionMarker

    from flkl.share import read

    # Step 1: 引数の読み込み
    parser = argparse.ArgumentParser(description="Run the flickr discrimination task.")
    parser.add_argument("--subject", "-s", required=True, help="Subject ID")
    parser.add_argument("--yaml", "-y", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    subject = args.subject
    yaml_path = args.yaml

    with open(yaml_path, "r") as f:
        config_data = yaml.safe_load(f)

    log_dir = "./log"
    if not exists(log_dir):
        mkdir(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_filename = f"{subject}-{timestamp}.yaml"
    log_path = join(log_dir, log_filename)

    with open(log_path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False)

    config = Config(log_path)
    com_output_config: Optional[dict] = config.comport.get("output")
    if com_output_config is None:
        raise ValueError("`com_output_config` are not defined.")
    com_output_config.update({"mode": Mode.user})

    if com_output_config is None:
        raise Exception()

    available_boards = check_connected_board_info()

    if len(available_boards) == 0:
        raise RuntimeError("No Arduino boards were detected. Please check the connection and try again.")
    elif len(available_boards) > 1:
        raise RuntimeError("Multiple Arduino boards were detected. Please connect only one board and try again.")

    board = available_boards[0]

    setting = ArduinoSetting.derive_from_portinfo(board)
    setting.apply_setting(com_output_config)
    serial_number = com_output_config.get("serial-number")
    print(f"Uploading sketch to controller arduino {serial_number}")
    ArduinoConnecter(setting).write_sketch()
    flkl = Flkl(ArduinoConnecter(setting).connect())
    [flkl.pin_mode(i, PinMode.OUTPUT) for i in range(0, 14)]

    controller = (
        Agent("CONTROLLER")
        .assign_task(flickr_discrimination, ino=flkl, expvars=config.experimental)
        .assign_task(self_terminate)
    )

    observer = Observer()

    agents = [controller, observer]
    register = Register(agents)
    env = Environment(agents)

    try:
        env.run()
    except KeyboardInterrupt:
        observer.send_all(SessionMarker.ABEND)
        observer.finish()
