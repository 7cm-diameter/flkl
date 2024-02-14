import numpy as np
from amas.agent import Agent, NotWorkingError
from utex.agent import AgentAddress
from utex.scheduler import SessionMarker, TrialIterator, mix, repeat

from flkl.share import Flkl, as_millis


async def present_reward(agent: Agent, ino: Flkl, expvars: dict):
    reward_pins = expvars.get("reward-pin", [4, 5])

    number_of_trial = expvars.get("number-of-trial", 200)
    reward_duration = expvars.get("reward-duration", 0.01)
    reward_duration_millis = as_millis(reward_duration)

    intertrial_interval = expvars.get("ITI", 10.)
    iti_range = expvars.get("ITI-range", 1.0)
    itis = np.random.uniform(
        intertrial_interval - iti_range,
        intertrial_interval + iti_range,
        number_of_trial,
    )

    left_ratio = expvars.get("left-ratio", 1)
    right_ratio = expvars.get("right-ratio", 1)
    lr_ratio = mix([0], [1], left_ratio, right_ratio)
    reward_per_trial = repeat(lr_ratio, number_of_trial // len(lr_ratio))
    np.random.shuffle(reward_per_trial)

    trials = TrialIterator(itis, reward_per_trial[:number_of_trial])

    try:
        for i, iti, reward in trials:
            print(f"Reward occurs {iti} sec after on {reward_pins[reward]} pin.")
            await agent.sleep(iti)
            ino.high_for(reward_pins[reward], reward_duration_millis)

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
        .assign_task(present_reward, ino=flkl, expvars=config.experimental)
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
