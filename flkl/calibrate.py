from amas.agent import Agent

from flkl.share import Flkl


async def reward_calibration(agent: Agent, ino: Flkl, n: int, interval: float, duration: float, pin: int):
    from amas.agent import NotWorkingError
    from numpy import arange
    from utex.agent import AgentAddress
    from utex.scheduler import SessionMarker

    from flkl.share import as_millis

    reward_duration_millis = as_millis(duration)

    try:
        while agent.working():
            for i in range(n):
                print(f"Trial = {i + 1} / n")
                await agent.sleep(interval)
                ino.high_for(pin, reward_duration_millis)
            agent.send_to(AgentAddress.OBSERVER.value, SessionMarker.NEND)
            agent.finish()
    except NotWorkingError:
        pass

if __name__ == "__main__":
    import argparse

    from amas.connection import Register
    from amas.env import Environment
    from pyno.com import check_connected_board_info
    from pyno.ino import ArduinoConnecter, ArduinoSetting, PinMode
    from utex.agent import AgentAddress, Observer, self_terminate
    from utex.scheduler import SessionMarker

    # Step 1: 引数の読み込み
    parser = argparse.ArgumentParser(description="Run the flickr discrimination task.")
    parser.add_argument("--pin", "-p", deault=4, help="Number of reward pin")
    parser.add_argument("--duration", "-d", required=True, help="Duration of reward")
    parser.add_argument("--interval", "-i", required=True, help="Interval between each reward")
    parser.add_argument("--number-of-reward", "-n", required=True, help="Number of rerward presentation")
    args = parser.parse_args()

    available_boards = check_connected_board_info()

    if len(available_boards) == 0:
        raise RuntimeError("No Arduino boards were detected. Please check the connection and try again.")
    elif len(available_boards) > 1:
        raise RuntimeError("Multiple Arduino boards were detected. Please connect only one board and try again.")

    board = available_boards[0]

    setting = ArduinoSetting.derive_from_portinfo(board)
    connector = ArduinoConnecter(setting)
    connector.write_sketch()
    flkl = Flkl(connector.connect())
    [flkl.pin_mode(i, PinMode.OUTPUT) for i in range(0, 14)]

    controller = (
        Agent("CONTROLLER")
        .assign_task(reward_calibration, ino=flkl, n=args.n, interval=args.interval, duration=args.duration, pin=args.pin)
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
