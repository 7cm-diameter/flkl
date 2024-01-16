from amas.agent import Agent, NotWorkingError
from pyno.ino import ArduinoFlicker, ArduinoLineReader, as_bytes


class Flkl(ArduinoFlicker):
    from pyno.ino import ArduinoConnecter

    def __init__(self, connecter: ArduinoConnecter):
        super().__init__(connecter)

    def flick_for2(self, pin1: int, pin2: int, hz1: float, hz2: float, millis: int):
        hz1 = int(hz1 * 10)
        hz2 = int(hz2 * 10)
        message = (
            b"\x15"
            + as_bytes(pin1, 1)
            + as_bytes(pin2, 1)
            + as_bytes(hz1, 1)
            + as_bytes(hz2, 1)
            + as_bytes(millis, 2)
        )
        self.connection.write(message)

    def high_for(self, pin: int, millis: int):
        message = b"\x17" + as_bytes(pin, 1) + as_bytes(millis, 2)
        self.connection.write(message)


def show_progress(trial: int, iti: float, hz: float, pin: int):
    print(f"Trial {trial}: flickr ({hz}) follows after {iti} sec on {pin} pin")


def as_eventtime(readline: str) -> tuple[int, int]:
    event_id = int(readline[0])
    if event_id == 1:
        event_id = 10
        timeidx = 2
    else:
        timeidx = 1
    micros = int(readline[timeidx:])
    return event_id, micros


async def detect_lick(agent: Agent, duration: float, target):
    from time import perf_counter

    while duration >= 0.0 and agent.working():
        s = perf_counter()
        mail = await agent.try_recv(duration)
        if mail is None:
            continue
        _, mess = mail
        if mess == target:
            return True
    return False


async def read(agent: Agent, ino: ArduinoLineReader, expvars: dict):
    from utex.agent import AgentAddress

    response_pin = expvars.get("response-pin", 6)

    try:
        while agent.working():
            readline: bytes = await agent.call_async(ino.readline)
            if readline is None:
                continue
            decoded_readline = readline.rstrip().decode("utf-8")
            event, _ = as_eventtime(decoded_readline)
            if event == response_pin:
                agent.send_to(AgentAddress.CONTROLLER.value, event)
            agent.send_to(AgentAddress.RECORDER.value, decoded_readline)

    except NotWorkingError:
        pass
