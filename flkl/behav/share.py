from amas.agent import Agent, NotWorkingError
from pyno.ino import ArduinoFlicker, ArduinoLineReader, as_bytes


class Flkl(ArduinoFlicker):
    from pyno.ino import ArduinoConnecter

    def __init__(self, connecter: ArduinoConnecter):
        super().__init__(connecter)

    def flick_for(self, pin: int, hz: float, flickr_duration: int, pulse_duration: int = 20):
        hz = int(hz * 10)
        message = b"\x13" + as_bytes(pin, 1) + as_bytes(hz, 1) + as_bytes(flickr_duration, 2) + as_bytes(pulse_duration, 2)
        self.connection.write(message)

    def flick_on(self, pin: int, hz: float, flickr_duration: int, pulse_duration: int = 20):
        hz = int(hz * 10)
        message = b"\x14" + as_bytes(pin, 1) + as_bytes(hz, 1) + as_bytes(flickr_duration, 2) + as_bytes(pulse_duration, 2)
        self.connection.write(message)

    def flick_for2(self, pin1: int, pin2: int, hz1: float, hz2: float, flickr_duration: int, pulse_duration: int = 20):
        hz1 = int(hz1 * 10)
        hz2 = int(hz2 * 10)
        message = (
            b"\x15"
            + as_bytes(pin1, 1)
            + as_bytes(pin2, 1)
            + as_bytes(hz1, 1)
            + as_bytes(hz2, 1)
            + as_bytes(flickr_duration, 2)
            + as_bytes(pulse_duration, 2)
        )
        self.connection.write(message)

    def flick_on2(self, pin1: int, pin2: int, hz1: float, hz2: float, flickr_duration: int, pulse_duration: int = 20):
        hz1 = int(hz1 * 10)
        hz2 = int(hz2 * 10)
        message = (
            b"\x16"
            + as_bytes(pin1, 1)
            + as_bytes(pin2, 1)
            + as_bytes(hz1, 1)
            + as_bytes(hz2, 1)
            + as_bytes(flickr_duration, 2)
            + as_bytes(pulse_duration, 2)
        )
        self.connection.write(message)

    def high_for(self, pin: int, millis: int):
        message = b"\x17" + as_bytes(pin, 1) + as_bytes(millis, 2)
        self.connection.write(message)


def as_millis(s: float) -> int:
    return int(s * 1000)


def show_progress(trial: int, iti: float, hz: float, pin: int):
    print(f"Trial {trial}: flickr ({hz}) follows after {iti} sec on {pin} pin")


def as_eventtime(readline: str) -> tuple[int, int]:
    try:
        event_id = int(readline[0])
    except IndexError:
        return -1, -1
    if event_id == 1:
        event_id = 10
        timeidx = 2
    else:
        timeidx = 1
    micros = int(readline[timeidx:])
    return event_id, micros


async def flush_message_for(agent: Agent, duration: float):
    from time import perf_counter

    while duration >= 0.0 and agent.working():
        s = perf_counter()
        await agent.try_recv(duration)
        e = perf_counter()
        duration -= e - s


async def count_lick(agent: Agent, duration: float, target) -> int:
    from time import perf_counter

    nlick = 0

    while duration >= 0.0 and agent.working():
        s = perf_counter()
        mail = await agent.try_recv(duration)
        duration -= perf_counter() - s
        if mail is None:
            continue
        _, mess = mail
        if mess == target:
            nlick += 1
    return nlick


async def go_with_limit(
    agent: Agent,
    correct: int,
    decision_duration: float,
    max_duration: float,
):
    from time import perf_counter

    while max_duration >= 0.0 and agent.working():
        s = perf_counter()
        mail = await agent.try_recv(max_duration)
        ellapsed_time = perf_counter() - s
        decision_duration -= ellapsed_time
        max_duration -= ellapsed_time

        if mail is None:
            break

        _, mess = mail
        if mess == correct and decision_duration <= 0.0:
            break
        else:
            continue


async def nogo_with_postpone(
    agent: Agent, incorrect: int, decision_duration: float, max_duration: float
):
    from time import perf_counter

    _decission_duration = decision_duration

    while max_duration >= 0.0 and agent.working():
        s = perf_counter()
        mail = await agent.try_recv(decision_duration)
        ellapsed_time = perf_counter() - s
        max_duration -= ellapsed_time
        if mail is None:
            break
        _, mess = mail
        if mess == incorrect:
            continue


async def fixed_interval_with_postpone(agent: Agent, correct: int, decision_duration: float,
                                       min_duration: float, max_duration: float, postpone: float):
    from time import perf_counter

    max_duration -= (min_duration - decision_duration)

    await flush_message_for(agent, min_duration - decision_duration)

    while max_duration >= 0.0 and agent.working():
        s = perf_counter()
        mail = await agent.try_recv(max_duration)
        ellapsed_time = perf_counter() - s

        max_duration -= ellapsed_time
        decision_duration -= ellapsed_time

        if mail is None:
            break

        _, mess = mail

        if mess != correct:
            decision_duration = postpone
        elif mess == correct and decision_duration <= 0.:
            break


async def fixed_time_with_error(agent: Agent, correct: int,
                                stimulus_duration: float, decision_duration: float) -> bool :
    from time import perf_counter

    await flush_message_for(agent, stimulus_duration - decision_duration)

    ncorrect = 0
    while decision_duration >= 0.:
        s = perf_counter()
        mail = await agent.try_recv(decision_duration)
        decision_duration -= perf_counter() - s

        if mail is None:
            continue

        _, mess = mail
        if mess == correct:
            ncorrect += 1
        else:
            ncorrect -= 1

    return ncorrect > 0

async def read(agent: Agent, ino: ArduinoLineReader, expvars: dict):
    from utex.agent import AgentAddress

    response_pin = expvars.get("response-pin", [6, 7])

    try:
        while agent.working():
            readline: bytes = await agent.call_async(ino.readline)
            if readline is None:
                continue
            decoded_readline = readline.rstrip().decode("utf-8")
            event, time = as_eventtime(decoded_readline)
            if event in response_pin:
                agent.send_to(AgentAddress.CONTROLLER.value, event)
            agent.send_to(AgentAddress.RECORDER.value, (time, event))

    except NotWorkingError:
        pass
