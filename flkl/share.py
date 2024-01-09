from pyno.ino import ArduinoFlicker, as_bytes


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
