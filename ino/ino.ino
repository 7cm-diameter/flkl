struct StateSwitchPin {
  int pins[13];
  int currState[13];
  int prevState[13];
  int pinNum;
};


typedef void (*ptrDigitalWrite)(void);
typedef void (*func)(int);


void digiHIGH0() { PORTD |= _BV(0); }
void digiHIGH1() { PORTD |= _BV(1); }
void digiHIGH2() { PORTD |= _BV(2); }
void digiHIGH3() { PORTD |= _BV(3); }
void digiHIGH4() { PORTD |= _BV(4); }
void digiHIGH5() { PORTD |= _BV(5); }
void digiHIGH6() { PORTD |= _BV(6); }
void digiHIGH7() { PORTD |= _BV(7); }
void digiHIGH8() { PORTB |= _BV(0); }
void digiHIGH9() { PORTB |= _BV(1); }
void digiHIGH10() { PORTB |= _BV(2); }
void digiHIGH11() { PORTB |= _BV(3); }
void digiHIGH12() { PORTB |= _BV(4); }
void digiHIGH13() { PORTB |= _BV(5); }

ptrDigitalWrite digiHIGH[14] = {
  digiHIGH0, digiHIGH1, digiHIGH2, digiHIGH3, digiHIGH4, digiHIGH5, digiHIGH6, digiHIGH7,
  digiHIGH8, digiHIGH9, digiHIGH10, digiHIGH11, digiHIGH12, digiHIGH13
};

void digiLOW0() { PORTD &= ~_BV(0); }
void digiLOW1() { PORTD &= ~_BV(1); }
void digiLOW2() { PORTD &= ~_BV(2); }
void digiLOW3() { PORTD &= ~_BV(3); }
void digiLOW4() { PORTD &= ~_BV(4); }
void digiLOW5() { PORTD &= ~_BV(5); }
void digiLOW6() { PORTD &= ~_BV(6); }
void digiLOW7() { PORTD &= ~_BV(7); }
void digiLOW8() { PORTB &= ~_BV(0); }
void digiLOW9() { PORTB &= ~_BV(1); }
void digiLOW10() { PORTB &= ~_BV(2); }
void digiLOW11() { PORTB &= ~_BV(3); }
void digiLOW12() { PORTB &= ~_BV(4); }
void digiLOW13() { PORTB &= ~_BV(5); }

ptrDigitalWrite digiLOW[14] = {
  digiLOW0, digiLOW1, digiLOW2, digiLOW3, digiLOW4, digiLOW5, digiLOW6, digiLOW7,
  digiLOW8, digiLOW9, digiLOW10, digiLOW11, digiLOW12, digiLOW13
};

typedef int (*ptrDigiRead)(void);

int digiRead0() { return PIND & _BV(0); }
int digiRead1() { return PIND & _BV(1); }
int digiRead2() { return PIND & _BV(2); }
int digiRead3() { return PIND & _BV(3); }
int digiRead4() { return PIND & _BV(4); }
int digiRead5() { return PIND & _BV(5); }
int digiRead6() { return PIND & _BV(6); }
int digiRead7() { return PIND & _BV(7); }
int digiRead8() { return PINB & _BV(0); }
int digiRead9() { return PINB & _BV(1); }
int digiRead10() { return PINB & _BV(2); }
int digiRead11() { return PINB & _BV(3); }
int digiRead12() { return PINB & _BV(4); }
int digiRead13() { return PINB & _BV(5); }

ptrDigiRead digiRead[14] = {
  digiRead0, digiRead1, digiRead2, digiRead3, digiRead4, digiRead5, digiRead6, digiRead7,
  digiRead8, digiRead9, digiRead10, digiRead11, digiRead12, digiRead13
};

/* Struct and functions for SSINPUT and SSINPUT_PULLUPv*/
StateSwitchPin initSSPin() {
  StateSwitchPin sspin;
  for(int i=0; i<13; i++) {
    sspin.pins[i] = 0;
    sspin.currState[i] = 0;
    sspin.prevState[i] = 0;
  };
  sspin.pinNum = 0;
  return sspin;
}

void setPinModeSS(StateSwitchPin *sspin, int pin, int mode) {
  pinMode(pin, mode);
  sspin->pins[sspin->pinNum] = pin;
  sspin->pinNum++;
}

void resetPinModeSS(StateSwitchPin *sspin, int pin) {
  if (sspin->pinNum > 0) {
    for (int i=0; i<sspin->pinNum; i++) {
      if (sspin->pins[i] == pin) {
        sspin->pinNum--;
        for (int j=i; j<13; j++) {
          if (j < 13) {
            sspin->pins[j] = sspin->pins[j+1];
          } else {
            sspin->pins[j] = 0;
          }
        }
      }
    }
  }
}

void checkPinState(StateSwitchPin *sspin) {
  for(int i=0; i<sspin->pinNum; i++) {
    int pin = sspin->pins[i];
    sspin->currState[i] = digitalRead(pin);
    if (sspin->prevState[i] && !sspin->currState[i]) {
      Serial.println(pin);
    }
    if (!sspin->prevState[i] && sspin->currState[i]) {
      Serial.println(-pin);
    }
    sspin->prevState[i] = sspin->currState[i];
  }
}

StateSwitchPin sspin = initSSPin();

/* Functions and vairables for flicker */
unsigned long calculate_interval(double hz, double duration) {
  return 1000 * (duration / (2 * hz + 1));
}

unsigned long from_millis_to_micros(double x) {
  return x * 1000;
}

double read_2bytes() {
  int readbytes[2];
  for (int i = 0; i < 2; i++) {
    while((readbytes[i] = Serial.read()) == -1) {};
  }
  return readbytes[1] * 256 + readbytes[0];
}

int flicker_states[14] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
double hz1;
double hz2;


void setup() {
  Serial.begin(115200);
}

void loop() {
  int pin1;
  int pin2;
  int rpin;
  int command;

  while (1) {
    while ((command = Serial.read()) == -1) {
      checkPinState(&sspin);
    };

    while ((pin1 = Serial.read() ) == -1) {
      checkPinState(&sspin);
    };

    switch (command) {
      // pinMode: '\x00' - '\x09'
      case '\x00': {
        pinMode(pin1, INPUT);
        resetPinModeSS(&sspin, pin1);
        break;
      }

      case '\x01': {
        pinMode(pin1, INPUT_PULLUP);
        resetPinModeSS(&sspin, pin1);
        break;
      }

      case '\x02': {
        pinMode(pin1, OUTPUT);
        resetPinModeSS(&sspin, pin1);
        break;
      }

      case '\x03': {
        setPinModeSS(&sspin, pin1, INPUT);
        break;
      }

      case '\x04': {
        setPinModeSS(&sspin, pin1, INPUT_PULLUP);
        break;
      }

      // write: '\x10' - '\x19'
      case '\x10': {
        digiLOW[pin1]();
        break;
      }

      case '\x11': {
        digiHIGH[pin1]();
        break;
      }

      case '\x12': {
        int v;
        while ( (v = Serial.read()) == -1) {
          checkPinState(&sspin);
        };
        analogWrite(pin1, v);
        break;
      }

      /*
      case '\x13': {
        int angle;
        while ( (angle = Serial.read()) == -1) {
          checkPinState(&sspin);
        };
        servos[pin1].write(angle);
        break;
      }
      */

      // Flkl.flick_for
      case '\x13': {
        while((hz1 = Serial.read()) == -1) {};
        double flickr_duration = read_2bytes();
        double pulse_duration = read_2bytes();
        while((rpin = Serial.read()) == -1) {};
        double reward_duration = read_2bytes();
        double pulse_interval = 10000 / hz1 - pulse_duration;


        unsigned long pulse_interval_micros = from_millis_to_micros(pulse_interval);
        unsigned long pulse_duration_micros = from_millis_to_micros(pulse_duration);
        unsigned long flickr_duration_micros = from_millis_to_micros(flickr_duration);
        unsigned long waitingtime  = pulse_duration_micros;
        unsigned long reward_duration_micros = from_millis_to_micros(reward_duration);
        digiHIGH[pin1]();
        flicker_states[pin1] = 1;
        unsigned long starttime = micros();
        unsigned long swtichtimer = starttime;

        while(micros() - starttime < flickr_duration_micros) {
          if ((micros() - swtichtimer) >= waitingtime) {
            if (flicker_states[pin1]) {
              digiLOW[pin1]();
              flicker_states[pin1] = 0;
              waitingtime = pulse_interval_micros;
            } else {
              digiHIGH[pin1]();
              flicker_states[pin1] = 1;
              waitingtime = pulse_duration_micros;
            }
            swtichtimer = micros();
          }
        }

        digiLOW[pin1]();
        flicker_states[pin1] = 0;

        if (rpin > 0) {
          unsigned long intervaltimer1 = micros();
          digiHIGH[rpin]();
          while(micros() - intervaltimer1 < reward_duration_micros) {};
          digiLOW[rpin]();
        }

        break;
      }

      // Flkl.flick_on
      case '\x14': {
        while((hz1 = Serial.read()) == -1) {};
        double flickr_duration = read_2bytes();
        double pulse_duration = read_2bytes();
        double pulse_interval = 10000 / hz1 - pulse_duration;

        unsigned long pulse_interval_micros = from_millis_to_micros(pulse_interval);
        unsigned long pulse_duration_micros = from_millis_to_micros(pulse_duration);
        unsigned long flickr_duration_micros = from_millis_to_micros(flickr_duration);
        unsigned long waitingtime  = pulse_duration_micros;
        digiHIGH[pin1]();
        flicker_states[pin1] = 1;
        unsigned long starttime = micros();
        unsigned long swtichtimer = starttime;

        while(micros() - starttime < flickr_duration_micros && (command = Serial.read()) == -1) {
          if ((micros() - swtichtimer) >= waitingtime) {
            if (flicker_states[pin1]) {
              digiLOW[pin1]();
              flicker_states[pin1] = 0;
              waitingtime = pulse_interval_micros;
            } else {
              digiHIGH[pin1]();
              flicker_states[pin1] = 1;
              waitingtime = pulse_duration_micros;
            }
            swtichtimer = micros();
          }
          if (command == '\x19') { break; }
        }

        /*
        double duration = read_2bytes();
        double flicktime = hz1 * duration / 10000;

        flicker_states[pin1] = 0;
        unsigned long interval1 = calculate_interval(flicktime , duration);
        unsigned long duration_micros = from_millis_to_micros(duration);
        unsigned long starttimer = micros();
        unsigned long intervaltimer1 = micros();

        while(micros() - starttimer < duration_micros && (command = Serial.read()) == -1) {
          if ((micros() - intervaltimer1) >= interval1) {
            if (!flicker_states[pin1]) {
              digiHIGH[pin1]();
              flicker_states[pin1] = 1;
            } else {
              digiLOW[pin1]();
              flicker_states[pin1] = 0;
            }
            intervaltimer1 = micros();
          }
          if (command == '\x19') { break; }
        };
        */

        digiLOW[pin1]();
        flicker_states[pin1] = 0;
        break;
      }

      // flick_for2
      case '\x15': {
        while((pin2 = Serial.read()) == -1) {};
        while((hz1 = Serial.read()) == -1) {};
        while((hz2 = Serial.read()) == -1) {};
        double flickr_duration = read_2bytes();
        double pulse_duration = read_2bytes();
        while((rpin = Serial.read()) == -1) {};
        double reward_duration = read_2bytes();
        double pulse_interval1 = 10000 / hz1 - pulse_duration;
        double pulse_interval2 = 10000 / hz2 - pulse_duration;


        unsigned long pulse_interval_micros1 = from_millis_to_micros(pulse_interval1);
        unsigned long pulse_interval_micros2 = from_millis_to_micros(pulse_interval2);
        unsigned long pulse_duration_micros = from_millis_to_micros(pulse_duration);
        unsigned long flickr_duration_micros = from_millis_to_micros(flickr_duration);
        unsigned long waitingtime1 = pulse_duration_micros;
        unsigned long waitingtime2 = pulse_duration_micros;
        unsigned long reward_duration_micros = from_millis_to_micros(reward_duration);
        digiHIGH[pin1]();
        digiHIGH[pin2]();
        flicker_states[pin1] = 1;
        flicker_states[pin2] = 1;
        unsigned long starttime = micros();
        unsigned long swtichtimer1 = starttime;
        unsigned long swtichtimer2 = starttime;

        while(micros() - starttime < flickr_duration_micros) {
          unsigned long current_time = micros();
          if ((current_time - swtichtimer1) >= waitingtime1) {
            if (flicker_states[pin1]) {
              digiLOW[pin1]();
              flicker_states[pin1] = 0;
              waitingtime1 = pulse_interval_micros1;
            } else {
              digiHIGH[pin1]();
              flicker_states[pin1] = 1;
              waitingtime1 = pulse_duration_micros;
            }
            swtichtimer1 = micros();
          }

          if ((current_time - swtichtimer2) >= waitingtime2) {
            if (flicker_states[pin2]) {
              digiLOW[pin2]();
              flicker_states[pin2] = 0;
              waitingtime2 = pulse_interval_micros2;
            } else {
              digiHIGH[pin2]();
              flicker_states[pin2] = 1;
              waitingtime2 = pulse_duration_micros;
            }
            swtichtimer2 = micros();
          }
        }

        /*
        double duration = read_2bytes();
        double flicktime1 = hz1 * duration / 10000;
        double flicktime2 = hz2 * duration / 10000;

        flicker_states[pin1] = 0;
        unsigned long interval1 = calculate_interval(flicktime1, duration);
        unsigned long interval2 = calculate_interval(flicktime2, duration);
        unsigned long duration_micros = from_millis_to_micros(duration);
        unsigned long starttimer = micros();
        unsigned long intervaltimer1 = micros();
        unsigned long intervaltimer2 = micros();

        while(micros() - starttimer < duration_micros) {
          if ((micros() - intervaltimer1) >= interval1) {
            if (!flicker_states[pin1]) {
              digiHIGH[pin1]();
              flicker_states[pin1] = 1;
            } else {
              digiLOW[pin1]();
              flicker_states[pin1] = 0;
            }
            intervaltimer1 = micros();
          }
          if ((micros() - intervaltimer2) >= interval2) {
            if (!flicker_states[pin2]) {
              digiHIGH[pin2]();
              flicker_states[pin2] = 1;
            } else {
              digiLOW[pin2]();
              flicker_states[pin2] = 0;
            }
            intervaltimer2 = micros();
          }
        };
        */

        digiLOW[pin1]();
        digiLOW[pin2]();
        flicker_states[pin1] = 0;
        flicker_states[pin2] = 0;

        if (rpin > 0) {
          unsigned long intervaltimer1 = micros();
          digiHIGH[rpin]();
          while(micros() - intervaltimer1 < reward_duration_micros) {};
          digiLOW[rpin]();
        }

        break;
      }

      // flick_on2
      case '\x16': {
        while((pin2 = Serial.read()) == -1) {};
        while((hz1 = Serial.read()) == -1) {};
        while((hz2 = Serial.read()) == -1) {};
        double flickr_duration = read_2bytes();
        double pulse_duration = read_2bytes();
        double pulse_interval1 = 10000 / hz1 - pulse_duration;
        double pulse_interval2 = 10000 / hz2 - pulse_duration;

        unsigned long pulse_interval_micros1 = from_millis_to_micros(pulse_interval1);
        unsigned long pulse_interval_micros2 = from_millis_to_micros(pulse_interval2);
        unsigned long pulse_duration_micros = from_millis_to_micros(pulse_duration);
        unsigned long flickr_duration_micros = from_millis_to_micros(flickr_duration);
        unsigned long waitingtime1 = pulse_duration_micros;
        unsigned long waitingtime2 = pulse_duration_micros;
        digiHIGH[pin1]();
        digiHIGH[pin2]();
        flicker_states[pin1] = 1;
        flicker_states[pin2] = 1;
        unsigned long starttime = micros();
        unsigned long swtichtimer1 = starttime;
        unsigned long swtichtimer2 = starttime;

        while(micros() - starttime < flickr_duration_micros && (command = Serial.read()) == -1) {
          unsigned long current_time = micros();
          if ((current_time - swtichtimer1) >= waitingtime1) {
            if (flicker_states[pin1]) {
              digiLOW[pin1]();
              flicker_states[pin1] = 0;
              waitingtime1 = pulse_interval_micros1;
            } else {
              digiHIGH[pin1]();
              flicker_states[pin1] = 1;
              waitingtime1 = pulse_duration_micros;
            }
            swtichtimer1 = micros();
          }

          if ((current_time - swtichtimer2) >= waitingtime2) {
            if (flicker_states[pin2]) {
              digiLOW[pin2]();
              flicker_states[pin2] = 0;
              waitingtime2 = pulse_interval_micros2;
            } else {
              digiHIGH[pin2]();
              flicker_states[pin2] = 1;
              waitingtime2 = pulse_duration_micros;
            }
            swtichtimer2 = micros();
          }
          if (command == '\x19') { break; }
        }

        /*
        double duration = read_2bytes();
        double flicktime1 = hz1 * duration / 10000;
        double flicktime2 = hz2 * duration / 10000;

        flicker_states[pin1] = 0;
        unsigned long interval1 = calculate_interval(flicktime1, duration);
        unsigned long interval2 = calculate_interval(flicktime2, duration);
        unsigned long duration_micros = from_millis_to_micros(duration);
        unsigned long starttimer = micros();
        unsigned long intervaltimer1 = micros();
        unsigned long intervaltimer2 = micros();

        while(micros() - starttimer < duration_micros && (command = Serial.read()) == -1) {
          if ((micros() - intervaltimer1) >= interval1) {
            if (!flicker_states[pin1]) {
              digiHIGH[pin1]();
              flicker_states[pin1] = 1;
            } else {
              digiLOW[pin1]();
              flicker_states[pin1] = 0;
            }
            intervaltimer1 = micros();
          }
          if ((micros() - intervaltimer2) >= interval2) {
            if (!flicker_states[pin2]) {
              digiHIGH[pin2]();
              flicker_states[pin2] = 1;
            } else {
              digiLOW[pin2]();
              flicker_states[pin2] = 0;
            }
            intervaltimer2 = micros();
          }
          if (command == '\x19') { break; }
        };
        */

        digiLOW[pin1]();
        digiLOW[pin2]();
        flicker_states[pin1] = 0;
        flicker_states[pin2] = 0;
        break;
      }

      case '\x17': {
        double duration = read_2bytes();
        unsigned long duration_micros = from_millis_to_micros(duration);
        unsigned long intervaltimer1 = micros();

        digiHIGH[pin1]();
        while(micros() - intervaltimer1 < duration_micros) {};
        digiLOW[pin1]();
        break;
      }

      // read: '\x20' - '\x29'
      case '\x20': {
        int state = digiRead[pin1]();
        Serial.write(state);
        break;
      }

      case '\x21': {
        int v = analogRead(pin1);
        Serial.write(v);
        break;
      }

      default: {
        break;
      }
    }
  }
}
