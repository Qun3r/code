#include <Arduino.h>

#define UART_MODE 1

// ====== Piny L298N ======
const uint8_t ENA = 11; // PWM (lewy)
const uint8_t IN1 = 12;
const uint8_t IN2 = 13;

const uint8_t ENB = 6;  // PWM (prawy)
const uint8_t IN3 = 8;
const uint8_t IN4 = 7;

// ====== OGRANICZENIA ======
const uint8_t MIN_PWM = 130;     // nie schodzimy poniżej (poza STOP)
const uint8_t MAX_PWM = 200;

// ====== Prędkości bazowe ======
const uint8_t SPEED_FWD = 150;

// ŁUK (soft turn) - oba do przodu, różnica mała
const uint8_t SPEED_ARC_IN  = 135;   // wewnętrzne koło (>= MIN_PWM!)
const uint8_t SPEED_ARC_OUT = 160;   // zewnętrzne koło

// PIVOT (sharp) - impuls, ale nadal bez “0” (pivot będzie delikatny)
const uint8_t SPEED_PIVOT = 165;

// ====== Timing UART ======
const uint16_t COMMAND_HOLD_MS = 120;  // jak krótko trzymać ostatnią komendę
const uint16_t LOST_STOP_MS    = 600;  // po tylu ms braku komend -> STOP

// ====== NOWE: Burst forward (spowolnienie średnie) ======
const uint16_t FWD_ON_MS  = 160;  // ile jedzie
const uint16_t FWD_OFF_MS = 340;  // ile stoi -> razem ~0.5 s

// ====== NOWE: impuls skrętu i przerwa na ocenę ======
const uint16_t TURN_PULSE_MS   = 70;   // sam skręt krótko
const uint16_t TURN_PAUSE_MS   = 260;  // po skręcie nie rób kolejnego skrętu (czas na nową decyzję)

// Sharp (pivot) - jeszcze krócej, bo łatwo robi kółka
const uint16_t PIVOT_PULSE_MS  = 55;
const uint16_t PIVOT_PAUSE_MS  = 320;

// ====== Rampa PWM (delikatna) ======
const uint8_t  RAMP_STEP = 8;
const uint16_t RAMP_DELAY_MS = 6;

// ---------- stan ----------
unsigned long lastRxMs = 0;
char desiredCmd = 'S';     // ostatnia komenda z Pythona

// gating skrętów
unsigned long turnBlockUntilMs = 0;   // do kiedy blokować kolejne skręty
unsigned long actionUntilMs    = 0;   // do kiedy trwa aktualny impuls (turn/pivot)

// burst forward
unsigned long fwdPhaseStartMs = 0;
bool fwdOn = true;

// aktualne PWM
int curL = 0, curR = 0;

// ---------- low-level ----------
int clampMin(int v) {
  if (v == 0) return 0;
  return constrain(v, MIN_PWM, MAX_PWM);
}

void setDirLeft(bool forward) {
  digitalWrite(IN1, forward ? LOW  : HIGH);
  digitalWrite(IN2, forward ? HIGH : LOW);
}
void setDirRight(bool forward) {
  digitalWrite(IN3, forward ? HIGH : LOW);
  digitalWrite(IN4, forward ? LOW  : HIGH);
}

void applyPWMNow() {
  analogWrite(ENA, constrain(curL, 0, 255));
  analogWrite(ENB, constrain(curR, 0, 255));
}

void rampTo(int targetL, int targetR) {
  targetL = clampMin(targetL);
  targetR = clampMin(targetR);

  while (curL != targetL || curR != targetR) {
    if (curL < targetL) curL = min(curL + RAMP_STEP, targetL);
    else if (curL > targetL) curL = max(curL - RAMP_STEP, targetL);

    if (curR < targetR) curR = min(curR + RAMP_STEP, targetR);
    else if (curR > targetR) curR = max(curR - RAMP_STEP, targetR);

    applyPWMNow();
    delay(RAMP_DELAY_MS);
  }
}

void stopCoast() {
  rampTo(0, 0);
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
}

// ---------- ruchy bazowe ----------
void driveForwardNow() {
  setDirLeft(true);
  setDirRight(true);
  rampTo(SPEED_FWD, SPEED_FWD);
}

void arcLeftNow() {   // łagodny skręt w lewo (łuk)
  setDirLeft(true);
  setDirRight(true);
  rampTo(SPEED_ARC_IN, SPEED_ARC_OUT);
}

void arcRightNow() {  // łagodny skręt w prawo (łuk)
  setDirLeft(true);
  setDirRight(true);
  rampTo(SPEED_ARC_OUT, SPEED_ARC_IN);
}

// delikatny pivot (sharp) - krótki impuls
void pivotLeftNow() {
  setDirLeft(false);
  setDirRight(true);
  rampTo(MIN_PWM, SPEED_PIVOT);
}

void pivotRightNow() {
  setDirLeft(true);
  setDirRight(false);
  rampTo(SPEED_PIVOT, MIN_PWM);
}

bool isTurnCmd(char c) {
  return (c == 'L' || c == 'R' || c == 'l' || c == 'r');
}

// ---------- UART rx ----------
void onCommand(char c) {
  unsigned long now = millis();
  lastRxMs = now;

  // jeśli po skręcie jest pauza na ocenę, a przychodzi kolejny skręt -> ignoruj skręt (traktuj jak F)
  if (isTurnCmd(c) && now < turnBlockUntilMs) {
    c = 'F';
  }

  desiredCmd = c;

  // reset burst forward fazy przy każdej komendzie (żeby nie było “dziwnego” timingu)
  fwdPhaseStartMs = now;
  fwdOn = true;

  // jeśli to skręt -> wykonaj tylko impuls, a potem zablokuj skręty na czas “oceny”
  if (c == 'L') {
    arcLeftNow();
    actionUntilMs = now + TURN_PULSE_MS;
    turnBlockUntilMs = now + TURN_PULSE_MS + TURN_PAUSE_MS;
  } else if (c == 'R') {
    arcRightNow();
    actionUntilMs = now + TURN_PULSE_MS;
    turnBlockUntilMs = now + TURN_PULSE_MS + TURN_PAUSE_MS;
  } else if (c == 'l') {
    pivotLeftNow();
    actionUntilMs = now + PIVOT_PULSE_MS;
    turnBlockUntilMs = now + PIVOT_PULSE_MS + PIVOT_PAUSE_MS;
  } else if (c == 'r') {
    pivotRightNow();
    actionUntilMs = now + PIVOT_PULSE_MS;
    turnBlockUntilMs = now + PIVOT_PULSE_MS + PIVOT_PAUSE_MS;
  } else if (c == 'F') {
    // forward będzie realizowany jako burst w loop()
  } else if (c == 'S') {
    stopCoast();
  }
}

// ---------- setup ----------
void setup() {
  pinMode(ENA, OUTPUT); pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(ENB, OUTPUT); pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);

  stopCoast();
  Serial.begin(9600);
  Serial.println("Motor controller: burst+turn-pulse+pause");
}

// ---------- loop ----------
void loop() {
#if UART_MODE
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') continue;
    onCommand(c);
  }

  unsigned long now = millis();

  // 1) Jeżeli trwa impuls skrętu/pivota -> po czasie zakończ i daj forward burst (lub stop)
  if (actionUntilMs != 0 && now >= actionUntilMs) {
    actionUntilMs = 0;
    // Po skręcie NIE kręć dalej — przejdź na forward (burst) jako stabilizacja
    desiredCmd = 'F';
    fwdPhaseStartMs = now;
    fwdOn = true;
  }

  // 2) fail-safe: brak komend -> STOP
  if (now - lastRxMs > LOST_STOP_MS) {
    stopCoast();
    desiredCmd = 'S';
    return;
  }

  // 3) Forward jako burst (spowolnienie średniej prędkości)
  if (desiredCmd == 'F') {
    unsigned long phaseDt = now - fwdPhaseStartMs;

    if (fwdOn) {
      if (phaseDt >= FWD_ON_MS) {
        // koniec ON -> OFF
        fwdOn = false;
        fwdPhaseStartMs = now;
        stopCoast(); // przerwa
      } else {
        driveForwardNow();
      }
    } else {
      if (phaseDt >= FWD_OFF_MS) {
        // koniec OFF -> ON
        fwdOn = true;
        fwdPhaseStartMs = now;
        driveForwardNow();
      } else {
        // stoimy
      }
    }
  }

  // 4) STOP
  if (desiredCmd == 'S') {
    stopCoast();
  }

#else
  // test
  onCommand('F'); delay(3000);
  onCommand('L'); delay(1500);
  onCommand('R'); delay(1500);
  onCommand('l'); delay(1500);
  onCommand('r'); delay(1500);
  onCommand('S'); delay(2000);
#endif
}
