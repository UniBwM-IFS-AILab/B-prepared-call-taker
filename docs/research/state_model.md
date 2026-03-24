# State Modelling

## Official Documents

given:
  - state variables (symptoms)
  - corresponding outcome: RD2, TCPR, ImmediateDisposition+First-Aid-Algo
  - we can group symptoms by outcomes as higher-order state
  - State Transitions (Questions) for determining RD1 and the precondition global state

Needed:
  - function to map user-provided Text to Symptoms (Outcome can be determined deterministically)
  - Function to Map user-provided Text to State Transitions

## Possible Approaches

### 1. Flat List

All variables are in a flat list, their role is differentiated by Type (`RD1_Boolean`, `RD2_Boolean`, ...) and sometimes by name (`RD1`, `RD2`, `TCPR`, `Immediate_disposition`)

### 2. Group by Outcome

- Symptoms are nested by the respective Outcome of them being true: RD1, RD2, T-CPR, and Immediate Disposition
- Problems:
  - How to deal with Symptoms belonging to multiple outcomes
    - Separate Model? (call by copy or by reference)
    - can Fields be declared outside of a Model and have linked values somehow?
  -

### 3. Group by Triage / Symptom Section

- Group Symptoms by categories (Breathing, Consciousness, Circulation, ...)
- can be endlessly nested -> needs recursive traversal to determine outcomes

## Current Implementations

- **flat list** of boolean symptoms where RD1 and RD2 is computed based on respective related symptoms
- breathing and consciousness had their semantics flipped in order to make computation easier
  - both RD1 question need to be answered with Yes now instead of No to trigger an RD1 state
  - Question stays the same: *Do they get enough air?* but Variable changes
  - Original Version `now_sufficient_air` -> Now: `now_insufficient_air`

### State Variables

1. Initially Caller and Location are determined (Who / Where)
2. Open ended Question to determine the type of Problem (What)
3. In case of Medical:
   1. The System should have a global state dictionary that describes describes the state of the patient (`python -> bool | None` for yes, no and unknown).
   2. There is a list of symptoms that indicate RD2, Immediate Disposition and T-CPR
      - "Hinweiszeichen" <- Strukturierte Notrufabfrage ILS Bayern
   3. RD1 necessity is determined via Dialogue Flow
      - Once RD1 is determined the patient is asked for further detail to detect possible RD2
4. Fire: TODO
5. None: TODO
