# ems-prepared

## Known Bugs

- Determination of Emergency Type
  -
- Handling of Fire- and Non-Emergency Cases
- sometimes RD1 and RD2 Variables are triggered together because they are too similar
  - now_unresponsive & unconscious
- Completion of data after triggering disposition causes endless loop (disabled for now)
- When the agent could not deduce new state from a caller's response, it will ask for some variables (*fine*) in a weird manner (*not fine*)
  - examples
    - include the state key verbatim in the question (with underscores)
    - return a json string with value sit wants filled
  - We need to give better instructions
- the agent cannot self-loop, ie it can only ask once for more information
  - no message history is attached
  - RD1 takes precedence over repeated question loops
  - Solution:
    - attach history of all messages between first return of str, until a struct is returned,
    - Third return_type that indicates that the agent decided to not ask further but that there is no point in asking further (does this allow the agent to cheat?)

## Todos

- Decouple state filling from conversation
  - two separate agents, conversation agent decided how long they want to continue asking
- ditch `situation_description` and use message history instead
  - let agent decide how long he want to talk to determine Emrgency type while passing history along
- remove global variables, find other ways to deal with it (probably graph state)
- Could give the merger (for system state) to agent as a tool
- Handle if we run out of question for "subgraph"
  - currently if medical questions run through, they are just repeated
- Proper Subgraph for Medical Emergency
  - some nodes could be shared with main graph while others are only required in either, reducing complexity
- Transform Emergency type into a queue / stack /list
  - if none, force agent to fill it
  - if empty, start end of dialogue
