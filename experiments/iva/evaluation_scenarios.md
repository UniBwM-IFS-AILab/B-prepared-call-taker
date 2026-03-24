# Scenarios

## Aim

Re-use example Scenarios from Munich ILS for evaluating performance of the model. Currently **focus on medical cases** since we do not have other paths implemented. Prefer to have **only one patient** as we don't know how to deal with multiple yet.

Should have at least **one scenario per outcome**:

-   RD1 (We evaluate that the system does not escalate to RD2 by mistake)
-   RD2
-   T-CPR -\> Cardiac Arrest / Agonal Breathing
-   (High Urgency / immediate disposition -\> Heavy Injury) -\> I cannot judge what injuries are truly urgent
-   Maybe Last case with varying outcomes

| Outcome                              | Col2                              |
|--------------------------------------|-----------------------------------|
| RD1                                  |                                   |
| RD2                                  |                                   |
| T-CPR                                | Cardiac Arrest / Agonal Breathing |
| High Urgency / immediate disposition | Some form of Heavy Injury         |

: Scenarios by outcome

### Candidates

**RD1**: Notfallrettung  -\> **1**, **9**, **14**, 17, **21** [Notarzteinsatz / Notfalleinsatz]

**RD2**: Notfallrettung  -\> **5**, **6**, 8, **16** [Notarzteinsatz / Notfalleinsatz]

**Varying**: Notfallrettung -\> **7**, **13**, 19, **20**

**T-CPR**: Notfalleinsatz T-CPR -\> **1**, **3**, 4 \[Not **2** because the caller needs convincing\]

**Immediate disposition**: I cannot judge what injuries are truly urgent

### Found Scenarios:

Notfalleinsatz 9, 4, 13, 14, 20, 16, 18, 6, 9, 2, 15, 6, 5, 4, 4, 5, 1, 10, 16, 12, 15, 21, 15, 13

Notarzteinsatz 7

TCPR 1, 1, 2, 3,
