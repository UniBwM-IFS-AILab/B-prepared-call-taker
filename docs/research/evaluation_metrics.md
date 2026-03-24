# “How to Evaluate Your Dialogue Models: A Review of Approaches” ([pdf](zotero://open-pdf/library/items/7ZMUD9W6?page=1))(8/11/2025, 9:51:15 AM)

## Metrics

In general we want to know 3 things:

1. Did we reach the **correct outcome**, i.e. rd1, rd2, t-cpr, ...
2. Did we terminate the dialogue using the expected optimal **number of turns**
    - easily determined by following the flowchart from the official material
    - Too little and too many will get penalized
3. Is the **extracted State** correct?
    - Mainly to make sure that the outcome has been reached for the right reasons, ie. we could end up in RD1 for many subsets of symptoms that don't reflect the actual conversation
    - Looking at the correctsness of individual slots will mainly evaluate the power of the NLU / internal reasoning

---

### “task completion rate” ([pdf](zotero://open-pdf/library/items/7ZMUD9W6?page=4&annotation=C27438A9)) Correct outcome determination (RD2, TCPR, ...)

- Check if the subset of outcome variables (the outcome state is correct).
  - RD2 with and without T-CPR / High Urgency
  - RD1 only

Goal: Simply average for all dialogues whether the correct outcome is reached. (compare sets of outcome variables between observed and expected)

#### Additional Sources

“Walker, M.A., Kamm, C.A., Litman, D.J.: Towards developing general models of usability with PARADISE. Nat. Lang. Eng. 6(3&4) (2000)” ([pdf](zotero://open-pdf/library/items/7ZMUD9W6?page=12&annotation=L3578KDU))

“Wen, T., Vandyke, D., Mrksˇ ́ıc, N., Gaˇs ́ıc, M., Rojas-Barahona, L., Su, P., Ultes, S., Young, S.: A network-based end-to-end trainable task-oriented dialogue system. In: EACL 2017” ([pdf](zotero://open-pdf/library/items/7ZMUD9W6?page=12&annotation=XGRIVLRA))

### Slot filling / "Dialogue state tracking" (DST) ([pdf](zotero://open-pdf/library/items/7ZMUD9W6?page=4&annotation=QPDLP6I8))

- usually done turn-based ("tracking")
- existing metrics (Slot Accuracy and Joint Goal Accuracy) are flawed due to their cumulative nature
  - There are a bunch of other proposed metrics that try to improve on this with no consensus -> messy

### Turn-based accuracy vs evaluating only final state

- Emergency phone calls are often done in stressful situations. When The call taker asks for the location, but the caller answers with some patient symptoms, it is more important to capture the symptoms correctly than to penalize the turn-dependent score
- Only evaluating final state side-steps the issues joint goal accuracy and slot accuracy have [TODO: Source]

Turn-Based Variatons:

- Turn-based on **system turn**
  - makes no sense,
  - we want to be able to handle cases where the caller specifically provides info different from the question, this would be penalized here
- Turn-based on **user-turn**
  - is difficult when doing human evaluation (as opposed to simulated users / automatic evaluation)
  - i.e. whether the utterance leasds to the human-expected state change
  - since there is no control what the caller will say, we would need to manually label ideal dialogue state per turn after recording
  - this is tedious, can introduce biases and is vulnerable to human error or human subjectivity

We do not really care about the correct order of state changes, since we want to reach the optimal state subset in the expected number of turns. **While the official material dictates an order, even the simulated calls used for training show that this is not feasible in practice. Enforcing the order in real emergencies could lead to unnecessary dialogue turns and causes avoidable stress both for the caller and the disponent.**

Hence, **Joint goal Accuracy (JGA) and turn-based Slot Accuracy (SA) does not seem useful** for our use case, since it is cumulative and evaluates if each turn is correct. We don’t necessarily care about the order of these state events. We want to reach the exact set of state as fast as *realistically* possible.

Turn-based could be useful to enforce additional constrains /introduce extra punishments:

- asking for a RD2 Symptom before RD1 is true (won't happen in State machine unless buggy)
- asking for multiple unrelated symptoms (Won't happen in State machine and LLM if we enforce pre-defined questions)

### State comparisons for "Task completion" and "Final Slot Accuracy"

#### Average over Pairwise-comparisons (Accuracy)

$$
C_{Accuracy} \;=\; \frac{\sum_{i=1}^{n}\delta_i}{n}
\qquad
\begin{cases}
1, & o_i = e_i\\
0, & \text{otherwise}
\end{cases}
$$

Basically correct slots divided by the number of slots.

With Weights:
$$
C_{\text{Weighted}} \;=\;
\frac{\sum_{i=1}^{n} w_i\,\delta_i}
     {\sum_{i=1}^{n} w_i},
\qquad
\delta_i \;=\;
\begin{cases}
1, & o_i = e_i\\
0, & \text{otherwise}
\end{cases}
\qquad 0 \le C_{\text{slots}}\le 1
$$

- w - optional weight
- o - observed slot outcome
- e - expected slot value

unless we penalize extra state (i.e. not unknown / init value), this will bie ignored, could instead do:

<!--
$$
\delta_i \;=\;
\begin{cases}
1, & o_i = e_i
\\
-1, & o_i \neq e_i%%o_i \neq e_i \; \text{and} \; o_i \neq unknown
\\
% 0, & \text{otherwise}
\end{cases}
$$
-->

$$
\delta_i \;=\;
\begin{cases}
1, & o_i = e_i
\\
0, & o_i = unknown
\\
-1, & \text{otherwise}
\end{cases}
$$

The extra case only punished extra / incorrect state, while missing one is ignored.

##### Fuse Outcome accuracy with Slot accuracy

Instead evaluating both subsets separately we could combine them and **apply different weights**. I.e. Outcomes could use a stronger weight, and metadata could use a lower one (since they are mandatory anyway), but how to determine the values for weights?

- we could use the jaccard index both for task completion rate and slot accuracy, however, for different reasons.
  - For Task completion we would use it because we have multiple outcome variables that can imply other outcomes (T-CPR -> RD2). We could avoide this by implementing an ordering of Outcomes (T-CPR = Urgency > RD2 > RD1) so the highest order is considered first (pairwise comparison) and so on
  - For Slot Accuracy we can use it since we want to reach an outcome with the minimum number of state variables, no less and no more. The Jaccard Index provides that if unknown values are filtered first.

#### Jaccard Index

$$
C_{\text{Jaccard}}
\;=\;
\frac{\bigl|\,E_k \cap O_k \,\bigr|}
     {\bigl|\,E_k  \cup O_k \,\bigr|},
\qquad 0 \le C_{\text{Jaccard}}\le 1
$$

- E - Expected State for known values (True / False)
- O - Observed State for known values (True / False)

Compare sets as whole, need to filter unknown values as a pre-step in the implementation. Missing or unnecessary state will be penalized equally.

---

### “Dialogue efficiency” ([pdf](zotero://open-pdf/library/items/7ZMUD9W6?page=5&annotation=TLQQCUWF))

“effective metric is the **turn number**” ([pdf](zotero://open-pdf/library/items/7ZMUD9W6?page=5&annotation=R58Y9827))

Problem: LLM can reduce Turn number by guessing too much correctly \
Solution: Punish too few turns as well to account for model hallucinations. / Measure deviation from expected turn number

Variables:

- ETC = Expected Turn Count
  - Normally the ETC is equal to the expected number of known state variables (perfect dialogue)
  - This can deviate based on the scenario!
- RTC = Recorderd Turn count
- Additional: Classify errounous systems turns:
  - Clarifying questions (error occurred before, due to vague question/answer or unsuccessufull state mapping)
  - Questions for state, not in the ground truth **Needs Implementation**
    - we need to pair questions to state for this, would make sense for other reasons as well, eg. autocompletion of minimum necessary state, if Name, location is still missing after reaching an outcome state
    - we could also just exclude those variables from the score (they are mandatory anyway), might make some scenarios unusable
- Variations:
  - Turns‑per‑slot (MultiWOZ benchmarking scripts)
  - Turns until fail (PARADISE)






## Appendix

### other Metrics

#### Turn-Based metrics

##### Slot Accuracy

$$ SA_t = \frac{T-M_t-W_t}{T} $$

- T: total number of predefined slots
- M_t: number of missed slots
- W_t: number of wrongly predicted slots

##### Joint Goal Accuracy

$$ JGA = \begin{cases}     1 & \text{if predicted state = gold state},\\     0 & \text{otherwise}. \end{cases} $$

##### Slot accuracy definition (Chat-GPT)

Definition:

$$\begin{align} m_{j,s} &= f_s\!\bigl(p_{j,s},\,g_{j,s}\bigr)\in[0,1] \tag{1}\\[6pt] \end{align}$$

$$\begin{align} \text{SA}(j) &=   \frac{\displaystyle\sum_{s\in S_j} w_s\,m_{j,s}}        {\displaystyle\sum_{s\in S_j} w_s} \tag{2}\\[10pt] \end{align}$$

$$\begin{align} \text{SlotAccuracy} &=   \frac{\displaystyle\sum_{j\in D}\sum_{s\in S_j} w_s\,m_{j,s}}        {\displaystyle\sum_{j\in D}\sum_{s\in S_j} w_s} \tag{3} \end{align}$$

$S_j$ & Set of slots in record $j$\
$g_{j,s}$ & Gold (ideal) value of slot $s$ in record $j$\
$p_{j,s}$ & Predicted value of slot $s$ in record $j$\
$m_{j,s}$ & Slot-level match score in $[0,1]$ (eq.,(1))\
$w_s$ & Non-negative weight for slot $s$ (default $1$)\
$\text{SA}(j)$ & Per-record slot accuracy (eq.,(2))\
\text{SlotAccuracy} & Corpus-level slot accuracy (eq.,(3))\
$f_s(\cdot)$ & Type-specific matching function (exact, tolerant, similarity, …)

“Takanobu, R., Zhu, Q., Li, J., Peng, B., Gao, J., Huang, M.: Is your goal-oriented dialog model performing really well? empirical analysis of system-wise evaluation. In: Proceedings of the 21st SIGDial” ([pdf](zotero://open-pdf/library/items/7ZMUD9W6?page=12&annotation=E4JWBLYA))
