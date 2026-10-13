# SARA: A Wearable Currency and Reading Glass for Blind Users in Bangladesh — System and Registered Study Design

*Draft for an accessibility venue (ASSETS / CHI style), 2026-10-13. **This draft has no user-study results, because no study has been run.** It is written as a system description with a registered study design. It cannot be submitted as a study paper until Section 6 is filled with data. Every system number is from a result file of the project.*

## Abstract

Cash is still the main way to pay in Bangladesh, and banknotes of different values are close in size. A blind person who cannot tell a 500 from a 1,000 Taka note, or who is handed a counterfeit, depends on the honesty of whoever is nearby. We present SARA, a camera glass that names Taka notes, gives a deliberately cautious answer about counterfeits, reads Bangla and English text and names objects, in Bangla and without the internet. We describe the design decisions that follow from one fact: the user cannot check the device's answer. We then give a pre-registered within-subject study design with its power analysis. The study itself has not been run; we report no outcome with users.

## 1. The design problem: a user who cannot verify

A sighted person who is told "this note is counterfeit" can look again. A blind person cannot, and may accuse someone wrongly, in public. Three decisions follow.

1. **The glass never says "counterfeit".** It says "likely genuine" only above a threshold that no validation counterfeit reached, and otherwise "check by hand whether it is genuine". With n = 88 validation counterfeits, a new counterfeit of the same kind passes with probability 1/(n + 1) = 1.1 %. On the test notes, 0 of 88 counterfeits passed. In 1,889 runs of the application the word "counterfeit" was not spoken.
2. **The user cannot promise how many pictures the device gets.** A counterfeit model trained on six views of a note fell to 71.2 % with one view. Trained on a random number of views, it holds 98.2 % with one.
3. **The device says what is wrong with the picture.** For the watermark check the glass asks for the note against a light and rejects dark or blurred frames with a spoken reason ("hold the note closer to the light", "keep the note still"). Of 222 clean back-lit test photographs 213 pass; darkened or defocused copies do not.

## 2. Interaction

- Five buttons found by touch: mode, action, read, volume up, volume down. No screen.
- Capture and reading are separate: ACTION reads what is in view, READ repeats it, so the camera has to be held still only for a moment.
- Every change of state is spoken; a short vibration confirms that a note was found, also in a noisy market.
- Object names are not repeated within four seconds.
- Speech: Bangla sentences stay whole (an earlier version broke them at every comma and digit); a long text is spoken complete and in order; whole pages are spoken line by line while the rest is still being read.
- Several notes in one picture are counted and the total is spoken (`NEW_FEATURES.md`).

## 3. What the device can do today (measured without users)

| Function | Result | Caveat |
|---|---|---|
| Names the Taka note | 91.5 % on 1,536 photographs by other people | 18.5 % on close-ups with fingers over a folded note |
| Counterfeit answer | 0 of 88 test counterfeits passed | On whole notes it confirms 0.4–2 % of genuine notes: it mostly asks for a check by hand |
| Watermark check | 92.6 % on unseen counterfeit prints; 50 ms on the Pi 5 | One counterfeit collection, about 24 unseen prints |
| Reads text | 2.5 % character error on rendered text | 39 % (Bangla) and 62 % (Latin script) on real street words |
| Names objects | mAP@0.5 0.551 on COCO | The 80 COCO classes; no stairs, rickshaws or open drains |
| Speed on the Pi 5 | Currency answer 327 ms; with speech under 3 s | One text capture 18.7 s before the latest speed work; not re-timed |

A first hands-on test through the glass's own camera found currency recognition unreliable and Bangla text unread. That is the state a study would start from, and it has to be repaired first.

## 4. Why a study is needed, and what it cannot be replaced by

Every number above describes a model on photographs. None describes a person. Whether the cautious verdict is trusted or ignored, whether the spoken guidance helps or irritates, whether a user can hold a note in view of a head-worn camera at all: these cannot be measured on a dataset.

## 5. Registered study design

- **Question.** Does spoken capture guidance make the counterfeit check faster or less often inconclusive for blind and low-vision users?
- **Participants.** Blind or low-vision adults who handle cash. Consent in Bangla, read aloud. Ethics approval is required before recruitment and has not been obtained.
- **Design.** Within-subject, two conditions (guided, unguided), order counterbalanced by participant number. The same list of notes for everyone.
- **Primary outcomes.** Denomination accuracy and time per note. The counterfeit policy is scored as how often the glass asks for a human check, not as accuracy of a verdict it does not give.
- **Secondary outcomes.** NASA-TLX workload, System Usability Scale, requests for help, errors corrected on a second attempt.
- **Sample size.** Exact noncentral-t calculation for a paired design at α = 0.05 and 80 % power: 20 participants detect a standardised effect of 0.66, 50 detect 0.40, 105 detect 0.28.
- **Analysis.** Paired difference with 95 % interval and Cohen's d, the participant as the unit. The analysis script prints DATA_NOT_COLLECTED until the data file exists.
- **Recording.** The field log writes every button press, result and timing; a voice questionnaire asks six yes/no questions in Bangla. Voices are stored under a participant number.

## 6. Results with users

NOT COLLECTED. Trial sessions with one visually impaired participant have started; they are not reported here and no conclusion rests on them.

## 7. Risks to participants and how the design answers them

- A wrong "genuine" could cost money; a wrong "counterfeit" could cause a public accusation. The device says neither with certainty: "likely genuine" is advice, and it never says "counterfeit".
- The glass's camera sees bystanders. Pictures are processed on the device and not stored in the offline modes.
- Emotion recognition of other people is used only to name an expression on request; it is not recorded.

## 8. What a reviewer should hold us to

A study paper needs Section 6. Until then this is a system and design document. We would rather submit it late with data than early without.
