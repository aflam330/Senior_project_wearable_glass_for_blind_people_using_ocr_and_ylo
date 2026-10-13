# A synthetic second currency: not built (2026-10-13)

The request was to make a synthetic Indian rupee counterfeit set (public images, watermark removed, serial numbers duplicated) and to test the watermark localizer and the prefix network on it. It was not built. This note says why, so that the absence is a decision and not an oversight.

## Why it would not measure anything

1. **Both models read physical detail that public images do not have.** The watermark models work on a photograph of a note held against a lamp. Public rupee images are ordinary front-lit scans or photographs; the watermark is not visible in them. "Removing the watermark" from an image that does not show one changes nothing.
2. **A counterfeit made by editing a genuine image is not a counterfeit.** It carries the editing artefact and nothing else. A model trained or tested on it learns to find the edit. The result would be a number about our editing, with the look of a cross-currency result. The same conclusion was reached on 30 September for Taka (`SYNTHETIC_COUNTERFEIT.md`: "stripping security features is not a dataset").
3. **The prefix network needs six close-up views per note in a fixed order.** No public rupee collection has them.
4. **Today's UV result is a warning about simulated negatives.** A simulation was useful there because it exposed a failure (the check accepted printed sheets). It could not have proved a success, and the note on it says so. A synthetic rupee set would be asked to prove a success.

## What can be said without data

The lamp-removal step (quotient image) and the fluorescence-residual step do not depend on the currency: Propositions 2, 3 and 5 of `THEORY_STRONG.md` do not mention Taka. Everything learned from data does depend on it: the genuine portrait of each note value, the watermark window, every threshold. A second currency therefore needs its own photographs.

## What a real second-currency test needs

- Back-lit photographs of genuine notes of the second currency, with a note ID, for the portrait and the localizer (a few hundred notes per value).
- Counterfeit notes of that currency photographed the same way, grouped by print.
- For the prefix network, the same six views per note.

With genuine notes only, one honest experiment is possible: build the portrait of the second currency and report how well the residual separates genuine notes from plain paper and from the other currency's notes. That is a statement about the method's portability, not about counterfeits.
