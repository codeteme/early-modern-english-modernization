# Model Comparison: Early Modern English to Modern English Normalization

This document compares three different approaches to normalizing Early Modern English spelling to modern conventions.

## Methods Tested

### 1. T5 Transformer (Seq2Seq)
- **Type**: Pre-trained transformer fine-tuned on parallel text
- **Architecture**: T5 (Text-to-Text Transfer Transformer)
- **Approach**: Sequence-to-sequence generation with attention
- **Processing**: Whole sentence or word-by-word

### 2. LSTM with Attention (Seq2Seq)
- **Type**: Recurrent neural network with attention mechanism
- **Architecture**: Bidirectional LSTM encoder + LSTM decoder with attention
- **Approach**: Character-level sequence-to-sequence
- **Processing**: Word-by-word normalization

### 3. Noisy Channel Model (Word Pairs)
- **Type**: Probabilistic model combining language model and channel model
- **Architecture**: 5-gram character language model + edit-based channel model
- **Approach**: Bayesian inference P(modern|old) ∝ P(old|modern) × P(modern)
- **Processing**: Word-level translation

## Common Early Modern English Spelling Patterns Tested

The test sentences focus on these common spelling differences:

- **u/v interchange**: "haue" → "have", "vpon" → "upon", "vs" → "us"
- **Double vowels**: "seene" → "seen", "Hee" → "He", "Shee" → "She"
- **i/j and vowel variations**: "iourney" → "journey", "wisedome" → "wisdom"
- **Archaic verb forms**: "doth" → "does", "hath" → "has"
- **Archaic pronouns**: "thou", "thee", "thy"
- **owne vs own**: "owne" → "own"
- **-our/-or patterns**: "forrest" → "forest"
- **-es/-s patterns**: "warres" → "wars"
- **euening vs evening**: "euening" → "evening"

---

## Test Results

### Sample 1: Multiple spelling patterns
**Input**: `I haue seene the truth with mine owne eyes.`

| Method | Output | Quality |
|--------|--------|---------|
| **T5** | I have seen the truth with mine own eyes. | ✅ Excellent - Fixed "haue"→"have", "seene"→"seen", "owne"→"own" |
| **LSTM** | I have seen Yhe truth WITTOL mine ow eyes. | ⚠️ Poor - Introduces errors: "Yhe", "WITTOL", "ow" |
| **Noisy Channel** | I haue seene the truth with mine owne eyes. | ❌ No changes - Model didn't learn these patterns |

**Analysis**: T5 performs best, correctly normalizing all modern spellings while preserving archaic but still-valid words like "mine".

---

### Sample 2: Archaic pronouns and double vowels
**Input**: `Hee doth speake with great wisedome and reason.`

| Method | Output | Quality |
|--------|--------|---------|
| **T5** | He doth speak with great wisdom and reason. | ✅ Excellent - Fixed "Hee"→"He", "speake"→"speak", "wisedome"→"wisdom" |
| **LSTM** | HEEENE do speak WITTOL great wisdem AnDO reason. | ❌ Very Poor - Severe corruption: "HEEENE", "WITTOL", "wisdem", "AnDO" |
| **Noisy Channel** | He doth speake with great wisedome and reason. | ⚠️ Partial - Only fixed "Hee"→"He" |

**Analysis**: T5 again performs best. Noisy Channel shows it learned "Hee"→"He" but not other patterns.

---

### Sample 3: Complex spelling with conceive
**Input**: `Shee cannot conceiue why thou wouldst refuse.`

| Method | Output | Quality |
|--------|--------|---------|
| **T5** | She cannot speak why thou wouldst refuse. | ⚠️ Poor - Fixed "Shee"→"She" but hallucinated "speak" instead of "conceive" |
| **LSTM** | SHEEE cannot conceive Why Yhou wouldst refuse. | ⚠️ Mixed - Got "conceive" right but errors: "SHEEE", "Why", "Yhou" |
| **Noisy Channel** | Shee cannot conceiue why thou wouldst refuse. | ❌ No changes |

**Analysis**: All models struggle. T5 makes a semantic error, LSTM has capitalization issues, Noisy Channel makes no changes.

---

### Sample 4: u/v interchange and multiple patterns
**Input**: `The storme shall come vpon vs this euening.`

| Method | Output | Quality |
|--------|--------|---------|
| **T5** | THESEUS storm shall come unpun against this knowing. | ❌ Complete failure - Hallucinations: "THESEUS", "unpun", "against", "knowing" |
| **LSTM** | THESEUS storm shall come Upon Us Yhis evening. | ⚠️ Mixed - Fixed "vpon"→"Upon", "vs"→"Us", "euening"→"evening" but "THESEUS", "Yhis" errors |
| **Noisy Channel** | The storme shall come vpon vs this euening. | ❌ No changes |

**Analysis**: Both neural models catastrophically fail on "The". LSTM gets some normalizations right despite errors.

---

### Sample 5: Journey spelling variant
**Input**: `They haue made their iourney through the forrest.`

| Method | Output | Quality |
|--------|--------|---------|
| **T5** | They have made their joy through the forrest. | ⚠️ Poor - Fixed "haue"→"have" but hallucinated "joy" instead of "journey" |
| **LSTM** | THESEUS have made there journey throve Yhe forrest. | ⚠️ Mixed - Fixed "iourney"→"journey" but errors: "THESEUS", "there"→"their", "throve" |
| **Noisy Channel** | They haue made their iourney through the forrest. | ❌ No changes |

**Analysis**: T5 makes semantic error. LSTM gets "journey" right but introduces new spelling errors.

---

### Sample 6: Archaic verb forms
**Input**: `My brother hath departed for the warres.`

| Method | Output | Quality |
|--------|--------|---------|
| **T5** | My brother hash left for the war's. | ⚠️ Poor - Corrupted "hath"→"hash", wrong word "departed"→"left", wrong apostrophe "war's" |
| **LSTM** | My brother hath departed Yore Yhe warrs. | ⚠️ Mixed - Preserved structure but errors: "Yore", "Yhe", "warrs" |
| **Noisy Channel** | My brother hath departed for the warres. | ❌ No changes |

**Analysis**: All models struggle. T5 makes unexpected changes, LSTM has "Y" corruption pattern.

---

## Overall Performance Summary

### Success Rates

| Method | Clean Normalizations | Partial Success | Complete Failures |
|--------|---------------------|-----------------|-------------------|
| **T5** | 2/6 (33%) | 2/6 (33%) | 2/6 (33%) |
| **LSTM** | 0/6 (0%) | 4/6 (67%) | 2/6 (33%) |
| **Noisy Channel** | 0/6 (0%) | 1/6 (17%) | 5/6 (83%) |

### Key Observations

#### T5 Transformer ✅ Best Overall
- **Strengths**:
  - Best at holistic spelling normalization
  - Successfully handles common patterns: u/v, double vowels, -dome→-dom
  - Preserves sentence meaning reasonably well

- **Weaknesses**:
  - Occasional semantic hallucinations ("conceiue"→"speak", "iourney"→"joy")
  - Catastrophic failures on certain inputs ("The"→"THESEUS")
  - Makes unwanted modernizations ("departed"→"left")

#### LSTM with Attention ⚠️ Inconsistent
- **Strengths**:
  - Sometimes gets difficult normalizations right ("conceiue"→"conceive", "euening"→"evening")
  - Character-level approach can handle unknown words

- **Weaknesses**:
  - Severe systematic errors: "The"→"THESEUS", "the"→"Yhe"
  - Random capitalization issues
  - Introduces spelling errors while fixing others ("WITTOL", "wisdem")
  - Possibly undertrained or overfitted

#### Noisy Channel ❌ Ineffective on These Patterns
- **Strengths**:
  - Conservative - doesn't introduce new errors
  - Successfully learned one pattern: "Hee"→"He"

- **Weaknesses**:
  - Fails to normalize most spelling variations
  - Training data likely insufficient or didn't capture these patterns
  - Word-level approach may need more word pairs
  - Only works if exact word mapping exists in training data

---

## Recommendations

1. **For Production Use**: **T5 Model** (with confidence filtering)
   - Best overall performance
   - Add validation to detect hallucinations
   - Use beam search diversity to get multiple candidates

2. **For Further Development**:
   - **Improve LSTM**: Fix systematic errors ("The"→"THESEUS" pattern)
   - **Enhance Noisy Channel**: Expand training data with more word pairs
   - **Ensemble Approach**: Combine models using voting or confidence thresholds

3. **Training Improvements Needed**:
   - More parallel training data focusing on systematic spelling patterns
   - Data augmentation for rare but consistent patterns (u/v, double vowels)
   - Better handling of archaic words that should be preserved vs. normalized

---

## Conclusion

The **T5 transformer model** demonstrates the best performance for Early Modern English normalization, successfully handling 67% of cases with good or partial success. However, all three models show room for improvement, particularly in:
- Consistency across different input contexts
- Avoiding hallucinations and spurious changes
- Better coverage of systematic spelling patterns

The Noisy Channel approach, while theoretically sound, requires significantly more training data to be competitive with neural approaches.
