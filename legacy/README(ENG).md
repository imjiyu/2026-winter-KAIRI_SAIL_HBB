## Semantic-Logic Inconsistency Functions
This dataset probes whether a model answers by (A) function-name semantics (semantic prior) or (B) the actual function body (implementation/execution), and is designed to surface the conflict circuit where these two signals compete in the residual stream.

## Emotions
This dataset is designed to observe how a preceding situation forms a residual direction inside the model and constrains the emotion choice that follows “I feel …”.  
The key goal is to identify a **situation inference circuit** that activates before any explicit emotion word appears and biases the emotion slot.  
By doing so, we analyze where and how situation-based signals combine with emotion-lexical circuits to determine the final output.

## Matching event&year
This dataset probes whether large language models contain **circuits that retrieve years associated with well-known events.**
It investigates how specific events (keys) are internally linked to specific years (values).
By intervening on year-related representations, we test whether modifying stored values causally changes the model’s output.

## chemical formula
This dataset examines how a model decides whether to express the same knowledge as a word or as a symbolic formula.
It also observes which internal circuits follow format instructions and switch the output representation accordingly.

## Typo Correlation
This dataset evaluates whether a model can retrieve correct knowledge despite spelling errors in the query.
Without explicit typo-correction instructions, it probes spontaneous normalization and entity recovery circuits during knowledge retrieval.

## Commonsense
This dataset evaluates whether a model can distinguish real-world facts using statements that contradict basic commonsense (world knowledge).
Each sentence requires a True/False judgment and can be used to analyze commonsense-based reasoning circuits.

## Social Bias
This dataset examines whether the model interprets socially biased statements as fairness judgment problems rather than factual queries.
Its goal is to analyze the involvement of normative reasoning circuits when responding to generalized claims about specific groups.

## Harmful Actions Refusal
This dataset analyzes the internal mechanisms by which the model chooses refusal over knowledge generation when faced with clearly harmful or illegal requests.
It focuses on identifying where and when harmfulness signals suppress normal knowledge retrieval.

## Jailbreak
This dataset evaluates whether the model can recognize and block indirect or obfuscated harmful requests (jailbreaks) framed as fictional, educational, or hypothetical scenarios.
The goal is to reveal circuits that detect mismatches between surface meaning and underlying intent.