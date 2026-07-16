"""Generation stage: synthesis + citation over retrieval's ranked, clustered chunks.

RetrievalOutput -> GeneratedAnswer, via five phases (prompt_builder, synthesizer,
claim_parser, validator, formatter). Inform, not advise — never produces buy/sell/hold
calls; every surviving claim is grounded and cited. See README.md.
"""
