"""
demo_dataset.py

The scripted 20-interaction conversation used to demonstrate measurable
improvement . It's a single simulated session with a "work assistant" agent that initially makes 3 different
kinds of repeatable mistakes (wrong timezone, wrong date format, too
verbose), gets corrected on each, and should avoid repeating them for the
rest of the conversation.

  - Interactions 1-5 are deliberately front-loaded with the first-time
    mistakes (no rules exist yet, so nothing can be retrieved) this is
    the "before" window.
  - Interactions 16-20 are a clean stretch with no mistakes -- the "after"
    window. Comparing these two windows is exactly the evaluation
    criterion: "Error rate decreases measurably between interactions 1-5
    and interactions 16-20."
  - Interactions 7-20 deliberately reuse the same 3 underlying situations
    (timezone, date format, brevity) with varied wording each time, to
    show that rule retrieval generalizes beyond exact repeated phrasing --
    not just literal copy-paste of the original mistake.
  - The consolidation mechanism (5 corrections -> 1 high-confidence rule)
    is deliberately NOT forced into this 20-turn script -- it's already
    rigorously demonstrated on its own in the standalone rule_store demo.
    Forcing a 5th correction in here would mean scripting an artificial
    "mistake" the agent didn't actually make, which would undermine the
    honesty of this demo's error-rate numbers.
"""

CONVERSATION = [
    "What time should I schedule the client call, and what timezone should I mention for the deadline?",
    "No, always specify the deadline timezone as EST, not PST.",
    "Can you write today's date at the top of the report?",
    "Always use YYYY-MM-DD format for dates, not MM/DD/YYYY.",
    "Give me a quick answer: is the server deployment done?",
    "From now on, keep answers brief when I ask for something quick.",
    "What timezone should I use for the submission deadline?",
    "What date format should go on the invoice?",
    "Quick one -- are the tests passing?",
    "When is the next deadline and what timezone is that in?",
    "Format the date on this memo, please.",
    "Give me a brief update on the CI pipeline.",
    "Tell me the timezone we use for deadlines.",
    "What date should be printed on the shipping label?",
    "Briefly, did the deployment succeed?",
    "Note the deadline timezone on the shipping form.",
    "What date goes on the compliance certificate?",
    "Quick status: is CI green?",
    "What timezone applies to our contract deadlines?",
    "Quick check: did the migration finish?",
]
