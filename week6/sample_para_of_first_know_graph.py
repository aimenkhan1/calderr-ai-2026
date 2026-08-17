"""
sample_paragraphs.py — offline fallback dataset for the knowledge graph lab

These 20 short paragraphs are written from general knowledge (not copied
from Wikipedia or any other source) so the lab can run fully offline in
any environment. They cover an interconnected cluster of real people,
places, and institutions on purpose — Einstein, Curie, the Nobel Prize,
Princeton — so the resulting graph actually has shared entities linking
paragraphs together, which is what makes a "knowledge graph" interesting
to look at instead of 20 disconnected islands.

In real use, replace this with actual paragraphs fetched live from
Wikipedia via fetch_wikipedia_paragraphs() in knowledge_graph.py.
"""

SAMPLE_PARAGRAPHS = [
    "Albert Einstein was a theoretical physicist born in Germany in 1879. "
    "He is best known for developing the theory of relativity, one of the "
    "two pillars of modern physics alongside quantum mechanics.",

    "Einstein won the Nobel Prize in Physics in 1921 for his explanation "
    "of the photoelectric effect. The award is presented annually by the "
    "Royal Swedish Academy of Sciences in Stockholm.",

    "In 1933, Einstein moved to the United States to escape political "
    "persecution in Germany. He joined the Institute for Advanced Study "
    "in Princeton, New Jersey, where he worked until his death in 1955.",

    "Before moving to Princeton, Einstein studied at ETH Zurich in "
    "Switzerland, where he trained as a physicist and mathematics teacher.",

    "Marie Curie was a physicist and chemist born in Warsaw, Poland in "
    "1867. She later moved to Paris to continue her studies at the "
    "Sorbonne University.",

    "Curie conducted pioneering research on radioactivity together with "
    "her husband, Pierre Curie. Their work led to the discovery of two "
    "new chemical elements, polonium and radium.",

    "In 1903, Marie Curie won the Nobel Prize in Physics, sharing the "
    "award with Pierre Curie and Henri Becquerel for their research on "
    "radioactivity.",

    "Marie Curie later won a second Nobel Prize, this time in Chemistry "
    "in 1911, making her the first person to win Nobel Prizes in two "
    "different scientific fields.",

    "The Curie Institute in Paris was founded partly through funding "
    "connected to Marie Curie's research, and it remains an important "
    "center for cancer research today.",

    "Princeton University is a private research university located in "
    "Princeton, New Jersey. It is one of the oldest universities in the "
    "United States, founded in 1746.",

    "The Institute for Advanced Study, though located near Princeton "
    "University, is a separate institution. It was founded in 1930 as an "
    "independent center for theoretical research.",

    "The Nobel Prize was established through the will of Alfred Nobel, a "
    "Swedish chemist and inventor of dynamite. The first prizes were "
    "awarded in 1901.",

    "Alfred Nobel was born in Stockholm, Sweden in 1833. He held patents "
    "in many countries and made much of his fortune through the "
    "manufacture of explosives.",

    "The theory of relativity developed by Einstein is divided into two "
    "parts: special relativity, published in 1905, and general "
    "relativity, published in 1915.",

    "General relativity describes gravity not as a force but as a "
    "curvature of space and time caused by mass and energy. It replaced "
    "earlier explanations proposed by Isaac Newton.",

    "Isaac Newton was an English physicist and mathematician who lived in "
    "the 17th and 18th centuries. His laws of motion and universal "
    "gravitation dominated physics for over two hundred years.",

    "Radioactivity, the phenomenon studied extensively by Marie Curie, "
    "refers to the spontaneous emission of radiation from unstable atomic "
    "nuclei.",

    "Pierre Curie was a French physicist who worked closely with his wife "
    "Marie Curie. He died in a street accident in Paris in 1906, before "
    "she received her second Nobel Prize.",

    "Poland, where Marie Curie was born, was under the control of the "
    "Russian Empire during much of the 19th century, which limited "
    "educational opportunities for women like Curie.",

    "Germany, where Einstein was born, later became the site of major "
    "political upheaval in the 1930s that led many scientists, including "
    "Einstein, to emigrate to other countries such as the United States.",
]