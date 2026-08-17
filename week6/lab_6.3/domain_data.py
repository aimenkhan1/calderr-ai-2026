"""
domain_data.py

A small, self-contained, FICTIONAL knowledge domain about a made-up tech
ecosystem (people, companies, products, cities). Every fact below is
represented TWICE, in two different shapes:

  1. As a plain-English sentence  -> goes into the VECTOR store (ChromaDB)
  2. As a structured graph edge   -> goes into the KNOWLEDGE GRAPH (NetworkX)

This dual representation is what makes it possible to compare vector search
vs. graph traversal fairly: both are working from the exact same underlying
facts, just accessed in different ways.
"""

# DOCUMENTS — plain text chunks for the vector store.



DOCUMENTS = [
    {"id": "d1", "text": "Aria Solano founded NimbusCloud in 2015."},
    {"id": "d2", "text": "NimbusCloud is headquartered in Austin."},
    {"id": "d3", "text": "NimbusCloud launched its flagship product, CloudSync Pro, in 2017."},
    {"id": "d4", "text": "In 2020, NimbusCloud acquired PixelForge Studios."},
    {"id": "d5", "text": "PixelForge Studios was founded by Marcus Bell in 2012."},
    {"id": "d6", "text": "PixelForge Studios is based in Seattle."},
    {"id": "d7", "text": "PixelForge Studios is known for creating the PixelForge Engine."},
    {"id": "d8", "text": "Marcus Bell left PixelForge Studios in 2019, shortly before its acquisition."},
    {"id": "d9", "text": "Marcus Bell went on to found QuantumLeap Robotics in 2021."},
    {"id": "d10", "text": "QuantumLeap Robotics is headquartered in Denver."},
    {"id": "d11", "text": "QuantumLeap Robotics built a warehouse robot called LeapBot X1."},
    {"id": "d12", "text": "Priya Rao co-founded QuantumLeap Robotics together with Marcus Bell."},
    {"id": "d13", "text": "Before that, Priya Rao worked at Verdant Energy Co as Head of Engineering from 2016 to 2020."},
    {"id": "d14", "text": "Verdant Energy Co is based in Portland."},
    {"id": "d15", "text": "Verdant Energy Co created a smart-grid product called Verdant Grid."},
    {"id": "d16", "text": "In 2022, Verdant Energy Co partnered with Solstice AI to build the Solstice Assistant."},
    {"id": "d17", "text": "Solstice AI was founded by Layla Haddad in 2018."},
    {"id": "d18", "text": "Solstice AI is headquartered in Boston."},
    {"id": "d19", "text": "Tomasz Nowak founded Orbital Dynamics in 2019."},
    {"id": "d20", "text": "Orbital Dynamics is based in Munich."},
    {"id": "d21", "text": "Orbital Dynamics developed a navigation system called Orbital Nav."},
    {"id": "d22", "text": "In 2022, Orbital Dynamics partnered with QuantumLeap Robotics to integrate Orbital Nav into LeapBot X1."},
    {"id": "d23", "text": "Kenji Watanabe joined NimbusCloud as Chief Technology Officer in 2021."},
    {"id": "d24", "text": "Before joining NimbusCloud, Kenji Watanabe worked at PixelForge Studios as a lead engineer from 2014 to 2019."},
]



# GRAPH EDGES — the same facts, structured as (source, relation, target).


GRAPH_EDGES = [
    ("Aria Solano",      "founded",           "NimbusCloud",         {"year": 2015}),
    ("NimbusCloud",      "based_in",          "Austin",              {}),
    ("NimbusCloud",      "created_product",   "CloudSync Pro",       {"year": 2017}),
    ("NimbusCloud",      "acquired",          "PixelForge Studios",  {"year": 2020}),
    ("Marcus Bell",      "founded",           "PixelForge Studios",  {"year": 2012}),
    ("PixelForge Studios", "based_in",        "Seattle",             {}),
    ("PixelForge Studios", "created_product", "PixelForge Engine",   {}),
    ("Marcus Bell",      "left",              "PixelForge Studios",  {"year": 2019}),
    ("Marcus Bell",      "founded",           "QuantumLeap Robotics", {"year": 2021}),
    ("QuantumLeap Robotics", "based_in",      "Denver",              {}),
    ("QuantumLeap Robotics", "created_product", "LeapBot X1",        {}),
    ("Priya Rao",        "co_founded",        "QuantumLeap Robotics", {}),
    ("Priya Rao",        "worked_at",         "Verdant Energy Co",   {"role": "Head of Engineering", "start": 2016, "end": 2020}),
    ("Verdant Energy Co", "based_in",         "Portland",            {}),
    ("Verdant Energy Co", "created_product",  "Verdant Grid",        {}),
    ("Verdant Energy Co", "partnered_with",   "Solstice AI",         {"year": 2022}),
    ("Verdant Energy Co", "created_product",  "Solstice Assistant",  {"joint_with": "Solstice AI"}),
    ("Solstice AI",       "created_product",  "Solstice Assistant",  {"joint_with": "Verdant Energy Co"}),
    ("Layla Haddad",     "founded",           "Solstice AI",         {"year": 2018}),
    ("Solstice AI",      "based_in",          "Boston",              {}),
    ("Tomasz Nowak",     "founded",           "Orbital Dynamics",    {"year": 2019}),
    ("Orbital Dynamics", "based_in",          "Munich",              {}),
    ("Orbital Dynamics", "created_product",   "Orbital Nav",         {}),
    ("Orbital Dynamics", "partnered_with",    "QuantumLeap Robotics", {"year": 2022, "product": "LeapBot X1 navigation"}),
    ("Kenji Watanabe",   "works_at",          "NimbusCloud",         {"role": "CTO", "start": 2021}),
    ("Kenji Watanabe",   "worked_at",         "PixelForge Studios",  {"role": "lead engineer", "start": 2014, "end": 2019}),
]
