"""
Semantic Search CLI

Builds a semantic search tool over 100 sentences.
Compares all-MiniLM-L6-v2 vs BAAI/bge-small-en models.
Visualizes embeddings using 2D PCA.

Run: python Lab_3.1.py

load model->create sentences->embeds all sentences->stores in vector space ->pca visualise->user enters query->generate query embedding->pca of query on same->calculate cosine similarlity across all sentences->sort similarity score->select top_k most similar->display
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


#100 sentences covering 10 topics (10 sentences per topic)

SENTENCES = [
    # Science
    "The theory of relativity was developed by Albert Einstein.",
    "DNA carries the genetic information of living organisms.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "The speed of light in a vacuum is approximately 299792 km per second.",
    "Black holes are regions of spacetime where gravity is extremely strong.",
    "The human brain contains approximately 86 billion neurons.",
    "Quantum mechanics describes the behavior of particles at atomic scale.",
    "The periodic table organizes chemical elements by atomic number.",
    "Evolution by natural selection was proposed by Charles Darwin.",
    "The Big Bang theory describes the origin of the universe.",

    # Technology
    "Artificial intelligence enables machines to simulate human intelligence.",
    "Machine learning algorithms improve through experience and data.",
    "The internet connects billions of devices worldwide.",
    "Blockchain is a decentralized ledger technology.",
    "Python is a popular programming language for data science.",
    "Neural networks are inspired by the structure of the human brain.",
    "Cloud computing delivers services over the internet.",
    "Cybersecurity protects systems from digital attacks.",
    "Smartphones have transformed how people communicate.",
    "Electric vehicles are powered by rechargeable batteries.",

    # History
    "World War II ended in 1945 with the surrender of Germany and Japan.",
    "The Roman Empire was one of the largest empires in ancient history.",
    "The French Revolution began in 1789 and transformed French society.",
    "Christopher Columbus reached the Americas in 1492.",
    "The Industrial Revolution began in Britain in the late 18th century.",
    "The Berlin Wall fell in 1989 marking the end of the Cold War.",
    "The Egyptian pyramids were built as tombs for pharaohs.",
    "The Renaissance was a period of cultural and artistic rebirth in Europe.",
    "Mahatma Gandhi led India to independence through nonviolent resistance.",
    "The Apollo 11 mission landed humans on the Moon in 1969.",

    # Geography
    "The Amazon River is the largest river in the world by discharge.",
    "Mount Everest is the highest mountain above sea level on Earth.",
    "The Sahara Desert is the largest hot desert in the world.",
    "The Pacific Ocean is the largest and deepest ocean on Earth.",
    "Australia is both a country and a continent.",
    "The Great Barrier Reef is the worlds largest coral reef system.",
    "The Nile River flows northward through northeastern Africa.",
    "Antarctica is the coldest and driest continent on Earth.",
    "The Alps are the highest mountain range in Europe.",
    "Brazil is the largest country in South America.",

    # Biology
    "Cells are the basic structural and functional units of life.",
    "Mitochondria are known as the powerhouse of the cell.",
    "Viruses are microscopic agents that replicate inside living cells.",
    "The immune system protects the body from infections and disease.",
    "Mammals are warm-blooded vertebrates that nurse their young.",
    "Ecosystems consist of living organisms and their physical environment.",
    "Bacteria are single-celled microorganisms found everywhere on Earth.",
    "The human heart pumps blood throughout the body.",
    "Genes are segments of DNA that encode proteins.",
    "Plants produce oxygen through the process of photosynthesis.",

    # Physics
    "Newtons laws of motion describe the relationship between force and motion.",
    "Energy cannot be created or destroyed only transformed.",
    "Gravity is the force that attracts objects with mass toward each other.",
    "Electricity is the flow of electric charge through a conductor.",
    "Atoms are the smallest units of ordinary matter.",
    "Sound travels as a wave through air and other materials.",
    "Light behaves as both a wave and a particle.",
    "Temperature measures the average kinetic energy of particles.",
    "Magnetism is a force produced by moving electric charges.",
    "Radioactivity is the spontaneous emission of particles from atomic nuclei.",

    # Economics
    "Supply and demand determine the price of goods in a market.",
    "Inflation refers to the general increase in prices over time.",
    "Gross domestic product measures the economic output of a country.",
    "Interest rates influence borrowing and spending in an economy.",
    "Unemployment occurs when people who want jobs cannot find them.",
    "International trade allows countries to specialize in production.",
    "Stock markets allow companies to raise capital from investors.",
    "Taxes are compulsory payments collected by governments.",
    "Microeconomics studies individual consumer and firm behavior.",
    "Central banks control money supply and interest rates.",

    # Sports
    "Football is the most popular sport in the world.",
    "The Olympic Games bring together athletes from around the world.",
    "Cricket is especially popular in South Asia and England.",
    "Basketball was invented by James Naismith in 1891.",
    "Tennis is played on grass clay and hard court surfaces.",
    "The FIFA World Cup is held every four years.",
    "Swimming is both a competitive sport and a recreational activity.",
    "Marathon running covers a distance of 42.195 kilometers.",
    "Chess is a strategic board game played by millions worldwide.",
    "Formula One is the highest class of single-seater auto racing.",

    # Art and Culture
    "The Mona Lisa was painted by Leonardo da Vinci.",
    "Shakespeare wrote 37 plays and 154 sonnets.",
    "Jazz music originated in New Orleans in the early 20th century.",
    "The Louvre in Paris is the worlds largest art museum.",
    "Architecture combines art and engineering to design buildings.",
    "Cinema was invented in the late 19th century.",
    "Classical music refers to Western art music from 1750 to 1820.",
    "Literature encompasses written works including fiction and poetry.",
    "Photography captures images using light on a sensitive surface.",
    "Dance is a form of artistic expression through body movement.",

    # Health
    "Regular exercise improves cardiovascular health and mental wellbeing.",
    "A balanced diet provides essential nutrients for the body.",
    "Sleep is essential for physical and cognitive restoration.",
    "Vaccines prevent infectious diseases by training the immune system.",
    "Mental health is as important as physical health.",
    "Meditation reduces stress and improves focus and clarity.",
    "Diabetes is a condition where blood sugar levels are too high.",
    "Cancer is caused by uncontrolled growth of abnormal cells.",
    "Antibiotics are used to treat bacterial infections.",
    "Hygiene practices reduce the spread of infectious diseases.",
]

# Topic label for each sentence (10 per topic)
TOPICS = (
    ["Science"] * 10 + ["Technology"] * 10 + ["History"] * 10 +
    ["Geography"] * 10 + ["Biology"] * 10 + ["Physics"] * 10 +
    ["Economics"] * 10 + ["Sports"] * 10 + ["Art"] * 10 + ["Health"] * 10
)

TOPIC_COLORS = {
    "Science":    "#e74c3c",
    "Technology": "#3498db",
    "History":    "#f39c12",
    "Geography":  "#2ecc71",
    "Biology":    "#9b59b6",
    "Physics":    "#1abc9c",
    "Economics":  "#e67e22",
    "Sports":     "#34495e",
    "Art":        "#e91e63",
    "Health":     "#00bcd4",
}


#loading models

def load_models():

    print("Loading all-MiniLM-L6-v2...")
    model_mini = SentenceTransformer("all-MiniLM-L6-v2")
    print("Loading BAAI/bge-small-en...")
    model_bge  = SentenceTransformer("BAAI/bge-small-en")
    print("Both models ready.")
    return model_mini, model_bge


#embedding all sentences with a given model and returning the embeddings as a numpy array

def embed_all(model, model_name):
 
    print(f"\nEmbedding 100 sentences with {model_name}...")
    embeddings = model.encode(SENTENCES, show_progress_bar=True)
    print(f"Done. Shape: {embeddings.shape}")
    return embeddings


#semantic search function->it takes a query string, embeddings of sentences, the model used for embedding, and the number of top results to return. It returns a list of tuples containing the most similar sentences, their similarity scores, and their corresponding topics.

def search(query, embeddings, model, top_k=5):

    query_vec = model.encode([query])
    scores    = cosine_similarity(query_vec, embeddings)[0]
    top_idx   = scores.argsort()[::-1][:top_k]

    return [
        (SENTENCES[i], round(float(scores[i]), 4), TOPICS[i])
        for i in top_idx
    ]


#pca visualization->it enables to visualize the embeddings in 2D space using PCA and saves the plot as a PNG file.

def visualize_pca(embeddings, model_name):

    print(f"\nGenerating PCA visualization for {model_name}...")

    pca     = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    plt.figure(figsize=(14, 9))
    plt.style.use("dark_background")

    # Plot each topic with its color
    seen_topics = set()
    for i, (x, y) in enumerate(reduced):
        topic = TOPICS[i]
        color = TOPIC_COLORS[topic]
        label = topic if topic not in seen_topics else None
        plt.scatter(x, y, c=color, s=40, alpha=0.8, label=label)
        seen_topics.add(topic)

    plt.title(f"Sentence Embeddings — 2D PCA — {model_name}",
              fontsize=14, pad=15)
    plt.xlabel(f"PCA Component 1  (explains {pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    plt.ylabel(f"PCA Component 2  (explains {pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    plt.legend(loc="upper right", fontsize=8, markerscale=1.5)
    plt.tight_layout()

    filename = f"pca_{model_name.replace('/', '_')}.png"
    plt.savefig(filename, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


#compare the results of two models on a sample query    

def compare_models(query, embeddings_mini, embeddings_bge,
                   model_mini, model_bge):
 
    results_mini = search(query, embeddings_mini, model_mini)
    results_bge  = search(query, embeddings_bge,  model_bge)

    print(f"\nQuery: '{query}'")
    print()
    print(f"  {'all-MiniLM-L6-v2':<55}  {'BAAI/bge-small-en'}")
    print(f"  {'-'*55}  {'-'*55}")

    for i in range(5):
        m_text  = results_mini[i][0][:52]
        m_score = results_mini[i][1]
        b_text  = results_bge[i][0][:52]
        b_score = results_bge[i][1]
        print(f"  {m_text:<52} {m_score:.4f}   |   {b_text:<52} {b_score:.4f}")


#it run cli and allow user to write query find similar sentences,compare models and quit 

def run_cli(embeddings_mini, embeddings_bge, model_mini, model_bge):
    print()
    print()
    print("SEMANTIC SEARCH CLI - Lab 3.1")
    print()
    print("Commands:")
    print("  Type any query  → search with MiniLM model")
    print("  'compare'       → compare both models on a sample")
    print("  'quit'          → exit")
    print()

    while True:
        try:
            query = input("Query: ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break

        if not query:
            continue

        if query.lower() in ["quit", "exit", "q"]:
            print("Goodbye.")
            break

        if query.lower() == "compare":
            sample = "how does the human body fight disease"
            compare_models(sample, embeddings_mini, embeddings_bge,
                           model_mini, model_bge)
            continue

        # Search with MiniLM
        print()
        print(f"Results — all-MiniLM-L6-v2:")
        for i, (sentence, score, topic) in enumerate(
            search(query, embeddings_mini, model_mini), 1
        ):
            print(f"  {i}. [{score:.4f}] ({topic}) {sentence}")

        # Search with BGE
        print()
        print(f"Results — BAAI/bge-small-en:")
        for i, (sentence, score, topic) in enumerate(
            search(query, embeddings_bge, model_bge), 1
        ):
            print(f"  {i}. [{score:.4f}] ({topic}) {sentence}")
        print()


#main entry point

if __name__ == "__main__":

    # Load both models
    model_mini, model_bge = load_models()

    # Embed all sentences
    emb_mini = embed_all(model_mini, "all-MiniLM-L6-v2")
    emb_bge  = embed_all(model_bge,  "BAAI/bge-small-en")

    # Show embedding info
    print(f"\nModel comparison:")
    print(f"  all-MiniLM-L6-v2 : {emb_mini.shape} — {emb_mini.shape[1]} dimensions")
    print(f"  BAAI/bge-small-en : {emb_bge.shape}  — {emb_bge.shape[1]} dimensions")

    # Generate PCA visualizations
    visualize_pca(emb_mini, "all-MiniLM-L6-v2")
    visualize_pca(emb_bge,  "BAAI-bge-small-en")

    # Run sample comparisons
    print("\nRunning sample model comparisons...")
    compare_models("space exploration and rockets",
                   emb_mini, emb_bge, model_mini, model_bge)
    compare_models("how does the economy work",
                   emb_mini, emb_bge, model_mini, model_bge)
    compare_models("human body and health",
                   emb_mini, emb_bge, model_mini, model_bge)

    # Start interactive CLI
    run_cli(emb_mini, emb_bge, model_mini, model_bge)

    '''
sample questions

Who developed the theory of relativity?
How do machines learn from data?
When did humans first land on the Moon?
What is the highest mountain in the world?
Which organ pumps blood throughout the human body?
What force pulls objects toward the Earth?
What determines the price of products in a market?
Which sport has the FIFA World Cup every four years?
Who painted the Mona Lisa?
How can I reduce stress and improve focus?
What converts sunlight into chemical energy?
What is blockchain technology?
Who led India to independence through nonviolent resistance?
Which is the largest ocean on Earth?
Why are mitochondria called the powerhouse of the cell?
What is the smallest unit of ordinary matter?
What does GDP measure?
Who invented basketball?
Where did jazz music originate?
Why are vaccines important?
    '''