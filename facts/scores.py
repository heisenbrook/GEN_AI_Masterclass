import os
import numpy as np
import textwrap
import matplotlib.pyplot as plt
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# ======================VISUALIZING EMBEDDINGS======================
# python function to calculate L2 distance between embeddings of 
# different words/sentences to find similarities. Plots a matrix to visualize
# the L2 metric in action
# ==================================================================

# Utils functions

def calculate_L2(v1, v2):
    return np.linalg.norm(v1 - v2)**2

def wrap_labels(labels, width):
    return ['\n'.join(textwrap.wrap(label, width)) for label in labels]

def plot(data, words):
    fig, ax = plt.subplots()
    ax.imshow(data, cmap='Blues')

    labels = wrap_labels(words, 30)
    ax.set_xticks(np.arange(len(words)), labels=labels)
    ax.set_yticks(np.arange(len(words)), labels=labels)

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="left",
            rotation_mode="anchor")

    for i in range(len(words)):
        for j in range(len(words)):
            text = ax.text(j, i, round(data[i, j], 2),
                        ha="center", va="center")

    fig.tight_layout()
    plt.savefig('scores.png')



# Load API KEY
load_dotenv('/workspaces/GEN_AI_Masterclass/.env')
print('COURSE_KEY present?', os.getenv('COURSE_KEY') is not None)

api_key = os.getenv('COURSE_KEY')

# Embeddings
# ===================

embeddings = OpenAIEmbeddings(
    api_key=api_key
)

# Plotting examples

words = [
    "The happy child jumped bravely from rock to rock",
    "The child was not timid and had a good time jumping from rock to rock",
    "Although filled with great fear, the child jumped from rock to rock",
]

embs = [
    np.array(embeddings.embed_query(word)) for word in words
]

data = np.array([
    [calculate_L2(e1,e2) for e1 in embs] for e2 in embs
])

plot(data, words)