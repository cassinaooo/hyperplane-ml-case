from normalize_vendor import normalize_vendor
import numpy as np
import pandas as pd
from tqdm import tqdm

"""
OBJECTIVE

Compute and persist the trigram similarities between all vendors. 

"""

### DEPENDS ON:
# %%
fname = "./data/01-parsed-amounts-and-dates.parquet"

### GENERATES:
vendors_similarities_output_fname = "./data/vendors_similarities_trigrams_only.pkl"

# %%
df = pd.read_parquet(fname)

# %%
df.head()

# %%
vendors_str = list(set(df.vendor))
len(vendors_str)

# %%
for round in range(5):
    print("|", "| |".join(np.random.choice(vendors_str, 5)), "|", sep="")


# %%
samples = 10

# search_filter = list(filter(lambda x: "-" in x, vendors_str))

for s in np.random.choice(vendors_str, samples):
    print(s, "->", normalize_vendor(s))

# %%
norm_vendors_str = sorted(
    list(set(filter(lambda x: len(x) > 0, map(normalize_vendor, vendors_str))))
)

# %%
len(norm_vendors_str)

# %%
len(norm_vendors_str) / len(vendors_str)

# %% [markdown]
# #### Approach 1
#
# Use simple n-gram similarity

# %%
import nltk


def extract_trigrams(s):
    return set(["".join(tri) for tri in nltk.trigrams(s)])


# %%
print(extract_trigrams(np.random.choice(norm_vendors_str)))

# %%
vendors_ngrams = {}

for vendor in tqdm(norm_vendors_str, total=len(norm_vendors_str)):
    trigrams = extract_trigrams(vendor)

    vendors_ngrams[vendor] = {
        "trigrams": trigrams,
    }

# %%
samples = 1

for s in np.random.choice(norm_vendors_str, samples):
    print(s, "->", vendors_ngrams[s]["trigrams"])

# %% [markdown]
# ### n-grams distance matrix (bag of n-grams)


# %%
# non-commutative
def set_similarity(a, b):
    return 1 - len(a - b) / len(a)


# %%
samples = 5

for i in range(samples):
    a, b = np.random.choice(norm_vendors_str, 2)
    print(
        a,
        "|",
        b,
        "->",
        set_similarity(vendors_ngrams[a]["trigrams"], vendors_ngrams[b]["trigrams"]),
    )


# %%
def compute_similarities(vendor_1, cuttoff=0.5):
    result = {"vendor_name": vendor_1, "others": []}

    for vendor_2 in norm_vendors_str:
        if vendor_1 != vendor_2:
            sim = set_similarity(
                vendors_ngrams[vendor_1]["trigrams"],
                vendors_ngrams[vendor_2]["trigrams"],
            )
            # small memory optimization (pls send help)
            if sim > cuttoff:
                result["others"].append(
                    {
                        "vendor_name": vendor_2,
                        "sim": sim,
                    }
                )

    return result


# %%
similarities = []
# this is O(scary)
for vendor in tqdm(norm_vendors_str, total=len(norm_vendors_str)):
    similarities.append(compute_similarities(vendor))

# %%
samples = 5

for vendor in np.random.choice(similarities, samples):
    print(vendor["vendor_name"])
    print(sorted(vendor["others"], key=lambda x: -x["sim"])[:3])

# %%
import pickle as pkl

# %%
with open(vendors_similarities_output_fname, "wb") as f:
    pkl.dump(similarities, f)

# %%
