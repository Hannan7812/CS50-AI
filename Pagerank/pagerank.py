import os
import random
import re
import sys
import numpy as np

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # Create a dictionary which we will return
    return_dict = dict()
    # Edge Case: If it links to no page then choose among any with equal probability
    if len(corpus[page]) == 0:
        for x in corpus:
            return_dict[x] = 0
        for page, prob in return_dict.items():
            prob += 1 / len(return_dict)
        return return_dict
    # Iterate over all the pages that the current page links to and add them to the return dictionary
    for x in corpus[page]:
        return_dict[x] = 0
    # Add the probability for each page
    for x in return_dict:
        return_dict[x] += damping_factor / len(corpus[page])
    # Iterate over dictionary again and add all the missing pages to the return dictionary
    for x in corpus:
        if x not in return_dict:
            return_dict[x] = 0
    # Increase the probability for each because of random choosing
    for x in return_dict:
        return_dict[x] += (1 - damping_factor) / len(corpus)
    # Return the dictionary
    return return_dict


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    distribution = dict()
    for page in corpus:
        distribution[page] = 0
    pagef = random.choice(list(corpus.keys()))
    for i in range(1, n):
        current_distribution = transition_model(corpus, pagef, damping_factor)
        for page in distribution:
            distribution[page] = ((i-1) * distribution[page] + current_distribution[page]) / i

        pagef = random.choices(list(distribution.keys()), list(distribution.values()), k=1)[0]
    return distribution 


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    dictionary = dict()
    for x in corpus.keys():
        dictionary[x] = 0
    
        N = len(dictionary)
    for x in dictionary:
        dictionary[x] = 1 / N
    threshold = 0.0005

    while True:
        count = 0
        for x in dictionary:
            adding_fac1 = (1 - damping_factor) / N
            adding_fac2 = 0
            for y in corpus:
                if x in corpus[y]:
                    adding_fac2 += dictionary[y] / len(corpus[y])
            adding_fac2 = damping_factor * adding_fac2
            to_add = adding_fac1 + adding_fac2
            if abs(dictionary[x] - to_add) < threshold:
                count += 1
            dictionary[x] = to_add
        if count == N:
            break
    return dictionary


if __name__ == "__main__":
    main()
