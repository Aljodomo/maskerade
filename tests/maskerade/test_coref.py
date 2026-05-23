from maskerade.coref import find_coref_clusters

def test_find_coref_clusters_alice_smith():
    text = "Alice Smith lives in Berlin. She prefers to be called Ms. Smith."
    clusters = find_coref_clusters(text)
    
    # Verify that "Alice Smith" and "She" are resolved as coreferent,
    # but "Ms. Smith" is not grouped into the cluster by the model.
    assert clusters == [['Alice Smith', 'She']]
