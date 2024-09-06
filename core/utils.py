

def generate_sup_language_list():
    """
    generate support languages list
    """
    mapping = {}
    tsv_path = "639_1toPrefix.tsv"
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            language, prefix = parts[0], parts[1]
            mapping[language] = prefix
    return mapping
