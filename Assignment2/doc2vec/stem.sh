for f in data/pos/*; do
    python PorterStemmer.py "$f" >> stemmed_data/pos/"${f#data/POS}"
done

for f in data/neg/*; do
    python PorterStemmer.py "$f" >> stemmed_data/neg/"${f#data/NEG}"
done

