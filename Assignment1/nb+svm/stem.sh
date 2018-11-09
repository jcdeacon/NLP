for f in data/POS/*; do
    python PorterStemmer.py "$f" >> stemmed_data/POS/"${f#data/POS}"
done

for f in data/NEG/*; do
    python PorterStemmer.py "$f" >> stemmed_data/NEG/"${f#data/NEG}"
done

