rm -r pos/
rm -r neg/

mkdir pos/
mkdir neg/

for f in aclImdb/train/pos/*
do
    cp "$f" "pos/a_${f#"aclImdb/train/pos/"}"
done
for f in aclImdb/train/neg/*
do
    cp "$f" "neg/a_${f#"aclImdb/train/neg/"}"
done
for f in aclImdb/test/pos/*
do
    cp "$f" "pos/b_${f#"aclImdb/test/pos/"}"
done
for f in aclImdb/test/neg/*
do
    cp "$f" "neg/b_${f#"aclImdb/test/neg/"}"
done


