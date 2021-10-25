
file=$1

grep '^D' $1 | cut -f3- > ~/tmp/hypo
grep '^T' $1 | cut -f2- > ~/tmp/ref
cat ~/tmp/hypo | sacrebleu ~/tmp/ref