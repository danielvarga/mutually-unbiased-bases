
unzip -q quantum.zip
cd quantum
( ls *.cout | while read f ; do echo -n "$f " ; tail -1 $f ; done ) > tails_names
cat tails_names | awk '(sqrt(($3-0.083)^2)<0.01)' | cut -f1 -d' ' | sort -n | awk '{ printf("%03d.npy\n", $1) }' > optima
