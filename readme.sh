
unzip -q quantum.zip
cd quantum
( ls *.cout | while read f ; do echo -n "$f " ; tail -1 $f ; done ) > tails_names
cat tails_names | awk '(sqrt(($3-0.083)^2)<0.01)' | cut -f1 -d' ' | sort -n | awk '{ printf("%03d.npy\n", $1) }' > optima


ls quantum/[0-9]*.cout | while read f ; do echo -n "$f " ; cat $f | awk '($1==19000) { a = $2 }  ($1==20000) { b = $2 }  END { print a-b, a, b }' ; done > foo
cat foo | awk '(($3<0.0831668) && ($3>0))' | sort -k4 -g > optima_filtered
cat optima_filtered | cut -f1 -d' ' | cut -f2 -d'/' | cut -f1 -d'.' | sort -n | awk '{ printf("quantum/mub_%03d.npy\n", $1) }' > optima_filenames


python normal_form.py quantum/mub_120.npy | tr '234' '.bc'
python normal_form.py quantum/mub_120.npy mub_120_normal.npy

python reverse.py optimum.py
# -> works
python reverse.py mub_120_normal.npy
# -> works, but it has a row structure different from what i assumed
python reverse.py mub_300_normal.npy
# -> does not work, verify_constraints(x, y) fails

python reverse.py mub_120_normal.npy
# -> generated code output copied to reverse_symbolic.py

