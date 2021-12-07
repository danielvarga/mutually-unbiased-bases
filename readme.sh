
unzip -q quantum.zip
cd quantum
( ls *.cout | while read f ; do echo -n "$f " ; tail -1 $f ; done ) > tails_names
cat tails_names | awk '(sqrt(($3-0.083)^2)<0.01)' | cut -f1 -d' ' | sort -n | awk '{ printf("%03d.npy\n", $1) }' > optima


ls quantum/[0-9]*.cout | while read f ; do echo -n "$f " ; cat $f | awk '($1==19000) { a = $2 }  ($1==20000) { b = $2 }  END { print a-b, a, b }' ; done > foo
cat foo | awk '(($3<0.0831668) && ($3>0))' | sort -k4 -g > optima_filtered
cat optima_filtered | cut -f1 -d' ' | cut -f2 -d'/' | cut -f1 -d'.' | sort -n | awk '{ printf("quantum/mub_%03d.npy\n", $1) }' > optima_filenames


python normal_form.py quantum/mub_120.npy | tr '234' '.bc'
python normal_form.py quantum/mub_120.npy mub_120_normal.npy

python reverse.py optimum.npy
# -> works
python reverse.py mub_120_normal.npy
# -> works, but it has a row structure different from what i assumed
python reverse.py mub_300_normal.npy
# -> does not work, verify_constraints(x, y) fails

python reverse.py mub_120_normal.npy
python reverse.py mub_120_normal.npy mub_120_verynormal.npy
# -> generated code output copied to reverse_symbolic.py,
#    and at the same time put it into normal form mub_120_verynormal.npy


# product_angles() with print commented in:
python vis.py mub_120_normal.npy | tr ' ' '\t' > mub_120_polars.txt
# tells us that the product elements have nothing much in common, except for two facts:
# - what we already know, that magnitudes only have 3 possible values,
# - that each exact product element appears 3 times, all three times in the same matrix product:
cat mub_120_polars.txt | cut -f1,2,5,6 | sort | uniq -c | wc -l
36
# 36 x 3 = 108

# we normalize everything we have:
time cat optima_filtered | cut -f1 -d' ' | sed "s/\//\/mub_/" | sed "s/cout/npy/" | while read f ; do fnormed=`echo $f | sed "s/^quantum/normalized/"` ; python normal_form.py $f $fnormed > /dev/null ; done
# -> for some reason there are files not found, but we still collect 854 quasi-MUBs.

# let's check a bunch of them, what are the parameters of their Fourier bases?
# not too many, because most of them have a
# different zrow or graph structure, and they just
# bury the console with assert failed messages.
ls normalized/* | head | while read f ; do echo $f ; python reverse.py $f | awk 'f;/Okay/{f=1}' ; done

# Here's three reverse engineered:
# we can only tell values up to 180 degrees rotations.
normalized/mub_100.npy
a       b
-60+b   60-a+b
-60-a-b 60-a
a = -55.1187968716524
b = -0.0005514574849350036

normalized/mub_120.npy
120+c    d
60+c     c-d
120+d-c  120+d
c = 0.00750938849902
d = 4.885233682186406

# this one is weird, it has just one free parameter:
optimum.npy
a   a
0   a
60  120+a
a = 55.11852114360842
