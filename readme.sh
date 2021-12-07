
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


# implemented the Haagerup trick, can tell if two bases are equivalent.
# verified that F(x, y) ~ F(y, x) and F(x, y) ~ F(plusminus x, plusminus y)

# let's inspect a couple of bases.
# code still fails if mub does not have appropriate combinatorical structure, zrow and stuff.
# now let's present bases in this normal form:
# degrees between 0 and 180 (mirrored otherwise), and the two parameters are sorted.

# the first short awk only keeps lines after the line matching /Okay/.
# the second discards filename printing for files that did not produce output.
ls normalized/* | head -100 | while read f ; do echo $f ; python reverse.py $f 2> /dev/null | awk 'f;/Okay/{f=1}' ; done | awk '/npy$/ { l = $0 }  /^basis/ { if ($2==1) { print l ; print } else { print } }'

optimum.npy
basis 1 fourier_params_in_degrees 55.118521143608426 55.11852114363142 haagerup_distance 1.799560500614924e-12
basis 2 fourier_params_in_degrees 55.11852114360783 179.99999999997635 haagerup_distance 1.5929366233982608e-12
basis 3 fourier_params_in_degrees 60.00000000002455 175.11852114363128 haagerup_distance 2.4594215553008708e-12
normalized/mub_100.npy
basis 1 fourier_params_in_degrees 0.0005514574849350037 55.1187968716524 haagerup_distance 4.014721294667324e-13
basis 2 fourier_params_in_degrees 60.000551457485095 115.11824541416748 haagerup_distance 5.567767316530352e-13
basis 3 fourier_params_in_degrees 115.11879687165253 175.11824541416752 haagerup_distance 3.9923615222136125e-13
normalized/mub_10006.npy
basis 1 fourier_params_in_degrees 64.88793828463234 119.9870819221207 haagerup_distance 4.2510396244810345e-13
basis 2 fourier_params_in_degrees 59.987081922120666 115.12497979324696 haagerup_distance 4.285493401118279e-13
basis 3 fourier_params_in_degrees 115.11206171536766 175.12497979324704 haagerup_distance 3.7692006633893715e-13
normalized/mub_1003.npy
basis 1 fourier_params_in_degrees 59.997177875439576 64.88006781268074 haagerup_distance 4.147793762100671e-13
basis 2 fourier_params_in_degrees 60.00282212456034 64.88288993724116 haagerup_distance 3.987507210274216e-13
basis 3 fourier_params_in_degrees 55.11711006275879 55.11993218731921 haagerup_distance 3.3190114610663624e-13
normalized/mub_10088.npy
basis 1 fourier_params_in_degrees 120.007927962045 175.1145570159659 haagerup_distance 4.227735782985631e-13
basis 2 fourier_params_in_degrees 59.99207203795493 115.1224849780109 haagerup_distance 4.779226060042108e-13
basis 3 fourier_params_in_degrees 115.11455701596596 175.12248497801096 haagerup_distance 4.036212553418239e-13
normalized/mub_10091.npy
basis 1 fourier_params_in_degrees 64.88016000854286 120.00263772814168 haagerup_distance 3.6326516339273135e-13
basis 2 fourier_params_in_degrees 115.11983999145723 175.11720226331548 haagerup_distance 4.041205575473078e-13
basis 3 fourier_params_in_degrees 60.002637728141686 115.11720226331548 haagerup_distance 4.764522923330489e-13
normalized/mub_10141.npy
basis 1 fourier_params_in_degrees 64.88138879031048 175.11843107739793 haagerup_distance 4.442002728101066e-13
basis 2 fourier_params_in_degrees 60.00018013229164 175.11861120968956 haagerup_distance 4.329800373217753e-13
basis 3 fourier_params_in_degrees 60.00018013229169 64.88156892260213 haagerup_distance 4.0674404389571834e-13
normalized/mub_10227.npy
basis 1 fourier_params_in_degrees 59.996143353743314 64.87955056795164 haagerup_distance 3.869691568049449e-13
basis 2 fourier_params_in_degrees 55.11659278579166 179.99614335374335 haagerup_distance 3.623616706173449e-13
basis 3 fourier_params_in_degrees 55.11659278579171 55.12044943204834 haagerup_distance 5.550847325189179e-13
normalized/mub_10257.npy
basis 1 fourier_params_in_degrees 59.99537789829308 115.12083214463246 haagerup_distance 3.6481923168171776e-13
basis 2 fourier_params_in_degrees 0.0046221017069178714 55.11621004292554 haagerup_distance 5.245425946896753e-13
basis 3 fourier_params_in_degrees 4.883789957074434 64.87916785536753 haagerup_distance 3.7644908891520945e-13
normalized/mub_10295.npy
basis 1 fourier_params_in_degrees 4.8838259781184465 115.12086816254357 haagerup_distance 4.979341049725361e-13
basis 2 fourier_params_in_degrees 115.12086816254372 120.00469414066197 haagerup_distance 5.894738907066821e-13
basis 3 fourier_params_in_degrees 0.004694140662026676 124.88382597811832 haagerup_distance 4.123172614965698e-13
