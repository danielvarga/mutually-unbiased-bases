( for i in triplets.couts/triplet_mub.*.cout ; do echo -n "$i " ; tail -1 < $i ; done ) > triplets.finishes
cat triplets.finishes | grep "e-28" | cut -f1 -d' ' | cut -f3 -d '.' | awk '{ print "triplets/triplet_mub_" $0 ".npy" }' > triplets.bests
cat triplets.finishes | grep "e-20" | cut -f1 -d' ' | cut -f3 -d '.' | awk '{ print "triplets/triplet_mub_" $0 ".npy" }' > triplets.e-20
cat triplets.finishes | grep "e-2." | cut -f1 -d' ' | cut -f3 -d '.' | awk '{ print "triplets/triplet_mub_" $0 ".npy" }' > triplets.e-2x


# verifying if every MUB-triplet is two Fouriers with a Szollosi transfer (plus sporadic ones)
git checkout 8814a59
# on geforce2:
time ls cout | while read f ; do echo -n "$f " ; tail -1 < cout/$f ; done > cout.line_ends
# -> 3 mins
cat cout.line_ends | grep -v termin | grep "e-2." | awk '{ print "triplets/" $1 }' | sed "s/triplet_mub\./triplet_mub_/" | sed "s/cout/npy/" > triplets.e-2x
cat triplets.e-2x | while read f ; do echo -n "$f " ; python hadamard_cube.py $f ; echo ; done > is_fourier.cout 2> is_fourier.cerr
# -> the above is now is_fourier.sh
nohup bash is_fourier.sh &
cat is_fourier.cout | grep -v "^$" | grep -v Szollosi | grep -v sporadic
# -> triplets/triplet_mub_52295.npy triplets/triplet_mub_57638.npy 
scp triplets/triplet_mub_52295.npy triplets/triplet_mub_57638.npy hexagon.renyi.hu:./ai-shared/daniel/mub/
# also added to git under data.




git checkout 5daa92f
python search_cubes.py 1
mv cube_00001.npy data/cube_00001.just_hadamards.npy


python search_cubes.py 4
mv cube_00004.npy data/cube_00004.1d_slices_summing_to_1.npy 


git checkout c126edd
python search_cubes.py 1
for ((i=5; i<1000; ++i)) ; do python search_cubes.py $i > couts.two_1d_slices/$i.cout 2> /dev/null ; done
# -> cubes.two_1d_slices/cube_*.two_1d_slices_summing_to_1.npy
#    turns out this is not really what we need


mkdir straight_triplets
mkdir straight_cubes
mkdir canonized_cubes
cat ./data/classify.couts.all | grep -v WTF | cut -f2 -d' ' | cut -f3 -d'_' | cut -f1 -d'.' | head -100 | while read code ; do echo $code ; python canonize_cube.py $code mat ; done
cat ./data/classify.couts.all | grep -v WTF | head -100 > mub_sample_100.txt
zip -r mub_sample_100.zip mub_sample_100.txt straight_triplets straight_cubes canonized_cubes
scp -P 2820 mub_sample_100.zip hexagon.renyi.hu:./ai-shared/daniel/mub/
mv mub_sample_100.zip mub_sample_100.txt straight_triplets straight_cubes canonized_cubes matlab-sample-100
mkdir straight_triplets
mkdir straight_cubes
mkdir canonized_cubes
cat ./data/classify.couts.all | grep -v WTF | cut -f2 -d' ' | cut -f3 -d'_' | cut -f1 -d'.' | while read code ; do echo $code ; python canonize_cube.py $code npy ; done
# -> that takes 17 hours, let's see when i will stop it.
