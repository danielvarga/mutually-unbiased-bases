( for i in triplets.couts/triplet_mub.*.cout ; do echo -n "$i " ; tail -1 < $i ; done ) > triplets.finishes
cat triplets.finishes | grep "e-28" | cut -f1 -d' ' | cut -f3 -d '.' | awk '{ print "triplets/triplet_mub_" $0 ".npy" }' > triplets.bests
cat triplets.finishes | grep "e-20" | cut -f1 -d' ' | cut -f3 -d '.' | awk '{ print "triplets/triplet_mub_" $0 ".npy" }' > triplets.e-20
cat triplets.finishes | grep "e-2." | cut -f1 -d' ' | cut -f3 -d '.' | awk '{ print "triplets/triplet_mub_" $0 ".npy" }' > triplets.e-2x


time ls cout | while read f ; do echo -n "$f " ; tail -1 < cout/$f ; done > cout.line_ends
# -> 3 mins
cat cout.line_ends | grep -v termin | grep "e-2." | awk '{ print "triplets/" $1 }' | sed "s/triplet_mub\./triplet_mub_/" | sed "s/cout/npy/" > triplets.e-2x


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
