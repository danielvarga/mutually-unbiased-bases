mkdir -p cout
mkdir -p triplets
for ((i=0; i<10000; ++i))
do
    ipad=$(printf "%05d\n" $i)
    echo $ipad
    CUDA_VISIBLE_DEVICES= python search_triplets.py $i > cout/triplet_mub.$ipad.cout
done
