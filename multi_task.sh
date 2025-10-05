bash tools/dist_test.sh   configs/gres/InstanceVG-grefcoco.py 1 --load-from  work_dir/gres/InstanceVG-grefcoco/InstanceVG-grefcoco.pth 
bash tools/dist_test.sh  configs/refcoco/InstanceVG-B-refcoco/InstanceVG-B-refcoco.py  1 --load-from  work_dir/refcoco/InstanceVG-B-refcoco/InstanceVG-B-refcoco.pth
bash tools/dist_test.sh  configs/refcoco/InstanceVG-L-refcoco/InstanceVG-L-refcoco.py  1 --load-from  work_dir/refcoco/InstanceVG-L-refcoco/InstanceVG-L-refcoco.pth
bash tools/dist_test.sh  configs/refzom/InstanceVG-refzom.py  1 --load-from  work_dir/refzom/InstanceVG-refzom/InstanceVG-refzom.pth
bash tools/dist_test.sh  configs/rrefcoco/InstanceVG-rrefcoco.py  1 --load-from  work_dir/rrefcoco/InstanceVG-rrefcoco/InstanceVG-rrefcoco.pth