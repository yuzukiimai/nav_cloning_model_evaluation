for i in `seq 10`
do
  roslaunch nav_cloning nav_cloning_pytorch.launch
  sleep 20
done