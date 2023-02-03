for i in `seq 50`
do
  roslaunch nav_cloning nav_cloning_pytorch.launch
  sleep 30
done
