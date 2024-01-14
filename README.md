roslaunch tbot3_dqn stage.launch

conda activate py27
roslaunch tbot3_dqn result_graph.launch

conda activate py27
roslaunch tbot3_dqn dqn_1_train.launch
roslaunch tbot3_dqn dqn_2_train.launch
roslaunch tbot3_dqn ddqn_train.launch

roslaunch tbot3_dqn agent_test.launch 
