# Derin Q Ağları (DQN) ile Mobil Robotlarda Otonom Hareket
(Autonumous Navigation in Mobile Robot with Deep Q Networks)

Bu çalışmada, Derin Pekiştirmeli Öğrenme yöntemlerinden Derin Q Ağları ile TurtleBot3 mobil robotu eğitilerek Gazebo ortamında otonom hareket planlaması ele alınmaktadır.
<h3>Kullanılan Araçlar:</h3>
<ul>
  <li>Ubuntu 18.04 LTS</li>
  <li>Anaconda python 2.7</li>
  <li>catkin-pkg 0.5.2 (for ROS)</li>
  <li>tensorflow 1.15.0</li>
  <li>keras 2.2.4</li>
  <li>numpy 1.16.5</li>
  <li>pyqt 5.9.2</li>
  <li>ROS Melodic</li>
  <li>Gazebo</li>
  <li>TurtleBot3 Package</li>
</ul>
<h3>Ros Kurulumu:</h3>
<p>sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'</p>
<p>sudo apt install curl</p>
<p>curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -</p>
<p>sudo apt update</p>
<p>sudo apt install ros-melodic-desktop-full</p>
<br>
<p><b>Her yeni kabuk başlatıldığında, ROS ortam değişkenlerinin bash oturumunuza otomatik olarak eklenmesi için</b></p>
<p>echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc</p>
<p>source ~/.bashrc</p>
<br>
<p><b>Bağımlılıkları kurun</b></p>
<p>sudo apt install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential</p>
<p>sudo apt install python-rosdep</p>
<p>sudo rosdep init</p>
<p>rosdep update</p>
<h3>TurtleBot3 Paketlerini Kurun</h3>
<p>$ sudo apt-get install ros-melodic-turtlebot3-msgs</p>
<p>$ sudo apt-get install ros-melodic-turtlebot3</p>
<p>$ echo "export TURTLEBOT3_MODEL=burger" >> ~/.bashrc</p>
<p>mkdir -p ~/catkin_ws/src</p>
<p>cd ~/catkin_ws/src/</p>
<p>
  $ git clone -b melodic-devel https://github.com/ROBOTIS-GIT/DynamixelSDK.git<br>
  $ git clone -b melodic-devel https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git<br>
  $ git clone -b melodic-devel https://github.com/ROBOTIS-GIT/turtlebot3.git<br>
  $ cd ~/catkin_ws && catkin_make<br>
  $ echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
</p>

<h3>Uygulama:</h3>
<p>
terminal penceresi üç bölüme ayrılarak aşağıdaki başlatma dosyaları ile uygulama çalıştırılır.<br><br>
<b>terminal1:</b><br>
"roslaunch tbot3_dqn stage.launch" <br>
gazebo ortamını başlatarak oluşturulan haritayı yükler<br><br>

<b>terminal2:</b><br>
"roslaunch tbot3_dqn dqn_1_train.launch", (Mnih vd., 2013) modeline göre eğitimi başlatır.<br>
"roslaunch tbot3_dqn dqn_2_train.launch", (Mnih vd., 2015) modeline göre eğitimi başlatır.<br>
"roslaunch tbot3_dqn ddqn_train.launch", (V. Hasselt vd., 2016) modeline göre eğitimi başlatır.<br>
"roslaunch tbot3_dqn agent_test.launch", test işlemini başlatır.<br><br>

<b>terminal3:</b><br>
"roslaunch tbot3_dqn result_graph.launch", eğitim ya da test grafiklerini oluşturur.<br><br>
</p>

<h3>Test Videosu</h3>
<a href="https://www.youtube.com/watch?v=wl5inXl6BeM" target="_blank">DQN ve TurtleBot3 ile Otonom Hareket</a>

# Kaynaklar
- Playing Atari with Deep Reinforcement Learning (Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller) (2013)
- Human-level control through deep reinforcement learning (Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D.) (2015)
- Deep Reinforcement Learning with Double Q-learning (Hado van Hasselt, Arthur Guez, David Silver) (2016)
- ROS Packages for TurtleBot3 Machine Learning, <a href="https://github.com/ROBOTIS-GIT/turtlebot3_machine_learning">ROBOTIS Official GitHub</a>
