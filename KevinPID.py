import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation
from IPython import display

# Constants
initX, initY = 0, 0
targetX, targetY = 20, 0
currentHeading = 0
tolerance = 0.1
Kp_lin = 50
Ki_lin = 0.001
Kd_lin = 2


numOfFrames = 120
dt = 50 

currentPos = [initX, initY]
targetPos = [targetX, targetY]


def kevin_is_homo(currentPos, currentHeading, targetPos, Kp_lin, linMax=127):
  target_x = targetPos[0]
  error = target_x - currentPos[0]
  linVel = error * Kp_lin

  return linVel, 0

fig = plt.figure()
trajectory_lines = plt.plot([], '--', color='black')
trajectory_line = trajectory_lines[0]
heading_lines = plt.plot([], '-', color='red')
heading_line = heading_lines[0]
poses = plt.plot([], 'o', color='orange', markersize=10)
pose = poses[0]
rect_lines_1 = plt.plot([], '-', color='orange')
rect_lines_2 = plt.plot([], '-', color='orange')
rect_lines_3 = plt.plot([], '-', color='orange')
rect_lines_4 = plt.plot([], '-', color='orange')
rect_line_1 = rect_lines_1[0]
rect_line_2 = rect_lines_2[0]
rect_line_3 = rect_lines_3[0]
rect_line_4 = rect_lines_4[0]

# Graph Setup
plt.plot([initX, targetX], [initY, targetY], 'x',color='red',markersize=10)
plt.axis("scaled")
plt.xlim (min([initX, targetX])-5, max([initX, targetX])+5)
plt.ylim (min([initY, targetY])-5, max([initY, targetY])+5)

xs = [currentPos[0]]
ys = [currentPos[1]]

f = 0

def draw_square (length, center, orientation):
  global rect_line_1, rect_line_2, rect_line_3, rect_line_4
  corner1 = [center[0] + length/np.sqrt(2)*np.cos((orientation + 45) *pi/180), center[1] + length/np.sqrt(2)*np.sin((orientation + 45) *pi/180)]
  corner2 = [center[0] + length/np.sqrt(2)*np.cos((orientation + 135) *pi/180), center[1] + length/np.sqrt(2)*np.sin((orientation + 135) *pi/180)]
  corner3 = [center[0] + length/np.sqrt(2)*np.cos((orientation - 135) *pi/180), center[1] + length/np.sqrt(2)*np.sin((orientation - 135) *pi/180)]
  corner4 = [center[0] + length/np.sqrt(2)*np.cos((orientation - 45) *pi/180), center[1] + length/np.sqrt(2)*np.sin((orientation - 45) *pi/180)]
  rect_line_1.set_data([corner1[0], corner2[0]], [corner1[1], corner2[1]])
  rect_line_2.set_data([corner2[0], corner3[0]], [corner2[1], corner3[1]])
  rect_line_3.set_data([corner3[0], corner4[0]], [corner3[1], corner4[1]])
  rect_line_4.set_data([corner4[0], corner1[0]], [corner4[1], corner1[1]])


def robot_animation (frame) :
  global currentPos, currentHeading, f

  linearVel, turnVel = kevin_is_homo(currentPos, currentHeading, targetPos, Kp_lin)

  if f < 40:
    linearVel, turnVel = 0, 0


  maxLinVelfeet = 200 / 60 * pi*4 / 12
  maxTurnVelDeg = 200 / 60 * pi*4 / 9 *180/pi


  leftSideVel = linearVel - turnVel
  rightSideVel = linearVel + turnVel
  stepDis = (leftSideVel + rightSideVel)/100 * maxLinVelfeet * dt/1000


  currentPos[0] += stepDis * np.cos(currentHeading*pi/180)
  currentPos[1] += stepDis * np.sin(currentHeading*pi/180)
  currentHeading += (rightSideVel - leftSideVel)/2/100 * maxTurnVelDeg * dt/1000

  currentHeading = currentHeading%360
  if currentHeading < 0: currentHeading += 360

  xs.append(currentPos[0])
  ys.append(currentPos[1])

  draw_square(3, currentPos, currentHeading)
  heading_line.set_data ([currentPos[0], currentPos[0] + 0.75*np.cos(currentHeading/180*pi)], [currentPos[1], currentPos[1] + 0.75*np.sin(currentHeading/180*pi)])
  pose.set_data ((currentPos[0], currentPos[1]))
  trajectory_line.set_data (xs, ys)
  f += 1

anim = animation.FuncAnimation (fig, robot_animation, frames = numOfFrames, interval = dt)
plt.show()