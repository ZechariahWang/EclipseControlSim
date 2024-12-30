import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation
from IPython import display
# hi lol

path1 = [[0.0, 0.0], [0.571194595265405, -0.4277145118491421], [1.1417537280142898, -0.8531042347260006], [1.7098876452457967, -1.2696346390611464], [2.2705328851607995, -1.6588899151216996], [2.8121159420106827, -1.9791445882187304], [3.314589274316711, -2.159795566252656], [3.7538316863009027, -2.1224619985315876], [4.112485112342358, -1.8323249172947023], [4.383456805594431, -1.3292669972090994], [4.557386228943757, -0.6928302521681386], [4.617455513800438, 0.00274597627737883], [4.55408382321606, 0.6984486966257434], [4.376054025556597, 1.3330664239172116], [4.096280073621794, 1.827159263675668], [3.719737492364894, 2.097949296701878], [3.25277928312066, 2.108933125822431], [2.7154386886417314, 1.9004760368018616], [2.1347012144725985, 1.552342808106984], [1.5324590525923942, 1.134035376721349], [0.9214084611203568, 0.6867933269918683], [0.30732366808208345, 0.22955002391894264], [-0.3075127599907512, -0.2301742560363831], [-0.9218413719658775, -0.6882173194028102], [-1.5334674079795052, -1.1373288016589413], [-2.1365993767877467, -1.5584414896876835], [-2.7180981380280307, -1.9086314914221845], [-3.2552809639439704, -2.1153141204181285], [-3.721102967810494, -2.0979137913841046], [-4.096907306768644, -1.8206318841755131], [-4.377088212533404, -1.324440752295139], [-4.555249804461285, -0.6910016662308593], [-4.617336323713965, 0.003734984720118972], [-4.555948690867849, 0.7001491248072772], [-4.382109193278264, 1.3376838311365633], [-4.111620918085742, 1.8386823176628544], [-3.7524648889185794, 2.1224985058331005], [-3.3123191098095615, 2.153588702898333], [-2.80975246649598, 1.9712114570096653], [-2.268856462266256, 1.652958931009528], [-1.709001159778989, 1.2664395490411673], [-1.1413833971013372, 0.8517589252820573], [-0.5710732645795573, 0.4272721367616211], [0, 0], [0.571194595265405, -0.4277145118491421]]
# path1 = [[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.5, 0.5], [0.6, 0.6], [0.7, 0.7], [0.8, 0.8], [0.9, 0.9], [1, 1], [1.1, 1.1], [1.2, 1.2], [1.3, 1.3], [1.4, 1.4], [2, 2], [4, 6]]

x_state_estimate = 0
x_error_covariance = 1
x_process_noise_covariance = 0.01
x_measurement_noise_covariance = 0.1

x_predicted_value = 0
x_predicted_p = 0

y_state_estimate = 0
y_error_covariance = 1
y_process_noise_covariance = 0.01
y_measurement_noise_covariance = 0.1

y_predicted_value = 0 
y_predicted_p = 0


filtered_x = 0
filtered_y = 0

def update_x_components():
  global x_state_estimate 
  global x_error_covariance 
  global x_process_noise_covariance 
  global x_measurement_noise_covariance 

  global x_predicted_value 
  global x_predicted_p

  x_state_estimate = 0
  x_error_covariance = 1


def x_prediction_step():
    global x_state_estimate 
    global x_error_covariance 
    global x_process_noise_covariance 
    global x_measurement_noise_covariance 

    global x_predicted_value 
    global x_predicted_p

    x_predicted_value = x_state_estimate
    x_predicted_p = x_error_covariance + x_measurement_noise_covariance


def x_update_filter_step(raw_position):
    global x_state_estimate 
    global x_error_covariance 
    global x_process_noise_covariance 
    global x_measurement_noise_covariance 

    global x_predicted_value 
    global x_predicted_p

    residual = raw_position - x_predicted_value
    gain = x_predicted_p / (x_predicted_p + x_measurement_noise_covariance)
    x_state_estimate = x_predicted_value + gain * residual
    x_error_covariance = (1 - gain) * x_predicted_p
    return x_state_estimate


def update_y_components():
    global y_state_estimate 
    global y_error_covariance 
    global y_process_noise_covariance 
    global y_measurement_noise_covariance 

    global y_predicted_value 
    global y_predicted_p 
    global y_state_estimate 
    global y_error_covariance 
    y_state_estimate = 0
    y_error_covariance = 1


def y_prediction_step():
    global y_state_estimate 
    global y_error_covariance 
    global y_process_noise_covariance 
    global y_measurement_noise_covariance 

    global y_predicted_value 
    global y_predicted_p 
    global y_state_estimate 
    global y_error_covariance 

    y_predicted_value = y_state_estimate
    y_predicted_p = y_error_covariance + y_measurement_noise_covariance
    return y_predicted_p


def y_update_filter_step(raw_position):
    global y_state_estimate 
    global y_error_covariance 
    global y_process_noise_covariance 
    global y_measurement_noise_covariance 

    global y_predicted_value 
    global y_predicted_p 
    global y_state_estimate 
    global y_error_covariance 

    residual = raw_position - y_predicted_value
    gain = y_predicted_p / (y_predicted_p + y_measurement_noise_covariance)
    y_state_estimate = y_predicted_value + gain * residual
    y_error_covariance = (1 - gain) * y_predicted_p
    return y_state_estimate



def pt_to_pt_distance (pt1,pt2):
    distance = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
    return distance

def sgn (num):
  if num >= 0:
    return 1
  else:
    return -1

currentPos = [0, 0]
currentHeading = 330
lastFoundIndex = 0
lookAheadDis = 0.4
linearVel = 50

using_rotation = False
numOfFrames = 400

def pure_pursuit_step (path, currentPos, currentHeading, lookAheadDis, LFindex) :
  global filtered_x
  global filtered_y
  currentX = currentPos[0]
  currentY = currentPos[1]
  lastFoundIndex = LFindex
  intersectFound = False
  startingIndex = lastFoundIndex

  for i in range (startingIndex, len(path)-1):
    plt.plot(filtered_x, filtered_y, marker="o", markersize=5, markerfacecolor="green")
    x1 = path[i][0] - currentX
    y1 = path[i][1] - currentY
    x2 = path[i+1][0] - currentX
    y2 = path[i+1][1] - currentY
    dx = x2 - x1
    dy = y2 - y1
    dr = math.sqrt (dx**2 + dy**2)
    D = x1*y2 - x2*y1
    discriminant = (lookAheadDis**2) * (dr**2) - D**2


    if discriminant >= 0:
      sol_x1 = (D * dy + sgn(dy) * dx * np.sqrt(discriminant)) / dr**2
      sol_x2 = (D * dy - sgn(dy) * dx * np.sqrt(discriminant)) / dr**2
      sol_y1 = (- D * dx + abs(dy) * np.sqrt(discriminant)) / dr**2
      sol_y2 = (- D * dx - abs(dy) * np.sqrt(discriminant)) / dr**2

      sol_pt1 = [sol_x1 + currentX, sol_y1 + currentY]
      sol_pt2 = [sol_x2 + currentX, sol_y2 + currentY]

      minX = min(path[i][0], path[i+1][0])
      minY = min(path[i][1], path[i+1][1])
      maxX = max(path[i][0], path[i+1][0])
      maxY = max(path[i][1], path[i+1][1])

      if ((minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY)) or ((minX <= sol_pt2[0] <= maxX) and (minY <= sol_pt2[1] <= maxY)):

        foundIntersection = True
        if ((minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY)) and ((minX <= sol_pt2[0] <= maxX) and (minY <= sol_pt2[1] <= maxY)):
          if pt_to_pt_distance(sol_pt1, path[i+1]) < pt_to_pt_distance(sol_pt2, path[i+1]):
            goalPt = sol_pt1
          else:
            goalPt = sol_pt2

        else:
          if (minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY):
            goalPt = sol_pt1
          else:
            goalPt = sol_pt2

        if pt_to_pt_distance (goalPt, path[i+1]) < pt_to_pt_distance ([currentX, currentY], path[i+1]):
          lastFoundIndex = i
          break
        else:
          lastFoundIndex = i + 1
      else:
        foundIntersection = False
        goalPt = [path[lastFoundIndex][0], path[lastFoundIndex][1]]

      x_prediction_step()
      y_prediction_step()

      filtered_x = x_update_filter_step(currentPos[0])
      filtered_y = y_update_filter_step(currentPos[1])
      print("filtered x: " + str(filtered_x))
      print("filtered y " + str(filtered_y))
      print("real x: " + str(currentPos[0]))
      print("real y: " + str(currentPos[1]))


  Kp = 3
  absTargetAngle = math.atan2 (goalPt[1]-currentPos[1], goalPt[0]-currentPos[0]) *180/pi
  if absTargetAngle < 0: absTargetAngle += 360

  turnError = absTargetAngle - currentHeading
  if turnError > 180 or turnError < -180 :
    turnError = -1 * sgn(turnError) * (360 - abs(turnError))
  
  turnVel = Kp * turnError
  
  return goalPt, lastFoundIndex, turnVel


pi = np.pi
fig = plt.figure()
trajectory_lines = plt.plot([], '-', color='blue', linewidth = 4)
trajectory_line = trajectory_lines[0]
heading_lines = plt.plot([], '-', color='red')
heading_line = heading_lines[0]
connection_lines = plt.plot([], '-', color='green')
connection_line = connection_lines[0]
poses = plt.plot([], 'o', color='black', markersize=10)
pose = poses[0]

pathForGraph = np.array(path1)
plt.plot(pathForGraph[:, 0], pathForGraph[:, 1], '--', color='grey')
plt.plot(pathForGraph[:, 0], pathForGraph[:, 1], 'o', color='black', markersize=5)
plt.plot(filtered_x, filtered_y, marker="o", markersize=0.2, markerfacecolor="green")

plt.axis("scaled")
plt.xlim (-3, 6)
plt.ylim (-4, 4)
dt = 50
xs = [currentPos[0]]
ys = [currentPos[1]]

update_x_components()
update_y_components()

def pure_pursuit_animation (frame) :
  global currentPos
  global currentHeading
  global lastFoundIndex
  global linearVel

  if lastFoundIndex >= len(path1)-2 : lastFoundIndex = 0

  goalPt, lastFoundIndex, turnVel = pure_pursuit_step (path1, currentPos, currentHeading, lookAheadDis, lastFoundIndex)

  maxLinVelfeet = 200 / 60 * pi * 4 / 12
  maxTurnVelDeg = 200 / 60 * pi * 4 / 9 * (180 / pi)

  stepDis = linearVel/100 * maxLinVelfeet * dt/1000
  currentPos[0] += stepDis * np.cos(currentHeading*pi/180)
  currentPos[1] += stepDis * np.sin(currentHeading*pi/180)

  heading_line.set_data ([currentPos[0], currentPos[0] + 0.5*np.cos(currentHeading/180*pi)], [currentPos[1], currentPos[1] + 0.5*np.sin(currentHeading/180*pi)])
  connection_line.set_data ([currentPos[0], goalPt[0]], [currentPos[1], goalPt[1]])

  currentHeading += turnVel/100 * maxTurnVelDeg * dt/1000
  if using_rotation == False :
    currentHeading = currentHeading%360
    if currentHeading < 0: currentHeading += 360

  xs.append(currentPos[0])
  ys.append(currentPos[1])

  pose.set_data ((currentPos[0], currentPos[1]))
  trajectory_line.set_data (xs, ys)


anim = animation.FuncAnimation (fig, pure_pursuit_animation, frames = numOfFrames, interval = 50)
plt.show()