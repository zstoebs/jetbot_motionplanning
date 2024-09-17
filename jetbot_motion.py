from math import pi, atan2, sqrt, degrees
import time
import numpy as np

from IPython.display import display
import ipywidgets
import traitlets

from jetbot import Robot, Camera, bgr8_to_jpeg
import cv2
import numpy as np
import torchvision

mean = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])

normalize = torchvision.transforms.Normalize(mean, stdev)

def preprocess(camera_value):
    global device, normalize
    x = camera_value
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x).float()
    x = normalize(x)
    x = x.to(device)
    x = x[None, ...]
    return x

class JetBotMotion:
    def __init__(self, robot, dt, orient=0):
        self.robot = robot # jetbot robot object
        self.dt = dt
        self.pos = (0,0)
        self.init_orient = orient
        self.orient = orient # orientation relative to the origin
        
        self.speed_gain_slider = ipywidgets.FloatSlider(min=0.0, max=1.0, step=0.01, value=0.1, description='speed gain')
        self.steering_gain_slider = ipywidgets.FloatSlider(min=0.0, max=1.0, step=0.01, value=0.2, description='steering gain')
        self.steering_dgain_slider = ipywidgets.FloatSlider(min=0.0, max=0.5, step=0.001, value=0.0, description='steering kd')
        self.steering_bias_slider = ipywidgets.FloatSlider(min=-0.3, max=0.3, step=0.01, value=0.0, description='steering bias')

        display(self.speed_gain_slider, self.steering_gain_slider, self.steering_dgain_slider, self.steering_bias_slider)
        
        self.x_slider = ipywidgets.FloatSlider(min=-1.0, max=1.0, description='x')
        self.y_slider = ipywidgets.FloatSlider(min=0, max=1.0, orientation='vertical', description='y')
        self.steering_slider = ipywidgets.FloatSlider(min=-1.0, max=1.0, description='steering')
        self.speed_slider = ipywidgets.FloatSlider(min=0, max=1.0, orientation='vertical', description='speed')
        self.sleep_slider = ipywidgets.FloatSlider(min=0.0, max=1.0, step=0.001, value=0.1, description='sleep')

        display(ipywidgets.HBox([self.y_slider, self.speed_slider]))
        display(self.x_slider, self.steering_slider, self.sleep_slider)
    
    def MoveTo(self,x,y,vx,vy):
        x_diff = x - self.pos[0]
        y_diff = y - self.pos[1]
        
        if abs(x_diff) > 0.2 or abs(y_diff) > 0.2: 
            self.x_slider.value = x_diff
            self.y_slider.value = y_diff

            angle = np.arctan2(y_diff, x_diff)
            self.TurnBy(angle - self.orient)
            self.orient = angle

            time.sleep(0.01)

            dist = sqrt(x_diff**2 + y_diff**2) # takes 0.75 seconds to go 40 cm forward
            t = dist*0.75
            self.ForwardFor(t)

            self.pos = (x,y)
            print('Moved to ', self.pos)
        
    
    def MoveWithPIDTo(self,x,y, vx, vy):
        x_diff = x - self.pos[0]
        y_diff = y - self.pos[1]
        self.x_slider.value = x_diff
        self.y_slider.value = y_diff

        self.speed_slider.value = self.speed_gain_slider.value

        angle = np.arctan2(y_diff, x_diff)
        pid = angle * self.steering_gain_slider.value + (angle - self.orient) * self.steering_dgain_slider.value
        self.orient = angle
        print('Turned by angle ', angle)

        self.steering_slider.value = pid + self.steering_bias_slider.value

        self.robot.left_motor.value = max(min(self.speed_slider.value + self.steering_slider.value, 1.0), 0.0)
        self.robot.right_motor.value = max(min(self.speed_slider.value - self.steering_slider.value, 1.0), 0.0)
        
        t = self.sleep_slider.value if vx+vy==0 else sqrt((x_diff**2 + y_diff**2) / (vx**2 + vy**2)) # time to travel to next point at current velocity
        time.sleep(t)
        
        self.pos = (x,y)
        print('Moved to ', self.pos)
        
    def TurnLeft(self):
        self.robot.set_motors(0.4,-0.1)
        time.sleep(1)
        self.robot.stop()
    
    def TurnRight(self):
        self.robot.set_motors(-0.1,0.4)
        time.sleep(1)
        self.robot.stop()
    
    def ForwardFor(self, t):
        self.robot.forward(1)
        time.sleep(t)
        self.robot.stop()
    
    def LeftFor(self,t):
        self.TurnLeft()
        self.ForwardFor(t)
        
    def RightFor(self,t):
        self.TurnRight()
        self.ForwardFor(t)
    
    def TurnBy(self,angle):
        if angle < 0: 
            self.robot.right(0.9)
        elif angle > 0: 
            self.robot.left(0.9)
            
        time.sleep(abs(angle)/(2*pi))
        self.robot.stop()
        
        print('Turned by angle ', degrees(angle))
    
    def Manhattan(self,x,y,vx,vy):
        x_diff = x - self.pos[0]
        y_diff = y - self.pos[1]
        
        tx = abs(x_diff)
        ty = abs(y_diff)
        if x_diff > 0.0:
            self.LeftFor(tx)
            
            if y_diff > 0.0:
                self.LeftFor(ty)
            elif y_diff < -0.0:
                self.RightFor(ty)
    
        elif x_diff < -0.0:
            self.RightFor(tx)
            
            if y_diff > 0.0:
                self.LeftFor(ty)
            elif y_diff < -0.0:
                self.RightFor(ty)
        
        self.pos = (x,y)
        print('Moved to ', self.pos)
    
    def Reset(self):
        self.pos = (0,0)
        self.orient = self.init_orient



        