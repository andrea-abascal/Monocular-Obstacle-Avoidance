from djitellopy import Tello
import time
import math

def flight_circle(radius_cm=40, tangential_speed_cm_s=20, steps=18):
    """
    Make the Tello fly in a circle in the Y–Z plane (parallel to the wall),
    by sending low-level velocity commands using send_rc_control.

    Parameters:
        radius_cm: radius of the circle in cm
        tangential_speed_cm_s: constant speed along the circle
        steps: number of segments to approximate the circle
    """
    tello.takeoff()
    time.sleep(2)

    # Duration per segment
    total_time = 2 * math.pi * radius_cm / tangential_speed_cm_s
    dt = total_time / steps

    for i in range(steps):
        theta = (i + 0.5) / steps * 2 * math.pi

        # Compute velocity components
        v_y = int(tangential_speed_cm_s * math.cos(theta))  # left/right
        v_z = int(tangential_speed_cm_s * math.sin(theta))  # up/down

        tello.send_rc_control(v_y, 0, v_z, 0)
        time.sleep(dt)

    # Stop and land
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(1)
    tello.land()

def flight_rectangle(long_side_cm=50, short_side_cm=20):
    """
    Fly a vertical rectangle parallel to a wall in exact-distance steps.
    Rounds side_cm to nearest 20 cm for Tello's move_* commands.
    
    Parameters:
        long_side_cm: longest side in centimeters
        short_side_cm: shortest side in centimeters
    """
    step_cm = max(20, min(500, int(round(long_side_cm / 20)) * 20))
    print(f"[flight_rectangle] using step_cm = {long_side_cm} cm")

    tello.takeoff()
    time.sleep(2)
    tello.move_up(step_cm);    time.sleep(step_cm/100 + 1.5)
    tello.move_right(step_cm); time.sleep(step_cm/100 + 1.5)
    
    step_cm = max(20, min(500, int(round(short_side_cm / 20)) * 20))
    print(f"[flight_rectangle] using step_cm = {step_cm} cm")
    tello.move_down(step_cm);  time.sleep(step_cm/100 + 1.5)
    
    step_cm = max(20, min(500, int(round(long_side_cm / 20)) * 20))
    print(f"[flight_rectangle] using step_cm = {step_cm} cm")
    tello.move_left(step_cm);  time.sleep(step_cm/100 + 1.5)
    
    step_cm = max(20, min(500, int(round(short_side_cm / 20)) * 20))
    print(f"[flight_rectangle] using step_cm = {step_cm} cm")
    tello.move_up(step_cm);    time.sleep(step_cm/100 + 1.5)
    print("Battery (end of flight):", tello.get_battery(), "%")
    tello.land()
    
def flight_basic (distance=30):
    tello.takeoff()
    time.sleep(2)
    tello.move_left(30)
    tello.rotate_counter_clockwise(90)
    tello.move_forward(30)
    time.sleep(1)
    tello.land()
        
# Initialize and connect to Tello
tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")
# Uncomment selected routine/flight trajectory 
#flight_rectangle()
#flight_circle()
flight_basic()
tello.end()
