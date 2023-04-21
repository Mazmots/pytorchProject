import math

def calculate_angle(p1 ,p2 ,p3):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    x3 = p3[0]
    y3 = p3[1]
    # calculate distances between the points
    a = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    b = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
    c = math.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)

    # calculate angle using law of cosines
    angle_in_radians = math.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))

    # convert angle from radians to degrees
    angle_in_degrees = math.degrees(angle_in_radians)

    return angle_in_degrees