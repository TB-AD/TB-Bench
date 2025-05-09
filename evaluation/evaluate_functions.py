import json
import numpy as np
import random
import argparse 
import re
import os

def extract_distance(text):
    # Define the regex pattern to match distances in the form of 1.43 or 1 meter
    pattern = r'(\d+(?:\.\d+)?)\s*meters?'
    match = re.search(pattern, text)
    if match:
        return float(match.group(1))
    return -1000000000.0

def relative_distance_score(pred, gt):
    pred_distance = extract_distance(pred)
    gt_distance = extract_distance(gt)
    if pred_distance == -1000000000.0 or gt_distance == -1000000000.0:
        return 0

    acceptable_range = 0.25
    ### if the prediction is within 25% of the ground truth
    out = 1 if pred_distance >= gt_distance*(1-acceptable_range) and pred_distance <= gt_distance*(1+acceptable_range) else 0
    return out

def extract_angle(text):
    text = text.lower()
    text = text.replace('-degree', ' degrees')
    pattern = r'(-?\d+(?:\.\d+)?)\s*degrees?'
    match = re.search(pattern, text)
    if match:
        return float(match.group(1))
    return -1000000000.0

def calculate_angular_difference(angle1, angle2):
    diff = abs(angle1 - angle2)
    return min(diff, 360 - diff)

def orientation_reasoning_score(pred, short_gt):
    perpendicular_list = ['perpendicular', 'perpendicular to the ego-vehicle']
    opposite_list = ['opposite', 'opposite direction', 'opposite to the ego-vehicle', 'diametrically opposed', 'directly contrary', 'diametrically opposed']
    similar_list = ['similar', 'similar direction', 'similar to the ego-vehicle', 'closely resembles']

    perpendicular_matches = 1 if sum(word in pred for word in perpendicular_list) >= 1 else 0
    opposite_matches = 1 if sum(word in pred for word in opposite_list) >= 1 else 0
    similar_matches = 1 if sum(word in pred for word in similar_list) >= 1 else 0

    total_matches = perpendicular_matches + opposite_matches + similar_matches

    # Check for cheating or multiple matches
    if total_matches > 1:
        return 0

    if short_gt == 'perpendicular':
        return 1 if perpendicular_matches == 1 else 0
    elif short_gt == 'opposite':
        return 1 if opposite_matches == 1 else 0
    elif short_gt == 'similar':
        return 1 if similar_matches == 1 else 0
    else:
        pred_angle = extract_angle(pred)
        gt_angle = extract_angle(short_gt)
        if pred_angle == -1000000000.0 or gt_angle == -1000000000.0:
            return 0

        ### if the prediction is +- 15 degrees of the ground truth
        acceptable_degree_range= 15
        angular_difference = calculate_angular_difference(pred_angle, gt_angle)
        out = 1 if angular_difference <= acceptable_degree_range else 0
    return out

def other_lane_to_ego_score(pred, short_gt):
    front_lane_list = ['front lane']
    front_left_lane_list = ['front-left lane', 'front left lane']
    front_right_lane_list = ['front-right lane', 'front right lane']
    oncoming_traffic_lane_list = ['oncoming traffic lane', 'oncoming lane', 'oncoming traffic']

    front_lane_matches = 1 if sum(word in pred for word in front_lane_list) >= 1 else 0
    front_left_lane_matches = 1 if sum(word in pred for word in front_left_lane_list) >= 1 else 0
    front_right_lane_matches = 1 if sum(word in pred for word in front_right_lane_list) >= 1 else 0
    oncoming_traffic_lane_matches = 1 if sum(word in pred for word in oncoming_traffic_lane_list) >= 1 else 0

    total_matches = front_lane_matches + front_left_lane_matches + front_right_lane_matches + oncoming_traffic_lane_matches
    # Check for cheating or multiple matches
    if total_matches != 1:
        return 0  # Multiple matches or no match indicates cheating or incorrect prediction

    if short_gt == 'front_lane':
        return 1 if front_lane_matches == 1 else 0
    elif short_gt == 'front_left_lane':
        return 1 if front_left_lane_matches == 1 else 0
    elif short_gt == 'front_right_lane':
        return 1 if front_right_lane_matches == 1 else 0
    elif short_gt == 'oncoming_traffic_lane':
        return 1 if oncoming_traffic_lane_matches == 1 else 0
    return 0

def other_lane_changing_score(pred, short_gt):
    no_change_list = ['maintains its lane', 'maintaining its lane', 'no change', 'go straight', 'straight']
    left_lane_change_list = ['change to the left lane', 'changes to the left lane', 'changes to the left', 'change to the left', 'left lane change', "from the right to the left", "transitions to the left"]
    right_lane_change_list = ['change to the right lane',  'changes to the right lane','changes to the right','change to the right', 'right lane change', "from the left to the right", "transitions to the right"]

    no_change_matches = 1 if sum(word in pred for word in no_change_list) >= 1 else 0
    left_change_matches = 1 if sum(word in pred for word in left_lane_change_list) >= 1 else 0
    right_change_matches = 1 if sum(word in pred for word in right_lane_change_list) >= 1 else 0

    total_matches = no_change_matches + left_change_matches + right_change_matches
    # Check for cheating or multiple matches
    if total_matches != 1:
        return 0  # Multiple matches or no match indicates cheating or incorrect prediction

    if short_gt == 'no_change':
        return 1 if no_change_matches == 1 else 0
    elif short_gt == 'left_lane_change':
        return 1 if left_change_matches == 1 else 0
    elif short_gt == 'right_lane_change':
        return 1 if right_change_matches == 1 else 0
    return 0

def other_turning_score(pred, short_gt):
    left_turn_list = ['turn left', 'turns left', 'left turn', 'left-turn maneuver', 'turn left maneuver', 'turns left maneuver', 'turning left']
    right_turn_list = ['turn right', 'turns right', 'right turn', 'right-turn maneuver', 'turn right maneuver', 'turns right maneuver', 'turning right']
    go_straight_list = ['go straight', 'no turn', 'no turns', 'no turning', 'no turning maneuver', 'proceeds directly ahead']

    left_turn_matches = 1 if sum(word in pred for word in left_turn_list) >= 1 else 0
    right_turn_matches = 1 if sum(word in pred for word in right_turn_list) >= 1 else 0
    go_straight_matches = 1 if sum(word in pred for word in go_straight_list) >= 1 else 0

    total_matches = left_turn_matches + right_turn_matches + go_straight_matches
    # Check for cheating or multiple matches
    if total_matches != 1:
        return 0  # Multiple matches or no match indicates cheating or incorrect prediction

    if short_gt == 'left_turn':
        return 1 if left_turn_matches == 1 else 0
    elif short_gt == 'right_turn':
        return 1 if right_turn_matches == 1 else 0
    elif short_gt == 'go_straight':
        return 1 if go_straight_matches == 1 else 0
    return 0

def spatial_reasoning_score(pred, short_gt):
    front_list = ['positioned directly ahead of our car', 'positioned directly ahead', 'directly ahead']
    front_right_list = [
        'front right', 'to the front and to the right', 'forward-right direction', 'positioned to the front and to the right',
        'directly ahead and to the right of', 'positioned to the front and to the right', 'forward-right direction', 'front-right quadrant'
        'forward-right direction', 'front-right quadrant'
    ]
    front_left_list = [
        'front left', 'to the front and to the left', 'forward-left direction', 'positioned to the front and to the left', 
        'directly ahead and to the left of', 'positioned to the front and to the left', 'forward-left direction', 'front-left quadrant'
    ]
    back_list = ['positioned directly behind', 'located directly behind', 'directly behind', 'behind']
    back_right_list = ['back right','positioned directly behind and to the right', 'back right', 'rear right quadrant', 'rear-right quadrant']
    back_left_list = ['back left','positioned directly behind and to the left', 'back left', 'rear-left quadrant', 'rear left quadrant']

    front_matches = 1 if sum(word in pred for word in front_list) >= 1 else 0
    front_right_matches = 1 if sum(word in pred for word in front_right_list) >= 1 else 0
    front_left_matches = 1 if sum(word in pred for word in front_left_list) >= 1 else 0
    back_matches = 1 if sum(word in pred for word in back_list) >= 1 else 0
    back_right_matches = 1 if sum(word in pred for word in back_right_list) >= 1 else 0
    back_left_matches = 1 if sum(word in pred for word in back_left_list) >= 1 else 0

    total_matches = front_matches + front_right_matches + front_left_matches + back_matches + back_right_matches + back_left_matches
    # Check for cheating or multiple matches
    if total_matches != 1:
        print("!!!!!total_matches>1: ", total_matches)
        print("pred: ", pred)
        # return 0  # Multiple matches or no match indicates cheating or incorrect prediction

    if short_gt == 'front':
        return 1 if front_matches == 1 else 0
    elif short_gt == 'front left':
        return 1 if front_left_matches == 1 else 0
    elif short_gt == 'front right':
        return 1 if front_right_matches == 1 else 0
    elif short_gt == 'back':
        return 1 if back_matches == 1 else 0
    elif short_gt == 'back left':
        return 1 if back_left_matches == 1 else 0
    elif short_gt == 'back right':
        return 1 if back_right_matches == 1 else 0
    return 0

def ego_turning_score(pred, short_gt):
    right_turn_list = ['executed a precise right-turn maneuver','right turn']
    left_turn_list = ['executed a precise left-turn maneuver','left turn']
    go_straight_list = ['proceeds directly ahead','go straight', 'proceeds in a linear trajectory']

    right_turn_matches = 1 if sum(word in pred for word in right_turn_list) >= 1 else 0
    left_turn_matches = 1 if sum(word in pred for word in left_turn_list) >= 1 else 0
    go_straight_matches = 1 if sum(word in pred for word in go_straight_list) >= 1 else 0

    total_matches = right_turn_matches + left_turn_matches + go_straight_matches
    # Check for cheating or multiple matches
    if total_matches != 1:
        return 0  # Multiple matches or no match indicates cheating or incorrect prediction

    if short_gt == 'right_turn':
        return 1 if right_turn_matches == 1 else 0
    elif short_gt == 'left_turn':
        return 1 if left_turn_matches == 1 else 0
    elif short_gt == 'go_straight':
        return 1 if go_straight_matches == 1 else 0
    return 0

def ego_traverse_distance_score(pred, gt):
    pred_distance = extract_distance(pred)
    gt_distance = extract_distance(gt)
    if pred_distance == -1000000000.0 or gt_distance == -1000000000.0:
        return 0

    acceptable_range = 0.25
    ### if the prediction is within 25% of the ground truth
    out = 1 if pred_distance >= gt_distance*(1-acceptable_range) and pred_distance <= gt_distance*(1+acceptable_range) else 0
    if gt_distance < 1.0:
        out = 1 if pred_distance >= 0 and pred_distance <= 1*(1+acceptable_range) else 0
    return out