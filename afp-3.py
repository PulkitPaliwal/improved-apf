"""
人工势场寻路算法实现
改进人工势场，解决不可达问题，仍存在局部最小点问题
"""
from apf import APF, Vector2d
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Circle
import random
import numpy as np
import time
def check_vec_angle(v1: Vector2d, v2: Vector2d):
    v1_v2 = v1.deltaX * v2.deltaX + v1.deltaY * v2.deltaY
    angle = math.acos(v1_v2 / (v1.length * v2.length)) * 180 / math.pi
    return angle


class APF_Improved(APF):
    def __init__(self, start: (), goal: (), obstacles: [], k_att: float, k_rep: float, rr: float,
                 step_size: float, max_iters: int, goal_threshold: float, is_plot=False):
        self.start = Vector2d(start[0], start[1])
        self.current_pos = Vector2d(start[0], start[1])
        self.goal = Vector2d(goal[0], goal[1])
        self.obstacles = [Vector2d(OB[0], OB[1]) for OB in obstacles]
        self.k_att = k_att
        self.k_rep = k_rep
        self.rr = rr  # 斥力作用范围
        self.step_size = step_size
        self.max_iters = max_iters
        self.iters = 0
        self.goal_threashold = goal_threshold
        self.path = list()
        self.is_path_plan_success = False
        self.is_plot = is_plot
        self.delta_t = 0.01
        self.obstacles_list = obstacles
    
    def repulsion(self):
        """
        斥力计算, 改进斥力函数, 解决不可达问题
        :return: 斥力大小
        """
        rep = Vector2d(0, 0)  # 所有障碍物总斥力
        for obstacle in self.obstacles:
            # obstacle = Vector2d(0, 0)
            obs_to_rob = self.current_pos - obstacle
            rob_to_goal = self.goal - self.current_pos
            if (obs_to_rob.length > self.rr):  # 超出障碍物斥力影响范围
                pass
            else:
                rep_1 = Vector2d(obs_to_rob.direction[0], obs_to_rob.direction[1]) * self.k_rep * (
                        1.0 / obs_to_rob.length - 1.0 / self.rr) / (obs_to_rob.length ** 2) * (rob_to_goal.length ** 2)
                rep_2 = Vector2d(rob_to_goal.direction[0], rob_to_goal.direction[1]) * self.k_rep * ((1.0 / obs_to_rob.length - 1.0 / self.rr) ** 2) * rob_to_goal.length
                rep +=(rep_1+rep_2)
        return rep

def distance(x,y):
    return math.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)
if __name__ == '__main__':
    # 相关参数设置
    k_att, k_rep = 1, 0.8
    rr = 6
    step_size, max_iters, goal_threashold = .2, 500, .2  # 步长0.5寻路1000次用时4.37s, 步长0.1寻路1000次用时21s
    step_size_ = 2

    # 设置、绘制起点终点
    start, goal = (0,0), (25, 25)
    is_plot = True
    circles = False
    if is_plot:
        fig = plt.figure(figsize=(7, 7))
        subplot = fig.add_subplot(111)
        subplot.set_xlabel('X-distance: m')
        subplot.set_ylabel('Y-distance: m')
        subplot.plot(start[0], start[1], '*r')
        subplot.plot(goal[0], goal[1], '*r')
    # 障碍物设置及绘制
    # obs = [[12,12],[13,13]]
    # obs = [[1, 1], [2, 2],[1, 4], [2, 4], [3, 3], [6, 7], [7,7], [10, 6], [11, 12], [14, 14], [15,14], [16,14], [19,18], [20,19], [20,18], [18,17], [22,22], [22,18], [22,16], [20,14], [18,13]]
    # obs = [[25,20], [15,22], [15,6], [13,17], [14,0],[1, 1], [2, 2],[1, 4], [2, 4], [3, 3], [6, 7], [7,7], [10, 6], [11, 12], [14, 14], [15,14], [16,14], [19,18], [20,19], [20,18], [18,17], [22,22], [22,18], [22,16], [20,14], [18,13]]
    # obs = [[14,14], [14,15], [15,15], [16,14]]
    # obs = [[1,7],[2,7],[1,8], [3,9],[5,6], [6,6], [25,20], [25,22], [15,22], [15,6], [13,17], [14,0],[1, 1], [2, 2],[1, 4], [2, 4], [3, 3], [6, 7], [7,7], [10, 6], [11, 12], [14, 14], [15,14], [16,14], [19,18], [20,19], [20,18], [18,17], [22,22], [22,18], [22,16], [20,14], [18,13]]
    # obs = [[4,0],[1,7],[2,7],[1,8], [3,9],[5,6], [6,6], [25,20], [25,22], [15,22], [15,6], [13,17], [14,0],[1, 1], [2, 2],[1, 4], [2, 4], [3, 3], [6, 7], [7,7], [10, 6], [11, 12], [14, 14], [15,14], [16,14], [19,18], [20,19], [20,18], [18,17], [22,22], [22,18], [22,16], [20,14], [18,13]]
    # obs = [[10,10],[8,14],[5,15],[15,5],[4,0],[1,7],[2,7],[1,8], [3,9],[5,6], [6,6], [25,20], [25,22], [15,22], [15,6], [13,17], [14,0],[1, 1], [2, 2],[1, 4], [2, 4], [3, 3], [6, 7], [7,7], [10, 6], [11, 12], [14, 14], [15,14], [16,14], [19,18], [20,19], [20,18], [18,17], [22,22], [22,18], [22,16], [20,14], [18,13]]
    # obs = [[25,0], [24,5], [5,24], [4,20], [-1,4], [-2,6], [6,10], [14,10], [15,10], [16,9], [2,20], [2,21], [3,19], [4,16],[10,10],[8,14],[5,15],[15,5],[4,0],[1,7],[2,7],[1,8], [3,9],[5,6], [6,6], [25,20], [25,22], [15,22], [15,6], [13,17], [14,0],[1, 1], [2, 2],[1, 4], [2, 4], [3, 3], [6, 7], [7,7], [10, 6], [11, 12], [14, 14], [15,14], [16,14], [19,18], [20,19], [20,18], [18,17], [22,22], [22,18], [22,16], [20,14], [18,13]]
    # obs = [[25,0], [-1,1], [11,1], [24,5], [5,24], [5,-1], [7,2], [10,3], [2,1], [4,20], [-1,4], [-2,6], [15,10], [16,9], [2,20], [2,21], [3,19], [4,16],[10,10],[8,14],[5,15],[15,5],[4,0],[1,7],[2,7],[1,8], [3,9],[5,6], [6,6], [25,20], [25,22], [15,22], [15,6], [13,17], [14,0],[1, 1], [2, 2],[1, 4], [2, 4], [3, 3], [6, 7], [7,7], [10, 6], [11, 12], [14, 14], [15,14], [16,14], [19,18], [20,19], [20,18], [18,17], [22,22], [22,18], [22,16], [20,14], [18,13]]
    # obs = [[0,-1],[25,0], [11,1], [24,5], [5,24], [5,-1], [7,2], [10,3], [2,1], [4,20], [-1,4], [-2,6], [15,10], [16,9], [2,20], [2,21], [3,19], [4,16],[10,10],[8,14],[5,15],[15,5],[4,0],[1,7],[2,7],[1,8], [3,9],[5,6], [6,6], [25,20], [25,22], [15,22], [15,6], [13,17], [14,0],[1, 1], [2, 2],[1, 4], [2, 4], [3, 3], [6, 7], [7,7], [10, 6], [11, 12], [14, 14], [15,14], [16,14], [19,18], [20,19], [20,18], [18,17], [22,22], [22,18], [22,16], [20,14], [18,13]]
    # obs = [[1, 1], [2, 2],[1, 4], [2, 4], [3, 3], [6, 7], [7,7], [10, 6], [11, 12], [14, 14], [15,14], [16,14], [19,18], [20,19], [20,18], [18,17], [22,22], [22,18], [22,16], [20,14], [18,13]]
    obs = [[24,24],[24,25],[25,24],[23,25],[23,24],[6,6],[7,7],[7,20], [8,21],[9,1], [8,0], [8,2], [3,5], [3,1], [8,8],[10,10],[11,17], [8,14],[5,15],[15,5],[1,7],[2,7],[1,8], [3,9],[5,6], [11,6], [12,12], [25,20], [25,22], [15,22], [15,6], [13,17], [14,0],[1, 1], [2, 2],[1, 4], [2, 4], [3, 3], [6, 7], [7,7], [10, 6], [11, 12], [14, 14], [15,14], [19,18], [20,19], [20,18], [18,17], [22,22], [22,18], [22,16], [20,14], [18,13]]
    # obs = [[20,25],[21,4],[18,7], [23,8], [15,20],[12,1],[12,2],[13,1],[13,2],[15,19],[19,5],[18,4],[17,4],[15,18],[23,9], [22,11],[5,0] ,[1,10],[21,12],[19,17],[16,13],[20,11],[0,9],[0,8], [8,12],[1,11],[5,5],[6,6],[7,7],[20,17], [24,24],[24,23],[12,6],[12,7],[10,8],[11,13],[15,20],[14,13], [8,21],[9,1], [8,0], [8,2], [3,5], [3,1], [8,8], [3, 3], [0,1], [19,19],[20,20],[21,21], [22,19], [3,10],[3,8],[4,4],[5,5],[6,4],[6,11], [11,14],[11,8],[6,14],[5,14], [4,14],[3,16],[15,5],[16,7],[17,5],[17,9],[17,11],[19,12],[13,24],[2,22],[12,21],[4,10],[1,15],[15,15],  [18,17], [22,22], [22,18], [22,16], [12,19],[16,15],[16,16],[15,17],[17,15]]
    obs = [[14,14], [14,15], [15,15],[15,16]]
    print('obstacles: {0}'.format(obs))
    # for i in range(0):
    #     obs.append([random.uniform(2, goal[1] - 1), random.uniform(2, goal[1] - 1)])

    if is_plot:
        for OB in obs:
            circle = Circle(xy=(OB[0], OB[1]), radius=rr, alpha=0.3)
            if circles: subplot.add_patch(circle)
            subplot.plot(OB[0], OB[1], 'xk')
    # t1 = time.time()
    # for i in range(1000):
    
    avoidance_list = []
    straight_path = []
    for i in range(start[0], goal[0] + 1):
        straight_path.append([i,i])
    straight_path = np.array(straight_path)
    for obstacle in obs:
        for points in straight_path:
            if (obstacle[0] == points[0]) and (obstacle[1]== points[1]):
                avoidance_list.append(obstacle)
    avoidance_list = np.array(avoidance_list)
    print("avoidance_list: ", str(avoidance_list))
    # if is_plot:
    #     apf = APF_Improved(start, goal, obs, k_att, k_rep, rr, step_size, max_iters, goal_threashold, is_plot)
    # else:
    #     apf = APF_Improved(start, goal, obs, k_att, k_rep, rr, step_size, max_iters, goal_threashold, is_plot)
    loc_start = []
    loc_goal = []
    thresh = 3
    i = 0
    avoidance_list = sorted(avoidance_list, key=lambda x: x[1], reverse=False)
    while i < (len(avoidance_list)):
        z = i
        while  z+1<len(avoidance_list) and distance(avoidance_list[z],avoidance_list[z+1]) < thresh:
            z = z + 1
        loc_start.append(np.array(avoidance_list[i])-1)
        loc_goal.append(np.array(avoidance_list[z])+1)
        i = z + 1
        # loc_start.append(np.array(avoidance_list[i])-1)
        # loc_goal.append(np.array(avoidance_list[i])+1)
        # i = i + 1

    print("loc_goal: " + str(loc_goal))
    print("loc_start: " + str(loc_start))

    MOVING_STRAIGHT = True

    current_pt = start
    delta_t = 0.01
    total_length = 0
    j = 0
    path_ = []
    path_2 = []
    t = time.time()
    while distance(current_pt, goal) > goal_threashold:
        path_.append(current_pt)
        if j < len(loc_goal) and distance(current_pt,loc_start[j]) < 1:
            print("executing at: ", current_pt)
            MOVING_STRAIGHT = False
            apf = APF_Improved(loc_start[j], loc_goal[j], obs, k_att, k_rep, rr, step_size, max_iters, goal_threashold, is_plot)
            apf.path_plan()
            path = apf.path
            if is_plot:
                # path_ = []
                i = int(step_size_ / step_size)
                while (i < len(path)):
                    path_2.append(path[i])
                    i += int(step_size_ / step_size)
            for point in path:
                path_.append(point)

            for i in range(len(path)):
                if i!=0:    
                    total_length = total_length + math.sqrt((path[i][0]-path[i-1][0])**2+(path[i][1]-path[i-1][1])**2)
            current_pt = path_[-1]
            if distance(current_pt,loc_goal[j]) < 1:
                MOVING_STRAIGHT = True
                current_pt = loc_goal[j]
                total_length = total_length + distance(current_pt, loc_goal[j])
                j = j + 1
                # path_.append(current_pt)
                continue

            
        if MOVING_STRAIGHT:
            print("exec")
            if is_plot:
                path_2.append(current_pt)
                for i in range(10):   
                    plt.plot(current_pt[0]+i*0.1, current_pt[1]+i*0.1, '.b')
                    plt.pause(0.01)
            current_pt = np.array(current_pt) + 1
            if is_plot:
                path_2.append(current_pt)
            total_length = total_length + np.sqrt(2)
            # path_.append(current_pt)
        
        print("time: "+str(time.time()-t)+"s")

    # apf.path_plan()
    # if apf.is_path_plan_success:
    #     path = apf.path
    #     path_ = []
    #     i = int(step_size_ / step_size)
    #     while (i < len(path)):
    #         path_.append(path[i])
    #         i += int(step_size_ / step_size)
        
    #     if path_[-1] != path[-1]:  # 添加最后一个点
    #         path_.append(path[-1])
    #     print('planed path points:{}'.format(path_))
    
    # for i in range(len(path)):
    #     if i!=0:    
    #         total_length = total_length + math.sqrt((path[i][0]-path[i-1][0])**2+(path[i][1]-path[i-1][1])**2)
        
    print('path plan success ' + 'length: '+ str(total_length))
    # print(path_)
    if is_plot:
        px, py = [K[0] for K in path_2], [K[1] for K in path_2]  # 路径点x坐标列表, y坐标列表
        plt.plot(px, py, '^k')
        plt.plot(loc_start[0][0], loc_start[0][1], 's')
        plt.text(loc_start[0][0],loc_start[0][1], 'S_local', horizontalalignment = 'right')
        plt.plot(loc_goal[0][0], loc_goal[0][1], 's')
        plt.text(loc_goal[0][0],loc_goal[0][1], 'G_local', horizontalalignment = 'right')
        plt.show()
    # t2 = time.time()
    # print('寻路1000次所用时间:{}, 寻路1次所用时间:{}'.format(t2-t1, (t2-t1)/1000))