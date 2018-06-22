# -*- coding: utf-8 -*

from sklearn.preprocessing import LabelBinarizer
import math
from Hotspot import Hotspot
from Point import Point
import numpy as np
import os


class Env:
    def __init__(self):
        # 当前环境state
        self.state = []
        
        # mc移动花费的时间
        self.move_time = 0
        self.move_dist = 0
        self.total_time = 0

        # 一个回合最大的时间，用秒来表示，早上八点到晚上10点，十四个小时，总共 14 * 3600 秒的时间
        # 如果self.get_evn_time() 得到的时间大于这个时间，则表示该回合结束
        self.one_episode_time = 14 * 3600
        
        # sensor 和 mc的能量信息
        self.sensors_mobile_charger = {}
        # 初始化所有的sensor,mc 的能量信息
        self.set_sensors_mobile_charger()
        
        # 初始化self.sensors_mobile_charger 和 self.sensors
        # self.set_sensors_mobile_charger()
        # 对剩余寿命进行独热编码
        self.rl = ['Greater than the threshold value, 0', 'Smaller than the threshold value, 1', 'dead, -1']
        self.rl_label_binarizer = LabelBinarizer()
        self.rl_one_hot_encoded = self.rl_label_binarizer.fit_transform(self.rl)
        
        # 对是否属于hotspot 独热编码
        self.belong = ['1', '0']
        self.belong_label_binarizer = LabelBinarizer()
        self.belong_one_hot_encoded = self.belong_label_binarizer.fit_transform(self.belong)
        
        # 获得所有的hotspot
        self.hotspots = []
        # 初始化hotspots
        self.set_hotspots()
        
        # 记录当前时刻所在的hotspot，在环境初始化的时候设置为base_station
        self.current_hotspot = self.hotspots[0]

        # mc移动速度
        self.speed = 5
        # mc 移动消耗的能量
        self.mc_move_energy_consumption = 0
        # mc 给sensor充电消耗的能量
        self.mc_charging_energy_consumption = 0
        # 充电惩罚值
        self.charging_penalty = -1
        
        # sensors_points 存储所有的sensor轨迹点信息，初始化时从文件中读入。字典表示，key：sensor编号，value：轨迹点，list表示
        self.sensors_points = {}
        self.set_sensors_points()

        # sensor 在每个时间段访问hotspot的次数
        self.set_sensors_visit_hotspot_times()

    def get_sensors_visit_hotspot_times_info(self,hour):
        return self.sensors_visit_hotspot_times[hour]

    def set_sensors_visit_hotspot_times_phase(self, hour):
        self.sensors_visit_hotspot_times[hour] = {}

        files = os.listdir('hotspot_sensor/' + str(hour+1))
        for file in files:
            hotspot_num = file.split('.')[0]
            with open('hotspot_sensor/' + str(hour+1) + '/' + file) as f:
                lines = f.readlines()
                self.sensors_visit_hotspot_times[hour][hotspot_num] = lines

    def set_sensors_visit_hotspot_times(self):
        sensor_num = len(os.listdir('hotspot_sensor/'))
        self.sensors_visit_hotspot_times = sensor_num * [None]
        
        for i in range(sensor_num):
            self.set_sensors_visit_hotspot_times_phase(i) 

    def set_sensors_points(self):
        for i in range(17):
            with open('sensors/' + str(i) + '.txt', 'r') as f:
                lines = f.readlines()
                self.sensors_points[i] = lines

    def set_sensors_mobile_charger(self):
        # [0.7 * 6 * 1000, 0.6, 0, True]  依次代表：上一次充电后的剩余能量，能量消耗的速率，上一次充电的时间，
        # 是否已经死掉(计算reward的惩罚值时候使用，避免将一个sensor计算死掉了多次)，
        # 最后一个标志位，表示senor在该hotpot，还没有被充过电，如果已经充过了为True，避免被多次充电
        self.sensors_mobile_charger['0'] = [0.7 * 6 * 1000, 0.5, 0, True, False]
        self.sensors_mobile_charger['1'] = [0.3 * 6 * 1000, 0.3, 0, True, False]
        self.sensors_mobile_charger['2'] = [0.9 * 6 * 1000, 0.5, 0, True, False]
        self.sensors_mobile_charger['3'] = [0.5 * 6 * 1000, 0.3, 0, True, False]
        self.sensors_mobile_charger['4'] = [0.2 * 6 * 1000, 0.2, 0, True, False]
        self.sensors_mobile_charger['5'] = [0.4 * 6 * 1000, 0.3, 0, True, False]
        self.sensors_mobile_charger['6'] = [1 * 6 * 1000, 0.6, 0, True, False]
        self.sensors_mobile_charger['7'] = [0.3 * 6 * 1000, 0.5, 0, True, False]
        self.sensors_mobile_charger['8'] = [1 * 6 * 1000, 0.3, 0, True, False]
        self.sensors_mobile_charger['9'] = [0.9 * 6 * 1000, 0.2, 0, True, False]
        self.sensors_mobile_charger['10'] = [0.8 * 6 * 1000, 0.2, 0, True, False]
        self.sensors_mobile_charger['11'] = [0.5 * 6 * 1000, 0.4, 0, True, False]
        self.sensors_mobile_charger['12'] = [0.4 * 6 * 1000, 0.2, 0, True, False]
        self.sensors_mobile_charger['13'] = [0.6 * 6 * 1000, 0.2, 0, True, False]
        self.sensors_mobile_charger['14'] = [0.3 * 6 * 1000, 0.2, 0, True, False]
        self.sensors_mobile_charger['15'] = [0.9 * 6 * 1000, 0.6, 0, True, False]
        self.sensors_mobile_charger['16'] = [0.8 * 6 * 1000, 0.4, 0, True, False]
        self.sensors_mobile_charger['MC'] = [1000 * 1000, 50]

    def set_hotspots(self):
        # 这是编号为0 的hotspot，也就是base_stattion,位于整个充电范围中心
        base_station = Hotspot((116.333 - 116.318) * 85000 / 2, (40.012 - 39.997) * 110000 / 2, 0)
        self.hotspots.append(base_station)
        # 读取hotspot.txt 的文件，获取所有的hotspot，放入self.hotspots中
        path = './hotspot.txt'
        with open(path) as file:
            for line in file:
                data = line.strip().split(',')
                hotspot = Hotspot(float(data[0]), float(data[1]), int(data[2]))
                self.hotspots.append(hotspot)

    # 根据hotspot 的编号，在self.hotspots 中找到对应的hotpot
    def find_hotspot_by_num(self, num):
        for hotspot in self.hotspots:
            if hotspot.get_num() == num:
                return hotspot

    def initial_is_charged(self):
        for key, value in self.sensors_mobile_charger.items():
            if key != 'MC':
                value[4] = False

    # 传入一个action, 得到下一个state，reward，和 done(是否回合结束)的信息
    def step(self, action):
        reward = 0
        
        if action != -1:
            hotspot_num = action + 1
            staying_time = 1
        else:
            hotspot_num = 0
            staying_time = 0
        	
        # 得到下一个hotspot
        hotspot = self.find_hotspot_by_num(hotspot_num)
        # 当前hotspot 和 下一个hotspot间的距离,得到移动花费的时间，添加到self.move_time 里
        distance = hotspot.get_distance_between_hotspot(self.current_hotspot)
        self.move_dist += distance

        time = distance / self.speed
        self.move_time += time
        self.total_time += time

        # 更新self.current_hotspot 为 action 中选择的 hotspot
        self.current_hotspot = hotspot
        
        # 更新mc的剩余能量，减去移动消耗的能量
        self.sensors_mobile_charger['MC'][0] = self.sensors_mobile_charger['MC'][0] \
                                               - self.sensors_mobile_charger['MC'][1] * distance
                                               
        start_wait_seconds = self.get_evn_time()
        #every 20 min
        hour = int(start_wait_seconds / 1200) 
        
        # 将在hotspot_num 等待的时间 添加到state中的CS
        for i in range(42):
            if i == hotspot_num - 1:
                self.state[3 * i] += 1
                self.state[3 * i + 1] += staying_time
                self.state[3 * i + 2] = 1
            else:
                self.state[3 * i + 2] = 0


        self.total_time += staying_time * 5 * 60


        # mc 结束等待后环境的时间
        end_wait_seconds = self.get_evn_time()

	#in total 42 hotspots
        hotspot_info = []
	for i in range(42):
	    #path = './hotspot_sensor/' + str(hour) + '/' + str(i+1) + '.txt'
            # 读取文件，得到在当前时间段，hotspot_num 的访问情况，用字典保存。key: sensor 编号；value: 访问次数
            hotspot_num_sensor_arrived_times = {}
            sensors_visit_hotspot_info = self.get_sensors_visit_hotspot_times_info(hour)[str(i+1)]
            for line in sensors_visit_hotspot_info:
                data = line.strip().split(',')
                hotspot_num_sensor_arrived_times[data[0]] = data[1]
		
	    hotspot_info.append(hotspot_num_sensor_arrived_times)

	for i in range(17):
	    start = 42 * 3 + i * (3 + 42)
            end = start + 3
 		
            # 取出sensor
            sensor = self.sensors_mobile_charger[str(i)]

            # 上一次充电后的电量
            sensor_energy_after_last_time_charging = sensor[0]
                
            # 当前sensor 电量消耗的速率
            sensor_consumption_ratio = sensor[1]
                
            # 上一次的充电时间
            previous_charging_time = sensor[2]

            sensor_alive = sensor[3]

            end_time = end_wait_seconds

            sensor_meet = False 

            # 读取第i 个sensor 的轨迹点信息
            # test if the sensor can meet MC 
            #sensor_path = './sensors/' + str(i) + '.txt'
            sensor_points = self.sensors_points[i]
            for point_line in sensor_points:
                data = point_line.strip().split(',')
                point_time = self.str_to_seconds(data[2])
                point = Point(float(data[0]), float(data[1]), data[2])
                    
                if (start_wait_seconds <= point_time <= end_wait_seconds) and (point.get_distance_between_point_and_hotspot(self.current_hotspot) < 60):
                    end_time = point_time
                    sensor_meet = True
                    break

                if point_time > end_wait_seconds:
                    break

            # 在mc等待了action 中的等待时间以后，sensor 的剩余电量
            sensor_reserved_energy = sensor_energy_after_last_time_charging - \
                                         (end_time - previous_charging_time) * sensor_consumption_ratio
            # 当前sensor 的剩余寿命
            rl = sensor_reserved_energy / sensor_consumption_ratio

	    # 如果剩余寿命大于两个小时
            if rl >= 2 * 3600:
            # 得到 大于阈值的 独热编码,转换成list,然后更新state 中的能量状态
                rl_one_hot_encoded = self.rl_label_binarizer.transform(['Greater than the threshold value, 0']).tolist()[0]       
                self.state[start:end] = rl_one_hot_encoded

            elif 0 < rl < 2 * 3600:
            # 更新state中 的剩余寿命信息的状态
            # 得到 小于阈值的 独热编码,转换成list,然后更新state 中的状态
                rl_one_hot_encoded = self.rl_label_binarizer.transform(['Smaller than the threshold value, 1']).tolist()[0]
                self.state[start:end] = rl_one_hot_encoded
                   
                if sensor_meet:
                # mc 给该sensor充电， 充电后更新剩余能量
                    self.sensors_mobile_charger['MC'][0] = self.sensors_mobile_charger['MC'][0] \
                                                                       - (6 * 1000 - sensor_reserved_energy)
                # 设置sensor 充电后的剩余能量 是满能量
                    sensor[0] = 6 * 1000
                # 更新被充电的时间
                    sensor[2] = end_time
                                
                # 加上得到的奖励,需要先将 rl 的单位先转化成小时
                    rl = rl / 3600
                    reward += math.exp(-rl) 

            else:
            # 更新state中 的剩余寿命信息的状态
            # 得到 死掉的 独热编码,转换成list,然后更新state 中的状态
                rl_one_hot_encoded = self.rl_label_binarizer.transform(['dead, -1']).tolist()[0]
                self.state[start:end] = rl_one_hot_encoded
                    
                if sensor_alive:
                    sensor[3] = False
                    reward += self.charging_penalty 

            hotspot_pro = []
            for j in range(42):
		pj = int(hotspot_info[j][str(i)])
		if pj == 0:
		    pj_encode = self.belong_label_binarizer.transform(['0']).tolist()[0][0]
		else:
		    pj_encode = self.belong_label_binarizer.transform(['1']).tolist()[0][0]
			
		hotspot_pro.append(pj_encode)
		 
	    self.state[end:end+42] = hotspot_pro   
	    #end of sensor infor update

        #phase = int(end_wait_seconds / 3600) + 8
        # mc 给到达的sensor 充电后，如果能量为负或者 self.get_evn_time() > self.one_episode_time，则回合结束，反之继续
        if self.sensors_mobile_charger['MC'][0] <= 0 or self.get_evn_time() + 10 * 60 > self.one_episode_time:
            done = 1
            #print self.move_dist, self.get_evn_time()
        else:
            done = 0

        observation = np.array(self.state)
        return observation, reward, done

    # 初始化整个环境
    def reset(self, RL):
        # 前面0~83 都初始化为 0。记录CS的信息，每个hotspot占两位
        for i in range(42 * 3):
            self.state.append(0)
            
        # 84位开始记录sensor的信息,每一个sensor需要4位，17个sensor，共68位
        for i in range(42 * 3, 42 * 3 + 17 * 45):
            self.state.append(0)
        	
        # 得到一个随机的8点时间段的action,例如 43,1 表示到43 号hotspot 等待1个t
        # print(len(self.state))
        action = -1
        self.current_hotspot = self.hotspots[0]      

        state_, reward_, done_ = self.step(action)
        
        return state_, reward_, done_

    # 传入时间字符串，如：09：02：03，转化成与 08:00:00 间的秒数差
    def str_to_seconds(self, input_str):
        data = input_str.split(':')
        hour = int(data[0]) - 8
        minute = int(data[1])
        second = int(data[2])
        
        return hour * 3600 + minute * 60 + second

    # 获得当前环境的秒
    def get_evn_time(self):
        return self.total_time

        #total_t = 0
        #for i in range(42 * 3):
        #    if i % 3 == 1:
        #        total_t += self.state[i]
        #total_time = total_t * 5 * 60 + self.move_time

        #return total_time

    def test(self):
        dis = 0
        for h1 in self.hotspots:
            for h2 in self.hotspots:
                temp = h1.get_distance_between_hotspot(h2)
                if temp > dis:
                    dis = temp
        print(dis)
        
#if __name__ == '__main__':
#    evn = Env()
#    evn.test()
    # state_, reward, done = evn.reset(RL)
    # print(state_)
    # print(reward)
    # print(done)
