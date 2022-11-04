import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
#import heapq
import math

class Scenario(BaseScenario):
    def __init__(self):
        self.f_c = 2e9#2Ghz
        self.f_c_mmwave = 38e9#2Ghz
        self.d_0 =5 #m
        self.light_speed = 3e8
        self.a = 10
        self.b= 0.6
        self.sigma_2 = math.pow(10,-95/10)/1000 # in W
        self.X_0 = 20*math.log(self.d_0*self.f_c*4*math.pi/self.light_speed, 10)
        self.central_node = np.array([0,0])
    
    def make_world(self, obs_range):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 4
        num_landmarks = num_agents
        self.num_adversaries = num_agents
#        top_n_list = [0,1,2,3,4,5,4,4]
        top_n_list = obs_range
#        [2]*num_agents
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            # agent.silent = True
            agent.silent = True
            agent.size = 0.06
#            agent.accel = 1 # defualt is 5
#            agent.partial=True
            agent.id = i
            agent.top_n = min(top_n_list[i], num_agents)

#            False if i==0 else True#partial observation 
#            agent.max_speed =0.5
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        self.num_agents = num_agents
        world.num_adversaries = self.num_adversaries
        world.done = 0
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-0.95, +0.95, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-0.95, +0.95, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
#        world.done = 0
        world.catch = 0

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
#                rew += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False
    
    def is_collision_list(self, agent1, agent2, k):
        delta_pos = agent1.state.p_pos_list[k] - agent2.state.p_pos_list[k]
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

#    def reward(self, agent, world):
#        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
#        rew_list = []
#        for k in range(5):
#           rew = 0
#           catch = 0
#           for l in world.landmarks:
#               dists = [np.sqrt(np.sum(np.square(a.state.p_pos_list[k] - l.state.p_pos))) for a in world.agents]
#               mindis =  min(dists)
#               if mindis<= l.size + agent.size:
#                   catch += 1
#                   rew+=10
#               else:
#                   rew -= mindis
#           if catch==len(world.landmarks):
#               world.done = 1
#               return 1
#           
#           if agent.collide:
#               for a in world.agents:
#                   if self.is_collision_list(a, agent, k):
#                       rew -= 1
#           rew_list.append(rew)
#        return np.mean(rew_list)
             
    
    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
#        catch = 0
        if agent.id==0:
            self.rewards_all(world)
        dis= self.dis_n[agent.id]
        rew = - dis
        if dis < 0.1:
            rew += 1
#        if catch==len(world.landmarks):
#               world.done = 1
#               return 1   
        if agent.collide:
            for a in world.agents:
                if a is agent: 
                    continue
                elif self.is_collision(a, agent):
                    rew -= 2
        return rew

    def rewards_all(self, world):
        dis_matrix = []
        for a in world.agents:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for l in world.landmarks]
            dis_matrix.append(dists)
        dis_matrix = np.array(dis_matrix)
        dis_n = [0]*self.num_agents
        for i in range(self.num_agents):
            # p = dis_matrix.argmin(dis_matrix)
            p = np.unravel_index(np.argmin(dis_matrix),dis_matrix.shape)
            dis_n[p[0]] = dis_matrix[p]
            dis_matrix[p[0] ] = np.array([100]*self.num_agents)
            dis_matrix[:, p[1]] = np.array([100]*self.num_agents)
        self.dis_n = dis_n
        world.catch = len([i for i in dis_n if i < 0.1])

    
#        rew = 0
#        for l in world.landmarks:
#            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
#            rew -= min(dists)
#        if agent.collide:
#            for a in world.agents:
#                if self.is_collision(a, agent):
#                    rew -= 1
#        return rew

#    def observation(self, agent, world, Net_obs, partial = True):
#        # get positions of all entities in this agent's reference frame
#        entity_pos = []
#        for entity in world.landmarks:  # world.entities:
#            p = entity.state.p_pos - agent.state.p_pos
#            if not partial or agent.R > np.sqrt(np.sum(np.square(p))):
#                entity_pos.append(p)
#            else: entity_pos.append(np.array([1.,1.]))
#        
#        # entity colors
#        entity_color = []
#        for entity in world.landmarks:  # world.entities:
#            entity_color.append(entity.color)
#        # communication of all other agents
#        other_pos, Com_Delay = [], []
#        
#        for other in world.agents: 
#            if other is agent: 
#                Com_Delay.append(0)
#                continue
#            p = other.state.p_pos - agent.state.p_pos 
#            Com_Delay.append(self.sending_delay(self.num_agents, np.sqrt(np.sum(np.square(p))), 0.1)/30 )
#            if  not partial or agent.R > np.sqrt(np.sum(np.square(p))):
#                other_pos.append(p)
#            else:
#                other_pos.append(np.array([1.,1.]))
#        if Net_obs:
#            return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + [Com_Delay])
#        else:
#            return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)
     
    def observation(self, agent, world, pack_size, map_size):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)


        other_pos, Com_Delay, other_p_vel = [], [], []

        other_pos_2 = []
        for other in world.agents: 
            if other is agent: 
                Com_Delay.append(0)
                continue
            p = other.state.p_pos - agent.state.p_pos 
            dis = np.sqrt(np.sum(np.square(p)))
            if dis==0:
                Com_Delay.append(0.1)
            else:
                Com_Delay.append(max(1, self.sending_delay(self.num_agents, dis/2*map_size, pack_size*0.1))/world.est_delay ) #dis in 100*100m^2
            other_pos.append(p)
            other_p_vel.append(other.state.p_vel)

            #-------ohter absolute  position-------#
            p2 = other.state.p_pos
            other_pos_2.append(p2)

        # --------useless com.-------#
        # entity_pos = sorted(entity_pos, key=lambda s: np.sum(np.square(s)))[:agent.top_n]
        # # index = sorted(range(len(other_pos)), key=lambda i: np.sum(np.square(other_pos[i])))[:agent.top_n]
        # index = sorted(range(len(other_pos)), key=lambda i: np.sum(np.square(other_pos[i])))[:agent.top_n - 1]
        # index2 = sorted(range(len(other_pos)), key=lambda i: np.sum(np.square(other_pos[i])))[:5]
        # # other_pos = [other_pos[i] for i in index2]
        # # other_p_vel = [other_p_vel[i] for i in index]
        # # other_pos2 = [other_pos_2[i] for i in index2]
        #
        # other_pos = [other_pos[i] for i in index]
        # other_p_vel = [other_p_vel[i] for i in index2]
        #
        # # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_p_vel + entity_pos + other_pos +[Com_Delay])
        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos + entity_pos + other_p_vel +[Com_Delay])


        # --------useless com.2-------#
        # entity_pos = sorted(entity_pos, key=lambda s: np.sum(np.square(s)))[:agent.top_n]
        # # index = sorted(range(len(other_pos)), key=lambda i: np.sum(np.square(other_pos[i])))[:agent.top_n]
        # index = sorted(range(len(other_pos)), key=lambda i: np.sum(np.square(other_pos[i])))[:agent.top_n - 1]
        # index2 = sorted(range(len(other_pos)), key=lambda i: np.sum(np.square(other_pos[i])))[:5]
        # other_pos = [other_pos[i] for i in index2 ]
        # other_p_vel = [other_p_vel[i] for i in index2]
        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_p_vel  +[Com_Delay])

        #------------useless com.3--------#
        # entity_pos = sorted(entity_pos, key=lambda s: np.sum(np.square(s)))[:agent.top_n]
        # # index = sorted(range(len(other_pos)), key=lambda i: np.sum(np.square(other_pos[i])))[:agent.top_n]
        # # index = sorted(range(len(other_pos)), key=lambda i: np.sum(np.square(other_pos[i])))[:agent.top_n - 1]
        # # index2 = sorted(range(len(other_pos)), key=lambda i: np.sum(np.square(other_pos[i])))[:5]
        # # index = sorted(range(len(other_pos)), key=lambda i: np.sum(np.square(other_pos[i])))[:agent.top_n - 1]
        # index2 = sorted(range(len(other_pos)), key=lambda i: np.sum(np.square(other_pos[i])))[:agent.top_n]
        # other_pos = [other_pos[i] for i in index2 ]
        # other_p_vel = [other_p_vel[i] for i in index2]
        # other_message = np.random.random(20)
        #
        #
        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] +
        #                       entity_pos + other_pos + other_p_vel + [other_message] + [Com_Delay])
        #




        # ---------usefull com.------#
        # entity_pos = sorted(entity_pos, key=lambda s: np.sum(np.square(s)))[:agent.top_n]
        # index = sorted(range(len(other_pos)), key=lambda i: np.sum(np.square(other_pos[i])))[:agent.top_n]
        # # #
        entity_pos = sorted(entity_pos, key=lambda s: np.sum(np.square(s)))[:4]
        index = sorted(range(len(other_pos)), key=lambda i: np.sum(np.square(other_pos[i])))[:4]

        other_pos = [other_pos[i] for i in index]
        other_p_vel = [other_p_vel[i] for i in index]
        # print("pvel:" +  str([agent.state.p_vel]))
        # print("ppos: " + str([agent.state.p_pos]))
        # print(other_p_vel)
        # print(entity_pos)
        # print(other_pos)
        # print([Com_Delay])
        # print(np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_p_vel + entity_pos + other_pos +[Com_Delay]))
        # print(len(np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_p_vel + entity_pos + other_pos +[Com_Delay])))
        # print("---")
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_p_vel + entity_pos + other_pos +[Com_Delay])


    def get_centralnode_delay(self, world, num_agents, pack_size, map_size):
        dalay_n = []
        for agent in world.agents: 
            p = agent.state.p_pos - self.central_node
            dis = np.sqrt(np.sum(np.square(p)))
            dalay_n.append(self.sending_delay(num_agents, dis/2*map_size, pack_size*0.1)/world.est_delay)
        dalay_n_max = np.max(dalay_n)        
        return dalay_n_max, dalay_n
    
    def sending_delay(self, Num, distance, size):  # distance in m, size of packages in Mb
#        power = 1#w
#        Bandwidth = 2e3 #2Mhz
#        zeta_NLoS = 2.4
#        g = -(10* zeta_NLoS *math.log(distance,10) + self.X_0 )
#        eta = math.pow(10, g/10)*power/self.sigma_2 # W/W
#        X_G2G = Bandwidth/Num* math.log(1+eta)/1e3 #in Mbps
#        retun size/X_G2G*1e3 
        return size*distance*10
#    
#    + np.random.uniform(0,20)#ms sending delay +  Que/processing