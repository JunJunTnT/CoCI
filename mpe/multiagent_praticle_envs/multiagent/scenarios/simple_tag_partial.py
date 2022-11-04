import numpy as np
from multiagent.multiagent.core import World, Agent, Landmark
from multiagent.multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self,obs_range):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = 4
        num_adversaries = 4
        num_agents = num_adversaries + num_good_agents
        self.num_agents = num_agents
        self.num_adversaries = num_adversaries
        num_landmarks = 2
        top_n_list = obs_range
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.id  = i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 6.0 if agent.adversary else 8.0
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 2.0 if agent.adversary else 2.6
#            agent.cover_radius = 0.5
            agent.top_n = min(top_n_list[i], num_agents)
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        world.num_adversaries=num_adversaries
        world.done = 0
        self.central_node = np.array([0,0])
        return world


    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)
        world.catch = 0


    def benchmark_data(self, agent, world):
        # # returns data for benchmarking purposes
        # if agent.adversary:
        #     collisions = 0
        #     for a in self.good_agents(world):
        #         if self.is_collision(a, agent):
        #             collisions += 1
        #     return collisions
        # else:
        #     return 0
        collisions = 0
        rew = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        if agent.adversary:
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
        return (rew, collisions)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
#        def bound(x):
#            if x < 0.9:
#                return 0
#            if x < 1.0:
#                return (x - 0.9) * 10
#            return min(np.exp(2 * x - 2), 10)
#        for p in range(world.dim_p):
#            x = abs(agent.state.p_pos[p])
#            rew -= bound(x)
#        return rew
    
    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = True
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 2
        return rew
                    
    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        agents = self.good_agents(world) #prey
        adversaries = self.adversaries(world)
        if agent.id==0:
            self.rewards_all(world, adversaries, agents)
        dis= self.dis_n[agent.id]
        rew = - dis
        if dis < 0.1:
            rew += 2
        if agent.collide:
            for a in world.agents:
                if a is agent: 
                    continue
                elif self.is_collision(a, agent):
                    rew -= 2
#        rew = 0
#        shape = True
#
#        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
#            for adv in adversaries:
#                rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
#        if agent.collide:
#            for ag in agents:
#                for adv in adversaries:
#                    if self.is_collision(ag, adv):
#                        rew += 2
#                        break
        return rew
    
    def rewards_all(self, world, adversaries, agents):
        dis_matrix = []
        for a in adversaries:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for l in agents]
            dis_matrix.append(dists)
        dis_matrix = np.array(dis_matrix)
        dis_n = [0]*self.num_adversaries
        for i in range(self.num_adversaries):
            # p = dis_matrix.argmin(dis_matrix)
            p = np.unravel_index(np.argmin(dis_matrix),dis_matrix.shape)
            dis_n[p[0]] = dis_matrix[p]
            dis_matrix[p[0] ] = np.array([100]*self.num_adversaries)
            dis_matrix[:, p[1]] = np.array([100]*self.num_adversaries)
        self.dis_n = dis_n
        world.catch = len([i for i in dis_n if i < 0.1])
    
#        
#    def is_collision_list(self, agent1, agent2, k):
#        delta_pos = agent1.state.p_pos_list[k] - agent2.state.p_pos_list[k]
#        dist = np.sqrt(np.sum(np.square(delta_pos)))
#        dist_min = agent1.size + agent2.size
#        return True if dist < dist_min else False
#    
#    def agent_reward(self, agent, world):#prey
#        # Agents are negatively rewarded if caught by adversaries
#        rew_list = []
#        for k in range(5):
#            rew = 0
#            shape = True
#            adversaries = self.adversaries(world)
#            if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
#                for adv in adversaries:
#                    rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos_list[k] - adv.state.p_pos_list[k])))
#            if agent.collide:
#                for a in adversaries:
#                    if self.is_collision_list(a, agent, k):
#                        rew -= 10
#            rew_list.append(rew)
#        return np.mean(rew_list)
#    
#    def adversary_reward(self, agent, world):
#        # Adversaries are rewarded for collisions with agents
#        rew_list = []
#        for k in range(5):
#            rew = 0
#            shape = True
#            agents = self.good_agents(world)
#            adversaries = self.adversaries(world)
#            if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
#                for adv in adversaries:
#                    rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos_list[k] - adv.state.p_pos_list[k]))) for a in agents])
#            if agent.collide:
#                for ag in agents:
#                    for adv in adversaries:
#                        if self.is_collision_list(ag, adv, k):
#                            rew += 10
#            rew_list.append(rew)
#        return np.mean(rew_list)




#    def observation(self, agent, world):
#        # get positions of all entities in this agent's reference frame
#        entity_pos = []
#        for entity in world.landmarks:
#            if not entity.boundary:
#                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
#        entity_pos = heapq.nsmallest(3,entity_pos, key=lambda s: np.sum(np.square(s)))
#        # communication of all other agents
##        comm = []
#        other_pos = []
#        other_vel = []
#        for other in world.agents:
#            if other is agent: continue
##            comm.append(other.state.c)
#            distance = other.state.p_pos - agent.state.p_pos
##            if other_pos[-1] <= other.cover_radius: # partial observation
#            other_pos.append(distance)
#            if not other.adversary:
#                other_vel.append(other.state.p_vel)
#        other_pos = heapq.nsmallest(3,other_pos, key=lambda s: np.sum(np.square(s))) # top 3
#        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)
#    
         
    # def observation(self, agent, world, pack_size, map_size):
    #     # get positions of all entities in this agent's reference frame
    #     entity_pos = []
    #     for entity in world.landmarks:  # world.entities:
    #         entity_pos.append(entity.state.p_pos - agent.state.p_pos)
    #     # entity colors
    #     entity_color = []
    #     for entity in world.landmarks:  # world.entities:
    #         entity_color.append(entity.color)
    #     # communication of all other agents
    #     other_pos, Com_Delay, other_p_vel = [], [], []
    #     prey_pos, prey_vels = [], []
    #     for other in world.agents: 
    #         if other is agent:
    #             if other.adversary:
    #                 Com_Delay.append(0)
    #             else:
    #                 p = other.state.p_pos - agent.state.p_pos 
    #                 prey_pos.append(p)
    #                 prey_vels.append(other.state.p_vel)
    #             continue
    #         p = other.state.p_pos - agent.state.p_pos 
    #         if not other.adversary: # prey
    #             prey_pos.append(p)
    #             prey_vels.append(other.state.p_vel)
    #         else:
    #             dis = np.sqrt(np.sum(np.square(p)))
    #             if dis==0:
    #                 Com_Delay.append(0.1)
    #             else:
    #                 Com_Delay.append(max(1, self.sending_delay(self.num_agents, dis/2*map_size, pack_size*0.1))/world.est_delay) #dis in map_size*map_sizem^2
    #             other_pos.append(p)
    #             other_p_vel.append(other.state.p_vel)
    #     entity_pos = sorted(entity_pos, key=lambda s: np.sum(np.square(s)))[:agent.top_n]
    #     index = sorted(range(len(other_pos)), key=lambda i: np.sum(np.square(other_pos[i])))[:agent.top_n]
    #     other_pos= [other_pos[i] for i in index]
    #     other_p_vel = [other_p_vel[i] for i in index]

    #     return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos
    #                           + prey_pos + other_p_vel +prey_vels + [Com_Delay])

    def observation(self, agent, world, pack_size, map_size):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        other_pos, Com_Delay, other_p_vel = [], [], []
        prey_pos, prey_vels = [], []
        for other in world.agents: 
            if other is agent:
                if other.adversary:
                    Com_Delay.append(0)
                else:
                    p = other.state.p_pos - agent.state.p_pos 
                    prey_pos.append(p)
                    prey_vels.append(other.state.p_vel)
                continue
            p = other.state.p_pos - agent.state.p_pos 
            if not other.adversary: # prey
                prey_pos.append(p)
                prey_vels.append(other.state.p_vel)
            else:
                dis = np.sqrt(np.sum(np.square(p)))
                if dis==0:
                    Com_Delay.append(0.1)
                else:
                    Com_Delay.append(max(1, self.sending_delay(self.num_agents, dis/2*map_size, pack_size*0.1))/world.est_delay) #dis in map_size*map_sizem^2
                other_pos.append(p)
                other_p_vel.append(other.state.p_vel)
#        entity_pos = sorted(entity_pos, key=lambda s: np.sum(np.square(s)))[:agent.top_n]
        index = sorted(range(len(other_pos)), key=lambda i: np.sum(np.square(other_pos[i])))[:agent.top_n]
        other_pos= [other_pos[i] for i in index]
        other_p_vel = [other_p_vel[i] for i in index]

        index_ = sorted(range(len(prey_pos)), key=lambda i: np.sum(np.square(prey_pos[i])))[:agent.top_n]
        prey_pos= [prey_pos[i] for i in index_]
        prey_vels = [prey_vels[i] for i in index_]

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos
                              + prey_pos + other_p_vel + prey_vels + [Com_Delay])

    def get_centralnode_delay(self, world, num_agents, pack_size, map_size):
        dalay_n = []
        for agent in world.agents: 
            p = agent.state.p_pos - self.central_node
            dis = np.sqrt(np.sum(np.square(p)))
            dalay_n.append(self.sending_delay(num_agents, dis/2*map_size, pack_size*0.1)/world.est_delay)
#        dalay_n_max = np.max(dalay_n)        
        return  dalay_n
        
    
    def sending_delay(self, Num, distance, size):  # distance in m, size of packages in Mb
#        power = 1#w
#        Bandwidth = 10e3 #10Mhz
#        zeta_NLoS = 2.4
#        g = -(10* zeta_NLoS *math.log(distance,10) + self.X_0)
#        eta = math.pow(10, g/10)*power/self.sigma_2 # W/W
#        X_G2G = Bandwidth/Num* math.log(1+eta)/1e3 #in Mbps
#        return size/X_G2G*1e3 + np.random.uniform(0,5)
        return size*distance*10