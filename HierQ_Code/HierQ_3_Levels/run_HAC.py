from tkinter import *
from tkinter import ttk
import time
import numpy as np
from grid_world import Grid_World
from setup_flags import set_up
import pickle as cpickle

TEST_FREQ = 2


def get_next_state(state_mat,state, action):

        num_col = state_mat.shape[1]
        num_row = state_mat.shape[0]

        state_row = int(state/num_col)    
        state_col = state % num_col
        

        # print("State row: ", state_row)
        # print("State col: ", state_col)

        # If action is "left"
        if action == 0:
            if state_col != 0 and state_mat[state_row][state_col - 1] == 1:
                # print("Moving Left")
                state -= 1
        # If action is "up"
        elif action == 1:
            if state_row != 0 and state_mat[state_row - 1][state_col] == 1:
                # print("Moving Up")
                state -= num_col
        # If action is "right"
        elif action == 2:
            if state_col != (num_col - 1) and state_mat[state_row][state_col + 1] == 1:
                # print("Moving Right")
                state += 1
        # If action is "down"
        else:
            if state_row != (num_row - 1) and state_mat[state_row + 1][state_col] == 1:
                # print("Moving Down")
                state += num_col

        return state



def run_HAC(state_mat,agent,env,FLAGS,time_limits):

    MAX_LAYER_0_ITR = time_limits[0]
    MAX_LAYER_1_ITR = time_limits[1]

    
    if FLAGS.test:  
        print("\nSubgoal Q Values: ")
        print(agent.critic_lay1[4][0])
        print(agent.critic_lay1[4][2])
        print("\n")  
        
    

    test_period = 0

    max_test_time = state_mat.shape[0] * state_mat.shape[1]

    for episode in range(201):

        """
        # Print Layer 2's Q Table
        print("\n\nEpisode %d Layer 2 Qs: " % episode)
        print(agent.critic_lay2)
        print("\n\n")
        """

        if FLAGS.mix and episode % TEST_FREQ == 0:
            FLAGS.test = True
            
        # Reset environment
        state = 0

        # Use below goal for regular grid world
        # goal = state_mat.shape[0] * state_mat.shape[1] - 1

        # Use below goal for four rooms environment
        goal = 10
        goal_row = 2
        goal_col = 7
        goal = goal_row * state_mat.shape[1] + goal_col
        

        if FLAGS.show:
            env.reset_env(state,goal)

        end_goal_achieved = False

        t2 = 0

        old_subgoal_1 = -1
        old_subgoal_2 = -1

        layer_1_trailing_states = [state]
        layer_2_trailing_states = [state]
        

        total_steps = 0

        while not end_goal_achieved:

            initial_state_lay_2 = np.copy(state)
            initial_state_lay_2 = int(initial_state_lay_2)

            # Track previous subgoal for visualization purposes
            if total_steps > 0:
                old_subgoal_2 = int(np.copy(subgoal_2))

            # Get next subgoal
            subgoal_2, act_2_type = agent.get_action(initial_state_lay_2,FLAGS,2,goal)

            # Print Subgoal
            num_cols = state_mat.shape[1]
            row_num_2 = int(subgoal_2 / num_cols)
            col_num_2 = subgoal_2 % num_cols
            # print("\n Layer 2 Itr %d Subgoal: %d or (%d,%d)" % (t2,subgoal_2, row_num_2,col_num_2))

            layer_1_achieved = False

            for t1 in range(MAX_LAYER_1_ITR):

                initial_state = np.copy(state)
                initial_state = int(initial_state)
                
                if total_steps > 0:
                    old_subgoal_1 = int(np.copy(subgoal_1))

                # Get next subgoal
                subgoal_1, act_1_type = agent.get_action(initial_state,FLAGS,1,subgoal_2)

                # Display subgoal
                if FLAGS.show:
                    env.display_subgoals(subgoal_1,old_subgoal_1, subgoal_2, old_subgoal_2, state,goal)

                # print("Next Subgoal: ", subgoal_1)
                # time.sleep(2)

                # Print Subgoal
                row_num = int(subgoal_1 / num_cols)
                col_num = subgoal_1 % num_cols
                # print("\n Layer 1 Itr %d Subgoal: %d or (%d,%d)" % (t1,subgoal_1, row_num,col_num))

                layer_0_achieved = False


                for t0 in range(MAX_LAYER_0_ITR):

                    old_state = np.copy(state)
                    old_state = int(old_state)

                    # Get epsilon-greedy action from agent
                    action, act_0_type = agent.get_action(old_state,FLAGS,0,subgoal_1)
                    # action = np.random.randint(0,4)
                    
                    act_strings = ["left","up","right","down"]
                    
                    """
                    if act_0_type is not "Random":
                        print("Layer 2 Itr %d, Layer 1 Itr %d, Layer 0 Itr %d Policy Action: %d" % (t2,t1,t0,action), act_strings[action])
                    else:
                        print("Layer 2 Itr %d, Layer 1 Itr %d, Layer 0 Itr %d Random Action: %d" % (t2,t1,t0,action), act_strings[action])
                    """
                    
                    

                    # Get next state
                    state = get_next_state(state_mat,old_state,action)
                    # print("Next State: ", state)
                    # print("\n\nEpisode %d, HL Itr %d, LL Itr %d" % (episode, t1, t0))
                    # print("Old State: %d, Action %d, New State: %d, Subgoal: %d" % (old_state,action,state,subgoal_1))
                    
                    total_steps += 1

                    if state != old_state:
                        layer_1_trailing_states.append(old_state)
                        layer_2_trailing_states.append(old_state)
                    if len(layer_1_trailing_states) > MAX_LAYER_0_ITR:
                        layer_1_trailing_states.pop(0)
                    if len(layer_2_trailing_states) > (MAX_LAYER_1_ITR * MAX_LAYER_0_ITR):
                        layer_2_trailing_states.pop(0)

                    # Visualize action if necessary
                    if FLAGS.show:
                        env.step(old_state,state,goal)
                    
                    # Determine reward and whether any of the goals achieved
                    reward_0 = -1   
                    if state == subgoal_1:
                        reward_0 = 0
                        layer_0_achieved = True
                        # print("Layer 0 Subgoal hit!")

                    reward_1 = -1
                    if state == subgoal_2:
                        reward_1 = 0
                        layer_1_achieved = True
                        # print("Layer 1 Subgoal hit!")

                    reward_2 = -1
                    if state == goal:
                        reward_2 = 0
                        end_goal_achieved = True
                        print("Episode %d, L2 Itr %d, L1 Itr %d, L0 Itr %d: Goal hit!" % (episode,t2,t1,t0))
                        print("Total Steps: ", total_steps)

                    # Update critic lookup tables
                    if not FLAGS.test:

                        # Create layer 0 transitions
                        
                        # Create series of transitions evaluating performance of action given every possible subgoal state
                        
                        num_states = state_mat.shape[0] * state_mat.shape[1]

                        for i in range(num_states):
                            if i != state:
                                hindsight_trans_lay0 = [old_state,action,-1,state,i,False] 
                            else:
                                hindsight_trans_lay0 = [old_state,action,0,state,i,True]
                            agent.update_critic_lay0(np.copy(hindsight_trans_lay0)) 
                        
                        # print("State %d Q-Values: " % old_state, agent.critic_lay0[subgoal_1][old_state])
                        
                        # Create layer 1 transitions

                        # Create high level hindsight transitions in which each of the past MAX_LAYER_0_ITR states serves as the inital state, "state" serves as the subgoal, and every possible sugoal state serves as goal 
                        # print("\nLayer 1 Trailing Trans: ", layer_1_trailing_states)                 
                        for i in range(len(layer_1_trailing_states)):
                            for j in range(num_states):
                                if j != state:
                                    agent.update_critic_lay1(layer_1_trailing_states[i],state,-1,state,j,False)
                                else:
                                    agent.update_critic_lay1(layer_1_trailing_states[i],state,0,state,j,True)
                                    # print("Updated Q for State %d, Goal 1 %d, Goal 2 %d: " % (layer_1_trailing_states[i],state,int(j)), agent.critic_lay1[int(j)][layer_1_trailing_states[i]][state])
                        
                        
                        # Create layer 2 transitions
                        for i in range(len(layer_2_trailing_states)):
                            agent.update_critic_lay2(layer_2_trailing_states[i],state,reward_2,state,goal,end_goal_achieved)
                    
                    if layer_0_achieved or layer_1_achieved or end_goal_achieved:
                        break           
                    

                if layer_1_achieved or end_goal_achieved:
                    break

            t2 += 1

            if (end_goal_achieved or total_steps >= max_test_time) and FLAGS.mix and episode % TEST_FREQ == 0:

                FLAGS.test = False
                print("Test Period %d Result: " % test_period, end_goal_achieved)
                test_period += 1
                    
                break
                

    # Save and Print Q-Table
    cpickle.dump(agent.critic_lay2,open("critic_lay2_table.p","wb"))
    cpickle.dump(agent.critic_lay1,open("critic_lay1_table.p","wb"))
    cpickle.dump(agent.critic_lay0,open("critic_lay0_table.p","wb"))
    
    print("Critic Tables Saved")

    """
    print("Q-Table: ")
    print(agent.critic)

    print("\nPolicy: ")
    policy = agent.get_policy(state_mat)
    for i in policy:
        print(i)
    print("\n")
    """
            

        

        
