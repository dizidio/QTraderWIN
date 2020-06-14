from agent.agent import Agent
from functions import *
import sys
import pandas as pd

window_size, episode_count = 15, 1000

agent = Agent(window_size)
data = pd.read_csv('./data/WIN$N_close.csv')['WIN$N_close'].iloc[-1000:]
data = list(data.values)
#data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32

for e in range(episode_count + 1):
	print ("Episode " + str(e) + "/" + str(episode_count))
	state = getState(data, 0, window_size + 1)

	total_profit = 0
	current_pos = 0
	last_price = 0

	for t in range(l):
		action = agent.act(state)

		# sit
		next_state = getState(data, t + 1, window_size + 1)
		reward = 0

		if action == 1 and current_pos == 0: # buy
			last_price = data[t]
			current_pos = 1
			print("Buy: " + formatPrice(data[t]))

		if action == 2 and current_pos == 0: # sell
			last_price = data[t]
			current_pos = 2
			print("Sell: " + formatPrice(data[t]))

		elif action == 1 and current_pos == 2: # buy
			reward = max(last_price - data[t], 0)
			total_profit += last_price - data[t]
			print("Buy: " + formatPrice(data[t]) + " | Profit: " + formatPrice(last_price - data[t]))
			current_pos = 0

		elif action == 2 and current_pos == 1: # sell
			reward = max(data[t] - last_price, 0)
			total_profit += data[t] - last_price
			print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - last_price))
			current_pos = 0

		done = True if t == l - 1 else False
		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			print ("--------------------------------")
			print ("Total Profit: " + formatPrice(total_profit))
			print ("--------------------------------")

		if len(agent.memory) > batch_size:
			agent.expReplay(batch_size)

	if e % 10 == 0:
		agent.model.save("./models/model_ep" + str(e))
