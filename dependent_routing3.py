"""
Implementation of dependent rounding in https://dl.acm.org/doi/abs/10.1145/1147954.1147956
"""

import numpy as np
import random
import math

EPS = 1e-7









def isFraction(value):

	"""
	Returns if value is a fraction between 0 and 1.
	"""

	return value > EPS and value < 1 - EPS

def isNotFraction(value):

	"""
	Returns if value is either 0 or 1.
	"""

	return not isFraction(value)


class DependentRounding:

	def __init__(self, weights):

		"""
		weights:	[num_clients x num_facilities] probability of assigning client i to facility j. 
		"""

		sum_weights = np.sum(weights, axis = 1)
		for i in range(len(sum_weights)):
			if abs(sum_weights[i]-1)>EPS:
				print(sum_weights[i]-1,i)
			assert sum_weights[i] > 1 - EPS and sum_weights[i] < 1 + EPS, "Sum of weights \
															over all facilities must sum to one."

		self.weights = weights

		self.num_clients = weights.shape[0]
		self.num_facilities = weights.shape[1]
		self.total_nodes = self.num_clients + self.num_facilities

		self.facilityCode = [i for i in range(self.num_facilities)]
		self.clientCode = [i + self.num_facilities for i in range(self.num_clients)]

		self.graph = self._buildGraph(weights)
		self.cycle_found = False
		self.maximal_path_found = False

		return None

	def _buildGraph(self, weights):

		"""
		Builds a bipartite graph with fractional edges only.
		"""

		graph = [[] for i in range(self.total_nodes)]

		for i in range(self.num_clients):
			for j in range(self.num_facilities):
				if(isFraction(weights[i][j])):
					graph[self.clientCode[i]].append(self.facilityCode[j])
					graph[self.facilityCode[j]].append(self.clientCode[i])

		# Maintain neighbours in a set for fast removal when edge becomes integral.
		for i in range(self.total_nodes):
			graph[i] = set(graph[i])

		return graph

	def _decompose(self, edge):

		"""
		Decompose an edge to client and facility component.
		"""

		assert len(edge) == 2, "Invalid edge"

		if(edge[0] < self.num_facilities):
			facility = edge[0]
			client = edge[1] - self.num_facilities
		else:
			facility = edge[1]
			client = edge[0] - self.num_facilities

		assert client >= 0 and facility >= 0, "Invalid encoding found"
		assert client < self.num_clients and facility < self.num_facilities, "Invalid encoding found"

		return client, facility

	def _rebalance(self, edges):

		"""
		Rebalance edges in the cycle or a maximal path.
		It is guaranteed that there are even number of edges starting and ending at a facility.
		"""

		assert len(edges) % 2 == 0, "Rebalancing doesn't work on odd set of edges"

		alpha = 1
		beta = 1

		for i in range(len(edges)):

			client, facility = self._decompose(edges[i])
			w = self.weights[client][facility]
			if(i % 2 == 0):
				alpha = min(alpha, 1 - w)
				beta = min(beta, w)
			else:
				alpha = min(alpha, w)
				beta = min(beta, 1 - w)

		alpha_chain_probability = beta / (alpha + beta)
		beta_chain_probability = alpha / (alpha + beta)

		rng = random.random()
		if(rng < alpha_chain_probability):
			to_add = alpha		
		else:
			to_add = beta
			# Rotate the chain by 1 and add
			last = edges.pop()
			edges.insert(0, last)

		for i in range(len(edges)):
			client, facility = self._decompose(edges[i])
			# Add to edge
			if(i % 2 == 0):
				self.weights[client][facility] += to_add
			# Subtract from edge
			else:
				self.weights[client][facility] -= to_add

			assert self.weights[client][facility] >= -EPS \
						and self.weights[client][facility] <= 1 + EPS, "Weight is out of bounds"

		# Remove integral edges from the graph
		for i in range(len(edges)):
			client, facility = self._decompose(edges[i])

			if(isNotFraction(self.weights[client][facility])):
				
				assert self.facilityCode[facility] in self.graph[self.clientCode[client]], \
																	"Facility not found in graph"
				assert self.clientCode[client] in self.graph[self.facilityCode[facility]], \
																	"Client not found in graph"

				self.graph[self.clientCode[client]].remove(self.facilityCode[facility])
				self.graph[self.facilityCode[facility]].remove(self.clientCode[client])


	def _cycle_cancel(self, node, par = -1, visited = None, dfs_stack = []):

		"""
		Runs an iterative dfs to detect cycles and cancel them probabilistically using the alpha 
		or beta chain.
		
		Probability of picking the alpha chain: beta / (alpha + beta)
		Probability of picking the beta chain: alpha / (alpha + beta)
		node 			: 	Node that is currently being processed.
		visited			: 	Set of facilities visited so far
		dfs_stack		: 	A set alternating between clients and facilities in the bipartite graph
							denoting the current dfs stack.
		"""


		if (visited is None):
			visited = np.zeros(self.total_nodes).astype(bool)

		if (self.cycle_found):
			return True

		# Cycle found
		if (visited[node]):
			source = node

			edges = []
			last_visit = dfs_stack.pop()
			while(last_visit != source):
				edges.append([node, last_visit])
				node = last_visit
				last_visit = dfs_stack.pop()
			edges.append([node, last_visit])

			self._rebalance(edges)
			self.cycle_found = True
			return self.cycle_found

		visited[node] = True
		dfs_stack.append(node)
		for child in self.graph[node]:
			if(child != par):
				self._cycle_cancel(child, node, visited, dfs_stack)
				if(self.cycle_found):
					return self.cycle_found
		dfs_stack.pop()

		return self.cycle_found

	def _maximal_path_cancel(self, node, par = -1, dfs_stack = []):

		"""
		Runs an iterative dfs to detect cycles and cancel them probabilistically using the alpha 
		or beta chain.
		
		Probability of picking the alpha chain: beta / (alpha + beta)
		Probability of picking the beta chain: alpha / (alpha + beta)
		node 			: 	Node that is currently being processed.
		visited			: 	Set of facilities visited so far
		dfs_stack		: 	A set alternating between clients and facilities in the bipartite graph
							denoting the current dfs stack.
		"""


		if (self.maximal_path_found):
			return True

		# Maximal path found
		if (par != -1 and len(self.graph[node]) == 1):
			edges = []
			last_visit = dfs_stack.pop()
			while(len(dfs_stack) > 0):
				edges.append([node, last_visit])
				node = last_visit
				last_visit = dfs_stack.pop()
			edges.append([node, last_visit])

			self._rebalance(edges)
			self.maximal_path_found = True
			return self.maximal_path_found

		dfs_stack.append(node)
		for child in self.graph[node]:
			if(child != par):
				self._maximal_path_cancel(child, node, dfs_stack)
				if(self.maximal_path_found):
					return self.maximal_path_found
		dfs_stack.pop()

		return self.maximal_path_found

	def round(self):

		"""
		Rounds the fractional weights such that the rounded values follow the distribution 
		specified in weights, with capacities (sum of clients over a facility) lower or upper 
		rounded to the nearest integer.
		"""

		capacities = np.sum(self.weights, axis = 0)

		# Cancel all the cycles in the graph
		for i in range(self.num_facilities):

			# Keep cancelling cycles as long as they exist
			while(self._cycle_cancel(self.facilityCode[i])):
				# This state needs to be reset for each cycle finding iteration.
				self.cycle_found = False

		# Cancel all maximal paths in the graph
		more_maximal_paths_exist = True
		while(more_maximal_paths_exist):
			
			more_maximal_paths_exist = False
			# A maximal path starts in a facility with degree = 1
			for i in range(self.num_facilities):
				if(len(self.graph[i]) == 1):
					self._maximal_path_cancel(self.facilityCode[i])
					self.maximal_path_found = False
					more_maximal_paths_exist = True
					break

		# Assert that all the weights are integral
		for i in range(self.num_clients):
			for j in range(self.num_facilities):
				assert isNotFraction(self.weights[i][j]), "Rounding failed, integer weights remain"

		# Assert that the assignments to a facility is at least floor of its capacity and at most
		# ceil of its capacity
		rebalanced_capacities = np.sum(self.weights, axis = 0)
		for i in range(self.num_facilities):
			if not (math.floor(capacities[i]) <= rebalanced_capacities[i] + EPS and rebalanced_capacities[i] - EPS <= math.ceil(capacities[i]) ):
				print(i,rebalanced_capacities[i],capacities[i], math.floor(capacities[i]),math.ceil(capacities[i]))
			assert math.floor(capacities[i]) <= rebalanced_capacities[i] + EPS \
				and rebalanced_capacities[i] - EPS <= math.ceil(capacities[i]), \
					"Capacity guarantee is not met by the algorithm"

		return self.weights