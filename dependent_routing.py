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
		weights:	[num_drivers x num_centers] probability of assigning driver i to center j. 
		"""

		sum_weights = np.sum(weights, axis = 1)
		for i in range(len(sum_weights)):
			if abs(sum_weights[i]-1)>EPS:
				print(sum_weights[i]-1,i)
			assert sum_weights[i] > 1 - EPS and sum_weights[i] < 1 + EPS, "Sum of weights \
															over all centers must sum to one."

		self.weights = weights

		self.num_drivers = weights.shape[0]
		self.num_centers = weights.shape[1]
		self.total_nodes = self.num_drivers + self.num_centers

		self.centerCode = [i for i in range(self.num_centers)]
		self.driverCode = [i + self.num_centers for i in range(self.num_drivers)]

		self.graph = self._buildGraph(weights)
		self.cycle_found = False
		self.maximal_path_found = False

		return None

	def _buildGraph(self, weights):

		"""
		Builds a bipartite graph with fractional edges only.
		"""

		graph = [[] for i in range(self.total_nodes)]

		for i in range(self.num_drivers):
			for j in range(self.num_centers):
				if(isFraction(weights[i][j])):
					graph[self.driverCode[i]].append(self.centerCode[j])
					graph[self.centerCode[j]].append(self.driverCode[i])

		# Maintain neighbours in a set for fast removal when edge becomes integral.
		for i in range(self.total_nodes):
			graph[i] = set(graph[i])

		return graph

	def _decompose(self, edge):

		"""
		Decompose an edge to driver and center component.
		"""

		assert len(edge) == 2, "Invalid edge"

		if(edge[0] < self.num_centers):
			center = edge[0]
			driver = edge[1] - self.num_centers
		else:
			center = edge[1]
			driver = edge[0] - self.num_centers

		assert driver >= 0 and center >= 0, "Invalid encoding found"
		assert driver < self.num_drivers and center < self.num_centers, "Invalid encoding found"

		return driver, center

	def _rebalance(self, edges):

		"""
		Rebalance edges in the cycle or a maximal path.
		It is guaranteed that there are even number of edges starting and ending at a center.
		"""

		assert len(edges) % 2 == 0, "Rebalancing doesn't work on odd set of edges"

		alpha = 1
		beta = 1

		for i in range(len(edges)):

			driver, center = self._decompose(edges[i])
			w = self.weights[driver][center]
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
			driver, center = self._decompose(edges[i])
			# Add to edge
			if(i % 2 == 0):
				self.weights[driver][center] += to_add
			# Subtract from edge
			else:
				self.weights[driver][center] -= to_add

			assert self.weights[driver][center] >= -EPS \
						and self.weights[driver][center] <= 1 + EPS, "Weight is out of bounds"

		# Remove integral edges from the graph
		for i in range(len(edges)):
			driver, center = self._decompose(edges[i])

			if(isNotFraction(self.weights[driver][center])):
				
				assert self.centerCode[center] in self.graph[self.driverCode[driver]], \
																	"center not found in graph"
				assert self.driverCode[driver] in self.graph[self.centerCode[center]], \
																	"driver not found in graph"

				self.graph[self.driverCode[driver]].remove(self.centerCode[center])
				self.graph[self.centerCode[center]].remove(self.driverCode[driver])


	def _cycle_cancel(self, node, par = -1, visited = None, dfs_stack = []):

		"""
		Runs an iterative dfs to detect cycles and cancel them probabilistically using the alpha 
		or beta chain.
		
		Probability of picking the alpha chain: beta / (alpha + beta)
		Probability of picking the beta chain: alpha / (alpha + beta)
		node 			: 	Node that is currently being processed.
		visited			: 	Set of centers visited so far
		dfs_stack		: 	A set alternating between drivers and centers in the bipartite graph
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
		visited			: 	Set of centers visited so far
		dfs_stack		: 	A set alternating between drivers and centers in the bipartite graph
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
		specified in weights, with capacities (sum of drivers over a center) lower or upper 
		rounded to the nearest integer.
		"""

		capacities = np.sum(self.weights, axis = 0)

		# Cancel all the cycles in the graph
		for i in range(self.num_centers):

			# Keep cancelling cycles as long as they exist
			while(self._cycle_cancel(self.centerCode[i])):
				# This state needs to be reset for each cycle finding iteration.
				self.cycle_found = False

		# Cancel all maximal paths in the graph
		more_maximal_paths_exist = True
		while(more_maximal_paths_exist):
			
			more_maximal_paths_exist = False
			# A maximal path starts in a center with degree = 1
			for i in range(self.num_centers):
				if(len(self.graph[i]) == 1):
					self._maximal_path_cancel(self.centerCode[i])
					self.maximal_path_found = False
					more_maximal_paths_exist = True
					break

		# Assert that all the weights are integral
		for i in range(self.num_drivers):
			for j in range(self.num_centers):
				assert isNotFraction(self.weights[i][j]), "Rounding failed, integer weights remain"

		# Assert that the assignments to a center is at least floor of its capacity and at most
		# ceil of its capacity
		rebalanced_capacities = np.sum(self.weights, axis = 0)
		for i in range(self.num_centers):
			if not (math.floor(capacities[i]) <= rebalanced_capacities[i] + EPS and rebalanced_capacities[i] - EPS <= math.ceil(capacities[i]) ):
				print(i,rebalanced_capacities[i],capacities[i], math.floor(capacities[i]),math.ceil(capacities[i]))
			assert math.floor(capacities[i]) <= rebalanced_capacities[i] + EPS \
				and rebalanced_capacities[i] - EPS <= math.ceil(capacities[i]), \
					"Capacity guarantee is not met by the algorithm"

		return self.weights