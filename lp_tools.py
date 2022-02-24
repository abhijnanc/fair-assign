
import numpy as np

def L2Distance(data):

  transposed = np.expand_dims(data, axis = 1)
  distance = np.power(data - transposed, 2)
  distance = np.power(np.abs(distance).sum(axis = 2), 0.5) 
  distance=distance*110


  return distance


def cost_function(dataset, centres, num_variables):

  transposed = np.expand_dims(dataset, axis = 1)  
  distance = np.power(transposed - centres, 2).sum(axis = 2)
  distance = distance.reshape(-1)		# Row major reshaping

  zero_vector = np.zeros(num_variables - len(distance))	# Pad the rest with 0s. They don't contribute to cost
  objective = np.concatenate([distance, zero_vector], axis = 0)
  return objective
  """


	Given the dataset and cluster centres, this function returns the distance between all samples and cluster centres and returns the lp objective coefficients
	Input:
		dataset - [ num_samples x dim ] 2D numpy matrix containing the dataset
		centres - [ num_clusters x dim ] 2D numpy matrix containing the cluster centres
	Output:
		objective - [ num_variables ] 1D numpy matrix containing weights for corresponding variables
	"""


def prepare_to_add_variables(dataset, centres):

  """
  Assumes TV distance as statistical distance metric. 
  Takes as input the dataset and cluster centres proposed by some clustering algorirthm and returns the weight vectors
  Input:
    dataset - [ num_samples x dim ] 2D numpy matrix containing the dataset
    centres - [ num_clusters x dim ] 2D numpy matrix containing the cluster centres
  Output:
    objective - [ num_variables ] 1D numpy array containing the weight of different variables
    lower_bound - [ num_variables ] 1D array specifying the minimum value of a variable
    upper_bound - [ num_variables ] 1D array specifying the maximum value of a variable
    variable_names - [ num_variables ] 1D sting list consisting of the names of variables 
    P, C - P and C array contain the id of the corresponding variables
  Details:
    P_i_k is a real probability variable that denotes the weight for connecting ith sample to kth clusterrr
    C_i_j_k is a intermediate variable that is an upper bound on | P_i_k - P_j_k | - A hack to convert non linear abs constraint to a linear constraint
  """

  num_samples = len(dataset)
  num_centres = len(centres)
  _id = 0

  # P and C array contain the id of the corresponding variables
  P = np.zeros([num_samples, num_centres]).astype(int)
  C = np.zeros([num_samples, num_samples, num_centres]).astype(int)
 
  probability_variables = []
  for _point in range(num_samples):
    for _centre in range(num_centres):

      probability_variables.append("P_{point}_{centre}".format(
          point = _point,
          centre = _centre
        ))
      # Keep track of P_i_j's position in the lp variable vector 
      P[_point][_centre] = _id
      _id += 1

  abs_constraint_variables = []
  for _point1 in range(num_samples):
    for _point2 in range(_point1 + 1, num_samples):
      for _centre in range(num_centres):

        abs_constraint_variables.append("C_{point1}_{point2}_{centre}".format(
            point1 = _point1,
            point2 = _point2,
            centre = _centre
          ))
        # Keep track of C_i_j_k's position in lp variable vector
        C[_point1][_point2][_centre] = C[_point2][_point1][_centre] = _id
        _id += 1


  # Concatenating the names of both the types of variables
  variable_names = probability_variables + abs_constraint_variables


  # Setting lower bound = 0 and upper bound = 1 for all the variables
  num_variables = len(variable_names)
  lower_bound = [0.0 for i in range(num_variables)]
  upper_bound = [1.0 for i in range(num_variables)]

  # Computing the coefficients for objective function
  objective = cost_function(dataset, centres, num_variables)

  return objective, lower_bound, upper_bound, variable_names, P, C



def prepare_to_add_constraints(dataset, centres, upper_cap, lower_cap, P, C, cons, fair_distance):

  
  num_samples = len(dataset)
  num_centres = len(centres)

  rhs = []
  senses = []
  row_names = []
  coefficients = []
  eqn_id = 0				# Denotes the id of the constraint being processed currently

  distance = L2Distance(dataset)


	# Constraint type 1: Summation of P values over all clusters = 1 for each sample
  for point in range(num_samples):
    rhs.append(1.0)
    senses.append("E")
    row_names.append("{eqn_id}_Total_probability_{pt}".format(
        pt = point,
        eqn_id=eqn_id
      ))
    
    for centre in range(num_centres):
      coefficients.append((eqn_id, int(P[point][centre]), 1))

    eqn_id += 1
  



  # Constraint type 2: Number of expected delivery agents assigned to ffc is lesser than max capacity of ffc
  for centre in range(num_centres):
    rhs.append(float(upper_cap[centre]))
    senses.append("L")
    row_names.append("{eqn_id}_Max_Cap_{centre}".format(
            eqn_id = eqn_id,
            centre = centre
          ))
    for point in range(num_samples):
      coefficients.append((eqn_id,int(P[point][centre]),1))

    eqn_id+=1

  


  # Constraint type 3: Number of expected delivery agents assigned to ffc is greater than min capacity of ffc
  for centre in range(num_centres):
    rhs.append(float(lower_cap[centre]))
    senses.append("G")
    row_names.append("{eqn_id}_Min_Cap_{centre}".format(
						eqn_id = eqn_id,
						centre = centre
					))
    for point in range(num_samples):
      coefficients.append((eqn_id,int(P[point][centre]),1))

    eqn_id+=1



  
  # Constraint type 4: Lower and upper bound the abs values using their corresponding C variables
  for _point1 in range(num_samples):
    for _point2 in range(_point1+1,num_samples):
      for centre in range(num_centres):
        if distance[_point1][_point2]>=fair_distance:
          continue

        # Upper bound - P[i][k] - P[j][k] <= C[i][j][k]
        rhs.append(0)
        senses.append("L")
        row_names.append("{eqn_id}_Upper_Bound_{pt1}_{pt2}_{centre}".format(
            eqn_id = eqn_id,
            pt1 = _point1,
            pt2 = _point2,
            centre = centre
          ))
        coefficients.append((eqn_id, int(P[_point1][centre]), 1))
        coefficients.append((eqn_id, int(P[_point2][centre]), -1))
        coefficients.append((eqn_id, int(C[_point1][_point2][centre]), -1))

        eqn_id += 1

        # Lower_bound - P[i][k] - P[j][k] >= -C[i][j][k]
        rhs.append(0)
        senses.append("G")
        row_names.append("{eqn_id}_Lower_bound_{pt1}_{pt2}_{centre}".format(
            eqn_id = eqn_id,
            pt1 = _point1,
            pt2 = _point2,
            centre = centre
          ))
        coefficients.append((eqn_id, int(P[_point1][centre]), 1))
        coefficients.append((eqn_id, int(P[_point2][centre]), -1))
        coefficients.append((eqn_id, int(C[_point1][_point2][centre]), 1))

        eqn_id += 1



	# Constraint type 5: Add fairness constraints in terms of C variables - trick to make mod constraints linear
  for _point1 in range(num_samples):
    for _point2 in range(_point1+1,num_samples):
      if distance[_point1][_point2]<fair_distance:
        rhs.append(cons* distance[_point1][_point2])		# Multiply by 2 to account for the division in TV distance expression
        senses.append("L")
        row_names.append("{eqn_id}_Fainess_{pt1}_{pt2}".format(
            eqn_id = eqn_id,
            pt1 = _point1,
            pt2 = _point2
          ))

        for centre in range(num_centres):
          coefficients.append((eqn_id, int(C[_point1][_point2][centre]), 1))

        eqn_id += 1

  


  return rhs, senses, row_names, coefficients