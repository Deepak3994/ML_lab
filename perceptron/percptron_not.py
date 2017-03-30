
def activation_function(yin):
	if yin == 0:
		return 1
	elif yin < 0:
		return -1
	else:
		return 0

def train_model(learn_rate,data):
	weights = []
	bias = 0
	weight1 = 0
	learn_rate = 1
	#print(len(data))
	for i in range(len(data)):
		x1 = data[i][0]
		#print(x1)
		y_actual = data[i][1]
		yin = bias + x1 * weight1 
		y_predicted = activation_function(yin)
		#print(y_predicted)
		error = y_actual - y_predicted
		#print(error)
		weight1 = weight1 + x1*learn_rate*error
		bias = bias + learn_rate* error 
		#print(weight1,bias)	
	weights.append(bias)
	weights.append(weight1)
	return(weights)


def predict_not(learn_rate,data,x1):
	weights = train_model(learn_rate,data)
	bias = weights[0]
	w1 = weights[1]
	yin = bias + x1 * w1 
	target = activation_function(yin)
	if target >0:
		y_predicted = 1
	else:
		y_predicted = 0
	print(y_predicted)



def main():
	dataset = [[1,0],
			  [-1,1],]
	l_rate = 1
	predict_not(l_rate,dataset,-1)

main()