class AND_GATE:
	def __init__(self):
		self.w1 = 1
		self.w2 = 1
		self.x1 = None
		self.x2 = None
		

	def activation_function(self):
		yin = self.w1*self.x1+self.w2*self.x2
		if yin >= 2:
			return 1
		else:
			return 0

	def and_ops(self,x1,x2):
		self.x1 = x1
		self.x2 = x2
		Y = self.activation_function()
		print('%d\t%d\t%d'% (x1,x2,Y))
		return Y


def main():
	and1 = AND_GATE()
	Y = and1.and_ops(1,1)
	Y = and1.and_ops(1,0)
	Y = and1.and_ops(0,1)
	Y = and1.and_ops(0,0)
	
main()