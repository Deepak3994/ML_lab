class NOT_GATE:
	def __init__(self):
		self.w1 = 1
		self.x = None
		
		

	def activation_function(self):
		yin = self.w1*self.x
		if yin <= 0:
			return 1
		else:
			return 0

	def not_ops(self,x):
		self.x = x
		Y = self.activation_function()
		print('%d\t%d'% (self.x,Y))
		return Y


def main():
	not1 = NOT_GATE()
	Y = not1.not_ops(1)
	Y = not1.not_ops(0)
	
	
main()