class Hook:
	def __init__(self):
		pass

	def before_run(self, runner):
		pass


	def after_run(self, runner):
		pass

class PrintHook(Hook):
	def __init__(self, configs):
		super(PrintHook, self).__init__()
		self.id = configs['ID']

	def before_run(self, runner):
		print("PrintHook: " + str(self.id))

class SaveHook(Hook):
	def __init__(self, configs):
		super(SaveHook, self).__init__()
		self.count = 0

	def after_run(self, runner):
		with open("./saveHook.txt", 'a') as f:
			line = "save" + str(self.count) + "\n"
			f.write(line)
		self.count += 1

__all__ = {
	'PrintHook': PrintHook,
	'SaveHook': SaveHook,
}