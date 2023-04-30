import yaml
import hook


class Runner:
	def __init__(self, hook_config_path):
		self.hooks = []
		self.resigter_hooks(hook_config_path)


	def resigter_hooks(self, hook_config_path):
		with open(hook_config_path, encoding="utf-8", mode="r") as f:
			hook_dict = yaml.load(stream=f, Loader=yaml.FullLoader)

		for key, value in hook_dict.items():
			print("registering " + key)
			hk = hook.__all__[value['TYPE']](value)
			self.hooks.append(hk)
		print('register all hooks over')


	def call_hooks(self, phase):
		for hk in self.hooks:
			getattr(hk, phase)(self)

runner = Runner("./hook_config.yaml")

for i in range(111):
	runner.call_hooks('before_run')
	runner.call_hooks('after_run')
