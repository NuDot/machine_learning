from nevergrad.instrumentation import FolderFunction
from nevergrad.optimization import optimizerlib
from concurrent import futures
folder = "/project/snoplus/machine_learning/hyperparameter"
command = ["python", "hyperparameter/ng_script.py"]  # command to run from right outside the provided folder
func = FolderFunction(folder, command, clean_copy=True)
print(func.dimension)  # will print the number of variables of the function
optimizer = optimizerlib.OnePlusOne(dimension=func.dimension, budget=50, num_workers=1)
recommendation = optimizer.optimize(func, executor=None, batch_mode=True)
print(func.get_summary(recommendation))