import numpy as np



max_cap = np.array([20, 9.5, 11.5])
sku_weights = np.array([1.0, 2.0, 3.0])
inventory = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

free_capacity = max_cap[0] - (inventory[0, :] * sku_weights).sum()

print(max_cap[0])
print(inventory[0, :])
print(sku_weights)
print(inventory[0, :] * sku_weights)
print((inventory[0, :] * sku_weights).sum())

print(free_capacity)