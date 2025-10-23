# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:40:21 2023

@author: thoma
"""

from shape import Shape
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sh1 = Shape()
sh1.set_random_shape(5)
plt.plot(sh1.values,color='black')
plt.axis('off')
plt.show()

sh2 = Shape()
sh2.set_random_shape(5)
plt.plot(sh2.values,color='black')
plt.axis('off')
plt.show()

sh3 = Shape()
sh3.set_random_shape(5)
plt.plot(sh3.values,color='black')
plt.axis('off')
plt.show()


for sh in [sh1,sh2,sh3]:
    std_dev = 0.3
    
    # Create an array to store the arrays with noise
    arrays_with_noise = []
    
    # Create arrays with noise and plot each one
    for i in range(3):
        # Generate random noise
        noise = np.random.normal(0, std_dev, np.array(sh.values).shape)
        
        # Create array with noise
        array_with_noise = sh.values + noise
        
        # Append the array to the list
        arrays_with_noise.append(array_with_noise)
    
    plt.figure(figsize=(8,12))
    for i, array_with_noise in enumerate(arrays_with_noise):
        plt.subplot(3, 1, i + 1)
        plt.plot(array_with_noise,color='black')
        plt.axis('off')
        #plt.title(f'Array with Noise {i + 1}')
    
    plt.tight_layout()
    plt.show()
    
    

plt.plot(pd.Series(sh1.values),color='black')
plt.plot(pd.Series([sh1.values[-1],0.7],index=[4,5]),color='grey')
plt.fill_between([4,5], [sh1.values[-1],0.6], [sh1.values[-1],0.8], color='grey', alpha=0.2)
plt.axis('off')
plt.show()

plt.plot(pd.Series(sh2.values),color='black')
plt.plot(pd.Series([sh2.values[-1],0.5],index=[4,5]),color='grey')
plt.fill_between([4,5], [sh2.values[-1],0.1], [sh2.values[-1],0.9], color='grey', alpha=0.2)
plt.axis('off')
plt.show()

plt.plot(pd.Series(sh3.values),color='black')
plt.plot(pd.Series([sh3.values[-1],0.4],index=[4,5]),color='grey')
plt.fill_between([4,5], [sh3.values[-1],0.35], [sh3.values[-1],0.45], color='grey', alpha=0.2)
plt.axis('off')
plt.show()