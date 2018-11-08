import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# a1=np.load("/data2/codes/pottery/data/npy_10000/one/generated_5_0_637448_part_1.npy")
# a2=np.load("/data2/codes/pottery/data/npy_10000/one/generated_5_0_637448_part_2.npy")
# a3=np.load("/data2/codes/pottery/data/npy_10000/one/generated_5_0_637448_part_3.npy")
# a4=np.load("/data2/codes/pottery/data/npy_10000/one/generated_5_0_637448_part_4.npy")
# a5=np.load("/data2/codes/pottery/data/npy_10000/one/generated_5_0_637448_part_5.npy")

# a1=np.load("/data2/codes/pottery/data/npy_10000/two/generated_5_0_454809_part_1.npy")
# a2=np.load("/data2/codes/pottery/data/npy_10000/two/generated_5_0_454809_part_2.npy")
# a3=np.load("/data2/codes/pottery/data/npy_10000/two/generated_5_0_454809_part_3.npy")
# a4=np.load("/data2/codes/pottery/data/npy_10000/two/generated_5_0_454809_part_4.npy")
# a5=np.load("/data2/codes/pottery/data/npy_10000/two/generated_5_0_454809_part_5.npy")

# a1=np.load("/data2/codes/pottery/data/npy_10000/three/generated_5_0_84096_part_1.npy")
# a2=np.load("/data2/codes/pottery/data/npy_10000/three/generated_5_0_84096_part_2.npy")
# a3=np.load("/data2/codes/pottery/data/npy_10000/three/generated_5_0_84096_part_3.npy")
# a4=np.load("/data2/codes/pottery/data/npy_10000/three/generated_5_0_84096_part_4.npy")
# a5=np.load("/data2/codes/pottery/data/npy_10000/three/generated_5_0_84096_part_5.npy")

# a1=np.load("/data2/codes/pottery/data/npy_10000/four/generated_5_0_50700_part_1.npy")
# a2=np.load("/data2/codes/pottery/data/npy_10000/four/generated_5_0_50700_part_2.npy")
# a3=np.load("/data2/codes/pottery/data/npy_10000/four/generated_5_0_50700_part_3.npy")
# a4=np.load("/data2/codes/pottery/data/npy_10000/four/generated_5_0_50700_part_4.npy")
# a5=np.load("/data2/codes/pottery/data/npy_10000/four/generated_5_0_50700_part_5.npy")

a1=np.load("/data2/codes/pottery/data/npy_10000/five/generated_5_0_56784_part_1.npy")
a2=np.load("/data2/codes/pottery/data/npy_10000/five/generated_5_0_56784_part_2.npy")
a3=np.load("/data2/codes/pottery/data/npy_10000/five/generated_5_0_56784_part_3.npy")
a4=np.load("/data2/codes/pottery/data/npy_10000/five/generated_5_0_56784_part_4.npy")
a5=np.load("/data2/codes/pottery/data/npy_10000/five/generated_5_0_56784_part_5.npy")

a=np.concatenate((a1,a2,a3,a4,a5),axis=0)
a_x=a[:,0]
a_y=a[:,1]
a_z=a[:,2]

fig=plt.figure()
plt.scatter(a_x,a_y)
plt.show()