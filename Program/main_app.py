import numpy as np
import cv2
from sympy import Eq , symbols , solve

s = cv2.imread('source/source1.bmp')
sstore = s

t = cv2.imread('target/target1.bmp')
tstore = t

s_mean, s_std = cv2.meanStdDev(s)
s_mean = np.hstack(np.around(s_mean,2))
s_std = np.hstack(np.around(s_std,2))

t_mean, t_std = cv2.meanStdDev(t)
t_mean = np.hstack(np.around(t_mean,2))
t_std = np.hstack(np.around(t_std,2))

height , width , vac = s.shape

alphao = []
betao=[]

for n in range (0 , vac):
            x = symbols('x')
            equation1 = Eq( (x**2) * s_std[n] - t_std[n] , 0)
            alphao.append(solve(equation1))
            y = symbols('y')
            equation2 = Eq(t_mean[n] - s_mean[n]*x , 0)
            betao.append(solve(equation2))

alpha = []
beta = []

for i in range (0 , len(alphao)):
    alpha.append(alphao[i][1])

for i in range (0 , len(betao)):
    beta.append ( betao[i][0] )

print("Converting Pitures...")

for i in range (0 , height):
    for j in range (0 , width):
        for n in range (0 , vac):
            pixel = s[i][j][n]
            pixel = pixel*alpha[n] + beta[n]
            pixel = round(pixel)
            if pixel < 0:
                pixel = 0
            elif pixel > 255:
                pixel = 255
            sstore[i][j][n] = pixel

print('Convertion completed')

cv2.imwrite('result/result.bmp',sstore)
cv2.imshow('result' , sstore)
cv2.waitKey(0)