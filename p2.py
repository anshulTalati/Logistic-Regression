D = [(6.4432, 9.6309, 50.9155), (3.7861, 5.4681, 29.9852),\
(8.1158, 5.2114, 42.9626), (5.3283, 2.3159, 24.7445),\
(3.5073, 4.8890, 27.3704), (9.3900, 6.2406, 51.1350),\
(8.7594, 6.7914, 50.5774), (5.5016, 3.9552, 30.5206),\
(6.2248, 3.6744, 31.7380), (5.8704, 9.8798, 49.6374),\
(2.0774, 0.3774, 10.0634), (3.0125, 8.8517, 38.0517),\
(4.7092, 9.1329, 43.5320), (2.3049, 7.9618, 33.2198),\
(8.4431, 0.9871, 31.1220), (1.9476, 2.6187, 16.2934),\
(2.2592, 3.3536, 19.3899), (1.7071, 6.7973, 28.4807),\
(2.2766, 1.3655, 13.6945), (4.3570, 7.2123, 36.9220),\
(3.1110, 1.0676, 14.9160), (9.2338, 6.5376, 51.2371),\
(4.3021, 4.9417, 29.8112), (1.8482, 7.7905, 32.0336),\
(9.0488, 7.1504, 52.5188), (9.7975, 9.0372, 61.6658),\
(4.3887, 8.9092, 42.2733), (1.1112, 3.3416, 16.5052),\
(2.5806, 6.9875, 31.3369), (4.0872, 1.9781, 19.9475),\
(5.9490, 0.3054, 20.4239), (2.6221, 7.4407, 32.6062),\
(6.0284, 5.0002, 35.1676), (7.1122, 4.7992, 38.2211),\
(2.2175, 9.0472, 36.4109), (1.1742, 6.0987, 25.0108),\
(2.9668, 6.1767, 29.8861), (3.1878, 8.5944, 37.9213),\
(4.2417, 8.0549, 38.8327), (5.0786, 5.7672, 34.4707) ]

m = len(D)
print m

theta = []

theta0 = 0.0
theta1 = 1.0
theta2 = 1.0
alpha = 0.01

def costOfRegression(t0, t1, t2):
    costOfRegression = (sum((t0 + t1 * D[i][0] + t2 * D[i][1] - D[i][2])**2 for i in range (m-1)))/(2*m)
    return costOfRegression

cost = costOfRegression(theta0,theta1,theta2)
print "cost: ", cost 
while (True):
    

    temp0 = theta0 - alpha*(sum(theta0 + theta1*D[i][0] +theta2*D[i][1] - D[i][2] for i in range(m-1)) / m)
    temp1 = theta1 - alpha*(sum((theta0 + theta1*D[i][0] +theta2*D[i][1] - D[i][2])*D[i][0] for i in range(m-1)) / m)
    temp2 = theta2 - alpha*(sum((theta0 + theta1*D[i][0] +theta2*D[i][1] - D[i][2])*D[i][1] for i in range(m-1)) / m)

    cost1 = costOfRegression(temp0,temp1,temp2)
    # print "cost: ", cost1
    if cost < cost1:
        break
    cost = cost1
    theta0 = temp0
    theta1 = temp1
    theta2 = temp2

print theta0, theta1, theta2
print "Error : ", costOfRegression(theta0, theta1, theta2) 