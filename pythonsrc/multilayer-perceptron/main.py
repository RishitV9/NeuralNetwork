import network

network = network.MultilayerPerceptronNetwork([4, 2, 1])

for i in network.network:
    print("-----")
    for j in i:
        print(j.parents)
