import network

net = network.MultilayerPerceptronNetwork([4, 2, 1])
net.export_network()

network2 = network.MultilayerPerceptronNetwork()
# for i in network2.network:
#     for j in i:
