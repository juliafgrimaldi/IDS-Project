from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import Controller, OVSKernelSwitch, RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel
import os

class CustomTopo(Topo):
    def build(self):
        # Adding 6 hosts
        h1 = self.addHost('h1', ip='10.1.1.1/24', mac='00:00:00:00:00:01')
        h2 = self.addHost('h2', ip='10.1.1.2/24', mac='00:00:00:00:00:02')
        h3 = self.addHost('h3', ip='10.1.1.3/24', mac='00:00:00:00:00:03')
        h4 = self.addHost('h4', ip='10.1.1.4/24', mac='00:00:00:00:00:04')
        h5 = self.addHost('h5', ip='10.1.1.5/24', mac='00:00:00:00:00:05')
        h6 = self.addHost('h6', ip='10.1.1.6/24', mac='00:00:00:00:00:06')

        # Adding 3 switches
        s1 = self.addSwitch('s1', protocols='OpenFlow13')
        s2 = self.addSwitch('s2', protocols='OpenFlow13')
        s3 = self.addSwitch('s3', protocols='OpenFlow13')

        # Adding links between hosts and switches
        self.addLink(h1, s1)
        self.addLink(h2, s1)

        self.addLink(h3, s2)
        self.addLink(h4, s2)

        self.addLink(h5, s3)
        self.addLink(h6, s3)

        # Adding links between switches
        self.addLink(s1, s2)
        self.addLink(s2, s3)

if __name__ == '__main__':
    setLogLevel('info')
    os.environ['LANG'] = 'C'
    topo = CustomTopo()
    net = Mininet(topo=topo, switch=OVSKernelSwitch, controller=RemoteController('c0', ip='127.0.0.1', port=6653), ipBase='10.1.1.0/24')
    net.start()
    CLI(net)
    net.stop()
