from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import Controller, OVSKernelSwitch, RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel
import time
import os  

class CustomTopo(Topo):
    def build(self):
        h1 = self.addHost('h1', ip='10.1.1.1/24', mac='00:00:00:00:00:01')
        h2 = self.addHost('h2', ip='10.1.1.2/24', mac='00:00:00:00:00:02')
        h3 = self.addHost('h3', ip='10.1.1.3/24', mac='00:00:00:00:00:03')
        h4 = self.addHost('h4', ip='10.1.1.4/24', mac='00:00:00:00:00:04')
        h5 = self.addHost('h5', ip='10.1.1.5/24', mac='00:00:00:00:00:05')
        h6 = self.addHost('h6', ip='10.1.1.6/24', mac='00:00:00:00:00:06')

        s1 = self.addSwitch('s1', protocols='OpenFlow13')
        s2 = self.addSwitch('s2', protocols='OpenFlow13')
        s3 = self.addSwitch('s3', protocols='OpenFlow13')

        self.addLink(h1, s1)
        self.addLink(h2, s1)

        self.addLink(h3, s2)
        self.addLink(h4, s2)

        self.addLink(h5, s3)
        self.addLink(h6, s3)

        self.addLink(s1, s2)
        self.addLink(s2, s3)

def simulate_normal_traffic(net):
    h1 = net.get('h1')
    h2 = net.get('h2')
    h3 = net.get('h3')
    h4 = net.get('h4')
    h5 = net.get('h5')
    h6 = net.get('h6')

    h2.cmd('iperf -s &')
    h4.cmd('iperf -s &')

    # normal traffic between h1 and h2
    h1.cmd('iperf -c 10.1.1.2 -t 20 &')  # 20 seconds of TCP traffic

    # normal traffic between h3 and h4
    h3.cmd('iperf -c 10.1.1.4 -t 20 &')  # 20 seconds of TCP traffic

    h5.cmd('iperf -c 10.1.1.6 -t 20 &')
    
    time.sleep(25)  

    print("Stopping iperf servers...")
    h2.cmd('killall iperf')
    h4.cmd('killall iperf')

def run_custom_topo():
    topo = CustomTopo()
    net = Mininet(topo=topo, controller=RemoteController, switch=OVSKernelSwitch)
    
    net.start()

    simulate_normal_traffic(net)

    CLI(net)

    net.stop()
    print("Network stopped")

if __name__ == '__main__':
    setLogLevel('info')
    os.environ['LANG'] = 'C'
    run_custom_topo()
