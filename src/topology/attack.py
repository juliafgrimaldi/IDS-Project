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

def simulate_attacks(net):
    h3 = net.get('h3')
    h5 = net.get('h5')
    h6 = net.get('h6')
    h4 = net.get('h4')

    # SYN flood w/ h3
    print("Starting SYN flood attack with h3...")
    h3.cmd('hping3 --flood -p 80 10.1.1.4 &')

    # UDP Flood w/ h5
    print("Starting UDP flood attack with h5...")
    h5.cmd('iperf -c 10.1.1.4 -u -b 10M -t 20 &')  # 10 Mbps for 20 seconds

    # ICMP flood w/ h6
    print("Starting ICMP flood attack with h6...")
    h6.cmd('hping3 --flood --icmp 10.1.1.4 &')

    time.sleep(30) 
    print("Stopping all the attacks...")
    h3.cmd('killall hping3')
    h5.cmd('killall iperf')
    h6.cmd('killall hping3')
    h4.cmd('killall iperf')

def run_custom_topo():
    topo = CustomTopo()
    net = Mininet(topo=topo, controller=RemoteController, switch=OVSKernelSwitch)
    
    net.start()
    print("Network started")

    simulate_attacks(net)

    CLI(net)

    net.stop()
    print("Attack network stopped")

if __name__ == '__main__':
    os.environ['LANG'] = 'C'
    setLogLevel('info')
    run_custom_topo()
