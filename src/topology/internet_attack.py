from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import Controller, OVSKernelSwitch, RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel
from time import sleep
import os
from random import choice, randrange

class CustomTopo(Topo):
    def build(self):
        h1 = self.addHost('h1', ip='10.1.1.1/24', mac='00:00:00:00:00:01')
        h2 = self.addHost('h2', ip='10.1.1.2/24', mac='00:00:00:00:00:02')
        h3 = self.addHost('h3', ip='10.1.1.3/24', mac='00:00:00:00:00:03')
        h4 = self.addHost('h4', ip='10.1.1.4/24', mac='00:00:00:00:00:04')
        h5 = self.addHost('h5', ip='10.1.1.5/24', mac='00:00:00:00:00:05')
        h6 = self.addHost('h6', ip='10.1.1.6/24', mac='00:00:00:00:00:06')

        h0 = self.addHost('h0', ip='192.168.1.100/24', mac='00:00:00:00:00:07')

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
        self.addLink(h0, s1)

def ip_generator():
    """Generate a random IP in the network."""
    return "10.1.1.{}".format(randrange(1,6))

def simulate_attacks(net):
    hosts = [net.get('h1'), net.get('h2'), net.get('h3'), net.get('h4'), net.get('h5'), net.get('h6')]
    internet_host = net.get('h0')

    for _ in range(5): 
        src = choice(hosts)
        dst_ip = ip_generator()
        print("Performing Ping Flood: Source={} -> Target={}".format(src.IP(), dst_ip))
        internet_host.cmd("ping -f -i 0.1 -c 5000 {} &".format(dst_ip))
        sleep(50)  

    for _ in range(3):
        dst = choice(hosts)
        print("ICMP Flood with iperf from Internet (h0) to {}".format(dst.IP()))
        dst.cmd("iperf -s > /dev/null 2>&1 &")  # Inicia servidor iperf no host de destino
        internet_host.cmd("iperf -c {} -u -b 500M -t 30 > /dev/null 2>&1 &".format(dst.IP()))
        sleep(30)


    #print("Stopping attacks...")
    #for host in hosts:
    #    host.cmd('killall ping iperf')  # Stop all running ping and iperf commands
    #internet_host.cmd('killall ping iperf')
    #sleep(2)

def run_custom_topo():
    topo = CustomTopo()
    net = Mininet(topo=topo, controller=RemoteController, switch=OVSKernelSwitch)

    net.start()
    print("Network started")

    simulate_attacks(net)

    print("Stopping the network...")
    net.stop()
    print("Attack network stopped")

    os._exit(0)

if __name__ == '__main__':
    os.environ['LANG'] = 'C'
    setLogLevel('info')
    run_custom_topo()