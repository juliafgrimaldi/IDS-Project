from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import Controller, OVSKernelSwitch, RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel
from mininet.link import TCLink
from random import choice
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

def simulate_traffic(net, duration=10, interval=5, traffic_multiplier=2):
    h1 = net.get('h1')
    h2 = net.get('h2')
    h3 = net.get('h3')
    h4 = net.get('h4')
    h5 = net.get('h5')
    h6 = net.get('h6')

    hosts = [net.get('h{}'.format(i)) for i in range(1,7)]
    
    for _ in range(interval * traffic_multiplier):
        src = choice(hosts)
        dst = choice(hosts)

        while dst  == src:
            dst = choice(hosts)

        dst_ip = dst.IP()
        traffic_type = choice(['iperf', 'ping', 'curl'])  

        if traffic_type == 'iperf':
            dst.cmd('iperf -s > /dev/null 2>&1 &')   
            print("Simulating traffic TCP between {} and {}".format(src.IP(), dst_ip))
            src.cmd('iperf -c {} -t {} > /dev/null 2>&1 &'.format(dst_ip, duration))
            time.sleep(duration)
            dst.cmd('killall iperf')

        elif traffic_type == 'ping':
            print("Simulating ICMP traffic (ping) between {} and {}".format(src.IP(), dst_ip))
            src.cmd('ping {} &'.format(dst_ip)) 
            time.sleep(10)

        elif traffic_type == 'curl':
            print("Simulating HTTP between {} and {}".format(src.IP(), dst_ip))
            src.cmd('curl http://{} &'.format(dst_ip))
            time.sleep(10)
        
        time.sleep(1)

def run_custom_topo():
    topo = CustomTopo()
    net = Mininet(topo=topo, link=TCLink, controller=RemoteController, switch=OVSKernelSwitch)
    
    net.start()

    print("Simulating traffic...")
    simulate_traffic(net)

    print("Stopping all processes...")
    for host in net.hosts:
        host.cmd('killall iperf ping curl')

    print("Stopping the network...")    
    net.stop()
    print("Network stopped")

if __name__ == '__main__':
    setLogLevel('info')
    os.environ['LANG'] = 'C'
    run_custom_topo()
