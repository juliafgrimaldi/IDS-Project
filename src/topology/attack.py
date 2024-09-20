from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import Controller, OVSKernelSwitch, RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel
import time
import os
import random

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
    hosts = [net.get('h1'), net.get('h2'), net.get('h3'), net.get('h4'), net.get('h5'), net.get('h6')]

    attack_interval = 10
    attack_types = ['syn', 'udp', 'icmp', 'http']
    attack_duration = random.randint(60,120)

    for _ in range(int(attack_duration / attack_interval)):
        attacker = random.choice(hosts) 
        victims = random.choice([host.IP() for host in hosts if host != attacker], k=3) # 3 targets
        attack_type = random.choice(attack_types) 

        if attack_type == 'syn':
            for victim in victims:
                print("Starting SYN flood attack with {} targeting {}...".format(attacker.name, victim))
                attacker.cmd('hping3 --flood -p 80 -i u1000 {} &'.format(victim))

        elif attack_type == 'udp':
            for victim in victims:
                print("Starting UDP flood attack with {} targeting {}...".format(attacker.name, victim))
                attacker.cmd('iperf -c {} -u -b 100M -t {} &'.format(victim, attack_interval))

        elif attack_type == 'icmp':
            for victim in victims:
                print("Starting ICMP flood attack with {} targeting {}...".format(attacker.name, victim))
                attacker.cmd('hping3 --flood --icmp -i u100 {} &'.format(victim))

        time.sleep(attack_interval)

    print("Stopping all the attacks...")
    for host in hosts:
        host.cmd('killall hping3')
        host.cmd('killall iperf')

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
