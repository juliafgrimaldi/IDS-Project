import math
import csv
import time
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER, CONFIG_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet, ether_types
from ryu.lib.packet import ipv4, icmp, tcp, udp
from ryu.lib.packet import in_proto
from ryu.topology.api import get_switch, get_link, get_host
from ryu.topology import event
import os

class TrafficMonitor(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(TrafficMonitor, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.mac_to_port = {}
        self.monitor_thread = hub.spawn(self._monitor)
        self.filename = 'train_traffic_stats.csv'
        self._initialize_csv()

    def _initialize_csv(self):
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as csvfile:
                fieldnames = ['time', 'dpid', 'ip_src', 'tp_src', 'packets', 'bytes', 'ip_proto', 'duration_sec', 'label']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            self.logger.info('Registering datapath: %016x', datapath.id if datapath.id else 0)
            self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            self.logger.info('Unregistering datapath: %016x', datapath.id if datapath.id else 0)
            if datapath.id in self.datapaths:
                del self.datapaths[datapath.id]

    def _monitor(self):
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(10)

    def _request_stats(self, datapath):
        self.logger.info('Sending flow stats request to: %016x', datapath.id if datapath.id else 0)
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        body = ev.msg.body
        timestamp = time.time()

        with open(self.filename, 'a', newline='') as csvfile:
            fieldnames = ['time', 'dpid', 'ip_src', 'tp_src', 'packets', 'bytes', 'ip_proto', 'duration_sec', 'label']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            for stat in body:
                writer.writerow({
                'time': timestamp,
                'dpid': ev.msg.datapath.id,
                'ip_src': stat.match.get('ipv4_src', 'NULL'),
                'tp_src': stat.match('tcp_src', stat.match.get('udp_src', 'NULL')) ,
                'packets': stat.packet_count,
                'bytes': stat.byte_count,
                'ip_proto': stat.match['ip_proto'],
                'duration_sec': stat.duration_sec,
                'label': '1'
            })

    @set_ev_cls(event.EventSwitchEnter)
    def switch_enter_handler(self, ev):
        switch = ev.switch
        self.logger.info('Switch entered: %016x', switch.dp.id if switch.dp.id else 0)
        self.install_default_flows(switch.dp)

    @set_ev_cls(event.EventSwitchLeave)
    def switch_leave_handler(self, ev):
        switch = ev.switch
        self.logger.info('Switch left: %016x', switch.dp.id if switch.dp.id else 0)

    def add_flow(self, datapath, priority, match, actions, buffer_id=None, idle=0, hard=0):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id, idle_timeout=idle, hard_timeout=hard, priority=priority,
                                    match=match, instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    idle_timeout=idle, hard_timeout=hard, match=match, instructions=inst)
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        self.install_default_flows(datapath)

    def install_default_flows(self, datapath):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # send all packets to controller
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        if ev.msg.msg_len < ev.msg.total_len:
            self.logger.debug("packet truncated: only %s of %s bytes",
                            ev.msg.msg_len, ev.msg.total_len)

        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']


        # analyse the received packets using the packet library.
        pkt = packet.Packet(msg.data)
        eth_pkt = pkt.get_protocols(ethernet.ethernet)[0]

        # ignore LLDP packets
        if eth_pkt.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        dst = eth_pkt.dst
        src = eth_pkt.src
        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})

        self.logger.info("packet in %s %s %s %s", dpid, src, dst, in_port)

        # learn a mac address to avoid FLOOD next time.
        self.mac_to_port[dpid][src] = in_port

        # if the destination mac address is already learned,
        # decide which port to output the packet, otherwise FLOOD.
        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        # construct action list.
        actions = [parser.OFPActionOutput(out_port)]

        # install a flow on switches to avoid packet_in next time.
        if out_port != ofproto.OFPP_FLOOD:
        # Verify if its an ip packet and creates an appropriate match
            if eth_pkt.ethertype == ether_types.ETH_TYPE_IP:
                ip = pkt.get_protocol(ipv4.ipv4)
                srcip = ip.src
                dstip = ip.dst
                protocol = ip.proto

                # ICMP
                if protocol == in_proto.IPPROTO_ICMP:
                    t = pkt.get_protocol(icmp.icmp)
                    match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP,
                                        ipv4_src=srcip, ipv4_dst=dstip,
                                        ip_proto=protocol,
                                        icmpv4_code=t.code,
                                        icmpv4_type=t.type)
                # TCP
                elif protocol == in_proto.IPPROTO_TCP:
                    t = pkt.get_protocol(tcp.tcp)
                    match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP,
                                        ipv4_src=srcip, ipv4_dst=dstip,
                                        ip_proto=protocol,
                                        tcp_src=t.src_port,
                                        tcp_dst=t.dst_port)
                # UDP
                elif protocol == in_proto.IPPROTO_UDP:
                    u = pkt.get_protocol(udp.udp)
                    match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP,
                                            ipv4_src=srcip, ipv4_dst=dstip,
                                            ip_proto=protocol,
                                            udp_src=u.src_port,
                                            udp_dst=u.dst_port)
                
                # Verify if packet has a valid buffer_id
                if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                    self.add_flow(datapath, 1, match, actions, msg.buffer_id, idle=20, hard=100)
                    return
                else:
                    self.add_flow(datapath, 1, match, actions, idle=20, hard=100)

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)
