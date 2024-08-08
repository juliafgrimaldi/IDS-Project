from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER, CONFIG_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet
import csv
import time
from ryu.topology.api import get_switch, get_link, get_host

class TrafficMonitor(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(TrafficMonitor, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)
        self.filename = 'traffic_stats.csv'
        self._initialize_csv()

    def _initialize_csv(self):
        with open(self.filename, 'w', newline='') as csvfile:
            fieldnames = ['time', 'dpid', 'port', 'rx_packets', 'tx_packets', 'rx_bytes', 'tx_bytes', 'rx_errors', 'tx_errors']
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
        self.logger.info('Sending stats request to: %016x', datapath.id if datapath.id else 0)
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        body = ev.msg.body
        timestamp = time.time()

        with open(self.filename, 'a', newline='') as csvfile:
            fieldnames = ['time', 'dpid', 'port', 'rx_packets', 'tx_packets', 'rx_bytes', 'tx_bytes', 'rx_errors', 'tx_errors']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            for stat in body:
                writer.writerow({
                    'time': timestamp,
                    'dpid': ev.msg.datapath.id,
                    'port': stat.port_no,
                    'rx_packets': stat.rx_packets,
                    'tx_packets': stat.tx_packets,
                    'rx_bytes': stat.rx_bytes,
                    'tx_bytes': stat.tx_bytes,
                    'rx_errors': stat.rx_errors,
                    'tx_errors': stat.tx_errors
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

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id, priority=priority,
                                    match=match, instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst)
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


    def add_default_host_flows(self, datapath):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # rules to allow communication between hosts
        # assuming a single switch with multiple hosts connected to different ports

        hosts = get_host(self, datapath.id)
        host_ports = [host.port.port_no for host in hosts]
        
        for in_port in host_ports:
            for out_port in host_ports:
                if in_port != out_port:
                    match = parser.OFPMatch(in_port=in_port)
                    actions = [parser.OFPActionOutput(out_port)]
                    self.add_flow(datapath, 1, match, actions)