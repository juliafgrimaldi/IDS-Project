from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet, ether_types
import csv
import time
import os


class TrafficMonitorSwitch(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(TrafficMonitorSwitch, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)
        self.filename = 'traffic_stats.csv'
        self._initialize_csv()

    def _initialize_csv(self):
        """Initialize the CSV file to store traffic statistics."""
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as csvfile:
                fieldnames = ['time', 'dpid', 'in_port', 'eth_src', 'eth_dst', 
                              'packets', 'bytes', 'duration_sec', 'label']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, CONFIG_DISPATCHER])
    def _state_change_handler(self, ev):
        """Handle switch connections and disconnections."""
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            self.logger.info('Registering datapath: %016x', datapath.id)
            self.datapaths[datapath.id] = datapath
        elif ev.state == CONFIG_DISPATCHER:
            self.logger.info('Unregistering datapath: %016x', datapath.id)
            if datapath.id in self.datapaths:
                del self.datapaths[datapath.id]

    def _monitor(self):
        """Periodically request statistics from all datapaths."""
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(10)

    def _request_stats(self, datapath):
        """Request flow and port statistics from a datapath."""
        self.logger.info('Sending stats request to datapath: %016x', datapath.id)
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        flow_req = parser.OFPFlowStatsRequest(datapath)
        port_req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)

        datapath.send_msg(flow_req)
        datapath.send_msg(port_req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        """Handle flow statistics replies and log to CSV."""
        body = ev.msg.body
        timestamp = time.time()

        with open(self.filename, 'a', newline='') as csvfile:
            fieldnames = ['time', 'dpid', 'in_port', 'eth_src', 'eth_dst',
                          'packets', 'bytes', 'duration_sec', 'label']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            for stat in body:
                writer.writerow({
                    'time': timestamp,
                    'dpid': ev.msg.datapath.id,
                    'in_port': stat.match.get('in_port', 'N/A'),
                    'eth_src': stat.match.get('eth_src', 'N/A'),
                    'eth_dst': stat.match.get('eth_dst', 'N/A'),
                    'packets': stat.packet_count,
                    'bytes': stat.byte_count,
                    'duration_sec': stat.duration_sec,
                    'label': '1'  
                })

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        """Add a flow entry to a datapath."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match, instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst)
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Install table-miss entry on a switch connection."""
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        msg = ev.msg
        in_port = msg.match['in_port']
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        dst = eth.dst
        src = eth.src

        match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        """Handle packet-in messages."""
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        dst = eth.dst
        src = eth.src
        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})

        self.logger.info("packet in %s %s %s %s", dpid, src, dst, in_port)

        self.mac_to_port[dpid][src] = in_port

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            self.add_flow(datapath, 1, match, actions)

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=msg.data)
        datapath.send_msg(out)
