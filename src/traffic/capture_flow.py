from ryu.base import app_manager
from ryu.controller import controller
from ryu.controller.handler import set_ev_cls
from ryu.controller.handler import MAIN_DISPATCHER
from ryu.lib import hub
from ryu.ofproto import ofproto_v1_3
import csv

class TrafficStats(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFPROTOVERSION]

    def __init__(self, *args, **kwargs):
        super(TrafficStats, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.csv_file = 'traffic_stats.csv'
        self.start_time = hub.get_time()
        self.monitor_interval = 10  
        self._schedule_stats()

    def _schedule_stats(self):
        self.event_loop.call_later(self.monitor_interval, self._get_stats)

    @set_ev_cls(controller.EventREGISTER, [MAIN_DISPATCHER])
    def register(self, ev):
        datapath = ev.msg.datapath
        self.datapaths[datapath.id] = datapath

    @set_ev_cls(controller.EventDEAD, [MAIN_DISPATCHER])
    def dead(self, ev):
        datapath = ev.msg.datapath
        if datapath.id in self.datapaths:
            del self.datapaths[datapath.id]

    def _get_stats(self):
        for datapath in self.datapaths.values():
            self._request_stats(datapath)
        self._schedule_stats()

    def _request_stats(self, datapath):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        req = parser.OFPFlowStatsRequest(datapath, 0, ofproto.OFPTT_ALL, ofproto.OFPP_ANY, ofproto.OFPG_ANY)
        datapath.send_msg(req)

        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)

    @set_ev_cls(controller.EventFlowStatsReply, [MAIN_DISPATCHER])
    def _flow_stats_reply(self, ev):
        body = ev.msg.body
        datapath = ev.msg.datapath
        self._write_stats_to_csv(datapath.id, body, 'flow')

    @set_ev_cls(controller.EventPortStatsReply, [MAIN_DISPATCHER])
    def _port_stats_reply(self, ev):
        body = ev.msg.body
        datapath = ev.msg.datapath
        self._write_stats_to_csv(datapath.id, body, 'port')

    def _write_stats_to_csv(self, dpid, stats, stat_type):
        with open(self.csv_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if stat_type == 'flow':
                for stat in stats:
                    writer.writerow([
                        dpid,
                        'flow',
                        stat.match['in_port'] if 'in_port' in stat.match else 'N/A',
                        stat.match['eth_src'] if 'eth_src' in stat.match else 'N/A',
                        stat.match['eth_dst'] if 'eth_dst' in stat.match else 'N/A',
                        stat.match['ipv4_src'] if 'ipv4_src' in stat.match else 'N/A',
                        stat.match['ipv4_dst'] if 'ipv4_dst' in stat.match else 'N/A',
                        stat.byte_count,
                        stat.packet_count,
                        stat.duration_sec,
                        stat.duration_nsec,
                        1  # ataque
                    ])
            elif stat_type == 'port':
                for stat in stats:
                    writer.writerow([
                        dpid,
                        'port',
                        stat.port_no,
                        stat.rx_packets,
                        stat.tx_packets,
                        stat.rx_bytes,
                        stat.tx_bytes,
                        stat.rx_dropped,
                        stat.tx_dropped,
                        stat.rx_errors,
                        stat.tx_errors,
                        1 
                    ])
