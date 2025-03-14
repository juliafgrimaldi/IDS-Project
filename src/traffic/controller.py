import re
import csv
import time
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER, CONFIG_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet, ether_types
from ryu.topology.api import get_switch, get_link, get_host
from ryu.topology import event
import os
from ML.knn import train_knn, predict_knn
from ML.svm import train_svm, predict_svm
from ML.decisiontree import train_decision_tree, predict_decision_tree
from ML.naivebayes import train_naive_bayes, predict_naive_bayes
from ML.randomforest import train_random_forest, predict_random_forest

class TrafficMonitor(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(TrafficMonitor, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.mac_to_port = {}
        self.monitor_thread = hub.spawn(self._monitor)
        self.train_file = 'traffic_stats.csv'
        self.filename = 'traffic_predict.csv'
        self.flow_model = None
        self._initialize_csv()
        self.models = {}
        self.accuracies = {}
        self._train_models()

    def _initialize_csv(self):
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as csvfile:
                fieldnames = ['time', 'dpid', 'in_port', 'eth_src', 'eth_dst', 'packets', 'bytes', 'duration_sec']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

    def _train_models(self):
        self.logger.info("Treinando todos os modelos...")
        self.models['decision_tree'], self.accuracies['decision_tree'], self.dt_selector, self.dt_encoder, self.dt_imputer, self.dt_scaler = train_decision_tree(self.train_file)
        self.models['knn'], self.accuracies['knn'], self.knn_selector, self.knn_encoder, self.knn_imputer, self.knn_scaler = train_knn(self.train_file)
        self.models['naive_bayes'], self.accuracies['naive_bayes'], self.nb_selector, self.nb_encoder, self.nb_imputer, self.nb_scaler = train_naive_bayes(self.train_file)
        self.models['random_forest'], self.accuracies['random_forest'], self.rf_selector, self.rf_encoder, self.rf_imputer, self.rf_scaler = train_random_forest(self.train_file)
        self.models['svm'], self.accuracies['svm'], self.svm_selector, self.svm_encoder, self.svm_imputer, self.svm_scaler = train_svm(self.train_file)

    def predict_all_models(self, data):
        predictions = {}
        predictions['decision_tree'] = predict_decision_tree(self.models['decision_tree'], self.dt_selector, self.dt_encoder, self.dt_imputer, self.dt_scaler, data)
        predictions['knn'] = predict_knn(self.models['knn'], self.knn_selector, self.knn_encoder, self.knn_imputer, self.knn_scaler, data)
        predictions['naive_bayes'] = predict_naive_bayes(self.models['naive_bayes'], self.nb_selector, self.nb_encoder, self.nb_imputer, self.nb_scaler, data)
        predictions['random_forest'] = predict_random_forest(self.models['random_forest'], self.rf_selector, self.rf_encoder, self.rf_imputer, self.rf_scaler, data)
        predictions['svm'] = predict_svm(self.models['svm'], self.svm_selector, self.svm_encoder, self.svm_imputer, self.svm_scaler, data)
        return predictions


    def weighted_vote(self, predictions):
        weighted_votes = {}
        for model_name, pred in predictions.items():
            weight = self.accuracies[model_name] 
            for i, p in enumerate(pred):
                if i not in weighted_votes:
                    weighted_votes[i] = 0
                weighted_votes[i] += p * weight  
        
        final_predictions = []
        for i in weighted_votes:
            final_predictions.append(1 if weighted_votes[i] > 0.5 else 0)  # Decisão final
        
        return final_predictions
    
    def predict_traffic(self):
        try:
            self.logger.info("Predição com todos os modelos...")
            predictions = self.predict_all_models(self.filename)  
            final_predictions = self.weighted_vote(predictions) 
        

            legitimate_traffic = 0
            ddos_traffic = 0
            for pred in final_predictions:  
                if pred == 0:
                    legitimate_traffic += 1  
                else:
                    ddos_traffic += 1  
        
            self.logger.info(f"Legitimate traffic: {legitimate_traffic}, DDoS traffic: {ddos_traffic}")
        except Exception as e:
            self.logger.error(f"Erro na predição: {e}")
    
    
    def is_high_volume(self, packets, bytes, duration_sec):
        """Verifica se há um alto volume de pacotes ou bytes em um curto período de tempo."""
        packets_per_sec = packets / duration_sec if duration_sec > 0 else 0
        bytes_per_sec = bytes / duration_sec if duration_sec > 0 else 0
        return packets_per_sec > 10000 or bytes_per_sec > 100000000

    def is_long_connection(self, duration_sec):
        """Verifica se a conexão é muito longa."""
        return duration_sec > 3600  

    def is_invalid_mac(self, mac):
        """Verifica se o endereço MAC é inválido ou tem padrões repetitivos."""
        invalid_patterns = [
            r"00:00:00:00:00:00",  # MAC inválido
            r"([0-9A-Fa-f]{2}:)\1{5}"  # Padrões repetitivos
        ]
        for pattern in invalid_patterns:
            if re.match(pattern, mac, re.IGNORECASE):
                return True
        return False
    
    def block_traffic(self, datapath, eth_src, eth_dst, in_port):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Cria uma regra de fluxo para descartar pacotes
        match = parser.OFPMatch(in_port=in_port, eth_src=eth_src, eth_dst=eth_dst)
        actions = []  # Nenhuma ação (descartar pacotes, n sao encaminhados)
        self.add_flow(datapath, 100, match, actions, idle=60, hard=120) 

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
            self.predict_traffic()

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
            fieldnames = ['time', 'dpid', 'in_port', 'eth_src', 'eth_dst', 'packets', 'bytes', 'duration_sec']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            for stat in body:
                eth_dst = stat.match.get('eth_dst', 'NULL')
                eth_src = stat.match.get('eth_src', 'NULL')
                in_port = stat.match.get('in_port', 'NULL')
                packets = stat.packet_count
                bytes = stat.byte_count
                duration_sec = stat.duration_sec


                if (
                    self.is_high_volume(packets, bytes, duration_sec) or
                    self.is_long_connection(duration_sec) or
                    self.is_invalid_mac(eth_src) or
                    self.is_invalid_mac(eth_dst)):

                    self.block_traffic(ev.msg.datapath, eth_src, eth_dst, in_port)

                    self.logger.warning(f"Tráfego suspeito detectado e bloqueado: "
                                       f"eth_src={eth_src}, eth_dst={eth_dst}, "
                                       f"packets={packets}, bytes={bytes}, duration_sec={duration_sec}")
                    continue

                writer.writerow({
                'time': timestamp,
                'dpid': ev.msg.datapath.id,
                'in_port': stat.match.get('in_port', 'NULL'),
                'eth_src': stat.match.get('eth_src', 'NULL'),
                'eth_dst': stat.match.get('eth_dst', 'NULL'),
                'packets': stat.packet_count,
                'bytes': stat.byte_count,
                'duration_sec': stat.duration_sec
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

        dpid = format(datapath.id, "d").zfill(16)
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
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            # verify if we have a valid buffer_id, if yes avoid to send both
            # flow_mod & packet_out
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                self.add_flow(datapath, 1, match, actions, msg.buffer_id)
                return
            else:
                self.add_flow(datapath, 1, match, actions)
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)
