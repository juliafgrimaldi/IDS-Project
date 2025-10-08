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
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Adicionar diretório ML ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ML'))

try:
    from ML.preprocessing import preprocess_data
    from ML.knn import predict_knn
    from ML.svm import predict_svm
    from ML.decisiontree import predict_decision_tree
    from ML.randomforest import predict_random_forest
except ImportError as e:
    print(f"ERRO ao importar módulos ML: {e}")
    print("Certifique-se de que os arquivos estão em ML/")


class TrafficMonitor(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(TrafficMonitor, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.mac_to_port = {}
        
        # Configurações de arquivos
        self.train_file = 'ML/backend/traffic_dataset.csv'  # Dataset para treino
        self.filename = 'traffic_predict.csv'
        self.processed_file = "traffic_predict_processed.csv"
        self.models_dir = 'models'
        
        self.flow_model = None
        self.start_time = time.time()
        self.last_processed_time = self.start_time
        
        self.logger.info("="*60)
        self.logger.info("TrafficMonitor iniciando...")
        self.logger.info("="*60)
        self.logger.info("Timestamp de início: {}".format(self.start_time))
        
        # Inicializar estruturas
        self._initialize_csv()
        self._backup_old_traffic()
        
        # Treinar ou carregar modelos
        self.models = {}
        self.accuracies = {}
        self.numeric_columns = []
        self.categorical_columns = []
        
        self._load_or_train_models()
        
        # Iniciar monitor
        self.monitor_thread = hub.spawn(self._monitor)
        self.logger.info("TrafficMonitor inicializado com sucesso!")

    def _load_or_train_models(self):
        """Carrega modelos existentes ou treina novos"""
        os.makedirs(self.models_dir, exist_ok=True)
        
        model_files = {
            'knn': 'knn_model_bundle.pkl',
            'random_forest': 'randomforest_model_bundle.pkl',
            'decision_tree': 'dt_model_bundle.pkl',
            'svm': 'svm_model_bundle.pkl'
        }
        
        all_models_exist = all(
            os.path.exists(os.path.join(self.models_dir, fname)) 
            for fname in model_files.values()
        )
        
        if all_models_exist:
            self.logger.info("Modelos encontrados! Carregando...")
            self._load_models(model_files)
        else:
            self.logger.warning("Modelos não encontrados. Iniciando treinamento...")
            if not os.path.exists(self.train_file):
                self.logger.error("ERRO: Dataset de treino não encontrado: {}".format(self.train_file))
                self.logger.error("Por favor, execute primeiro: python train_models_standalone.py")
                raise FileNotFoundError("Dataset de treino não encontrado")
            
            self._train_all_models()
            self._load_models(model_files)

    def _load_models(self, model_files):
        """Carrega os bundles dos modelos"""
        self.logger.info("Carregando modelos de: {}".format(self.models_dir))
        
        for model_name, filename in model_files.items():
            filepath = os.path.join(self.models_dir, filename)
            try:
                with open(filepath, 'rb', encoding="utf-8-sig") as f:
                    bundle = pickle.load(f)
                    self.models[model_name] = bundle
                    self.accuracies[model_name] = bundle.get('accuracy', 1.0)
                    
                    # Pegar colunas do primeiro modelo
                    if not self.numeric_columns:
                        self.numeric_columns = bundle.get('numeric_columns', [])
                        self.categorical_columns = bundle.get('categorical_columns', [])
                    
                    self.logger.info("✓ {} carregado (acurácia: {:.2f}%)".format(
                        model_name, self.accuracies[model_name] * 100
                    ))
            except Exception as e:
                self.logger.error("Erro ao carregar {}: {}".format(model_name, e))

    def _load_dataset(self, filepath):
        """Carrega dataset com tratamento de encoding"""
        encodings = ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                self.logger.info("Tentando carregar com encoding: {}".format(encoding))
                data = pd.read_csv(filepath, sep=",", encoding=encoding)
                self.logger.info("✓ Dataset carregado: {} registros com {}".format(
                    len(data), encoding
                ))
                return data
            except UnicodeDecodeError:
                continue
            except Exception as e:
                self.logger.error("Erro ao carregar dataset: {}".format(e))
                raise
        
        raise ValueError("Não foi possível carregar o dataset com nenhum encoding testado")

    def _train_all_models(self):
        """Treina todos os modelos"""
        self.logger.info("="*60)
        self.logger.info("INICIANDO TREINAMENTO DE MODELOS")
        self.logger.info("="*60)
        self.logger.info("Dataset: {}".format(self.train_file))
        
        # KNN
        try:
            self.logger.info("\n1/4 - Treinando KNN...")
            self._train_knn()
        except Exception as e:
            self.logger.error("Erro ao treinar KNN: {}".format(e))
        
        # Random Forest
        try:
            self.logger.info("\n2/4 - Treinando Random Forest...")
            self._train_random_forest()
        except Exception as e:
            self.logger.error("Erro ao treinar Random Forest: {}".format(e))
        
        # Decision Tree
        try:
            self.logger.info("\n3/4 - Treinando Decision Tree...")
            self._train_decision_tree()
        except Exception as e:
            self.logger.error("Erro ao treinar Decision Tree: {}".format(e))
        
        # SVM
        try:
            self.logger.info("\n4/4 - Treinando SVM...")
            self._train_svm()
        except Exception as e:
            self.logger.error("Erro ao treinar SVM: {}".format(e))
        
        self.logger.info("\n" + "="*60)
        self.logger.info("TREINAMENTO CONCLUÍDO!")
        self.logger.info("="*60)

    def _train_knn(self):
        """Treina modelo KNN"""
        self.logger.info("Carregando dataset: {}".format(self.train_file))
        
        try:
            # Tentar diferentes encodings
            try:
                df = pd.read_csv(self.train_file, sep=",", encoding='utf-8-sig')
                data = df.groupby("label", group_keys=False).apply(lambda x:x.sample(min(len(x), 2000)))
            except UnicodeDecodeError:
                self.logger.warning("Erro UTF-8, tentando latin-1...")
                data = pd.read_csv(self.train_file, encoding='latin-1')
            except:
                self.logger.warning("Erro latin-1, tentando ISO-8859-1...")
                data = pd.read_csv(self.train_file, encoding='ISO-8859-1')
        except Exception as e:
            self.logger.error("Erro ao carregar dataset: {}".format(e))
            raise
        
        if data.empty:
            raise ValueError("Dataset vazio")
        
        self.logger.info("Dataset carregado: {} registros".format(len(data)))

        X, y, imputer, scaler, encoder, selector, numeric_columns, categorical_columns = preprocess_data(data)
        
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.3, random_state=42
        )
        
        param_grid = {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }
        
        knn_model = KNeighborsClassifier()
        grid_search = GridSearchCV(knn_model, param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.logger.info("KNN Accuracy: {:.2f}%".format(accuracy * 100))
        
        bundle = {
            'model': best_model,
            'selector': selector,
            'encoder': encoder,
            'imputer': imputer,
            'scaler': scaler,
            'accuracy': accuracy,
            'numeric_columns': numeric_columns,
            'categorical_columns': categorical_columns
        }
        
        output_path = os.path.join(self.models_dir, 'knn_model_bundle.pkl')
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.logger.info("✓ KNN salvo em: {}".format(output_path))
        except Exception as e:
            self.logger.error("Erro ao salvar KNN: {}".format(e))
            raise

    def _train_random_forest(self):
        """Treina modelo Random Forest"""
        df = pd.read_csv(self.train_file, sep=",", encoding='utf-8-sig')
        data = df.groupby("label", group_keys=False).apply(lambda x:x.sample(min(len(x), 2000)))
        X, y, imputer, scaler, encoder, selector, numeric_columns, categorical_columns = preprocess_data(data)
        
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.3, random_state=42
        )
        
        rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=10,
            min_samples_leaf=5, random_state=42, n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.logger.info("Random Forest Accuracy: {:.2f}%".format(accuracy * 100))
        
        bundle = {
            'model': rf_model,
            'selector': selector,
            'encoder': encoder,
            'imputer': imputer,
            'scaler': scaler,
            'accuracy': accuracy,
            'numeric_columns': numeric_columns,
            'categorical_columns': categorical_columns
        }
        
        with open(os.path.join(self.models_dir, 'randomforest_model_bundle.pkl'), 'wb') as f:
            pickle.dump(bundle, f)

    def _train_decision_tree(self):
        """Treina modelo Decision Tree"""
        df = pd.read_csv(self.train_file, sep=",", encoding='utf-8-sig')
        data = df.groupby("label", group_keys=False).apply(lambda x:x.sample(min(len(x), 2000)))
        X, y, imputer, scaler, encoder, selector, numeric_columns, categorical_columns = preprocess_data(data)
        
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.3, random_state=42
        )
        
        dt_model = DecisionTreeClassifier(
            max_depth=20, min_samples_split=10, 
            min_samples_leaf=5, random_state=42
        )
        dt_model.fit(X_train, y_train)
        
        y_pred = dt_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.logger.info("Decision Tree Accuracy: {:.2f}%".format(accuracy * 100))
        
        bundle = {
            'model': dt_model,
            'selector': selector,
            'encoder': encoder,
            'imputer': imputer,
            'scaler': scaler,
            'accuracy': accuracy,
            'numeric_columns': numeric_columns,
            'categorical_columns': categorical_columns
        }
        
        with open(os.path.join(self.models_dir, 'dt_model_bundle.pkl'), 'wb') as f:
            pickle.dump(bundle, f)

    def _train_svm(self):
        """Treina modelo SVM"""
        df = pd.read_csv(self.train_file, sep=",", encoding='utf-8-sig')
        data = df.groupby("label", group_keys=False).apply(lambda x:x.sample(min(len(x), 2000)))
        X, y, imputer, scaler, encoder, selector, numeric_columns, categorical_columns = preprocess_data(data)
        
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.3, random_state=42
        )
        
        # Limitar para datasets grandes
        if len(X_train) > 30000:
            indices = np.random.choice(len(X_train), 30000, replace=False)
            X_train = X_train.iloc[indices] if hasattr(X_train, 'iloc') else X_train[indices]
            y_train = y_train.iloc[indices] if hasattr(y_train, 'iloc') else y_train[indices]
        
        svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        svm_model.fit(X_train, y_train)
        
        y_pred = svm_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.logger.info("SVM Accuracy: {:.2f}%".format(accuracy * 100))
        
        bundle = {
            'model': svm_model,
            'selector': selector,
            'encoder': encoder,
            'imputer': imputer,
            'scaler': scaler,
            'accuracy': accuracy,
            'numeric_columns': numeric_columns,
            'categorical_columns': categorical_columns
        }
        
        with open(os.path.join(self.models_dir, 'svm_model_bundle.pkl'), 'wb') as f:
            pickle.dump(bundle, f)

    def predict_all_models(self, data):
        """Faz predição com todos os modelos"""
        predictions = {}
        ddos_flows = {}
        
        try:
            predictions['knn'], ddos_flows['knn'] = predict_knn(
                self.models['knn'], data
            )
        except Exception as e:
            self.logger.error("Erro em KNN: {}".format(e))
            predictions['knn'] = np.array([])
        
        try:
            predictions['random_forest'], ddos_flows['random_forest'] = predict_random_forest(
                self.models['random_forest'], data
            )
        except Exception as e:
            self.logger.error("Erro em Random Forest: {}".format(e))
            predictions['random_forest'] = np.array([])
        
        try:
            predictions['decision_tree'], ddos_flows['decision_tree'] = predict_decision_tree(
                self.models['decision_tree'], data
            )
        except Exception as e:
            self.logger.error("Erro em Decision Tree: {}".format(e))
            predictions['decision_tree'] = np.array([])
        
        try:
            predictions['svm'], ddos_flows['svm'] = predict_svm(
                self.models['svm'], data
            )
        except Exception as e:
            self.logger.error("Erro em SVM: {}".format(e))
            predictions['svm'] = np.array([])
        
        return predictions, ddos_flows

    def weighted_vote(self, predictions):
        """Votação ponderada baseada na acurácia"""
        if not predictions:
            return []
        
        # Filtra predições vazias
        valid_predictions = {k: v for k, v in predictions.items() if len(v) > 0}
        
        if not valid_predictions:
            return []
        
        num_samples = len(list(valid_predictions.values())[0])
        weighted_votes = {}
        
        for model_name, pred in valid_predictions.items():
            weight = self.accuracies.get(model_name, 1.0)
            for i, p in enumerate(pred):
                if i not in weighted_votes:
                    weighted_votes[i] = 0
                weighted_votes[i] += p * weight
        
        final_predictions = []
        for i in range(num_samples):
            final_predictions.append(1 if weighted_votes.get(i, 0) > 0.5 else 0)
        
        return final_predictions

    def _backup_old_traffic(self):
        """Backup de tráfego antigo"""
        if os.path.exists(self.filename):
            df = pd.read_csv(self.filename)
            if not df.empty:
                if os.path.exists(self.processed_file):
                    df.to_csv(self.processed_file, mode='a', index=False, header=False)
                else:
                    df.to_csv(self.processed_file, index=False)
                df.head(0).to_csv(self.filename, index=False)

    def _initialize_csv(self):
        """Inicializa arquivo CSV"""
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as csvfile:
                fieldnames = ['time', 'dpid', 'in_port', 'eth_src', 'eth_dst', 'packets', 'bytes', 'duration_sec']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

    def predict_traffic(self):
        """Predição de tráfego malicioso"""
        try:
            if not os.path.exists(self.filename):
                self.logger.debug("CSV não existe ainda")
                return
            
            df = pd.read_csv(self.filename)
            if df.empty:
                self.logger.debug("Nenhum dado para predição")
                return

            df_unprocessed = df[df['time'] > self.last_processed_time].copy()
            
            if df_unprocessed.empty:
                self.logger.debug("Nenhum fluxo novo para processar")
                return

            processing_start_time = time.time()
            
            temp_filename = 'temp_predict.csv'
            df_unprocessed.to_csv(temp_filename, index=False)
            
            self.logger.info("Processando {} NOVOS fluxos".format(len(df_unprocessed)))
            
            predictions, features = self.predict_all_models(temp_filename)
            
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            
            final_predictions = self.weighted_vote(predictions)
            
            for model_name, pred in predictions.items():
                if len(pred) > 0:
                    malicious = sum(pred)
                    self.logger.info("Modelo {}: {} maliciosos de {}".format(
                        model_name, malicious, len(pred)
                    ))
            
            legitimate_traffic = 0
            ddos_traffic = 0
            
            for i, pred in enumerate(final_predictions):
                if i >= len(df_unprocessed):
                    break
                
                if pred == 0:
                    legitimate_traffic += 1
                else:
                    ddos_traffic += 1
                    row = df_unprocessed.iloc[i]
                    dpid = int(row['dpid']) if not pd.isna(row['dpid']) else None
                    eth_src = row['eth_src']
                    eth_dst = row['eth_dst']
                    in_port = int(row['in_port']) if not pd.isna(row['in_port']) else None

                    datapath = self.datapaths.get(dpid)
                    if datapath and in_port is not None:
                        self.block_traffic(datapath, eth_src, eth_dst, in_port)
                        self.logger.warning("MALICIOSO BLOQUEADO: src={}, dst={}, dpid={}".format(
                            eth_src, eth_dst, dpid
                        ))
            
            self.last_processed_time = processing_start_time
            
            total = legitimate_traffic + ddos_traffic
            if total > 0:
                self.logger.info("RESULTADO: {} legítimos, {} DDoS ({:.1f}% maliciosos)".format(
                    legitimate_traffic, ddos_traffic, (ddos_traffic/total)*100
                ))
            
        except Exception as e:
            self.logger.error("Erro na predição: {}".format(e))
            import traceback
            self.logger.error(traceback.format_exc())

    def is_high_volume(self, packets, bytes, duration_sec):
        """Verifica alto volume"""
        packets_per_sec = packets / duration_sec if duration_sec > 0 else 0
        bytes_per_sec = bytes / duration_sec if duration_sec > 0 else 0
        return packets_per_sec > 10000 or bytes_per_sec > 100000000

    def is_long_connection(self, duration_sec):
        """Verifica conexão longa"""
        return duration_sec > 3600

    def is_invalid_mac(self, mac):
        """Verifica MAC inválido"""
        invalid_patterns = [
            r"00:00:00:00:00:00",
            r"([0-9A-Fa-f]{2}:)\1{5}"
        ]
        for pattern in invalid_patterns:
            if re.match(pattern, mac, re.IGNORECASE):
                return True
        return False

    def block_traffic(self, datapath, eth_src, eth_dst, in_port):
        """Bloqueia tráfego malicioso"""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch(in_port=in_port, eth_src=eth_src, eth_dst=eth_dst)
        actions = []
        self.add_flow(datapath, 100, match, actions, idle=60, hard=120)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            self.logger.info('Registrando datapath: %016x', datapath.id if datapath.id else 0)
            self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                del self.datapaths[datapath.id]

    def _monitor(self):
        """Thread de monitoramento"""
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(10)
            self.predict_traffic()

    def _request_stats(self, datapath):
        """Requisita estatísticas"""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        """Handler de estatísticas de fluxo"""
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

                if (self.is_high_volume(packets, bytes, duration_sec) or
                    self.is_long_connection(duration_sec) or
                    self.is_invalid_mac(eth_src) or
                    self.is_invalid_mac(eth_dst)):
                    
                    self.block_traffic(ev.msg.datapath, eth_src, eth_dst, in_port)
                    self.logger.warning("Tráfego suspeito bloqueado: src={}, dst={}".format(
                        eth_src, eth_dst
                    ))
                    continue

                writer.writerow({
                    'time': timestamp,
                    'dpid': ev.msg.datapath.id,
                    'in_port': in_port,
                    'eth_src': eth_src,
                    'eth_dst': eth_dst,
                    'packets': packets,
                    'bytes': bytes,
                    'duration_sec': duration_sec
                })

    @set_ev_cls(event.EventSwitchEnter)
    def switch_enter_handler(self, ev):
        switch = ev.switch
        self.logger.info('Switch conectado: %016x', switch.dp.id if switch.dp.id else 0)
        self.install_default_flows(switch.dp)

    def add_flow(self, datapath, priority, match, actions, buffer_id=None, idle=0, hard=0):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id, 
                                   idle_timeout=idle, hard_timeout=hard, 
                                   priority=priority, match=match, instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                   idle_timeout=idle, hard_timeout=hard, 
                                   match=match, instructions=inst)
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        self.install_default_flows(datapath)

    def install_default_flows(self, datapath):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, 
                                         ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth_pkt = pkt.get_protocols(ethernet.ethernet)[0]

        if eth_pkt.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        dst = eth_pkt.dst
        src = eth_pkt.src
        dpid = format(datapath.id, "d").zfill(16)
        
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
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