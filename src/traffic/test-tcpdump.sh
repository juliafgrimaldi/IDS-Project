#!/bin/bash

# Define o arquivo CSV de saída
CSV_FILE="network_traffic.csv"

# Cabeçalho do arquivo CSV
echo "timestamp,source_ip,destination_ip,protocol,source_port,destination_port,length" > $CSV_FILE

# Captura o tráfego de rede e processa com tcpdump
sudo tcpdump -i any -l -nn ip | while read line
do
    # Extrai os campos necessários
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    SRC_IP=$(echo $line | awk '{print $3}' | cut -d '.' -f 1-4)
    DST_IP=$(echo $line | awk '{print $5}' | cut -d '.' -f 1-4)
    PROTOCOL=$(echo $line | awk '{print $6}' | tr -d ',')
    SRC_PORT=$(echo $line | awk '{print $3}' | awk -F. '{print $(NF-1)}')
    DST_PORT=$(echo $line | awk '{print $5}' | awk -F. '{print $(NF-1)}')
    LENGTH=$(echo $line | awk '{print $8}')
    
    # Grava os dados no arquivo CSV
    echo "$TIMESTAMP,$SRC_IP,$DST_IP,$PROTOCOL,$SRC_PORT,$DST_PORT,$LENGTH" >> $CSV_FILE
done
