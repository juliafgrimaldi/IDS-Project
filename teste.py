import requests
import json
import sys

def test_ryu_api():
    
    base_url = "http://127.0.0.1:8080"
    
    print("="*60)
    print("TESTANDO API REST DO RYU")
    print("="*60)
    
    # 1. Listar switches
    print("\n1. SWITCHES ATIVOS:")
    print("-"*60)
    try:
        response = requests.get(f"{base_url}/stats/switches", timeout=5)
        switches = response.json()
        print(f"Switches: {switches}")
        
        if not switches:
            print("\n✗ NENHUM SWITCH CONECTADO!")
            print("\nVerifique:")
            print("  1. O Ryu controller esta rodando?")
            print("  2. O Mininet esta conectado ao Ryu?")
            print("  3. A porta esta correta? (6653 ou 6633)")
            return
        
    except requests.exceptions.ConnectionError:
        print("\n✗ ERRO: Nao foi possivel conectar a API do Ryu")
        print("\nVerifique:")
        print("  1. O Ryu esta rodando com ofctl_rest?")
        print("     ryu-manager ryu.app.ofctl_rest seu_controller.py")
        print("  2. A porta e 8080? (padrao)")
        return
    except Exception as e:
        print(f"\n✗ Erro: {e}")
        return
    
    for dpid in switches:
        print(f"\n2. FLOWS DO SWITCH {dpid}:")
        print("-"*60)
        
        try:
            response = requests.get(f"{base_url}/stats/flow/{dpid}", timeout=5)
            data = response.json()
            
            flows = data.get(str(dpid), [])
            print(f"Total de flows: {len(flows)}\n")
            
            if not flows:
                print("✗ Nenhum flow encontrado!")
                continue
            
            # Analisar primeiro flow em detalhe
            print("PRIMEIRO FLOW (completo):")
            print(json.dumps(flows[0], indent=2))
            
            print("\n" + "-"*60)
            print("ANaLISE DO MATCH:")
            print("-"*60)
            
            for idx, flow in enumerate(flows[:5], 1):
                match = flow.get('match', {})
                packet_count = flow.get('packet_count', 0)
                byte_count = flow.get('byte_count', 0)
                
                print(f"\nFlow #{idx}:")
                print(f"  Packets: {packet_count}, Bytes: {byte_count}")
                print(f"  Match fields:")
                
                if not match:
                    print("    (match vazio)")
                else:
                    for key, value in match.items():
                        print(f"    {key}: {value}")
            
            # Estatísticas de campos
            print("\n" + "-"*60)
            print("ESTATiSTICAS DOS CAMPOS:")
            print("-"*60)
            
            field_stats = {}
            for flow in flows:
                match = flow.get('match', {})
                for field in match.keys():
                    field_stats[field] = field_stats.get(field, 0) + 1
            
            print(f"\nCampos encontrados em {len(flows)} flows:")
            for field, count in sorted(field_stats.items()):
                percentage = (count / len(flows)) * 100
                print(f"  {field:20s}: {count:4d} flows ({percentage:.1f}%)")
            
            # Verificar se tem campos de IP
            print("\n" + "-"*60)
            print("VERIFICACAO DE CAMPOS IP:")
            print("-"*60)
            
            ip_fields = [
                'ipv4_src', 'ipv4_dst', 'nw_src', 'nw_dst',
                'ip_src', 'ip_dst', 'IPv4_src', 'IPv4_dst'
            ]
            
            found_ip_fields = [f for f in ip_fields if f in field_stats]
            
            if found_ip_fields:
                print("✓ Campos IP encontrados:")
                for field in found_ip_fields:
                    print(f"  - {field} ({field_stats[field]} flows)")
            else:
                print("✗ NENHUM CAMPO IP ENCONTRADO!")
                print("\nIsso significa que os flows sao de camada 2 (Ethernet).")
                print("\nCampos presentes (provavelmente MAC):")
                for field in field_stats.keys():
                    print(f"  - {field}")
            
            # Verificar eth_type
            print("\n" + "-"*60)
            print("DISTRIBUICAO DE eth_type:")
            print("-"*60)
            
            eth_types = {}
            for flow in flows:
                match = flow.get('match', {})
                eth_type = match.get('eth_type', match.get('dl_type', 'N/A'))
                eth_types[eth_type] = eth_types.get(eth_type, 0) + 1
            
            for eth_type, count in sorted(eth_types.items()):
                eth_type_name = {
                    2048: 'IPv4 (0x0800)',
                    2054: 'ARP (0x0806)',
                    34525: 'IPv6 (0x86dd)',
                    '0x0800': 'IPv4',
                    '0x0806': 'ARP',
                    '0x86dd': 'IPv6',
                    'N/A': 'Não especificado'
                }.get(eth_type, f'Desconhecido ({eth_type})')
                
                percentage = (count / len(flows)) * 100
                print(f"  {eth_type_name:20s}: {count:4d} flows ({percentage:.1f}%)")
        
        except Exception as e:
            print(f"✗ Erro ao buscar flows: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("RESUMO E RECOMENDACOES:")
    print("="*60)
    
    if not found_ip_fields:
        print("PROBLEMA: Flows nao tem campos de IP!")
    else:
        print("\n✓ Campos IP encontrados! O problema pode estar na extracao.")
        print("\nVerifique se o controller esta usando os nomes corretos:")
        for field in found_ip_fields:
            print(f"  - {field}")


if __name__ == "__main__":
    test_ryu_api()