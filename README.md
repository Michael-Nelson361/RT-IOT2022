# RT-IoT2022
## Project Description


## Initial Hypotheses


## Data Dictionary
| # | Feature | Type | Data Type | Description |
|---|---------|------|-----------|-------------|
| 1 | id.orig_p | Integer | Integer | The network port used by the origin source |
| 2 | id.resp_p | Integer | Integer | The network port used by the responding device |
| 3 | proto | Categorical | Object | The protocol utilized in the connection |
| 4 | service | Continuous | Object | The type of service used in the connection |
| 5 | flow_duration | Continuous | Float | How long the connection lasted between devices |
| 6 | fwd_pkts_tot | Integer | Integer | Total packets forwarded in the session |
| 7 | bwd_pkts_tot | Integer | Integer | Total packets backward (returned) in the session |
| 8 | fwd_data_pkts_tot | Integer | Integer | Total data packets forwarded in the session (excluding control packets) |
| 9 | bwd_data_pkts_tot | Integer | Integer | Total data packets backward in the session (excluding control packets) |
| 10 | fwd_pkts_per_sec | Continuous | Float | Rate of packets forwarded per second |
| 11 | bwd_pkts_per_sec | Continuous | Float | Rate of packets backward per second |
| 12 | flow_pkts_per_sec | Continuous | Float | Combined rate of packets per second |
| 13 | down_up_ratio | Continuous | Float | Ratio of downstream to upstream traffic |
| 14 | fwd_header_size_tot | Integer | Integer | Total size of the header forwarded |
| 15 | fwd_header_size_min | Integer | Integer | Smallest header size sent |
| 16 | fwd_header_size_max | Integer | Integer | Largest header size sent |
| 17 | bwd_header_size_tot | Integer | Integer | Total size of the header returned |
| 18 | bwd_header_size_min | Integer | Integer | Smallest header size returned |
| 19 | bwd_header_size_max | Integer | Integer | Largest header size returned |
| 20 | flow_FIN_flag_count | Integer | Integer | Count of FIN flags signaling the end of data transmission in a session |
| 21 | flow_SYN_flag_count | Integer | Integer | Count of SYN flags used to initiate and establish sessions |
| 22 | flow_RST_flag_count | Integer | Integer | Count of RST flags used to abruptly terminate sessions |
| 23 | fwd_PSH_flag_count | Integer | Integer | Count of PSH flags in forward packets indicating the push of buffered data to the receiving application |
| 24 | bwd_PSH_flag_count | Integer | Integer | Count of PSH flags in backward packets indicating the push of buffered data to the sending application |
| 25 | flow_ACK_flag_count | Integer | Integer | Count of ACK flags used to acknowledge the receipt of packets |
| 26 | fwd_URG_flag_count | Integer | Integer | Count of URG flags in forwarded packets indicating data should be processed urgently |
| 27 | bwd_URG_flag_count | Integer | Integer | Count of URG flags in backward packets indicating data should be processed urgently |
| 28 | flow_CWR_flag_count | Integer | Integer | Count of CWR (Congestion Window Reduced) flags used by the sender to signal congestion control |
| 29 | flow_ECE_flag_count | Integer | Integer | Count of ECE (ECN Echo) flags indicating network congestion without dropping packets |
| 30 | fwd_pkts_payload.min | Continuous | Integer | Minimum payload size in forwarded packets |
| 31 | fwd_pkts_payload.max | Continuous | Integer | Maximum payload size in forwarded packets |
| 32 | fwd_pkts_payload.tot | Continuous | Integer | Total payload size in forwarded packets |
| 33 | fwd_pkts_payload.avg | Continuous | Float | Average payload size in forwarded packets |
| 34 | fwd_pkts_payload.std | Continuous | Float | Standard deviation of payload sizes in forwarded packets |
| 35 | bwd_pkts_payload.min | Continuous | Integer | Minimum payload size in backward packets |
| 36 | bwd_pkts_payload.max | Continuous | Integer | Maximum payload size in backward packets |
| 37 | bwd_pkts_payload.tot | Continuous | Integer | Total payload size in backward packets |
| 38 | bwd_pkts_payload.avg | Continuous | Float | Average payload size in backward packets |
| 39 | bwd_pkts_payload.std | Continuous | Float | Standard deviation of payload sizes in backward packets |
| 40 | flow_pkts_payload.min | Continuous | Integer | Minimum payload size in the flow |
| 41 | flow_pkts_payload.max | Continuous | Integer | Maximum payload size in the flow |
| 42 | flow_pkts_payload.tot | Continuous | Integer | Total payload size in the flow |
| 43 | flow_pkts_payload.avg | Continuous | Float | Average payload size in the flow |
| 44 | flow_pkts_payload.std | Continuous | Float | Standard deviation of payload sizes in the flow |
| 45 | fwd_iat.min | Continuous | Float | Minimum inter-arrival time of forwarded packets |
| 46 | fwd_iat.max | Continuous | Float | Maximum inter-arrival time of forwarded packets |
| 47 | fwd_iat.tot | Continuous | Float | Total inter-arrival time of forwarded packets |
| 48 | fwd_iat.avg | Continuous | Float | Average inter-arrival time of forwarded packets |
| 49 | fwd_iat.std | Continuous | Float | Standard deviation of inter-arrival times of forwarded packets |
| 50 | bwd_iat.min | Continuous | Float | Minimum inter-arrival time of backward packets |
| 51 | bwd_iat.max | Continuous | Float | Maximum inter-arrival time of backward packets |
| 52 | bwd_iat.tot | Continuous | Float | Total inter-arrival time of backward packets |
| 53 | bwd_iat.avg | Continuous | Float | Average inter-arrival time of backward packets |
| 54 | bwd_iat.std | Continuous | Float | Standard deviation of inter-arrival times of backward packets |
| 55 | flow_iat.min | Continuous | Float | Minimum inter-arrival time in the flow |
| 56 | flow_iat.max | Continuous | Float | Maximum inter-arrival time in the flow |
| 57 | flow_iat.tot | Continuous | Float | Total inter-arrival time in the flow |
| 58 | flow_iat.avg | Continuous | Float | Average inter-arrival time in the flow |
| 59 | flow_iat.std | Continuous | Float | Standard deviation of inter-arrival times in the flow |
| 60 | payload_bytes_per_second | Continuous | Float | Rate of payload transmission in bytes per second |
| 61 | fwd_subflow_pkts | Continuous | Float | Forward subflow packet count |
| 62 | bwd_subflow_pkts | Continuous | Float | Backward subflow packet count |
| 63 | fwd_subflow_bytes | Continuous | Float | Forward subflow byte count |
| 64 | bwd_subflow_bytes | Continuous | Float | Backward subflow byte count |
| 65 | fwd_bulk_bytes | Continuous | Float | Forward bulk byte count |
| 66 | bwd_bulk_bytes | Continuous | Float | Backward bulk byte count |
| 67 | fwd_bulk_packets | Continuous | Float | Forward bulk packet count |
| 68 | bwd_bulk_packets | Continuous | Float | Backward bulk packet count |
| 69 | fwd_bulk_rate | Continuous | Float | Rate of bulk data transmission in the forward direction |
| 70 | bwd_bulk_rate | Continuous | Float | Rate of bulk data transmission in the backward direction |
| 71 | active.min | Continuous | Float | Minimum time the flow was active before going idle |
| 72 | active.max | Continuous | Float | Maximum time the flow was active before going idle |
| 73 | active.tot | Continuous | Float | Total time the flow was active before going idle |
| 74 | active.avg | Continuous | Float | Average time the flow was active before going idle |
| 75 | active.std | Continuous | Float | Standard deviation of active times before the flow went idle |
| 76 | idle.min | Continuous | Float | Minimum time the flow was idle |
| 77 | idle.max | Continuous | Float | Maximum time the flow was idle |
| 78 | idle.tot | Continuous | Float | Total time the flow was idle |
| 79 | idle.avg | Continuous | Float | Average time the flow was idle |
| 80 | idle.std | Continuous | Float | Standard deviation of idle times |
| 81 | fwd_init_window_size | Integer | Integer | Initial window size in forwarded TCP connections, indicative of congestion handling |
| 82 | bwd_init_window_size | Integer | Integer | Initial window size in backward TCP connections, indicative of congestion handling |
| 83 | fwd_last_window_size | Integer | Integer | Last window size observed in forwarded TCP connections, can indicate changes in network conditions |
| 84 | Attack_type | Categorical | Object | Specifies the type of network traffic, whether normal or related to specific types of network attacks |


## Project Pipeline
1. Planning
2. Acquisition and Preparation (Wrangling)
3. Exploration
4. Pre-Processing and Modeling
5. Delivery

## Reproduction of Findings
1. Clone this repository.
2. Install the UC Irvine Python package:
```
pip install ucimlrepo
```
3. Run the notebooks

## Key Findings