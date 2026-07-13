# Comprehensive System Component Ablation Study

| Configuration Ablated | Avg Latency (s) | Avg Confidence | Avg Citations | Uncertainty Triggers | Success Rate |
| --- | --- | --- | --- | --- | --- |
| Full VYOR System (Control) | 14.179s | 51.40% | 0.6 | 2 | 100.00% |
| - Surprise Gate | 24.602s | 52.00% | 0.4 | 2 | 100.00% |
| - Titans LTM | 28.574s | 30.00% | 0.4 | 4 | 100.00% |
| - Critic Agent | 12.871s | 100.00% | 0.4 | 0 | 100.00% |
| - Debate Loop | 9.468s | 100.00% | 0.2 | 0 | 100.00% |
| - Hybrid Search | 7.848s | 100.00% | 0.6 | 0 | 100.00% |
| - RRF Fusion | 9.864s | 100.00% | 0.6 | 0 | 100.00% |
| - Time Decay | 12.453s | 100.00% | 0.6 | 0 | 100.00% |
| - Persistent Tokens | 11.954s | 100.00% | 0.4 | 0 | 100.00% |
