# AgentProxy OTEL Stack Example

This directory contains a complete observability stack for AgentProxy using OpenTelemetry, Tempo, Prometheus, and Grafana.

## Quick Start

1. **Start the stack**:
   ```bash
   docker-compose up -d
   ```

2. **Enable telemetry in AgentProxy**:
   ```bash
   export AGENTPROXY_ENABLE_TELEMETRY=1
   export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
   export OTEL_SERVICE_NAME=agentproxy
   export OTEL_SERVICE_NAMESPACE=dev
   ```

3. **Run AgentProxy**:
   ```bash
   pa "Create a hello world script"
   ```

4. **View telemetry**:
   - Grafana: http://localhost:3000 (no login required)
   - Prometheus: http://localhost:9090
   - Tempo: http://localhost:3200

## Stack Components

### OTEL Collector (Port 4317, 4318, 8889)
- Receives telemetry from AgentProxy via OTLP
- Routes traces to Tempo
- Exports metrics to Prometheus

### Tempo (Port 3200)
- Stores distributed traces
- Provides trace search and visualization
- Linked from Grafana

### Prometheus (Port 9090)
- Scrapes metrics from OTEL Collector
- Time-series storage
- Query interface

### Grafana (Port 3000)
- Unified visualization dashboard
- Pre-configured datasources for Prometheus and Tempo
- Pre-loaded AgentProxy dashboard

## What You'll See

### Traces
- `pa.run_task` - Full task execution trace
  - `pa.reasoning_loop` - PA decision-making spans
    - `gemini.api.call` - Gemini API calls
  - `claude.subprocess` - Claude Code execution spans
  - `pa.function.*` - Function execution (verify, test, etc.)

### Metrics
- Task counts and durations
- PA decision distributions
- Verification pass/fail rates
- Active session counts
- API call latencies

## Configuration

All services are configured via YAML files in this directory:
- `docker-compose.yml` - Stack orchestration
- `otel-collector-config.yaml` - OTEL routing
- `tempo.yaml` - Trace storage
- `prometheus.yml` - Metric scraping
- `grafana/provisioning/` - Auto-configured datasources and dashboards

## Customization

### Change Retention
Edit `tempo.yaml`:
```yaml
compactor:
  compaction:
    block_retention: 24h  # Change from 1h to 24h
```

### Add More Metrics
Edit `prometheus.yml` to add scrape targets:
```yaml
scrape_configs:
  - job_name: 'my-app'
    static_configs:
      - targets: ['my-app:8080']
```

### Custom Dashboard
1. Create dashboard in Grafana UI
2. Export JSON
3. Save to `grafana/dashboards/my-dashboard.json`
4. Restart Grafana: `docker-compose restart grafana`

## Troubleshooting

### No traces appearing
1. Check OTEL collector logs: `docker-compose logs otel-collector`
2. Verify endpoint: `echo $OTEL_EXPORTER_OTLP_ENDPOINT`
3. Verify telemetry enabled: `echo $AGENTPROXY_ENABLE_TELEMETRY`

### No metrics appearing
1. Check Prometheus targets: http://localhost:9090/targets
2. Check OTEL collector metrics endpoint: http://localhost:8889/metrics
3. Verify scrape config in `prometheus.yml`

### Grafana not connecting to datasources
1. Check datasource config: `cat grafana/provisioning/datasources/datasources.yaml`
2. Verify network connectivity: `docker-compose exec grafana ping prometheus`

## Stopping the Stack

```bash
# Stop but keep data
docker-compose down

# Stop and remove data
docker-compose down -v
```

## Production Notes

This example stack is for **development and testing only**. For production:

1. **Use external storage** - Configure Tempo with S3/GCS backend
2. **Add authentication** - Enable Grafana auth and OTEL collector auth
3. **Scale collectors** - Run multiple OTEL collectors behind load balancer
4. **Configure retention** - Adjust based on your data volume
5. **Add alerting** - Configure Prometheus alertmanager
6. **Monitor the stack** - Add health checks and monitoring for OTEL components

## Further Reading

- [OpenTelemetry Docs](https://opentelemetry.io/docs/)
- [Tempo Docs](https://grafana.com/docs/tempo/latest/)
- [Prometheus Docs](https://prometheus.io/docs/)
- [Grafana Docs](https://grafana.com/docs/grafana/latest/)
