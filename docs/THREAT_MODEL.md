# Proof-of-Concept Threat Model

This is a research threat model, not a compliance certification.

## Assets

- Raw industrial/sensor data.
- Feature and target names that may reveal proprietary process context.
- Problem descriptions entered by operators/researchers.
- LLM API credentials.
- Model artifacts and experiment logs.
- Structured LLM recommendations that can influence model configuration.

## Trust boundaries

1. **Local data boundary:** raw rows, train/validation/test splits, perturbations, quality diagnostics and H2O execution remain local to the research runtime.
2. **External LLM boundary:** only a constrained metadata summary is sent when OpenAI mode is enabled: problem text, sanitized predictor names, class prevalence, aggregate quality indicators and validation metrics during a retry.
3. **Execution boundary:** LLM output cannot execute code directly. Pydantic schema validation and an allowlist adapter translate only recognized fields into ML configuration.

## Threats and controls

| Threat | Control in this prototype | Remaining limitation |
|---|---|---|
| Raw data exfiltration | Raw dataframe rows are not included in LLM prompts. | Feature names and problem text may still contain proprietary context. |
| Prompt injection in uploaded data | Uploaded cell contents are not inserted into LLM prompts. | User-entered problem descriptions remain untrusted text and should be sanitized/controlled in production. |
| Malformed or unexpected LLM output | Structured output parsed against `PlannerDecision`; extra fields are forbidden. | Provider/API failures can still occur and must be handled operationally. |
| Arbitrary configuration/code execution | `decision_to_execution_config()` accepts only allowlisted algorithms, metrics and predictor names. | Production systems need stronger sandboxing and authorization. |
| Credential leakage | API key is read from environment; no key is stored in repository code. | Secret-management infrastructure is outside this proof of concept. |
| Sensitive logs | Experiment outputs contain aggregate metrics/configuration metadata, not raw rows. | Organizations must define retention/access-control policies. |
| Model/provider updates | Exact model ID and run date should be archived with results. | Hosted provider behavior can change over time; archived outputs are therefore essential. |
| Unauthorized human-facing action | Framework only recommends/exports model artifacts; it does not actuate machinery. | Production integration with MES/SCADA would require authentication, authorization, change control, rollback and safety interlocks. |

## Claims deliberately avoided

The architecture reduces unnecessary external data exposure but does **not** by itself establish GDPR, SOC 2, ISO 27001, HIPAA, PCI or other regulatory/compliance status. Those depend on the complete organizational deployment, contracts, retention controls, access policies and infrastructure.
