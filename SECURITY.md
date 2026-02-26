# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Please report security vulnerabilities to the AumOS security team:

**Email:** security@aumos.io

### What to Include

- Description of the vulnerability and its potential impact
- Steps to reproduce (proof-of-concept if possible)
- Affected versions
- Any suggested mitigations

### Response Timeline

- **Acknowledgement:** Within 48 hours
- **Initial assessment:** Within 5 business days
- **Fix + coordinated disclosure:** Within 90 days for critical issues

We follow responsible disclosure principles and will coordinate with you before
publishing any details publicly.

## Security Considerations for aumos-fairness-suite

### Protected Attribute Handling

- Protected attribute names and group memberships are tenant-scoped configuration.
  A tenant's protected attribute definitions are never exposed to another tenant.
- Protected attribute values from assessment datasets are never persisted in the
  database — only aggregate group statistics (counts, rates) are stored.
- Audit logs for all assessments and mitigation jobs are retained per your tenant's
  data retention policy.

### Regulatory Report Security

- Fairness reports generated for regulatory submission are immutable once created.
  Any subsequent assessment creates a new report rather than modifying an existing one.
- Report endpoints require the same JWT authentication as all other endpoints.
  Reports contain only aggregate statistics, not individual-level data.

### Multi-Tenancy

- Row-level security (RLS) is enforced at the database layer for all `fai_` tables.
- Never expose one tenant's assessment results, monitor configurations, or
  mitigation jobs to another tenant.
- All API endpoints require a valid JWT with the correct tenant claim.

### Model and Dataset References

- `model_id` and `dataset_id` fields are opaque UUID references to objects owned
  by other AumOS services. This service never fetches raw model weights or dataset
  rows on behalf of one tenant using another tenant's credentials.
- Mitigation strategies that require dataset access receive data payloads inline
  in the API request — they are not fetched from external storage by this service.

### Dependency Integrity

- AI Fairness 360 (Apache-2.0) and Fairlearn (MIT) are the approved bias detection
  libraries. Do not substitute alternative bias libraries without a license review.
- All dependencies are pinned in `pyproject.toml`. Pin updates require a security
  review for any dependency with filesystem or network access.
