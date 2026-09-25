"""Engine-facing scripts for explicit execution-host research steps.

This package is intentionally separate from medium_pc_audit. The
medium_pc_audit package defines deterministic control, analysis, provenance,
validation, and evidence contracts; subprocess-based engine work belongs here
so the boundary is structural rather than an informal convention.

Importing this package does not start engines. Tests may import pure parsing
or validation helpers, but deploying and running an engine-facing script is a
separate, explicit execution step on the chosen execution host.
"""
